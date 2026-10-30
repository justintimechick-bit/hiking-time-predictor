"""Plan a hike: upload a route, get one clear answer, then the details behind it."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json

import streamlit as st
from streamlit_folium import st_folium

from .cloud_repository import private_request_message
from .gpx import FEET_PER_METER, distance_meters
from .model_status import moving_status
from .solar import sunset
from .ui_format import (DIRECTION_LABELS, HIKE_TYPE_LABELS, RULE_DEFINITION, RULE_NAME, clock, feet,
                        humanize_note, md_escape, method_name, miles, pace, route_extremes)
from .ui_nav import flash, nav_button
from .web_charts import elevation_figure, segment_csv, segment_figure, time_along_route_figure, trail_map
from .web_components import duration, plot
from .web_upload import predict_upload, preview_upload, trip_statistics

BREAK_MODES = ('usual', 'custom', 'none')
BREAK_MODE_LABELS = {'usual': 'My usual', 'custom': 'Custom', 'none': 'None'}
GPX_HELP = ('Export a route as GPX from AllTrails, Gaia GPS, CalTopo, Komoot or onX. It needs elevation; '
            'timestamps are optional. Max 20 MB.')


def _on_route_change():
    # Removing the file with the uploader's x clears the remembered route too.
    if st.session_state.get('planned_gpx') is None:
        for key in ('trail_upload', 'cloud_route', 'cloud_route_key', 'prediction', 'prediction_key'):
            st.session_state.pop(key, None)


def _clear_route():
    for key in ('trail_upload', 'cloud_route', 'cloud_route_key', 'prediction', 'prediction_key'):
        st.session_state.pop(key, None)


def _prefill_breaks():
    if st.session_state.get('trail_break_mode') == 'custom' and st.session_state.get('trail_break_minutes') is None:
        prediction = st.session_state.get('prediction')
        predicted = prediction.break_forecast.predicted_minutes if prediction and prediction.break_forecast else None
        st.session_state['trail_break_minutes'] = int(5 * round((predicted or 30) / 5))


def _route_input():
    """(filename, bytes) of the route to plan, or None. Remembers the last route across pages."""
    uploaded = st.file_uploader('Planned route (GPX)', type=['gpx'], key='planned_gpx', max_upload_size=20,
                                help=GPX_HELP, on_change=_on_route_change)
    if uploaded is not None:
        return uploaded.name, uploaded.getvalue()
    stored = st.session_state.get('trail_upload')
    if stored:
        with st.container(horizontal=True, vertical_alignment='center', gap='small'):
            st.markdown(f':material/map: Planning **{md_escape(stored[0])}** from earlier. Upload another file to replace it.')
            st.button('Clear', key='trail_clear', type='tertiary', icon=':material/close:', on_click=_clear_route)
        return stored
    with st.popover('Where do I get a GPX file?', icon=':material/help:', type='tertiary'):
        st.markdown('- **AllTrails:** open a trail → ⋯ → *Download route* → GPX Track.\n'
                    '- **Gaia GPS / CalTopo / onX:** draw or open a route → *Export* → GPX.\n'
                    '- **Komoot:** open a tour → ⋯ → *Download GPX*.\n'
                    'For an out-and-back trail, a file with just the way out is fine: choose *Out and back* '
                    'and the app adds the return.')
    return None


def _settings(profile):
    preferred = (profile or {}).get('preferred_hike_type')
    columns = st.columns(3, gap='medium')
    hike_type = columns[0].segmented_control(
        'Hike type', list(HIKE_TYPE_LABELS), format_func=HIKE_TYPE_LABELS.get, required=True, key='trail_hike_type',
        default=preferred if preferred in HIKE_TYPE_LABELS else 'day_hike', persist_state='session',
        help='Changes the estimate only when your pace model has learned from both day hikes and backpacking trips.')
    direction = columns[1].segmented_control(
        'Route', list(DIRECTION_LABELS), format_func=DIRECTION_LABELS.get, required=True, key='trail_direction',
        default='one-way', persist_state='session',
        help='Out and back: upload only the way out. The app adds the return trip and predicts it separately.')
    mode = columns[2].segmented_control(
        'Breaks', BREAK_MODES, format_func=BREAK_MODE_LABELS.get, required=True, key='trail_break_mode',
        default='usual', persist_state='session', on_change=_prefill_breaks,
        help='My usual: your typical stopped time from your recorded hikes. Custom: set your own minutes.')
    hike_type, direction, mode = hike_type or 'day_hike', direction or 'one-way', mode or 'usual'
    columns = st.columns(3, gap='medium')
    start_day = columns[0].date_input('Date', value=_today(st.context.timezone), key='trail_date',
                                      persist_state='session', format='MM/DD/YYYY')
    start_time = columns[1].time_input('Start time (optional)', value=None, key='trail_start', step=900,
                                       format='12h', persist_state='session',
                                       help='Shows when you\'ll be back and how that compares with sunset. '
                                            'Times use this device\'s time zone.')
    allowance = 0
    if mode == 'custom':
        allowance = columns[2].number_input('Break minutes', min_value=0, max_value=10080, step=5,
                                            key='trail_break_minutes', persist_state='session') or 0
    return hike_type, direction, mode != 'none', mode == 'custom', allowance, start_day, start_time


def _today(tz):
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(tz)).date()
        except Exception:
            pass
    return date.today()


def _matches_trail_timezone(start, longitude):
    """Times are read in the device's timezone; only compare with sunset when that plausibly is the trail's."""
    offset = start.utcoffset()
    if offset is None:
        return False
    solar = longitude / 15  # hours east of UTC by the sun; real zones (and DST) stay within ~2.5 h
    return abs(offset.total_seconds() / 3600 - solar) <= 2.5


def _start_datetime(day, start, tz):
    if start is None:
        return None
    moment = datetime.combine(day or date.today(), start)
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return moment.replace(tzinfo=ZoneInfo(tz))
        except Exception:
            return moment
    return moment


def _timing(prediction, stats, start, direction):
    """Finish, turnaround and sunset facts for a planned start; empty without a start time."""
    if start is None:
        return {}
    total = stats['total_minutes']
    utc = start.astimezone(timezone.utc) if start.tzinfo else start
    finish = (utc + timedelta(minutes=total)).astimezone(start.tzinfo) if start.tzinfo else utc + timedelta(minutes=total)
    facts = {'finish': finish, 'days_later': (finish.date() - start.date()).days}
    if direction == 'out-and-back':
        legs = prediction.leg_totals
        if 'outbound' in legs.index and stats['moving_minutes']:
            outbound = float(legs.loc['outbound', 'predicted_time_min'])
            share = outbound / stats['moving_minutes']
            facts['turnaround'] = finish - timedelta(minutes=total - outbound - stats['break_minutes'] * share)
    profile = prediction.route.profile
    if (start.tzinfo is not None and not profile.empty and total <= 16 * 60
            and _matches_trail_timezone(start, float(profile.longitude.iloc[0]))):
        down = sunset(float(profile.latitude.iloc[0]), float(profile.longitude.iloc[0]), start.date())
        if down is not None:
            facts['sunset'] = down.astimezone(start.tzinfo)
            facts['margin'] = (facts['sunset'].timestamp() - finish.timestamp()) / 60
    return facts


def _when(moment, start):
    days = (moment.date() - start.date()).days
    return clock(moment) + (f' (+{days} day{"s" if days > 1 else ""})' if days > 0 else '')


def _daylight_badge(facts):
    margin = facts.get('margin')
    if margin is None or facts.get('days_later'):
        return
    sun = clock(facts['sunset'])
    if margin >= 60:
        st.badge(f'Back {duration(margin)} before sunset ({sun})', icon=':material/wb_sunny:', color='green')
    elif margin >= 0:
        st.badge(f'Only {duration(margin)} before sunset ({sun}). Consider starting earlier.',
                 icon=':material/wb_twilight:', color='orange')
    else:
        st.badge(f'Finishes {duration(-margin)} after sunset ({sun}). Start earlier or bring a headlamp.',
                 icon=':material/dark_mode:', color='red')


def _plan_text(route, stats, hike_type, direction, facts, start):
    lines = [route.name,
             f"{miles(stats['distance_miles'])} · ↑{feet(stats['elevation_gain_ft'])} · "
             f"{DIRECTION_LABELS[direction].lower()} · {HIKE_TYPE_LABELS[hike_type].lower()}",
             f"Estimate: {duration(stats['moving_minutes'])} moving + {duration(stats['break_minutes'])} breaks "
             f"= {duration(stats['total_minutes'])}"]
    if start is not None:
        lines.append(f"Start {clock(start)} on {start.strftime('%a %b')} {start.day} → back around "
                     f"{_when(facts['finish'], start)}")
        if facts.get('turnaround'):
            lines.append(f"Turnaround around {_when(facts['turnaround'], start)}")
        if facts.get('sunset') and not facts.get('days_later'):
            lines.append(f"Sunset {clock(facts['sunset'])}")
    return '\n'.join(lines)


def _save(repo, row, save_key):
    # Runs in the page body, after main() has revalidated the session and checked the deletion lock.
    try:
        repo.save_prediction(row)
    except Exception as error:
        st.session_state['trail_save_error'] = private_request_message(error)
        return
    st.session_state['saved_plan_key'] = save_key
    st.session_state.pop('trail_save_error', None)
    flash('Saved to your plans on the Dashboard.', icon=':material/bookmark_added:')
    st.rerun()


def _actions(repo, route, prediction, stats, models, hike_type, direction, include, override, facts, start):
    report = {'route_name': route.name, 'method': prediction.forecast.label, 'hike_type': hike_type,
              'direction': direction, **stats,
              'model_versions': [{'id': r['id'], 'type': r['model_type'], 'trained_at': r['trained_at']} for r in models]}
    if start is not None:
        report['planned_start'] = start.isoformat()
        report['expected_finish'] = facts['finish'].isoformat()
    by_type = {r['model_type']: r['id'] for r in models}
    row = {'moving_model_version_id': by_type.get('moving'), 'break_model_version_id': by_type.get('break'),
           'route_name': route.name[:250], 'distance_miles': stats['distance_miles'],
           'elevation_gain_ft': stats['elevation_gain_ft'], 'elevation_loss_ft': stats['elevation_loss_ft'],
           'hike_type': hike_type, 'route_direction': direction, 'predicted_moving_time_min': stats['moving_minutes'],
           'predicted_break_time_min': stats['break_minutes'] if stats['break_estimate_available'] or override or not include else None,
           'predicted_total_time_min': stats['total_minutes'], 'details': report}
    save_key = (st.session_state.get('prediction_key'), round(stats['break_minutes'], 3), include, override,
                start.isoformat() if start is not None else None)
    saved = st.session_state.get('saved_plan_key') == save_key
    with st.container(horizontal=True, gap='small', key='trip_actions'):
        clicked = st.button('Saved' if saved else 'Save to my plans', key='save_plan', type='primary', disabled=saved,
                            icon=':material/bookmark_added:' if saved else ':material/bookmark_add:')
        with st.popover('Download', icon=':material/download:'):
            st.download_button('Trip summary (JSON)', json.dumps(report, indent=2, default=str), 'hiking_prediction.json',
                               'application/json', on_click='ignore', icon=':material/description:')
            st.download_button('Segment table (CSV)', segment_csv(prediction), 'hiking_segment_predictions.csv',
                               'text/csv', key='download', on_click='ignore', icon=':material/table:')
        with st.popover('Share plan', icon=':material/share:'):
            st.caption('Copy this and send it to someone who\'ll check on you if you\'re late.')
            st.code(_plan_text(route, stats, hike_type, direction, facts, start), language=None, wrap_lines=True)
    if clicked:
        _save(repo, row, save_key)
    if st.session_state.get('trail_save_error'):
        st.error(st.session_state.pop('trail_save_error'))


def _basis(prediction, breaks, artifact, models, mstatus, stats, include):
    forecast = prediction.forecast
    selected = prediction.artifact.selected and forecast.experimental_minutes is None
    if selected:
        st.badge('Your pace model', icon=':material/person_pin_circle:', color='green')
        text = f'Moving time from your pace model ({method_name(forecast.label).split("(")[-1].rstrip(")")}).'
        if mstatus.personal_mae is not None:
            text += f' On your past hikes it was off by about {mstatus.personal_mae:.0f} min on average.'
        st.caption(text)
    else:
        st.badge('Standard hiking rule', icon=':material/straighten:', color='gray')
        st.caption(f'Moving time uses the {RULE_NAME}: {RULE_DEFINITION}.')
        if forecast.experimental_minutes is not None:
            delta = forecast.experimental_minutes - forecast.moving_minutes
            word = 'faster' if delta < 0 else 'slower'
            st.markdown(f'Your pace model, still learning, suggests **{duration(forecast.experimental_minutes)} '
                        f'moving** ({duration(abs(delta))} {word}). It takes over once it has been tested on '
                        f'{mstatus.minimum} complete hikes.')
            st.progress(mstatus.progress, text=f'{mstatus.verified} of {mstatus.minimum} complete test hikes')
        elif not models:
            st.caption('Add hikes you\'ve recorded and train your pace model to get estimates based on how you hike.')
            nav_button('Personalize with my hikes', 'plan_personalize', 'hikes', type='tertiary',
                       icon=':material/arrow_forward:')
    forecast_breaks = prediction.break_forecast
    if include and forecast_breaks and forecast_breaks.predicted_minutes is not None and not stats['break_override_active']:
        hikes = getattr(breaks, 'training_hikes', 0)
        count = forecast_breaks.expected_break_count
        stops = ''
        if count and forecast_breaks.expected_average_minutes:
            stops = (f' That is about {max(1, round(count))} stop{"s" if round(count) != 1 else ""} of '
                     f'~{forecast_breaks.expected_average_minutes:.0f} min plus short pauses.')
        st.caption(f'Breaks: your usual stopped time from {hikes} recorded hike{"s" if hikes != 1 else ""}.{stops}')
    elif include and not stats['break_override_active']:
        st.caption('Breaks aren\'t included yet: there\'s no break history to learn from. '
                   'Choose Custom breaks above to add your own.')


def _hero(repo, route, prediction, stats, breaks, artifact, models, mstatus, hike_type, direction,
          include, override, start):
    facts = _timing(prediction, stats, start, direction)
    with st.container(border=True, key='trip_hero'):
        st.subheader(md_escape(route.name), anchor=False)
        high = route.profile.smoothed_elevation_m.max() * FEET_PER_METER if not route.profile.empty else None
        chips = [f":material/straighten: {miles(stats['distance_miles'])}"
                 + (' round trip' if direction == 'out-and-back' else ''),
                 f":material/trending_up: {feet(stats['elevation_gain_ft'])} up",
                 f":material/trending_down: {feet(stats['elevation_loss_ft'])} down"]
        if high is not None:
            chips.append(f':material/landscape: high point {feet(high)}')
        chips.append(f"{DIRECTION_LABELS[direction]} · {HIKE_TYPE_LABELS[hike_type]}")
        st.markdown(' · '.join(chips))
        st.metric('Estimated total trip', duration(stats['total_minutes']),
                  help='Moving time plus breaks. A planning estimate, not a guaranteed arrival time.')
        with st.container(horizontal=True, gap='small', key='trip_supporting'):
            st.metric('Moving time', duration(stats['moving_minutes']), width=150)
            st.metric('Breaks', duration(stats['break_minutes']) if include else 'None', width=150,
                      help='Custom break time' if stats['break_override_active'] else None)
            st.metric('Moving pace', pace(60 / stats['average_speed_mph'] if stats['average_speed_mph'] else None),
                      width=150, help=f"{stats['average_speed_mph']:.1f} mph while moving")
            if facts:
                st.metric('Back around', _when(facts['finish'], start), width=190)
        if facts.get('turnaround'):
            st.caption(f":material/u_turn_left: Turn around by {_when(facts['turnaround'], start)} "
                       'to keep this pace.')
        _daylight_badge(facts)
        extremes = route_extremes({'distance_miles': stats['distance_miles'] / (2 if direction == 'out-and-back' else 1),
                                   'elevation_gain_ft': stats['elevation_gain_ft']}, artifact.route_ranges or {})
        for message in extremes:
            st.badge(message, icon=':material/warning:', color='orange')
        _basis(prediction, breaks, artifact, models, mstatus, stats, include)
        _actions(repo, route, prediction, stats, models, hike_type, direction, include, override, facts, start)
    return facts


def _loops_back(route):
    outbound = route.profile.loc[route.profile.leg.eq('outbound')] if 'leg' in route.profile else route.profile
    if len(outbound) < 2:
        return False
    first, last = outbound.iloc[0], outbound.iloc[-1]
    try:
        return distance_meters(first.latitude, first.longitude, last.latitude, last.longitude) < 150
    except ValueError:
        return False


def _details(route, prediction, stats, breaks, direction, start, route_key):
    st.subheader('Route', anchor=False)
    st_folium(trail_map(route), height=400, use_container_width=True, returned_objects=[],
              key='trail_' + hashlib.sha256(repr(route_key).encode()).hexdigest()[:16])
    st.caption('Green marks the start, orange the turnaround and the dotted line the return.'
               if direction == 'out-and-back' else 'Green marks the start and the blue flag the finish.')
    elevation = elevation_figure(route)
    elevation.update_layout(height=280, showlegend=direction == 'out-and-back')
    plot(elevation, 'elevation')
    forecast = prediction.break_forecast
    predicted = forecast.predicted_minutes if forecast else None
    with st.expander(f'Breaks: how we estimated {duration(predicted)}' if predicted is not None else 'Breaks',
                     icon=':material/local_cafe:'):
        st.metric('Original predicted break time', duration(predicted),
                  help='Your usual stopped time for a route like this, before any custom override.')
        if predicted is None:
            st.write('No personal break estimate yet. ' + ' '.join(forecast.warnings if forecast else []))
        else:
            st.write(f'Based on your typical stopped time per mile across {getattr(breaks, "training_hikes", 0)} '
                     'recorded hikes. It includes short pauses and unclear GPS time, so totals aren\'t too optimistic.')
            if forecast.expected_break_count:
                st.caption(f'Typical pattern: about {forecast.expected_break_count:.1f} rest breaks averaging '
                           f'{forecast.expected_average_minutes:.0f} min. These are averages, so the count can be fractional.')
            for name, value in forecast.experimental_minutes.items():
                st.caption(f'Experimental break model {name}: {duration(value)}')
        if stats['break_override_active']:
            st.caption(f"You set custom breaks of {duration(stats['break_minutes'])}.")
    if prediction.forecast.experimental_minutes is not None or prediction.artifact.selected:
        with st.expander('Compare estimates', icon=':material/compare_arrows:'):
            columns = st.columns(2)
            if prediction.forecast.experimental_minutes is not None:
                columns[0].metric('Your pace model (preview)', duration(prediction.forecast.experimental_minutes))
            else:
                columns[0].metric('Your pace model', duration(prediction.forecast.moving_minutes))
            columns[1].metric('Standard hiking rule', duration(stats['naismith_minutes']))
            st.caption('Moving time only. The CSV keeps both: predicted_time_min is the estimate shown above and '
                       'regression_time_min is your pace model\'s preview.')
    with st.expander('Segment-by-segment breakdown', icon=':material/stacked_line_chart:'):
        plot(time_along_route_figure(prediction, start), 'time_along')
        st.caption('Estimated moving time along the route' + (', shown as clock time from your start.' if start else '.'))
        plot(segment_figure(prediction), 'segments')
        st.dataframe(prediction.segments, hide_index=True, width='stretch')
        st.caption('Segment times exclude breaks. The CSV uses the same unrounded rows.')
    notes = [humanize_note(n) for n in dict.fromkeys([*route.notes, *prediction.forecast.warnings])]
    with st.expander('How reliable is this estimate?', icon=':material/info:'):
        for note in dict.fromkeys(notes):
            st.markdown(f'- {md_escape(note)}')
        st.caption('Weather, trail conditions, fatigue, navigation and pack weight can change your pace. '
                   'Past errors are not a guaranteed range for this route.')


def trail_page(repo, artifact, breaks, models, *, profile=None, model_error=None):
    st.title('Plan a hike')
    st.caption('Upload a route to see how long it will take you: moving time, breaks, and when you\'ll be back.')
    if model_error:
        st.error("Your saved pace model couldn't be verified, so predictions are paused. Retrain on My pace.",
                 icon=':material/gpp_bad:')
        st.caption(model_error)
        nav_button('Go to My pace', 'plan_fix_model', 'pace', type='primary')
        return
    mstatus = moving_status(artifact, models)
    with st.container(border=True, key='trip_settings'):
        route_file = _route_input()
        hike_type, direction, include, override, allowance, start_day, start_time = _settings(profile)
    if route_file is None:
        with st.container(border=True):
            st.badge(mstatus.badge, icon=mstatus.icon, color=mstatus.color)
            st.markdown(mstatus.headline)
            st.caption('Routes don\'t need timestamps. Only distance and elevation are used.')
        return
    name, contents = route_file
    key = (repo.user_id, hashlib.sha256(contents).hexdigest(), name, hike_type, direction,
           repr(artifact.config), id(type(artifact)), id(type(breaks)),
           tuple((r['id'], r['artifact_hash']) for r in models))
    if st.session_state.get('cloud_route_key') != key:
        for stale in ('prediction', 'prediction_key', 'cloud_route', 'cloud_route_key'):
            st.session_state.pop(stale, None)
        try:
            with st.spinner('Estimating your time…'):
                route = preview_upload(contents, name, hike_type, direction, config=artifact.config)
                prediction = predict_upload(contents, name, hike_type, direction, artifact=artifact,
                                            break_artifact=breaks)
        except ValueError as error:
            st.error(f'We couldn\'t read this GPX: {error}', icon=':material/error:')
            st.caption('Try exporting it again as a GPX track or route that includes elevation.')
            return
        st.session_state.update(cloud_route=route, cloud_route_key=key, prediction=prediction, prediction_key=key,
                                trail_upload=(name, contents))
    route, prediction = st.session_state['cloud_route'], st.session_state['prediction']
    if direction == 'out-and-back' and _loops_back(route):
        st.warning('This route already ends where it starts. If it\'s a loop or a full round trip, choose One way '
                   'so the distance isn\'t doubled.', icon=':material/sync_problem:')
    stats = trip_statistics(prediction, include_predicted_breaks=include,
                            manual_break_minutes=allowance if override and include else None)
    start = _start_datetime(start_day, start_time, st.context.timezone)
    _hero(repo, route, prediction, stats, breaks, artifact, models, mstatus, hike_type, direction,
          include, override, start)
    _details(route, prediction, stats, breaks, direction, start, key)
