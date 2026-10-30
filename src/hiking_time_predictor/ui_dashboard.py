"""Dashboard: first-run onboarding, model status, saved plans and hiking stats."""
import pandas as pd
import streamlit as st

from .model_status import break_status, is_stale, moving_status, rule_bias_sentence, selection_counts
from .personal_hiking import snapshot_fingerprint
from .ui_format import (DIRECTION_LABELS, HIKE_TYPE_LABELS, RULE_DEFINITION, RULE_NAME, feet, friendly_date,
                        md_escape, method_name, miles, pace, relative_day)
from .ui_nav import nav_button
from .web_charts import pace_vs_rule_figure
from .web_components import duration, plot


def status_card(mstatus, stale, models, selected, *, key_prefix):
    """Badge, plain-language status and the next model action. Shared by Dashboard and My hikes."""
    with st.container(border=True):
        st.badge(mstatus.badge, icon=mstatus.icon, color=mstatus.color)
        st.markdown(mstatus.headline)
        if mstatus.detail:
            st.caption(mstatus.detail)
        if mstatus.show_progress and models:
            st.progress(mstatus.progress, text=f'{mstatus.verified} of {mstatus.minimum} complete test hikes')
        if stale:
            st.warning('Your hikes or selections changed since your pace model was last trained. '
                       'Retrain so estimates include them.', icon=':material/update:')
            nav_button('Go to My pace to retrain', f'{key_prefix}_retrain', 'pace', type='primary', icon=':material/sync:')
        elif not models and selected:
            nav_button('Train my pace model', f'{key_prefix}_train', 'pace', type='primary',
                       icon=':material/model_training:')
        elif not models and key_prefix == 'home':
            nav_button('Choose hikes to learn from', f'{key_prefix}_choose', 'hikes', icon=':material/checklist:')


def _getting_started(selected, total):
    with st.container(border=True, key='get_started'):
        st.subheader("Let's learn your hiking pace")
        st.write('Add a few hikes you\'ve already recorded. The app learns how fast you really move and how long '
                 'you usually rest, then estimates time on trails you haven\'t hiked yet.')
        steps = st.columns(3)
        with steps[0]:
            st.markdown('<span class="step-number">1</span>**Add past hikes**', unsafe_allow_html=True)
            st.caption('GPX files with timestamps, exported from Strava, Garmin, AllTrails or any GPS watch.')
            nav_button('Add hikes', 'home_add_hikes', 'hikes', type='primary', icon=':material/upload_file:')
        with steps[1]:
            st.markdown('<span class="step-number">2</span>**Train your pace model**', unsafe_allow_html=True)
            st.caption('One click once your hikes are in. It takes a few seconds.')
        with steps[2]:
            st.markdown('<span class="step-number">3</span>**Plan a hike**', unsafe_allow_html=True)
            st.caption(f'Works right now with the {RULE_NAME} ({RULE_DEFINITION}). '
                       'Estimates get personal as you add hikes.')
            nav_button('Plan a hike', 'home_plan_now', 'plan', icon=':material/route:')


def _saved_plans(predictions, tz):
    st.subheader('Saved trip plans')
    if not predictions:
        st.caption('No saved plans yet. On Plan a hike, upload a route and select Save to my plans to keep it here.')
        return
    cards = st.columns(min(len(predictions), 3))
    for card, plan in zip(cards, predictions[:3]):
        with card, st.container(border=True):
            st.markdown(f"**{md_escape(plan.get('route_name') or 'Planned route')}**")
            st.metric('Total trip', duration(plan.get('predicted_total_time_min')), label_visibility='collapsed')
            breaks = plan.get('predicted_break_time_min')
            moving = duration(plan.get('predicted_moving_time_min'))
            st.caption(f'{moving} moving + {duration(breaks)} breaks' if breaks else f'{moving} moving, no breaks')
            st.caption(f"{miles(plan.get('distance_miles'))} · {feet(plan.get('elevation_gain_ft'))} climb · "
                       f"{HIKE_TYPE_LABELS.get(plan.get('hike_type'), 'Hike')} · "
                       f"{DIRECTION_LABELS.get(plan.get('route_direction'), '')}")
            method = method_name((plan.get('details') or {}).get('method'))
            st.caption(f"Saved {relative_day(plan.get('created_at'), tz=tz)} · {method}")
    if len(predictions) > 3:
        with st.expander(f'All saved plans ({len(predictions)})', icon=':material/list:'):
            table = pd.DataFrame([{
                'Saved': friendly_date(p.get('created_at'), tz), 'Route': p.get('route_name'),
                'Total': duration(p.get('predicted_total_time_min')),
                'Moving': duration(p.get('predicted_moving_time_min')),
                'Breaks': duration(p.get('predicted_break_time_min')) if p.get('predicted_break_time_min') is not None else '—',
                'Distance': p.get('distance_miles'), 'Climb': p.get('elevation_gain_ft'),
                'Type': HIKE_TYPE_LABELS.get(p.get('hike_type'), '—'),
                'Direction': DIRECTION_LABELS.get(p.get('route_direction'), '—')} for p in predictions])
            st.dataframe(table, hide_index=True, width='stretch', column_config={
                'Route': st.column_config.TextColumn(width='large'),
                'Distance': st.column_config.NumberColumn(format='%.1f mi'),
                'Climb': st.column_config.NumberColumn(format='%,.0f ft')})


def _hiking_stats(frame):
    st.subheader('Your hiking')
    ordered = frame.sort_values('activity_date')
    total_miles = float(frame.distance_miles.sum())
    moving = float(frame.moving_time_min.sum())
    per_hike_pace = (ordered.moving_time_min / ordered.distance_miles.replace(0, float('nan'))).dropna().round(2).tolist()
    row1 = st.columns(2, gap='small', wrap=False)
    last = ordered.iloc[-1] if len(ordered) else None
    row1[0].metric('Hikes recorded', str(len(frame)), border=True, height='stretch',
                   delta=f"Last: {friendly_date(last.activity_date)}" if last is not None else None,
                   delta_color='off', delta_arrow='off')
    row1[1].metric('Miles hiked', miles(total_miles), border=True, height='stretch',
                   chart_data=ordered.distance_miles.round(2).tolist() if len(frame) > 1 else None, chart_type='bar',
                   delta=f"{miles(last.distance_miles)} last hike" if last is not None else None,
                   delta_color='normal', delta_arrow='off',
                   help='Distance measured while moving, summed across your recordings.')
    row2 = st.columns(2, gap='small', wrap=False)
    row2[0].metric('Elevation climbed', feet(frame.elevation_gain_ft.sum()), border=True, height='stretch',
                   chart_data=ordered.elevation_gain_ft.fillna(0).round().tolist() if len(frame) > 1 else None,
                   chart_type='bar', delta=f"{feet(last.elevation_gain_ft)} last hike" if last is not None else None,
                   delta_color='normal', delta_arrow='off')
    row2[1].metric('Moving pace', pace(moving / total_miles if total_miles else None), border=True, height='stretch',
                   chart_data=per_hike_pace if len(per_hike_pace) > 1 else None,
                   delta=f"{pace(per_hike_pace[-1])} last hike" if per_hike_pace else None,
                   delta_color='normal', delta_arrow='off',
                   help='Average minutes per mile while moving. Stops are excluded. The line shows each hike, oldest first.')


def _rule_comparison(frame, tz, mstatus):
    valid = frame.loc[(frame.distance_miles > 0) & (frame.moving_time_min > 0)].copy()
    if len(valid) < 3:
        if len(valid):
            st.caption('A comparison with the standard hiking rule appears after 3 recorded hikes.')
        return
    rule = valid.distance_miles * 20 + valid.elevation_gain_ft.fillna(0) * 0.03
    ratio = float((valid.moving_time_min / rule).median())
    st.subheader('You vs. the standard hiking rule')
    if abs(ratio - 1) < 0.03:
        st.write(f'On your recorded hikes you move at about the pace the {RULE_NAME} predicts.')
    else:
        word = 'faster' if ratio < 1 else 'slower'
        st.write(f'On your recorded hikes you typically move **{abs(1 - ratio):.0%} {word}** than the '
                 f'{RULE_NAME} predicts.')
    valid['date_label'] = valid.activity_date.map(lambda d: friendly_date(d, tz))
    valid['elevation_gain_ft'] = valid.elevation_gain_ft.fillna(0)
    plot(pace_vs_rule_figure(valid), 'dashboard_distance')
    note = rule_bias_sentence(mstatus)
    st.caption('Each dot is one of your recordings. Dots below the dashed line are hikes where you were '
               'faster than the rule. ' + (note or 'Descriptive only: your pace model is tested separately on My pace.'))


def _stops(snapshot, ids, breaks, models):
    ordinary = [e for e in snapshot['breaks'] if e['classification'] == 'stationary_break' and e['hike_id'] in ids]
    if not ordinary:
        return
    st.subheader('Your stops')
    total = sum(e['duration_minutes'] for e in ordinary)
    columns = st.columns(3, gap='small')
    columns[0].metric('Rest breaks found', str(len(ordinary)), border=True)
    columns[1].metric('Typical break', duration(total / len(ordinary)), border=True)
    rate = getattr(breaks, 'minutes_per_mile', None) if models else None
    columns[2].metric('Stopped time per mile', f'{rate:.1f} min' if rate is not None else '—', border=True,
                      help='Your average stopped time per mile, including short pauses and unclear GPS time. '
                           'Appears after you train your pace model.')


def _recent(frame, tz):
    st.subheader('Recent hikes')
    recent = frame.head(5)
    table = pd.DataFrame({
        'Hike': recent.name, 'Date': recent.activity_date.map(lambda d: friendly_date(d, tz)),
        'Moving': recent.moving_time_min.map(duration),
        'Distance': recent.distance_miles, 'Climb': recent.elevation_gain_ft,
        'Pace': (recent.moving_time_min / recent.distance_miles.replace(0, float('nan'))).map(pace),
        'Learning from it': recent.training_enabled.astype(bool)})
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'Hike': st.column_config.TextColumn(pinned=True),
        'Distance': st.column_config.NumberColumn(format='%.1f mi'),
        'Climb': st.column_config.NumberColumn(format='%,.0f ft'),
        'Learning from it': st.column_config.CheckboxColumn(help='Selected for training your pace model')})
    nav_button('See all hikes', 'home_all_hikes', 'hikes', type='tertiary', icon=':material/arrow_forward:')


def dashboard(repo, models, *, artifact=None, breaks=None, profile=None):
    tz = st.context.timezone
    st.title('Dashboard')
    name = (profile or {}).get('display_name')
    st.caption(f'Welcome back, {md_escape(name)}.' if name else 'Welcome back.')
    hikes = [h for h in repo.list_hikes() if not h.get('deleting')]
    selected, total = selection_counts(hikes)
    if not hikes:
        _getting_started(selected, total)
        _saved_plans(repo.predictions(), tz)
        return
    snapshot = repo.snapshot()
    stale = is_stale(models, snapshot_fingerprint(snapshot))
    mstatus = moving_status(artifact, models, hikes)
    plan_col, status_col = st.columns([2, 3], gap='medium')
    with plan_col, st.container(border=True, key='home_plan_cta', height='stretch'):
        st.subheader('Plan your next hike')
        st.write('Upload a route and see how long it will take at your pace, including breaks and when you\'ll be back.')
        nav_button('Plan a hike', 'home_plan', 'plan', type='primary', icon=':material/route:', width='stretch')
    with status_col:
        status_card(mstatus, stale, models, selected, key_prefix='home')
        bstatus = break_status(breaks, models)
        if models:
            st.caption(f'Breaks: {bstatus.headline}')
    _saved_plans(repo.predictions(), tz)
    frame = pd.DataFrame(hikes)
    for column in ('distance_miles', 'moving_time_min', 'elevation_gain_ft'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    _hiking_stats(frame)
    _rule_comparison(frame, tz, mstatus)
    _stops(snapshot, set(frame.id), breaks, models)
    _recent(frame, tz)
