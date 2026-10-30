"""My hikes: import recordings, choose training hikes, inspect one hike, review stops, delete."""
import hashlib

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .breaks import BreakConfig, CLASSIFICATIONS
from .cloud_repository import private_request_message
from .gpx import ImportConfig
from .model_status import is_stale, selection_counts
from .personal_hiking import preview_recording, save_recording, snapshot_fingerprint, validate_break_edits
from .ui_format import (CLASSIFICATION_LABELS, HIKE_TYPE_LABELS, clock, feet, flag_labels, friendly_date,
                        humanize_filename, md_escape, miles, pace, parse_time, safe_filename)
from .ui_nav import flash, nav_button
from .web_charts import history_elevation_figure, history_pace_figure, recording_map
from .web_components import duration, plot

MAX_PREVIEW_FILES = 20
MAX_PREVIEW_BYTES = 60 * 1024 * 1024
HIKE_TYPE_CHOICES = ('day_hike', 'backpacking', 'unknown')
HIKE_TYPE_CHOICE_LABELS = {**HIKE_TYPE_LABELS, 'unknown': 'Not sure'}
PRIVACY_NOTE = ('Recordings show where you walked, often starting near home. Other users can never see them, they '
                'only train your own estimates, and you can delete any hike, or everything, at any time.')


def select_all_training(hike_ids):
    enabled = st.session_state['training_select_all']
    for hike_id in hike_ids:
        st.session_state['training_' + hike_id] = enabled


def _open_hike(hike_id):
    st.session_state['history_detail_hike_id'] = hike_id


def _close_hike():
    st.session_state.pop('history_detail_hike_id', None)


def _meta(hike, tz):
    parts = [friendly_date(hike.get('activity_date'), tz),
             HIKE_TYPE_LABELS.get(hike.get('hike_type'), 'Hike type not set'),
             miles(hike.get('distance_miles')), f"{feet(hike.get('elevation_gain_ft'))} climb",
             f"{duration(hike.get('moving_time_min'))} moving"]
    return ' · '.join(parts)


# --- Import -----------------------------------------------------------------

def _uploader_key():
    n = st.session_state.get('history_upload_n', 0)
    return 'historical_files' if n == 0 else f'historical_files_{n}'


def _import_result():
    result = st.session_state.pop('history_import_result', None)
    if not result:
        return
    message = f"Added {result['saved']} hike{'s' if result['saved'] != 1 else ''} to your history."
    if result['skipped']:
        message += f" {result['skipped']} {'was' if result['skipped'] == 1 else 'were'} already there and skipped."
    with st.container(border=True):
        st.success(message, icon=':material/check_circle:')
        if result['saved']:
            st.write('Next, train your pace model so estimates include these hikes.')
            nav_button('Train my pace model', 'history_train_next', 'pace', type='primary',
                       icon=':material/model_training:')


def _import_section(repo, existing, profile):
    config = ImportConfig(**existing[0]['import_config']) if existing else ImportConfig()
    break_config = BreakConfig(**existing[0]['break_config']) if existing else BreakConfig()
    previews = st.session_state.get('historical_previews', [])
    with st.expander('Add hikes from GPX files', expanded=not existing or bool(previews), icon=':material/upload_file:'):
        st.caption('Add recordings of hikes you\'ve finished: GPX files with timestamps from a GPS watch or app. '
                   'Up to 20 files and 60 MB at a time. Planning a new route? Use Plan a hike instead.')
        with st.popover('Where do I get GPX files?', icon=':material/help:', type='tertiary'):
            st.markdown('- **Strava:** open the activity → ⋯ menu → *Export GPX*.\n'
                        '- **Garmin Connect:** open the activity → gear icon → *Export to GPX*.\n'
                        '- **AllTrails:** open the recording → ⋯ → *Download route* → GPX.\n'
                        '- Most GPS watches and apps have a similar *Export GPX* option.')
        preferred = (profile or {}).get('preferred_hike_type')
        choice = st.segmented_control(
            'What kind of hikes are these?', HIKE_TYPE_CHOICES, format_func=HIKE_TYPE_CHOICE_LABELS.get,
            default=preferred if preferred in HIKE_TYPE_CHOICES else 'unknown', required=True, key='history_hike_type',
            help='Applies to every file in this batch. Hike type helps when you record both day hikes and backpacking trips.')
        hike_type = None if choice in (None, 'unknown') else choice
        files = st.file_uploader('Completed hike recordings', type=['gpx'], accept_multiple_files=True,
                                 key=_uploader_key(), label_visibility='collapsed', max_upload_size=20)
        upload_key = (tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files), hike_type,
                      repr(config), repr(break_config))
        if st.session_state.get('history_upload_key') != upload_key:
            st.session_state.pop('historical_previews', None)
            st.session_state.pop('history_preview_errors', None)
            if files:
                _preview(files, hike_type, config, break_config)
            # Mark the batch only once it has been read, so an interrupted read is retried.
            st.session_state['history_upload_key'] = upload_key
        for message in st.session_state.get('history_preview_errors', []):
            st.error(message, icon=':material/error:')
        previews = st.session_state.get('historical_previews', [])
        if previews:
            _review_previews(repo, previews, existing)


def _preview(files, hike_type, config, break_config):
    errors = []
    if len(files) > MAX_PREVIEW_FILES or sum(f.size for f in files) > MAX_PREVIEW_BYTES:
        st.session_state['history_preview_errors'] = [
            f'Add at most {MAX_PREVIEW_FILES} files (60 MB in total) at a time. Remove some and try again.']
        return
    previews = []
    with st.status(f'Reading {len(files)} recording{"s" if len(files) != 1 else ""}…', expanded=False) as status:
        for index, uploaded in enumerate(files, 1):
            status.update(label=f'Reading {index} of {len(files)}: {md_escape(uploaded.name)}')
            try:
                previews.append(preview_recording(uploaded.getvalue(), uploaded.name, hike_type,
                                                  config=config, break_config=break_config))
            except ValueError as error:
                errors.append(f'{md_escape(uploaded.name)} couldn\'t be read: {md_escape(error)}')
            except Exception:
                # gpxpy raises its own exception types for malformed numbers and tags.
                errors.append(f'{md_escape(uploaded.name)} couldn\'t be read. Check that it is a valid GPX file.')
        status.update(label=f'{len(previews)} of {len(files)} recordings ready to review',
                      state='complete' if previews else 'error')
    st.session_state['historical_previews'] = previews
    st.session_state['history_preview_errors'] = errors


def _review_previews(repo, previews, existing):
    known = {h['original_file_hash']: h for h in existing}
    duplicates = 0
    st.subheader('Review before adding', anchor=False)
    for preview in previews:
        summary = preview.summary
        duplicate = known.get(hashlib.sha256(preview.contents).hexdigest())
        duplicates += bool(duplicate)
        with st.container(border=True):
            st.text_input('Hike name', value=humanize_filename(preview.filename), key='name_' + preview.id,
                          max_chars=250, disabled=bool(duplicate),
                          help='Names can\'t be changed after you add a hike.')
            st.caption(f"{friendly_date(summary.get('start_time_utc'), st.context.timezone)} · "
                       f"{miles(summary['distance_miles'])} · {duration(summary['moving_time_min'])} moving · "
                       f"{duration(summary['stationary_time_min'])} stopped · {feet(summary['elevation_gain_ft'])} climb")
            badges = []
            if duplicate:
                badges.append(f":gray-badge[:material/check_circle: Already in your history as "
                              f"{md_escape(duplicate['name'])}. It will be skipped.]")
            badges += [f':orange-badge[{label}]' for label in flag_labels(summary.get('quality_flags'))]
            if summary.get('unknown_time_min'):
                badges.append(f":orange-badge[{duration(summary['unknown_time_min'])} of unclear GPS time]")
            if badges:
                st.markdown(' '.join(badges))
            st.checkbox('Use to learn my pace', value=not duplicate, key='eligible_' + preview.id,
                        disabled=bool(duplicate),
                        help='You can change this later on My hikes. Quality checks still apply when training.')
    new = len(previews) - duplicates
    label = f"Add {new} hike{'s' if new != 1 else ''} to my account" if new else 'Nothing new to add'
    if st.button(label, key='confirm_import', type='primary', disabled=not new, icon=':material/add_circle:'):
        _confirm_import(repo, previews)


def _confirm_import(repo, previews):
    saved, skipped, failed = 0, 0, []
    with st.status('Adding hikes…', expanded=False) as status:
        for index, preview in enumerate(previews, 1):
            status.update(label=f'Adding {index} of {len(previews)}…')
            try:
                _, created = save_recording(repo, preview, name=st.session_state.get('name_' + preview.id) or None,
                                            training_enabled=bool(st.session_state.get('eligible_' + preview.id)))
                saved += created
                skipped += not created
            except Exception:
                failed.append(preview)
        status.update(label='Done', state='complete' if not failed else 'error')
    if failed:
        for preview in failed:
            st.error(f'{md_escape(preview.filename)} wasn\'t added. Try again; the app checks for anything already saved.')
        st.session_state['historical_previews'] = failed
        return
    st.session_state['historical_previews'] = []
    st.session_state['history_upload_n'] = st.session_state.get('history_upload_n', 0) + 1
    st.session_state.pop('history_upload_key', None)
    st.session_state['history_import_result'] = {'saved': saved, 'skipped': skipped}
    st.rerun()


# --- Hike list and training selection --------------------------------------

def training_selection(repo, hikes, *, models=None):
    tz = st.context.timezone
    selectable = [h for h in hikes if not h.get('deleting')]
    for hike in hikes:
        key = 'training_' + hike['id']
        if hike.get('deleting'):
            st.session_state[key] = False
        elif key not in st.session_state:
            st.session_state[key] = hike['training_enabled']
    st.session_state['training_select_all'] = bool(selectable) and all(
        st.session_state['training_' + h['id']] for h in selectable)
    st.subheader('Your hikes')
    st.caption('Checked hikes are used to learn your pace. Changes save automatically and apply the next time you '
               'train on My pace.')
    with st.container(horizontal=True, vertical_alignment='center'):
        st.checkbox('Use all my hikes', key='training_select_all', disabled=not selectable,
                    on_change=select_all_training, args=([h['id'] for h in selectable],))
        selected = sum(st.session_state['training_' + h['id']] for h in selectable)
        st.caption(f'{selected} of {len(selectable)} selected')
    with st.container(border=True, height=620 if len(hikes) > 10 else 'content', key='hike_list'):
        for index, hike in enumerate(hikes):
            with st.container(horizontal=True, vertical_alignment='center', gap='small', wrap=False):
                st.checkbox(hike['name'], key='training_' + hike['id'], disabled=bool(hike.get('deleting')),
                            label_visibility='collapsed')
                badges = [f':orange-badge[{label}]' for label in flag_labels(hike.get('gps_quality_status'))]
                if hike.get('deleting'):
                    badges.insert(0, ':red-badge[Deletion pending]')
                st.markdown(f"**{md_escape(hike['name'])}**  \n:gray[{_meta(hike, tz)}]"
                            + (f"  \n{' '.join(badges)}" if badges else ''), width='stretch')
                st.button('Open', key='details_' + hike['id'], type='tertiary', icon=':material/chevron_right:',
                          icon_position='right', on_click=_open_hike, args=(hike['id'],),
                          help=f"Open {hike['name']}")
    changed = [h for h in selectable if st.session_state['training_' + h['id']] != h['training_enabled']]
    if changed:
        for hike in changed:
            repo.set_training(hike['id'], st.session_state['training_' + hike['id']])
        if len(changed) == 1:
            verb = 'will' if st.session_state['training_' + changed[0]['id']] else "won't"
            st.toast(f"{md_escape(changed[0]['name'])} {verb} be used to learn your pace.", icon=':material/check:')
        else:
            st.toast(f'Saved training choices for {len(changed)} hikes.', icon=':material/check:')
    if models is not None:
        stale = is_stale(models, snapshot_fingerprint(repo.snapshot())) if models else False
        if stale or (not models and selected):
            with st.container(border=True):
                st.write('Your hikes or selections changed since your pace model was last trained.' if stale
                         else f'{selected} hike{"s are" if selected != 1 else " is"} ready to learn from.')
                nav_button('Go to My pace to retrain' if stale else 'Go to My pace to train', 'history_train', 'pace',
                           type='primary', icon=':material/model_training:')


def _empty_state():
    with st.container(border=True):
        st.subheader('No hikes yet')
        st.write('Add a GPX recording of a hike you\'ve finished, using the section above. After a few hikes, '
                 'the app learns your pace and how long you usually rest.')
        st.markdown('1. Export a GPX file from Strava, Garmin or another app.\n'
                    '2. Add it above and check the details.\n'
                    '3. Train your pace model on My pace.\n'
                    '4. Plan a trail on Plan a hike.')
        st.caption('Plan a hike already works today, using the standard hiking rule.')
        nav_button('Plan a hike now', 'history_plan_now', 'plan', icon=':material/route:')


def history_page(repo, *, profile=None, models=None):
    st.title('My hikes')
    st.caption(f':material/lock: {PRIVACY_NOTE}')
    hikes = repo.list_hikes()
    hike = next((h for h in hikes if h['id'] == st.session_state.get('history_detail_hike_id')), None)
    if hike is not None:
        detail_view(repo, hike)
        return
    _import_result()
    _import_section(repo, hikes, profile)
    if not hikes:
        _empty_state()
        return
    training_selection(repo, hikes, models=models)


# --- One hike ---------------------------------------------------------------

def _load_detail(repo, hike):
    detail = st.session_state.get('activity_detail')
    if detail and detail[0] == hike['id']:
        return detail
    with st.spinner('Loading your recording…'):
        contents = repo.download('hike-gpx', hike['original_gpx_storage_path'])
        preview = preview_recording(contents, 'recording.gpx', hike['hike_type'],
                                    config=ImportConfig(**hike['import_config']),
                                    break_config=BreakConfig(**hike['break_config']))
    st.session_state['activity_detail'] = (hike['id'], contents, preview)
    return st.session_state['activity_detail']


def _stop_frame(repo, hike):
    events = [e for e in repo.snapshot()['breaks'] if e['hike_id'] == hike['id']]
    if not events:
        return pd.DataFrame()
    frame = pd.DataFrame([{**e['data'], 'id': e['id'], 'hike_id': hike['id'], 'break_classification': e['classification'],
                           'start_time': e['start_time'], 'end_time': e['end_time']} for e in events])
    for column in ('start_time', 'end_time'):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.loc[frame.break_classification.ne('moving')].sort_values('start_time').reset_index(drop=True)
    if frame.empty:
        return frame
    tz = st.context.timezone
    started = parse_time(hike.get('activity_date')) or frame.start_time.iloc[0]
    local = frame.start_time.dt.tz_convert(tz) if tz else frame.start_time
    frame['started'] = local.map(clock)
    frame['into_hike'] = (frame.start_time - started).map(lambda d: duration(d.total_seconds() / 60))
    frame['detected_label'] = frame.detected_classification.map(lambda c: CLASSIFICATION_LABELS.get(c, c))
    frame['correction_notes'] = frame.correction_notes.fillna('')
    frame['label'] = frame.break_classification.map(lambda c: CLASSIFICATION_LABELS.get(c, c))
    return frame


def _review_stops(repo, hike, stops):
    st.subheader('Stops on this hike')
    if stops.empty:
        st.caption('No stops were detected. This hike counts as continuous moving time.')
        return
    config = BreakConfig(**hike['break_config'])
    rests = stops.loc[stops.break_classification.eq('stationary_break')]
    st.write(f"We found {len(stops)} stop{'s' if len(stops) != 1 else ''}; {len(rests)} count as rest breaks "
             f"({duration(rests.duration_minutes.sum())}). Fix any the GPS got wrong.")
    st.caption(f'Only rest breaks of {config.min_break_minutes:g}–{config.max_break_minutes:g} min shape your '
               'break count and typical break length. Changes apply the next time you train on My pace.')
    version = st.session_state.get('break_review_n', 0)
    edited = st.data_editor(
        stops, key=f"break_review_{hike['id']}_{version}", hide_index=True, width='stretch',
        column_order=['started', 'into_hike', 'duration_minutes', 'distance_into_hike_miles',
                      'break_classification', 'detected_label', 'correction_notes'],
        disabled=[c for c in stops.columns if c not in ('break_classification', 'correction_notes')],
        column_config={
            'started': st.column_config.TextColumn('Started'),
            'into_hike': st.column_config.TextColumn('Into hike'),
            'duration_minutes': st.column_config.NumberColumn('Length', format='%.0f min'),
            'distance_into_hike_miles': st.column_config.NumberColumn('At mile', format='%.1f'),
            'break_classification': st.column_config.SelectboxColumn(
                'What was it?', options=[c for c in CLASSIFICATIONS if c != 'moving'],
                format_func=lambda c: CLASSIFICATION_LABELS.get(c, c), required=True, pinned=True,
                help='Only rest breaks count toward your typical break.'),
            'detected_label': st.column_config.TextColumn('GPS guessed'),
            'correction_notes': st.column_config.TextColumn('Your note', max_chars=2000)})
    changed = (edited.break_classification.ne(stops.break_classification)
               | edited.correction_notes.fillna('').ne(stops.correction_notes.fillna('')))
    count = int(changed.sum())
    if st.button(f"Save {count} change{'s' if count != 1 else ''}" if count else 'No changes to save',
                 key='save_breaks', type='primary', disabled=not count, icon=':material/save:'):
        try:
            validate_break_edits(stops, edited, config)
            rows = edited.loc[changed]
            repo.correct_breaks(hike['id'], [{'id': r.id, 'classification': r.break_classification,
                                             'notes': r.correction_notes or ''} for r in rows.itertuples()])
        except ValueError as error:
            st.error(f'Not saved: {error}', icon=':material/error:')
            return
        except Exception as error:
            st.error(private_request_message(error))
            return
        st.session_state['break_review_n'] = version + 1
        flash(f"Saved {count} stop correction{'s' if count != 1 else ''}. Retrain on My pace to use them.")
        st.rerun()


def _open_delete(hike_id):
    st.session_state['history_delete_hike'] = hike_id


def _close_delete():
    st.session_state.pop('history_delete_hike', None)


@st.dialog('Delete this hike?', icon=':material/delete:', on_dismiss=_close_delete)
def _confirm_delete(repo, hike):
    st.markdown(f"**{md_escape(hike['name'])}**  \n{_meta(hike, st.context.timezone)}")
    st.write('This permanently removes the recording, its original GPX file and its detected stops from your '
             'account. Your current pace model doesn\'t change until you retrain.')
    cancel, confirm = st.columns(2)
    if cancel.button('Cancel', key='delete_cancel', width='stretch'):
        _close_delete()
        st.rerun()
    if confirm.button('Delete permanently', key='delete_confirm', type='primary', width='stretch'):
        # Dialog reruns skip main()'s error handling, so keep SDK details private here too.
        try:
            repo.delete_hike(hike['id'])
        except Exception as error:
            st.error(private_request_message(error))
            return
        for key in ('history_delete_hike', 'history_detail_hike_id', 'activity_detail'):
            st.session_state.pop(key, None)
        flash(f"Deleted {md_escape(hike['name'])}.", icon=':material/delete:')
        st.rerun()


def detail_view(repo, hike):
    tz = st.context.timezone
    st.button('All hikes', key='history_back', type='tertiary', icon=':material/arrow_back:', on_click=_close_hike)
    st.subheader(md_escape(hike['name']), anchor=False)
    used = 'Used to learn your pace' if hike.get('training_enabled') else 'Not used to learn your pace'
    st.caption(f"{friendly_date(hike.get('activity_date'), tz, weekday=True)} · "
               f"{HIKE_TYPE_LABELS.get(hike.get('hike_type'), 'Hike type not set')} · {used}")
    if hike.get('deleting'):
        st.warning("Deleting this hike didn't finish. It's already excluded from training. "
                   'Select Delete hike to finish.', icon=':material/warning:')
    moving, elapsed = hike.get('moving_time_min'), hike.get('elapsed_time_min')
    with st.container(horizontal=True, gap='small', key='hike_stats'):
        for label, value in (('Distance', miles(hike.get('distance_miles'))), ('Moving time', duration(moving)),
                             ('Total time', duration(elapsed)),
                             ('Stopped', duration(elapsed - moving) if moving is not None and elapsed is not None else '—'),
                             ('Climb', feet(hike.get('elevation_gain_ft'))),
                             ('Moving pace', pace(moving / hike['distance_miles'] if hike.get('distance_miles') else None))):
            st.metric(label, value, border=True, width=155)
    try:
        detail = _load_detail(repo, hike)
    except Exception as error:
        st.warning('The original recording couldn\'t be loaded, so charts are unavailable. '
                   + (str(error) if isinstance(error, ValueError) else private_request_message(error)))
        detail = None
    stops = _stop_frame(repo, hike)
    if detail is not None:
        points, segments = detail[2].points, detail[2].segments
        left, right = st.columns([1, 1], gap='medium')
        with left:
            try:
                st_folium(recording_map(points, stops if not stops.empty else None), height=330,
                          use_container_width=True, returned_objects=[],
                          key='history_map_' + hashlib.sha256(hike['id'].encode()).hexdigest()[:16])
            except ValueError:
                st.caption('This recording has no coordinates to map.')
        with right:
            plot(history_elevation_figure(points, stops.loc[stops.break_classification.eq('stationary_break')]
                                          if not stops.empty else None), 'history_elevation')
            st.caption('Elevation along your recording. Gold dots are rest breaks.')
        if not segments.empty:
            plot(history_pace_figure(segments), 'history_pace')
            st.caption('Moving pace for each half-mile segment. Stops are excluded.')
    _review_stops(repo, hike, stops)
    with st.container(horizontal=True, gap='small'):
        if detail is not None:
            st.download_button('Download GPX', detail[1], safe_filename(hike['name'], '.gpx'), 'application/gpx+xml',
                               icon=':material/download:', on_click='ignore')
        st.button('Delete hike', key='delete_hike', icon=':material/delete:', on_click=_open_delete, args=(hike['id'],))
    if st.session_state.get('history_delete_hike') == hike['id']:
        _confirm_delete(repo, hike)
