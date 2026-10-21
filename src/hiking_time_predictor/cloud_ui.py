"""Streamlit account experience backed exclusively by the caller's Supabase JWT."""
import hashlib
import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from .auth_sessions import AuthManager, AuthenticationError
from .breaks import BreakConfig, CLASSIFICATIONS, NON_MOVING_TARGET
from .break_model import BreakArtifact
from .predictor import ModelArtifact
from .cloud_config import CloudConfig
from .cloud_repository import SupabaseRepository, private_request_message
from .gpx import ImportConfig
from .personal_hiking import (export_personal_data, load_personal_models, preview_recording,
    save_recording, snapshot_fingerprint, train_personal_models, validate_break_edits)
from .trusted_models import TrustedModelCodec
from .web_components import design, duration, number, performance_page, plot, predictor_page
from .web_upload import preview_upload, trip_statistics

PAGES = ('My Dashboard', 'My Hiking History', 'My Models', 'Trail Predictor', 'Account Settings')


def clear_private_state(user_id=None):
    if st.session_state.get('_bound_user') != user_id:
        credentials = st.session_state.get('_community_auth')
        st.session_state.clear()
        if user_id:
            st.session_state['_bound_user'] = user_id
            if credentials and credentials.user_id == user_id:
                st.session_state['_community_auth'] = credentials


def landing(config=None, message=None):
    st.title('Personal Hiking Time Predictor')
    st.write('Plan your next trail using your own hiking history.')
    st.write('Sign in with Google, import completed GPX recordings, and deliberately train your personal moving-time and break-time models. Your history, models and predictions stay private to your account.')
    if config:
        if getattr(config, 'auth_mode', 'service') == 'streamlit':
            from .community_auth import google_sign_in
            google_sign_in(config)
            st.caption('Refreshing this page or restarting the app requires signing in again. Your saved hiking data stays in Supabase.')
        else:
            st.link_button('Continue with Google', config.auth_url + '/auth/login', type='primary', width='stretch')
    else:
        st.button('Continue with Google', disabled=True)
    if message:
        st.info(message)
    st.caption('New hikers receive a labeled Naismith moving-time baseline until there is enough personal history. Break estimates remain unavailable until your own recordings support them.')


def dashboard(repo, models):
    st.title('My Dashboard')
    hikes = repo.list_hikes()
    frame = pd.DataFrame([h for h in hikes if not h.get('deleting')])
    miles = frame.distance_miles.sum() if len(frame) else 0
    moving = frame.moving_time_min.sum() if len(frame) else 0
    gain = frame.elevation_gain_ft.sum() if len(frame) else 0
    cols = st.columns(4)
    for col, label, value in zip(cols, ('Logged moving miles', 'Completed hikes', 'Elevation gain', 'Average moving pace'),
        (number(miles, ' mi', 2), str(len(frame)), number(gain, ' ft', 0), number(moving/miles if miles else None, ' min/mi', 2))):
        col.metric(label, value)
    snapshot = repo.snapshot()
    ordinary = [e for e in snapshot['breaks'] if e['classification'] == 'stationary_break']
    st.subheader('Historical breaks')
    cols = st.columns(3)
    cols[0].metric('Ordinary breaks', len(ordinary))
    total = sum(e['duration_minutes'] for e in ordinary)
    cols[1].metric('Total ordinary rest', duration(total))
    cols[2].metric('Average break', duration(total/len(ordinary) if ordinary else None))
    st.subheader('Current models')
    if not models:
        st.info('No trained models yet. Add completed recordings, confirm training eligibility, then visit My Models. Trail Predictor is ready with the Naismith baseline.')
    for row in models:
        st.write(f"**{row['model_type'].title()} model:** trained {row['trained_at'][:10]} on {row['training_hike_count']} eligible hikes.")
        if row['dataset_fingerprint'] != snapshot_fingerprint(snapshot):
            st.caption('Your history has changed since this version was trained. Use Retrain Models to include the latest data.')
    if len(frame):
        plot(px.scatter(frame, x='distance_miles', y='moving_time_min', hover_name='name',
                        labels={'distance_miles': 'Measured moving miles', 'moving_time_min': 'Moving minutes'}), 'dashboard_distance')
    predictions = repo.predictions()
    if predictions:
        st.subheader('Saved trail predictions')
        st.dataframe(pd.DataFrame(predictions).drop(columns=['user_id', 'details'], errors='ignore'), hide_index=True, width='stretch')


def select_all_training(hike_ids):
    enabled = st.session_state['training_select_all']
    for hike_id in hike_ids:
        st.session_state['training_' + hike_id] = enabled


def training_selection(repo, hikes):
    st.subheader('Hikes for training')
    st.caption('Choose individual hikes or use all uploaded data. Selections save automatically. Then visit My Models to train or retrain. Quality filters still apply.')
    selectable = [h for h in hikes if not h.get('deleting')]
    for hike in hikes:
        key = 'training_' + hike['id']
        if hike.get('deleting'):
            st.session_state[key] = False
        elif key not in st.session_state:
            st.session_state[key] = hike['training_enabled']
    st.session_state['training_select_all'] = bool(selectable) and all(
        st.session_state['training_' + h['id']] for h in selectable)
    st.checkbox('Use all uploaded data to train the models', key='training_select_all', disabled=not selectable,
                on_change=select_all_training, args=([h['id'] for h in selectable],))
    for hike in hikes:
        hike_type = {'day_hike': 'Day Hiking', 'backpacking': 'Backpacking'}.get(hike['hike_type'], 'Unknown hike type')
        with st.container(border=True):
            selection, details = st.columns([5, 1], vertical_alignment='center')
            with selection:
                st.checkbox(hike['name'], key='training_' + hike['id'], disabled=bool(hike.get('deleting')))
                st.caption(f"{hike['activity_date'][:10]} · {hike_type} · "
                           f"{number(hike['distance_miles'], ' mi', 2)} · "
                           f"{number(hike['elevation_gain_ft'], ' ft gain', 0)} · "
                           f"{duration(hike['moving_time_min'])} moving · {duration(hike['elapsed_time_min'])} elapsed · "
                           f"Quality: {hike['gps_quality_status'] or 'No flags'}")
                if hike.get('deleting'):
                    st.caption('Deletion pending; excluded from training.')
            if details.button('View details', key='details_' + hike['id']):
                st.session_state['history_detail_hike_id'] = hike['id']
    selected = sum(st.session_state['training_' + h['id']] for h in selectable)
    st.caption(f'{selected} of {len(selectable)} hikes selected for training.')
    changed = [h for h in selectable if st.session_state['training_' + h['id']] != h['training_enabled']]
    if changed:
        for hike in changed:
            repo.set_training(hike['id'], st.session_state['training_' + hike['id']])
        st.success('Training selections saved automatically. Visit My Models to train or retrain with these hikes.')


def history_page(repo):
    st.title('My Hiking History')
    st.write('Only completed, timestamped recordings become historical activities. Planned routes belong on Trail Predictor.')
    existing = repo.list_hikes()
    config = ImportConfig(**existing[0]['import_config']) if existing else ImportConfig()
    break_config = BreakConfig(**existing[0]['break_config']) if existing else BreakConfig()
    with st.expander('Upload historical GPX recordings', expanded=True):
        files = st.file_uploader('Historical recordings', type=['gpx'], accept_multiple_files=True, key='historical_files')
        label = st.selectbox('Hike type for these recordings', ['Unknown', 'Day Hiking', 'Backpacking'])
        hike_type = {'Unknown': None, 'Day Hiking': 'day_hike', 'Backpacking': 'backpacking'}[label]
        upload_key = (tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files), hike_type, repr(config), repr(break_config))
        if st.session_state.get('history_upload_key') != upload_key:
            st.session_state.pop('historical_previews', None)
        if st.button('Preview recordings', disabled=not files):
            if len(files) > 20 or sum(f.size for f in files) > 60 * 1024 * 1024:
                st.error('Preview at most 20 files and 60 MiB at a time.')
            else:
                previews = []
                for uploaded in files:
                    try:
                        previews.append(preview_recording(uploaded.getvalue(), uploaded.name, hike_type,
                                                         config=config, break_config=break_config))
                    except ValueError as error:
                        st.error(f'{uploaded.name}: {error}')
                st.session_state['historical_previews'] = previews
                st.session_state['history_upload_key'] = upload_key
        previews = st.session_state.get('historical_previews', [])
        for preview in previews:
            s = preview.summary
            st.markdown(f'**{preview.filename}**')
            st.write(f"{s['distance_miles']:.2f} moving miles · {duration(s['moving_time_min'])} moving · {duration(s['stationary_time_min'])} stopped · {duration(s['unknown_time_min'])} unresolved")
            st.caption(f"Elevation gain: {number(s['elevation_gain_ft'], ' ft', 0)}. Quality: {s['quality_flags'] or 'No flags'}.")
            st.text_input('Activity name', value=preview.filename, key='name_' + preview.id)
            st.checkbox('Include this completed activity in training (quality filters still apply)', key='eligible_' + preview.id)
        if previews and st.button('Confirm import to my account', type='primary'):
            saved = 0
            failed = []
            for preview in previews:
                try:
                    _, created = save_recording(repo, preview, name=st.session_state['name_' + preview.id],
                                                training_enabled=st.session_state['eligible_' + preview.id])
                    saved += created
                except Exception:
                    failed.append(preview)
                    st.error(f'{preview.filename}: import was not confirmed. Retry the retained preview; the app will check for an already completed import.')
            st.session_state['historical_previews'] = failed
            st.success(f'Imported {saved} recordings. Duplicates were retained once. Original local files are untouched.')
    hikes = repo.list_hikes()
    if not hikes:
        st.info('Your account has no recorded hikes yet.')
        return
    training_selection(repo, hikes)
    hike = next((h for h in hikes if h['id'] == st.session_state.get('history_detail_hike_id')), hikes[0])
    hike_id = hike['id']
    st.subheader('Activity details')
    st.write(hike['name'])
    if hike.get('deleting'):
        st.warning('Deletion was interrupted. Retry Delete this hike to finish; this hike is excluded from training.')
    if st.button('Load activity charts and original GPX'):
        contents = repo.download('hike-gpx', hike['original_gpx_storage_path'])
        preview = preview_recording(contents, 'recording.gpx', hike['hike_type'])
        st.session_state['activity_detail'] = (hike_id, contents, preview)
    detail = st.session_state.get('activity_detail')
    if detail and detail[0] == hike_id:
        st.download_button('Download original GPX', detail[1], 'recording.gpx', 'application/gpx+xml')
        points = detail[2].points
        plot(px.line(points, x='raw_distance_miles', y='smoothed_elevation_m', labels={'raw_distance_miles': 'Recorded miles', 'smoothed_elevation_m': 'Elevation (m)'}), 'history_elevation')
        segments = detail[2].segments
        if not segments.empty:
            pace = segments.assign(pace=segments.moving_time_min/segments.distance_miles.replace(0, float('nan')))
            plot(px.bar(pace, x='segment_index', y='pace', labels={'pace': 'Moving min/mile'}), 'history_pace')
    events = [e for e in repo.snapshot()['breaks'] if e['hike_id'] == hike_id]
    if events:
        st.subheader('Review detected breaks')
        st.caption('Moving intervals cannot also become breaks. Unknown, short, camping and non-hiking intervals are excluded from ordinary break targets. Saved corrections require explicit retraining.')
        original = pd.DataFrame([{**e['data'], 'id': e['id'], 'hike_id': hike_id,
                    'break_classification': e['classification'], 'start_time': e['start_time'], 'end_time': e['end_time']} for e in events])
        for c in ('start_time', 'end_time'):
            original[c] = pd.to_datetime(original[c], utc=True)
        edited = st.data_editor(original, key='review_' + hike_id, hide_index=True, width='stretch',
            disabled=[c for c in original if c not in ('break_classification','correction_notes')],
            column_config={'break_classification': st.column_config.SelectboxColumn('Classification', options=CLASSIFICATIONS, required=True)})
        if st.button('Save break corrections'):
            validate_break_edits(original, edited, BreakConfig(**hike['break_config']))
            changed = edited.loc[edited.break_classification.ne(original.break_classification) | edited.correction_notes.fillna('').ne(original.correction_notes.fillna(''))]
            edits = [{'id': r.id, 'classification': r.break_classification, 'notes': r.correction_notes or ''} for r in changed.itertuples()]
            repo.correct_breaks(hike_id, edits)
            st.success('Break corrections saved. Retrain models to use these labels.')
    st.subheader('Delete this hike')
    st.caption('Removes this recording, its segments and breaks. Existing trained versions remain historical snapshots; retrain to exclude the deleted hike from future predictions.')
    confirmed = st.checkbox('Permanently delete this selected hike and its GPX file', key='confirm_delete_' + hike_id)
    if st.button('Delete this hike', disabled=not confirmed):
        repo.delete_hike(hike_id)
        st.session_state.pop('activity_detail', None)
        st.rerun()


def models_page(repo, codec, artifact, breaks, models):
    st.title('My Models')
    st.write('Training uses only your confirmed completed recordings. Evaluation holds out entire hikes; GPS points and neighboring segments are not independent hikes.')
    if st.button('Retrain Models' if models else 'Train Models', type='primary'):
        with st.spinner('Preprocessing your training snapshot and evaluating both models…'):
            train_personal_models(repo, codec)
        st.session_state.pop('_models', None)
        st.rerun()
    st.info(artifact.reason)
    st.write('**Break predictor:** ' + breaks.reason)
    st.metric('Eligible break-training hikes', breaks.training_hikes)
    st.caption('Break allowance uses elapsed time minus moving time, including short pauses. '
               'Unresolved GPS intervals may include unmeasured movement. The moving-time model is evaluated separately.')
    if models and breaks.evaluation.get('target_definition') != NON_MOVING_TARGET:
        st.info('Break training now includes total non-moving time. Select Retrain Models to update your saved model using existing uploads.')
    metrics = breaks.evaluation.get('metrics', [])
    if metrics:
        st.dataframe(pd.DataFrame(metrics), hide_index=True, width='stretch')
        st.caption('Break comparisons use held-out whole-hike MAE in minutes. No RMSE is invented where the existing break evaluator supplies only MAE.')
    versions = repo.models()
    if versions:
        st.subheader('Version history')
        st.dataframe(pd.DataFrame(versions)[['model_type','version','trained_at','training_hike_count',
            'training_segment_count','is_active','status']], hide_index=True, width='stretch')
    with st.expander('Moving-time evaluation, coefficients and baseline comparison'):
        performance_page(artifact)


def trail_page(repo, artifact, breaks, models):
    with st.sidebar:
        uploaded = st.file_uploader('Upload planned GPX', type=['gpx'], key='planned_gpx')
        hike_label = st.selectbox('Hike type', ['Day Hiking', 'Backpacking'],
            index=0 if repo.profile()['preferred_hike_type']=='day_hike' else 1)
        direction_label = st.selectbox('Route direction', ['One Way', 'Out and Back'])
        if direction_label == 'Out and Back':
            st.caption('Upload only the outward leg. The return geometry is reversed and predicted separately.')
        include = st.checkbox('Include predicted break time', value=True)
        override = st.checkbox('Override break estimate', disabled=not include)
        allowance = st.number_input('Planned breaks (minutes)', min_value=0, max_value=10080, step=5, disabled=not(include and override))
        clicked = st.button('Predict Hiking Time', disabled=uploaded is None, type='primary')
    hike_type = 'day_hike' if hike_label == 'Day Hiking' else 'backpacking'
    direction = 'one-way' if direction_label == 'One Way' else 'out-and-back'
    route, contents, key = None, None, None
    if uploaded:
        contents = uploaded.getvalue()
        key = (repo.user_id, hashlib.sha256(contents).hexdigest(), uploaded.name, hike_type, direction,
               repr(artifact.config), id(type(artifact)), id(type(breaks)),
               tuple((r['id'],r['artifact_hash']) for r in models))
        if st.session_state.get('cloud_route_key') != key:
            st.session_state.pop('prediction', None)
            st.session_state['cloud_route'] = preview_upload(contents, uploaded.name, hike_type, direction, config=artifact.config)
            st.session_state['cloud_route_key'] = key
        route = st.session_state['cloud_route']
    for row in models:
        st.caption(f"{row['model_type'].title()} version trained {row['trained_at'][:10]} using {row['training_hike_count']} eligible hikes.")
    predictor_page(route, contents, key, uploaded, hike_type, direction, allowance, clicked, artifact,
                   breaks, include, override)
    prediction = st.session_state.get('prediction') if st.session_state.get('prediction_key') == key and key else None
    if prediction:
        stats = trip_statistics(prediction, include_predicted_breaks=include,
                                manual_break_minutes=allowance if override and include else None)
        report = {'route_name': route.name, 'method': prediction.forecast.label,
                  'hike_type': hike_type, 'direction': direction, **stats,
                  'model_versions': [{'id': r['id'], 'type': r['model_type'], 'trained_at': r['trained_at']} for r in models]}
        st.download_button('Download prediction report', json.dumps(report, indent=2), 'hiking_prediction.json', 'application/json')
        if st.button('Save prediction to my account'):
            by_type = {r['model_type']: r['id'] for r in models}
            repo.save_prediction({'moving_model_version_id': by_type.get('moving'), 'break_model_version_id': by_type.get('break'),
                'route_name': route.name[:250], 'distance_miles': stats['distance_miles'], 'elevation_gain_ft': stats['elevation_gain_ft'],
                'elevation_loss_ft': stats['elevation_loss_ft'], 'hike_type': hike_type, 'route_direction': direction,
                'predicted_moving_time_min': stats['moving_minutes'],
                'predicted_break_time_min': stats['break_minutes'] if stats['break_estimate_available'] or override or not include else None,
                'predicted_total_time_min': stats['total_minutes'], 'details': report})
            st.success('Prediction saved privately.')


def settings_page(repo, session, config):
    st.title('Account Settings')
    profile = repo.profile()
    st.write('Google account: ' + session.user.get('email', 'Email unavailable'))
    st.caption('Account created: ' + profile['created_at'])
    st.caption('Account UUID: ' + session.user_id)
    name = st.text_input('Display name', profile['display_name'])
    preferred = st.selectbox('Preferred hike type', ['day_hike','backpacking'], index=0 if profile['preferred_hike_type']=='day_hike' else 1)
    if st.button('Save preferences'):
        repo.save_profile(name, preferred)
        st.success('Preferences saved.')
    if st.button('Prepare my private data export'):
        st.session_state['account_export'] = export_personal_data(repo)
    if st.session_state.get('account_export'):
        st.download_button('Download my data', st.session_state['account_export'], 'my_hiking_data.zip', 'application/zip')
    st.caption('Exports include location history and model artifacts. Store downloaded copies privately.')
    if getattr(config, 'auth_mode', 'service') == 'streamlit':
        from .community_auth import account_actions, logout_button
        account_actions(config, session)
        logout_button(config, key='settings_logout')
    else:
        st.link_button('Delete all hiking application data', config.auth_url + '/account/clear')
        st.link_button('Delete my application account', config.auth_url + '/account/delete')
        st.caption('Each deletion opens an explicit confirmation page. Account deletion also removes the Supabase sign-in identity; your Google account is preserved. Sign in again immediately before requesting deletion.')
        st.link_button('Logout', config.auth_url + '/account/logout')
    with st.expander('Import the original single-user project'):
        st.write('The developer migration command runs locally on the owner’s machine, never on behalf of other web users. See docs/SUPABASE_SETUP.md. Original GPX files, datasets and models are preserved. Legacy model files are kept only as opaque backups and are never loaded.')
        archive = st.file_uploader('Developer migration ZIP', type=['zip'], key='legacy_bundle')
        consent = st.checkbox('This archive is my own history. Import it only into the account shown above.')
        if st.button('Import my legacy project once', disabled=archive is None or not consent):
            from .migrate_local import import_bundle
            with st.spinner('Verifying backup and importing your completed recordings…'):
                count = import_bundle(repo, archive.getvalue())
            st.success(f'Imported {count} recordings and verified the archive backup. Visit My Models to train trusted versions. Legacy models and processed datasets remain downloadable backup bytes.')


def main():
    st.set_page_config(page_title='Personal Hiking Time', page_icon='⛰', layout='wide')
    design()
    try:
        config = CloudConfig.from_env()
    except ValueError as error:
        clear_private_state()
        landing(message=str(error))
        return
    if getattr(config, 'auth_mode', 'service') == 'streamlit':
        from .community_auth import CommunityAuthManager
        try:
            session = CommunityAuthManager(config).restore(st.session_state.get('_community_auth'))
        except AuthenticationError as error:
            st.session_state.pop('_community_auth', None)
            clear_private_state()
            landing(config, str(error))
            return
    else:
        try:
            manager = AuthManager(config)
        except OSError:
            clear_private_state()
            landing(config, 'The private session store is unavailable. Check the auth service volume and permissions.')
            return
        sid = st.context.cookies.get(config.session_cookie)
        try:
            session = manager.restore(sid)
        except AuthenticationError as error:
            clear_private_state()
            landing(config, str(error))
            return
    clear_private_state(session.user_id)
    repo = SupabaseRepository(config, session)
    codec = TrustedModelCodec(config.model_key)
    with st.sidebar:
        st.markdown('### ⛰ Personal hiking time')
        st.caption('Your history. Your pace.')
        page = st.radio('Navigate', PAGES, key='account_page')
        if getattr(config, 'auth_mode', 'service') == 'streamlit':
            from .community_auth import logout_button
            logout_button(config, key='sidebar_logout')
        else:
            st.link_button('Logout', config.auth_url + '/account/logout')
    try:
        profile = repo.profile()  # Fail closed if migrations/profile initialization are unavailable.
        if profile.get('deletion_requested_at'):
            st.warning('Your confirmed cleanup is pending. Hiking data is locked against changes until the operator retries the cleanup job.')
            return
        rows = [r for r in repo.models() if r['is_active']]
        cache_key = (session.user_id, id(ModelArtifact), id(BreakArtifact),
                     tuple((r['id'],r['artifact_hash'],repr(r['provenance'])) for r in rows))
        model_error = None
        try:
            if st.session_state.get('_models_key') != cache_key or '_models' not in st.session_state:
                st.session_state['_models'] = load_personal_models(repo, codec)
                st.session_state['_models_key'] = cache_key
            artifact, breaks, models = st.session_state['_models']
        except ValueError as error:
            model_error = str(error)
            artifact, breaks, models = ModelArtifact(reason=model_error), BreakArtifact(reason=model_error), rows
            st.warning('Saved model verification failed. Visit My Models to retrain; your history and settings remain available.')
        if page == 'My Dashboard':
            dashboard(repo, models)
        elif page == 'My Hiking History':
            history_page(repo)
        elif page == 'My Models':
            models_page(repo, codec, artifact, breaks, models)
        elif page == 'Trail Predictor':
            if model_error:
                st.error(model_error)
            else:
                trail_page(repo, artifact, breaks, models)
        else:
            settings_page(repo, session, config)
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        # SDK errors can contain request details. Never echo tokens or raw responses.
        st.error(private_request_message(error))
