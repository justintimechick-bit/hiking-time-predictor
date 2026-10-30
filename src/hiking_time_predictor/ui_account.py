"""Account: identity, preferences, data export, deletion and the owner-only legacy import."""
import streamlit as st

from .personal_hiking import export_personal_data
from .ui_format import HIKE_TYPE_LABELS, friendly_date, md_escape
from .ui_nav import flash


def settings_page(repo, session, config):
    st.title('Account')
    profile = repo.profile()
    streamlit_mode = getattr(config, 'auth_mode', 'service') == 'streamlit'
    with st.container(border=True):
        st.markdown('**Signed in with Google**')
        st.text(session.user.get('email', 'Email unavailable'))
        st.caption(f"Member since {friendly_date(profile.get('created_at'), st.context.timezone)}")
        if streamlit_mode:
            from .community_auth import logout_button
            logout_button(config, key='settings_logout')
        else:
            st.link_button('Log out', config.auth_url + '/account/logout', icon=':material/logout:')
        with st.expander('Support details', type='compact'):
            st.caption('Account ID: ' + md_escape(session.user_id))
    with st.form('profile_preferences'):
        st.markdown('**Preferences**')
        name = st.text_input('Display name', profile.get('display_name') or '', max_chars=200,
                             help='Shown when you open your dashboard.')
        preferred = st.segmented_control(
            'Default hike type', list(HIKE_TYPE_LABELS), format_func=HIKE_TYPE_LABELS.get, required=True,
            default=profile.get('preferred_hike_type') if profile.get('preferred_hike_type') in HIKE_TYPE_LABELS else 'day_hike',
            help='Pre-selected when you plan a hike or add recordings.')
        if st.form_submit_button('Save preferences', type='primary'):
            repo.save_profile(name, preferred or 'day_hike')
            flash('Preferences saved.')
            st.rerun()
    with st.container(border=True):
        st.markdown('**Download your data**')
        st.write('A .zip with your GPX recordings, hike details, detected stops, trained models and saved plans. '
                 'It contains your location history, so store it somewhere private.')
        if st.session_state.get('account_export'):
            st.download_button('Download my data (.zip)', st.session_state['account_export'], 'my_hiking_data.zip',
                               'application/zip', type='primary', icon=':material/download:', on_click='ignore')
        elif st.button('Prepare my download', icon=':material/folder_zip:'):
            with st.spinner('Packaging your data…'):
                st.session_state['account_export'] = export_personal_data(repo)
            st.rerun()
    if streamlit_mode:
        from .community_auth import account_actions
        account_actions(config, session)
    else:
        with st.container(key='danger_zone'):
            st.markdown('**Delete data or account**')
            st.link_button('Delete all hiking data', config.auth_url + '/account/clear')
            st.link_button('Delete my account', config.auth_url + '/account/delete')
            st.caption('Each deletion opens a confirmation page. Deleting your account removes your sign-in for this '
                       'app; your Google account is not affected. Sign in again just before deleting.')
    with st.expander('Advanced: import from the offline desktop version', type='compact', icon=':material/upload_file:'):
        st.write('Only for the owner of the original single-user app. Upload the migration .zip created on that '
                 'computer; it is imported into this account only. Old model files are kept as backups and are '
                 'never used for predictions.')
        archive = st.file_uploader('Migration .zip', type=['zip'], key='legacy_bundle')
        consent = st.checkbox('This archive is my own history. Import it only into the account shown above.')
        if st.button('Import my legacy project once', disabled=archive is None or not consent):
            from .migrate_local import import_bundle
            with st.spinner('Verifying the backup and importing your recordings…'):
                count = import_bundle(repo, archive.getvalue())
            st.success(f'Imported {count} recordings. Train your pace model on My pace to use them.')
