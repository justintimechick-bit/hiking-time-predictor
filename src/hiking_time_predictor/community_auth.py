"""Google Identity Services -> Supabase sessions for a single Streamlit process.

Only a public client ID and hashed challenge enter the component. Supabase
validates Google's ID token and nonce; all app queries still use a user JWT.
Refresh credentials stay in server memory for this browser's Streamlit session.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import secrets
from threading import RLock
import time
from uuid import UUID

from supabase import ClientOptions, create_client

from .auth_sessions import AuthenticationError, UserSession


@dataclass
class BrowserCredentials:
    user_id: str
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    created_at: float
    lock: object = field(default_factory=RLock, repr=False, compare=False)


class CommunityAuthManager:
    SESSION_TTL = 7 * 24 * 3600
    FLOW_TTL = 600

    def __init__(self, config, *, client_factory=None):
        self.config = config
        self.client_factory = client_factory

    def client(self):
        if self.client_factory:
            return self.client_factory()
        return create_client(self.config.supabase_url, self.config.publishable_key,
                             options=ClientOptions(auto_refresh_token=False, persist_session=False,
                                                   postgrest_client_timeout=30))

    def exchange(self, credential, nonce, issued_at):
        if (not isinstance(credential, str) or not 1 <= len(credential) <= 16384
                or not isinstance(nonce, str) or len(nonce) < 32
                or not 0 <= time.time() - issued_at < self.FLOW_TTL):
            raise AuthenticationError('Sign-in expired. Start again.')
        try:
            client = self.client()
            result = client.auth.sign_in_with_id_token({'provider': 'google', 'token': credential, 'nonce': nonce})
            session = result.session
            user = client.auth.get_user(session.access_token).user
            if 'google' not in (user.app_metadata.get('providers') or [user.app_metadata.get('provider')]):
                raise ValueError('Unexpected identity provider')
            user_id = str(UUID(str(user.id)))
            if str(session.user.id) != user_id:
                raise ValueError('Identity mismatch')
            return BrowserCredentials(user_id, session.access_token, session.refresh_token,
                                      session.expires_at, time.time())
        except Exception:
            raise AuthenticationError('Google sign-in could not be verified. Try again.') from None

    def restore(self, credentials):
        if not isinstance(credentials, BrowserCredentials):
            raise AuthenticationError('Continue with Google to access your hiking history.')
        with credentials.lock:
            try:
                if not 0 <= time.time() - credentials.created_at < self.SESSION_TTL:
                    raise ValueError('Session expired')
                client = self.client()
                if credentials.expires_at <= time.time() + 60:
                    session = client.auth.refresh_session(credentials.refresh_token).session
                    # Keep rotated refresh credentials even if a later verification fails.
                    credentials.access_token = session.access_token
                    credentials.refresh_token = session.refresh_token
                    credentials.expires_at = session.expires_at
                user = client.auth.get_user(credentials.access_token).user
                if str(user.id) != credentials.user_id:
                    raise ValueError('Identity mismatch')
                return UserSession(credentials.user_id, credentials.access_token,
                                   user.model_dump(mode='json'), credentials.created_at)
            except Exception:
                raise AuthenticationError('Session expired or unavailable. Sign in again.') from None

    def logout(self, credentials):
        try:
            self.client().auth.admin.sign_out(credentials.access_token, 'global')
        except Exception:
            # Local credentials are always discarded by the caller.
            pass


def google_sign_in(config):
    import streamlit as st
    from streamlit.components.v1 import declare_component

    now = time.time()
    challenge = st.session_state.get('_google_challenge')
    if not challenge or not 0 <= now - challenge[1] < CommunityAuthManager.FLOW_TTL:
        challenge = (secrets.token_urlsafe(48), now)
        st.session_state['_google_challenge'] = challenge
    nonce, issued_at = challenge
    component = declare_component('hiking_google_login', path=str(Path(__file__).with_name('google_login')))
    value = component(client_id=config.google_client_id, nonce=hashlib.sha256(nonce.encode()).hexdigest(),
                      key='google_login_' + hashlib.sha256(nonce.encode()).hexdigest()[:16], default=None)
    if value is not None:
        # A submitted challenge is single use, including when verification fails.
        st.session_state.pop('_google_challenge', None)
        try:
            credentials = CommunityAuthManager(config).exchange(value, nonce, issued_at)
        except AuthenticationError:
            st.warning('Google sign-in failed or expired. Select Start sign-in again.')
            if st.button('Start sign-in again'):
                st.rerun()
        else:
            st.session_state['_community_auth'] = credentials
            st.rerun()


def logout_button(config, *, key):
    import streamlit as st
    with st.popover('Logout'):
        st.write('Are you sure you want to log out?')
        if st.button('Yes, log out', key=key):
            credentials = st.session_state.get('_community_auth')
            if credentials:
                CommunityAuthManager(config).logout(credentials)
            st.session_state.clear()
            st.rerun()
        st.caption('Close this panel to stay signed in. Logout also revokes your Supabase refresh sessions.')


def account_actions(config, session):
    import streamlit as st
    from .account_cleanup import admin_client, delete_account_data
    from .cloud_config import CloudConfig

    st.subheader('Delete hiking data or account')
    action = st.selectbox('Deletion action', ['Delete all hiking data', 'Delete my application account'])
    delete_identity = action == 'Delete my application account'
    phrase = 'DELETE MY ACCOUNT' if delete_identity else 'DELETE MY HIKING DATA'
    st.caption('This permanently removes your private hiking data and models. Your Google account is preserved. Sign in within the last ten minutes before confirming.')
    with st.form('community_account_delete'):
        confirmation = st.text_input('Type ' + phrase + ' to confirm')
        submitted = st.form_submit_button('Permanently delete')
    if submitted:
        if confirmation != phrase:
            st.error('Confirmation did not match.')
            return
        try:
            current = CommunityAuthManager(config).restore(st.session_state.get('_community_auth'))
            if current.user_id != session.user_id or time.time() - current.created_at > 600:
                st.error('Log out and sign in again, then confirm within ten minutes.')
                return
            # Privileged credentials are read only in this confirmed cleanup path.
            admin_config = CloudConfig.from_env(include_admin=True)
            delete_account_data(admin_client(admin_config), current.user_id, delete_identity=delete_identity)
        except Exception:
            st.error('Cleanup did not complete. Your data stays locked if cleanup started; contact the operator to retry the pending job.')
            return
        CommunityAuthManager(config).logout(st.session_state['_community_auth'])
        st.session_state.clear()
        st.rerun()
