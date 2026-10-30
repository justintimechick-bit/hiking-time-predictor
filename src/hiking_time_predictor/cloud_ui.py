"""Streamlit account experience backed exclusively by the caller's Supabase JWT."""
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from .auth_sessions import AuthManager, AuthenticationError
from .break_model import BreakArtifact
from .predictor import ModelArtifact
from .cloud_config import CloudConfig
from .cloud_repository import SupabaseRepository, private_request_message
from .personal_hiking import load_personal_models
from .trusted_models import TrustedModelCodec
from .ui_account import settings_page
from .ui_dashboard import dashboard
from .ui_format import RULE_DEFINITION, RULE_NAME
from .ui_history import history_page, select_all_training, training_selection  # noqa: F401 (public API)
from .ui_models import models_page
from .ui_nav import CONTEXT_KEY, PAGES_KEY, SPEC, arrived, build_pages, context, show_flash
from .ui_plan import trail_page
from .web_components import design

ASSETS = Path(__file__).with_name('assets')
ABOUT = 'Hiking Time: plan hikes using your own pace. Your hikes, models and plans are private to your account.'


@dataclass
class PageContext:
    """Per-run objects for the page callables. Lives in this session's state only."""
    repo: object
    codec: object
    config: object
    session: object
    profile: dict
    artifact: object
    breaks: object
    models: list
    model_error: str | None = None


def clear_private_state(user_id=None):
    if st.session_state.get('_bound_user') != user_id:
        credentials = st.session_state.get('_community_auth')
        st.session_state.clear()
        if user_id:
            st.session_state['_bound_user'] = user_id
            if credentials and credentials.user_id == user_id:
                st.session_state['_community_auth'] = credentials


def _home():
    c = context()
    dashboard(c.repo, c.models, artifact=c.artifact, breaks=c.breaks, profile=c.profile)


def _plan():
    c = context()
    trail_page(c.repo, c.artifact, c.breaks, c.models, profile=c.profile, model_error=c.model_error)


def _hikes():
    c = context()
    history_page(c.repo, profile=c.profile, models=c.models)


def _pace():
    c = context()
    models_page(c.repo, c.codec, c.artifact, c.breaks, c.models, model_error=c.model_error)


def _account():
    c = context()
    settings_page(c.repo, c.session, c.config)


RENDERERS = {'home': _home, 'plan': _plan, 'hikes': _hikes, 'pace': _pace, 'account': _account}


def _register_hidden_pages():
    # Signed out: register the pages without showing or running them, so a link like
    # /plan survives Google sign-in instead of hitting "Page not found".
    st.navigation(list(build_pages(RENDERERS).values()), position='hidden')


def landing(config=None, message=None):
    with st.container(horizontal_alignment='center'):
        with st.container(width=760, key='landing'):
            st.space('small')
            st.title('Personal Hiking Time Predictor')
            st.markdown('<p class="landing-lead">Know how long your next hike will take, based on how '
                        '<em>you</em> actually hike.</p>', unsafe_allow_html=True)
            if message and config is None:
                st.warning('Sign-in is temporarily unavailable. Please try again later.', icon=':material/cloud_off:')
                st.caption(message)
            elif message:
                st.warning(message, icon=':material/schedule:')
            if config:
                if getattr(config, 'auth_mode', 'service') == 'streamlit':
                    from .community_auth import google_sign_in
                    google_sign_in(config)
                else:
                    st.link_button('Continue with Google', config.auth_url + '/auth/login', type='primary',
                                   icon=':material/login:')
            else:
                st.button('Continue with Google', disabled=True)
            st.caption(':material/lock: Free. Your hikes, models and plans are private to your account.')
            st.space('small')
            steps = st.columns(3, gap='medium')
            for column, (number, title, body) in zip(steps, (
                    ('1', 'Add past hikes', 'Upload GPX recordings from Strava, Garmin, AllTrails or any GPS watch.'),
                    ('2', 'Learn your pace', 'The app learns how fast you move on climbs and descents, and how long you rest.'),
                    ('3', 'Plan a route', 'Upload any route for moving time, breaks, a total, and when you\'ll be back.'))):
                with column, st.container(border=True, height='stretch'):
                    st.markdown(f'<span class="step-number">{number}</span>**{title}**', unsafe_allow_html=True)
                    st.caption(body)
            with st.container(border=True, key='landing_card'):
                st.caption('Example plan')
                st.markdown('**Mount Diablo summit trail** · 8.8 mi · ↑2,606 ft')
                st.metric('Estimated total trip', '4 hr 36 min', label_visibility='collapsed')
                st.caption('4 hr 14 min moving + 22 min breaks · start 8:00 AM, back around 12:36 PM · '
                           'about 5 hr before sunset')
            st.caption(f'No history yet? You still get an estimate from the {RULE_NAME} ({RULE_DEFINITION}) '
                       'until your own hikes are enough to learn from.')
            if getattr(config, 'auth_mode', 'service') == 'streamlit':
                st.caption('For your security, refreshing or closing this tab signs you out. Your saved hikes are kept.')


def _logout_control(config, key):
    if getattr(config, 'auth_mode', 'service') == 'streamlit':
        from .community_auth import logout_button
        logout_button(config, key=key)
    else:
        st.link_button('Log out', config.auth_url + '/account/logout', icon=':material/logout:')


def main():
    st.set_page_config(page_title='Hiking Time', page_icon=':material/hiking:', layout='wide',
                       menu_items={'About': ABOUT})
    design()
    try:
        config = CloudConfig.from_env()
    except ValueError as error:
        clear_private_state()
        _register_hidden_pages()
        landing(message=str(error))
        return
    if getattr(config, 'auth_mode', 'service') == 'streamlit':
        from .community_auth import CommunityAuthManager
        had_credentials = st.session_state.get('_community_auth') is not None
        try:
            session = CommunityAuthManager(config).restore(st.session_state.get('_community_auth'))
        except AuthenticationError as error:
            st.session_state.pop('_community_auth', None)
            clear_private_state()
            _register_hidden_pages()
            # A first visit needs no notice; an expired or failed session does.
            landing(config, str(error) if had_credentials else None)
            return
    else:
        try:
            manager = AuthManager(config)
        except OSError:
            clear_private_state()
            _register_hidden_pages()
            landing(config, 'The private session store is unavailable. Check the auth service volume and permissions.')
            return
        sid = st.context.cookies.get(config.session_cookie)
        try:
            session = manager.restore(sid)
        except AuthenticationError as error:
            clear_private_state()
            _register_hidden_pages()
            landing(config, str(error) if sid else None)
            return
    # Another account signed in within this browser session: reset to the dashboard.
    switched = st.session_state.get('_bound_user') not in (None, session.user_id)
    clear_private_state(session.user_id)
    repo = SupabaseRepository(config, session)
    codec = TrustedModelCodec(config.model_key)
    pages = build_pages(RENDERERS)
    st.session_state[PAGES_KEY] = pages
    st.logo(str(ASSETS / 'logo.svg'), size='large', icon_image=str(ASSETS / 'mark.svg'))
    try:
        profile = repo.profile()  # Fail closed if migrations/profile initialization are unavailable.
        if profile.get('deletion_requested_at'):
            st.navigation(list(pages.values()), position='hidden')
            st.warning('Your deletion request is being processed, so your hiking data is locked for now. '
                       'If this message is still here tomorrow, contact the app owner.', icon=':material/lock:')
            _logout_control(config, 'locked_logout')
            return
        page = st.navigation(list(pages.values()), position='top')
        if switched:
            st.switch_page(pages['home'])
        rows = [r for r in repo.models() if r['is_active']]
        cache_key = (session.user_id, id(ModelArtifact), id(BreakArtifact),
                     tuple((r['id'], r['artifact_hash'], repr(r['provenance'])) for r in rows))
        model_error = None
        try:
            if st.session_state.get('_models_key') != cache_key or '_models' not in st.session_state:
                st.session_state['_models'] = load_personal_models(repo, codec)
                st.session_state['_models_key'] = cache_key
            artifact, breaks, models = st.session_state['_models']
        except ValueError as error:
            model_error = str(error)
            artifact, breaks, models = ModelArtifact(reason=model_error), BreakArtifact(reason=model_error), rows
        st.session_state[CONTEXT_KEY] = PageContext(repo, codec, config, session, profile, artifact, breaks,
                                                    models, model_error)
        current = next((key for key, value in pages.items() if value.url_path == page.url_path), 'home')
        st.set_page_config(page_title=f'{SPEC[current].title} · Hiking Time')
        arrived(current)
        show_flash()
        if model_error and current not in ('plan', 'pace'):
            st.warning('Your saved pace model couldn\'t be verified. Retrain on My pace; your hikes and settings '
                       'are safe.', icon=':material/gpp_bad:')
        page.run()
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        # SDK errors can contain request details. Never echo tokens or raw responses.
        st.error(private_request_message(error))
