"""Hosted sign-in keeps Supabase verification and user-token isolation."""
from dataclasses import replace
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hiking_time_predictor.auth_sessions import AuthenticationError, UserSession
from hiking_time_predictor.cloud_config import CloudConfig
from hiking_time_predictor.community_auth import BrowserCredentials, CommunityAuthManager

USER = '11111111-1111-4111-8111-111111111111'
OTHER = '22222222-2222-4222-8222-222222222222'


def setup_manager():
    user = SimpleNamespace(id=USER, app_metadata={'providers': ['google']},
                           model_dump=lambda **_: {'id': USER, 'email': 'synthetic@example.invalid'})
    session = SimpleNamespace(user=user, access_token='access-private', refresh_token='refresh-private',
                              expires_at=time.time() + 3600)
    client = Mock()
    client.auth.sign_in_with_id_token.return_value.session = session
    client.auth.get_user.return_value.user = user
    config = CloudConfig('https://project.supabase.co', 'public', app_url='https://hiking.streamlit.app',
                         model_key='a' * 48, auth_mode='streamlit',
                         google_client_id='synthetic.apps.googleusercontent.com')
    return CommunityAuthManager(config, client_factory=lambda: client), client, session


def test_google_nonce_exchange_and_verified_identity():
    manager, client, _ = setup_manager()
    raw_nonce = 'random-server-nonce-' * 3
    credentials = manager.exchange('google-token', raw_nonce, time.time())
    client.auth.sign_in_with_id_token.assert_called_once_with(
        {'provider': 'google', 'token': 'google-token', 'nonce': raw_nonce})
    assert credentials.user_id == USER
    assert manager.restore(credentials).user_id == USER
    assert 'access-private' not in repr(credentials) and 'refresh-private' not in repr(credentials)


@pytest.mark.parametrize('token,issued', [('', 0), ('token', -700), ('token', 10), ('x' * 16385, 0)])
def test_rejects_invalid_or_expired_challenge_without_exchange(token, issued):
    manager, client, _ = setup_manager()
    with pytest.raises(AuthenticationError):
        manager.exchange(token, 'nonce' * 12, time.time() + issued)
    client.auth.sign_in_with_id_token.assert_not_called()


@pytest.mark.parametrize('failure', ['provider', 'identity', 'server'])
def test_exchange_fails_closed_and_redacts_errors(failure):
    manager, client, session = setup_manager()
    if failure == 'provider':
        session.user.app_metadata = {'providers': ['email']}
    elif failure == 'identity':
        client.auth.get_user.return_value.user = SimpleNamespace(id=OTHER, app_metadata={'providers': ['google']})
    else:
        client.auth.get_user.side_effect = RuntimeError('access-private refresh-private')
    with pytest.raises(AuthenticationError) as error:
        manager.exchange('google-token', 'nonce' * 12, time.time())
    assert 'private' not in str(error.value)


def test_refresh_rotation_and_session_identity_check():
    manager, client, _ = setup_manager()
    credentials = BrowserCredentials(USER, 'old-access', 'old-refresh', time.time() - 1, time.time())
    client.auth.refresh_session.return_value.session = SimpleNamespace(
        access_token='new-access', refresh_token='new-refresh', expires_at=time.time() + 3600)
    assert manager.restore(credentials).access_token == 'new-access'
    client.auth.refresh_session.assert_called_once_with('old-refresh')
    assert credentials.refresh_token == 'new-refresh'
    manager.restore(credentials)
    assert client.auth.refresh_session.call_count == 1
    client.auth.get_user.return_value.user.id = OTHER
    with pytest.raises(AuthenticationError):
        manager.restore(credentials)


def test_session_expiry_and_missing_credentials():
    manager, client, _ = setup_manager()
    old = BrowserCredentials(USER, 'access', 'refresh', time.time() + 3600, time.time() - manager.SESSION_TTL)
    for value in [None, old]:
        with pytest.raises(AuthenticationError):
            manager.restore(value)
    client.auth.get_user.assert_not_called()


def test_streamlit_config_does_not_require_disk_key_or_matching_auth_host():
    manager, _, _ = setup_manager()
    manager.config.validate()
    assert manager.config.session_key == ''
    with pytest.raises(ValueError, match='AUTH_SESSION_KEY'):
        replace(manager.config, auth_mode='service').validate()
    with pytest.raises(ValueError, match='GOOGLE_CLIENT_ID'):
        replace(manager.config, google_client_id='').validate()
    with pytest.raises(ValueError, match='HTTPS'):
        replace(manager.config, app_url='http://hiking.streamlit.app').validate()


def test_streamlit_secrets_fallback_and_admin_separation(monkeypatch, tmp_path):
    import streamlit as st
    monkeypatch.chdir(tmp_path)
    for key in ['AUTH_MODE','SUPABASE_URL','SUPABASE_PUBLISHABLE_KEY','APP_URL','GOOGLE_CLIENT_ID',
                'MODEL_SIGNING_KEY','SUPABASE_SECRET_KEY']:
        monkeypatch.delenv(key, raising=False)
    secrets = {'AUTH_MODE': 'streamlit', 'SUPABASE_URL': 'https://project.supabase.co',
               'SUPABASE_PUBLISHABLE_KEY': 'public', 'APP_URL': 'https://hiking.streamlit.app',
               'GOOGLE_CLIENT_ID': 'synthetic.apps.googleusercontent.com',
               'MODEL_SIGNING_KEY': 'a' * 48, 'SUPABASE_SECRET_KEY': 'private-admin'}
    monkeypatch.setattr(st, 'secrets', secrets)
    assert CloudConfig.from_env().auth_mode == 'streamlit'
    assert CloudConfig.from_env().admin_key == ''
    assert CloudConfig.from_env(include_admin=True).admin_key == 'private-admin'


def hosted_settings(config, user_id, created_at):
    import streamlit as st
    from hiking_time_predictor.community_auth import account_actions
    from hiking_time_predictor.auth_sessions import UserSession
    account_actions(config, UserSession(user_id, 'private-access', {}, created_at))


def test_hosted_deletion_requires_confirmation_and_recent_identity(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from hiking_time_predictor import account_cleanup
    manager, _, _ = setup_manager()
    deleted = Mock()
    admin = Mock()
    monkeypatch.setattr(account_cleanup, 'delete_account_data', deleted)
    monkeypatch.setattr(account_cleanup, 'admin_client', lambda _: admin)
    monkeypatch.setattr(CloudConfig, 'from_env', lambda **_: manager.config)
    credentials = BrowserCredentials(USER, 'private-access', 'private-refresh', time.time() + 3600, time.time())
    monkeypatch.setattr(CommunityAuthManager, 'restore', lambda self, value:
                        UserSession(value.user_id, value.access_token, {}, value.created_at))
    monkeypatch.setattr(CommunityAuthManager, 'logout', lambda *_: None)
    app = AppTest.from_function(hosted_settings, args=(manager.config, USER, time.time())).run()
    app.session_state['_community_auth'] = credentials
    app.button[0].click().run()
    deleted.assert_not_called()
    app.text_input[0].set_value('DELETE MY HIKING DATA')
    app.session_state['_community_auth'].created_at = time.time() - 601
    app.button[0].click().run()
    deleted.assert_not_called()
    assert any('ten minutes' in e.value for e in app.error)
    app.session_state['_community_auth'].created_at = time.time()
    app.text_input[0].set_value('DELETE MY HIKING DATA')
    app.button[0].click().run()
    assert deleted.call_count == 1, [e.value for e in app.error]
    deleted.assert_called_once_with(admin, USER, delete_identity=False)
    assert not app.exception


def test_hosted_main_preserves_credentials_and_clears_other_users_state(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from hiking_time_predictor import cloud_ui
    from hiking_time_predictor.auth_sessions import UserSession
    from test_personal_hiking import FakeRepo
    manager, _, _ = setup_manager()
    repo = FakeRepo()
    repo.profile = lambda: {'id': USER, 'created_at': '2026-01-01', 'display_name': '', 'preferred_hike_type': 'day_hike'}
    repo.predictions = lambda: []
    monkeypatch.setattr(CloudConfig, 'from_env', lambda: manager.config)
    monkeypatch.setattr(CommunityAuthManager, 'restore', lambda self, value:
                        UserSession(value.user_id, value.access_token, {}, value.created_at))
    monkeypatch.setattr(cloud_ui, 'SupabaseRepository', lambda *_: repo)
    monkeypatch.setattr(cloud_ui, 'AuthManager', lambda *_: pytest.fail('Hosted mode opened the disk vault'))
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'streamlit_app.py'), default_timeout=15)
    credentials = BrowserCredentials(USER, 'private-access', 'private-refresh', time.time() + 3600, time.time())
    app.session_state['_community_auth'] = credentials
    app.session_state['_bound_user'] = OTHER
    app.session_state['account_export'] = b'other-users-private-export'
    app.run()
    assert not app.exception and not app.error
    assert app.session_state['_community_auth'].user_id == USER
    assert app.session_state['_bound_user'] == USER
    assert 'account_export' not in app.session_state
