"""Actual installed Supabase SDK with a mocked Auth HTTP service."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import httpx
import pytest
from supabase import ClientOptions, create_client

from hiking_time_predictor.auth_service import create_app
from hiking_time_predictor.auth_sessions import AuthManager, AuthenticationError, SessionVault
from hiking_time_predictor.cloud_config import CloudConfig


@pytest.fixture
def auth_fixture(tmp_path):
    config = CloudConfig('https://test.supabase.co', 'sb_publishable_test', 'https://hiking.test',
        'https://hiking.test', tmp_path/'vault'/'sessions.db', Fernet.generate_key().decode(), 'm'*48, secure_cookies=True)
    user = {'id':str(uuid4()),'aud':'authenticated','role':'authenticated','email':'test@example.invalid',
            'app_metadata':{'provider':'google','providers':['google']},'user_metadata':{'full_name':'Synthetic Hiker'},
            'created_at':'2026-01-01T00:00:00Z'}
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path.endswith('/token'):
            payload = json.loads(request.content)
            if request.url.params.get('grant_type') == 'pkce':
                assert payload['code_verifier'] and payload['auth_code'] == 'single-use-code'
            return httpx.Response(200,json={'access_token':'access-secret-'+str(len(calls)),
                'refresh_token':'refresh-secret-'+str(len(calls)),'expires_in':3600,
                'expires_at':int(time.time())+3600,'token_type':'bearer','user':user})
        if request.url.path.endswith('/user'):
            assert request.headers['Authorization'].startswith('Bearer access-secret-')
            return httpx.Response(200,json=user)
        if request.url.path.endswith('/logout'):
            return httpx.Response(204)
        raise AssertionError(str(request.url))
    def factory(storage):
        return create_client(config.supabase_url,config.publishable_key,options=ClientOptions(
            flow_type='pkce',storage=storage,auto_refresh_token=False,persist_session=False,
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    manager = AuthManager(config,client_factory=factory)
    return config,manager,user,calls,factory


def sign_in(manager):
    state,url=manager.begin()
    query=parse_qs(urlparse(url).query)
    assert query['provider']==['google']
    assert query['code_challenge_method']==['s256']
    assert '/auth/callback?state='+state in query['redirect_to'][0]
    return manager.finish(state,state,'single-use-code')


def test_pkce_survives_new_manager_and_browser_redirect(auth_fixture):
    config,manager,user,calls,factory=auth_fixture
    state,url=manager.begin()
    new_manager=AuthManager(config,client_factory=factory)
    sid=new_manager.finish(state,state,'single-use-code')
    session=new_manager.restore(sid)
    assert session.user_id==user['id']
    assert session.user['email']==user['email']
    assert 'access-secret' not in repr(session)
    assert 'access-secret' not in config.session_db.read_bytes().decode(errors='ignore')
    assert 'refresh-secret' not in config.session_db.read_bytes().decode(errors='ignore')
    assert config.session_db.stat().st_mode & 0o777 == 0o600
    query=parse_qs(urlparse(url).query)
    exchange=json.loads(next(r.content for r in calls if r.url.path.endswith('/token')))
    import base64
    challenge=base64.urlsafe_b64encode(hashlib.sha256(exchange['code_verifier'].encode()).digest()).decode().rstrip('=')
    assert query['code_challenge']==[challenge]
    with pytest.raises(AuthenticationError):
        new_manager.finish(state,state,'single-use-code')


def test_state_cookie_mismatch_and_expired_flow(auth_fixture):
    _,manager,_,calls,_=auth_fixture
    state,_=manager.begin()
    with pytest.raises(AuthenticationError): manager.finish(state,'wrong','single-use-code')
    assert not calls
    with manager.vault.transaction() as db:
        db.execute('UPDATE vault SET expires=0')
    with pytest.raises(AuthenticationError): manager.finish(state,state,'single-use-code')
    assert not calls


def test_refresh_rotation_is_saved_and_not_repeated(auth_fixture):
    _,manager,_,calls,_=auth_fixture
    sid=sign_in(manager)
    with manager.vault.transaction() as db:
        value=manager.vault.get(db,sid,'session')
        value['expires_at']=time.time()-1
        manager.vault.put(db,sid,'session',value,3600)
    first=manager.restore(sid)
    second=manager.restore(sid)
    assert first.access_token==second.access_token
    assert sum(r.url.params.get('grant_type')=='refresh_token' for r in calls)==1
    with manager.vault.transaction() as db:
        assert manager.vault.get(db,sid,'session')['refresh_token'].startswith('refresh-secret-')


def test_revoked_refresh_drops_session(auth_fixture):
    _,manager,_,_,_=auth_fixture
    sid=sign_in(manager)
    with manager.vault.transaction() as db:
        value=manager.vault.get(db,sid,'session'); value['expires_at']=0
        manager.vault.put(db,sid,'session',value,3600)
    manager.client_factory=lambda _: (_ for _ in ()).throw(RuntimeError('revoked'))
    with pytest.raises(AuthenticationError,match='expired or revoked'): manager.restore(sid)
    with pytest.raises(AuthenticationError): manager.restore(sid)


def test_callback_cookie_and_logout_csrf(auth_fixture):
    config,manager,_,_,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url='https://hiking.test')
    response=client.get('/auth/login',follow_redirects=False)
    state=client.cookies[config.flow_cookie]
    response=client.get('/auth/callback',params={'code':'single-use-code','state':state},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.app_url
    cookie=response.headers.get_list('set-cookie')[0]
    assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=lax' in cookie
    assert 'access-secret' not in cookie and 'refresh-secret' not in cookie
    assert response.headers['cache-control']=='no-store'
    assert response.headers['referrer-policy']=='strict-origin'
    sid=client.cookies[config.session_cookie]
    assert manager.restore(sid)
    assert client.post('/account/logout',data={}).status_code==403
    page=client.get('/account/logout')
    assert 'Are you sure you want to log out?' in page.text
    assert 'Yes, log out' in page.text and 'No, stay signed in' in page.text
    assert 'Type <strong>' not in page.text and '<input name="confirmation"' not in page.text
    assert page.headers['referrer-policy']=='strict-origin'
    import re
    token=re.search(r'name="csrf" value="([^"]+)"',page.text).group(1)
    response=client.post('/account/logout',data={'csrf':token,'confirmation':'yes'},
                         headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303
    assert response.headers['location']==config.app_url
    assert 'Max-Age=0' in response.headers['set-cookie']
    assert config.session_cookie not in client.cookies
    with pytest.raises(AuthenticationError): manager.restore(sid)


def test_deletion_requires_recent_session_and_explicit_phrase(auth_fixture):
    config,manager,user,_,_=auth_fixture
    deleted=[]
    client=TestClient(create_app(config,manager=manager,cleanup=lambda uid,identity:deleted.append((uid,identity))),base_url=config.auth_url)
    sid=sign_in(manager); client.cookies.set(config.session_cookie,sid)
    import re
    page=client.get('/account/delete')
    csrf=re.search(r'name="csrf" value="([^"]+)"',page.text).group(1)
    assert client.post('/account/delete',data={'csrf':csrf,'confirmation':'yes'},headers={'Origin':config.auth_url}).status_code==403
    assert not deleted
    result=client.post('/account/delete',data={'csrf':csrf,'confirmation':'DELETE MY ACCOUNT'},headers={'Origin':config.auth_url},follow_redirects=False)
    assert result.status_code==303 and deleted==[(user['id'],True)]
    with pytest.raises(AuthenticationError):manager.restore(sid)


def test_configuration_rejects_unsafe_origins_and_secret_app_key(auth_fixture):
    config,*_=auth_fixture
    config.validate()
    for invalid in (replace(config,app_url='https://evil.test'),replace(config,auth_url='http://hiking.test'),
                    replace(config,publishable_key='sb_secret_privileged'),replace(config,secure_cookies=False)):
        with pytest.raises(ValueError):invalid.validate()


def test_logout_no_preserves_session_and_does_not_revoke_tokens(auth_fixture):
    import re
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager)
    client.cookies.set(config.session_cookie,sid)
    page=client.get('/account/logout')
    token=re.search(r'name="csrf" value="([^"]+)"',page.text).group(1)
    response=client.post('/account/logout',data={'csrf':token,'confirmation':'no'},
        headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.app_url
    assert 'set-cookie' not in response.headers
    assert manager.restore(sid)
    assert not any(r.url.path.endswith('/logout') for r in calls)


@pytest.mark.parametrize('origin',[None,'null','https://evil.test','https://hiking.test.evil.test','http://localhost:8501'])
def test_logout_rejects_missing_null_or_foreign_origin_with_valid_csrf(auth_fixture,origin):
    import re
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    page=client.get('/account/logout')
    token=re.search(r'name="csrf" value="([^"]+)"',page.text).group(1)
    headers={'Origin':origin} if origin is not None else {}
    response=client.post('/account/logout',data={'csrf':token,'confirmation':'yes'},headers=headers)
    assert response.status_code==403
    assert manager.restore(sid)
    assert not any(r.url.path.endswith('/logout') for r in calls)


@pytest.mark.parametrize('data',[
    {'confirmation':'yes'},
    {'confirmation':'yes','csrf':'forged'},
    {'confirmation':'yes','csrf':'\u2603'},
    {'confirmation':'SIGN OUT'},
    {'confirmation':'unexpected'},
])
def test_logout_cannot_bypass_confirmation_or_csrf(auth_fixture,data):
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    result=client.post('/account/logout',data=data,headers={'Origin':config.auth_url},follow_redirects=False)
    assert result.status_code==303
    assert result.headers['location']==config.auth_url+'/account/logout'
    assert manager.restore(sid)
    assert not any(r.url.path.endswith('/logout') for r in calls)


def test_logout_on_localhost_ports_and_global_local_session_invalidation(auth_fixture):
    import re
    config,manager,user,calls,_=auth_fixture
    config=replace(config,app_url='http://localhost:8501',auth_url='http://localhost:8000',secure_cookies=False)
    config.validate()
    manager.config=config
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager)
    another_sid=sign_in(manager)
    client.cookies.set(config.session_cookie,sid)
    page=client.get('/account/logout')
    assert page.headers['referrer-policy']=='strict-origin'
    # A successful POST must also be allowed to navigate to Streamlit's
    # different origin. HTTP test clients do not enforce browser CSP.
    assert "form-action 'self' http://localhost:8501;" in page.headers['content-security-policy']
    token=re.search(r'name="csrf" value="([^"]+)"',page.text).group(1)
    result=client.post('/account/logout',data={'csrf':token,'confirmation':'yes'},headers={'Origin':config.auth_url},follow_redirects=False)
    assert result.status_code==303 and result.headers['location']=='http://localhost:8501'
    assert any(r.url.path.endswith('/logout') and r.url.params.get('scope')=='global' for r in calls)
    for session_id in (sid,another_sid):
        with pytest.raises(AuthenticationError):manager.restore(session_id)


def test_deletion_still_requires_typed_confirmation_and_action_bound_csrf(auth_fixture):
    import re
    config,manager,_,_,_=auth_fixture
    deleted=[]
    client=TestClient(create_app(config,manager=manager,cleanup=lambda *args:deleted.append(args)),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    logout_page=client.get('/account/logout')
    logout_csrf=re.search(r'name="csrf" value="([^"]+)"',logout_page.text).group(1)
    delete_page=client.get('/account/delete')
    assert 'Type <strong>DELETE MY ACCOUNT</strong>' in delete_page.text
    response=client.post('/account/delete',data={'csrf':logout_csrf,'confirmation':'DELETE MY ACCOUNT'},headers={'Origin':config.auth_url})
    assert response.status_code==403 and not deleted


class _Forms:
    """Read real rendered hidden fields without relying on submitter-button values."""
    @staticmethod
    def parse(document):
        from html.parser import HTMLParser
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__();self.forms=[];self.current=None
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if tag=='form':
                    self.current={'action':attrs.get('action'),'fields':{}}
                    self.forms.append(self.current)
                elif tag=='input' and self.current is not None and attrs.get('name'):
                    self.current['fields'][attrs['name']]=attrs.get('value','')
            def handle_endtag(self,tag):
                if tag=='form':self.current=None
        parser=Parser();parser.feed(document)
        return parser.forms


@pytest.mark.parametrize('choice',['yes','no'])
def test_rendered_logout_forms_work_without_submitter_button_fields(auth_fixture,choice):
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    page=client.get('/account/logout')
    forms=_Forms.parse(page.text)
    assert len(forms)==2
    assert {form['fields']['confirmation'] for form in forms}=={'yes','no'}
    form=next(form for form in forms if form['fields']['confirmation']==choice)
    response=client.post(form['action'],data=form['fields'],headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.app_url
    if choice=='yes':
        with pytest.raises(AuthenticationError):manager.restore(sid)
        assert any(r.url.path.endswith('/logout') for r in calls)
    else:
        assert manager.restore(sid)
        assert not any(r.url.path.endswith('/logout') for r in calls)


def test_old_typed_logout_form_gets_fresh_confirmation_without_logging_out(auth_fixture):
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    fields=_Forms.parse(client.get('/account/logout').text)[0]['fields']
    fields['confirmation']='SIGN OUT'
    response=client.post('/account/logout',data=fields,headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.auth_url+'/account/logout'
    assert manager.restore(sid)
    assert not any(r.url.path.endswith('/logout') for r in calls)
    refreshed=client.get(response.headers['location'])
    assert 'Yes, log out' in refreshed.text and 'No, stay signed in' in refreshed.text


def test_session_change_between_get_and_post_requires_fresh_confirmation(auth_fixture):
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    old_sid=sign_in(manager);client.cookies.set(config.session_cookie,old_sid)
    stale_fields=_Forms.parse(client.get('/account/logout').text)[0]['fields']
    current_sid=sign_in(manager);client.cookies.set(config.session_cookie,current_sid)
    response=client.post('/account/logout',data=stale_fields,headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.auth_url+'/account/logout'
    assert manager.restore(current_sid)
    assert not any(r.url.path.endswith('/logout') for r in calls)
    fresh_fields=_Forms.parse(client.get(response.headers['location']).text)[0]['fields']
    assert fresh_fields['csrf']!=stale_fields['csrf']
    confirmed=client.post('/account/logout',data=fresh_fields,headers={'Origin':config.auth_url},follow_redirects=False)
    assert confirmed.status_code==303 and confirmed.headers['location']==config.app_url
    with pytest.raises(AuthenticationError):manager.restore(current_sid)


def test_already_expired_logout_session_clears_cookie_and_returns_to_app(auth_fixture):
    config,manager,_,calls,_=auth_fixture
    client=TestClient(create_app(config,manager=manager),base_url=config.auth_url)
    sid=sign_in(manager);client.cookies.set(config.session_cookie,sid)
    fields=_Forms.parse(client.get('/account/logout').text)[0]['fields']
    manager.vault.delete(sid)
    response=client.post('/account/logout',data=fields,headers={'Origin':config.auth_url},follow_redirects=False)
    assert response.status_code==303 and response.headers['location']==config.app_url
    assert 'Max-Age=0' in response.headers['set-cookie']
    assert not any(r.url.path.endswith('/logout') for r in calls)
