from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from supabase import create_client as real_client

from hiking_time_predictor.auth_sessions import UserSession
from hiking_time_predictor.cloud_repository import CloudError, SupabaseRepository


def test_official_sdk_sends_user_jwt_to_both_database_and_storage(monkeypatch):
    requests=[]
    user_id=str(uuid4())
    def handler(request):
        requests.append(request)
        if '/rest/v1/profiles' in request.url.path:
            return httpx.Response(200,json=[{'id':user_id,'display_name':'Synthetic'}])
        if '/storage/v1/object/' in request.url.path:
            return httpx.Response(200,content=b'private gpx')
        raise AssertionError(str(request.url))
    def factory(url,key,options):
        options.httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        return real_client(url,key,options=options)
    monkeypatch.setattr('hiking_time_predictor.cloud_repository.create_client',factory)
    config=SimpleNamespace(supabase_url='https://test.supabase.co',publishable_key='sb_publishable_test')
    repo=SupabaseRepository(config,UserSession(user_id,'user-specific-access-token',{},0))
    assert repo.profile()['id']==user_id
    assert repo.download('hike-gpx',user_id+'/test.gpx')==b'private gpx'
    assert len(requests)==2
    assert all(r.headers['Authorization']=='Bearer user-specific-access-token' for r in requests)
    assert all(r.headers['apikey']=='sb_publishable_test' for r in requests)
    for path in (str(uuid4())+'/other.gpx',user_id+'/../other.gpx',user_id+'/nested/test.gpx'):
        with pytest.raises(CloudError):repo.download('hike-gpx',path)
    assert len(requests)==2


def test_missing_storage_permissions_propagate_without_switching_to_admin(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(403,json={'statusCode':'403','error':'Unauthorized','message':'RLS permission denied'})
    def factory(url,key,options):
        options.httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        return real_client(url,key,options=options)
    monkeypatch.setattr('hiking_time_predictor.cloud_repository.create_client',factory)
    uid=str(uuid4())
    repo=SupabaseRepository(SimpleNamespace(supabase_url='https://test.supabase.co',publishable_key='sb_publishable_test'),UserSession(uid,'user-only-token',{},0))
    with pytest.raises(Exception):repo.download('hike-gpx',uid+'/test.gpx')
    assert len(calls)==1 and calls[0].headers['Authorization']=='Bearer user-only-token'


@pytest.mark.parametrize('code,expected',[
    ('PGRST205','database setup is incomplete'),
    ('PGRST202','database setup is incomplete'),
    ('42501','denied'),
    ('PGRST301','could not be authorized'),
    ('unexpected','could not be completed'),
])
def test_database_errors_are_actionable_and_never_echo_raw_responses(code,expected):
    from postgrest.exceptions import APIError
    from hiking_time_predictor.cloud_repository import private_request_message
    error=APIError({'code':code,'message':'private-token-and-location','details':'private-response','hint':'private-hint'})
    message=private_request_message(error)
    assert expected in message
    assert 'private-token' not in message and 'private-response' not in message and 'private-hint' not in message
