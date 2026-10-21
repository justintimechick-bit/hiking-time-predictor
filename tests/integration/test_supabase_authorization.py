"""DIRECT PostgREST/Storage tests against a migrated disposable Supabase project.

Never point these at production. No UI filters or fake RLS engine are involved.
Enable RUN_SUPABASE_INTEGRATION=1 and set SUPABASE_TEST_{URL,PUBLISHABLE_KEY,SECRET_KEY}.
"""
from dataclasses import replace
from datetime import datetime, timezone
import os
import time
from uuid import uuid4

import httpx
import pytest
from postgrest.exceptions import APIError
from supabase import create_client, ClientOptions

from hiking_time_predictor.account_cleanup import delete_account_data
from hiking_time_predictor.auth_sessions import UserSession
from hiking_time_predictor.cloud_config import CloudConfig
from hiking_time_predictor.cloud_repository import SupabaseRepository
from hiking_time_predictor.personal_hiking import preview_recording, save_recording, train_personal_models, load_personal_models
from hiking_time_predictor.trusted_models import TrustedModelCodec

pytestmark=pytest.mark.integration


@pytest.fixture(scope='module')
def live_users():
    if os.getenv('RUN_SUPABASE_INTEGRATION')!='1':
        pytest.skip('Direct Supabase authorization tests require explicit opt-in and a disposable migrated project.')
    required=['SUPABASE_TEST_URL','SUPABASE_TEST_PUBLISHABLE_KEY','SUPABASE_TEST_SECRET_KEY']
    if not all(os.getenv(k) for k in required):
        pytest.fail('Integration opt-in requires all SUPABASE_TEST_* credentials.')
    url,key,secret=[os.environ[k] for k in required]
    admin=create_client(url,secret,options=ClientOptions(auto_refresh_token=False,persist_session=False))
    config=CloudConfig(url,key)
    users=[]
    try:
        for _ in range(2):
            email='hiking-rls-'+uuid4().hex+'@example.invalid';password=uuid4().hex+uuid4().hex
            user=admin.auth.admin.create_user({'email':email,'password':password,'email_confirm':True}).user
            users.append({'id':user.id})  # Cleanup even if sign-in/setup later fails.
            client=create_client(url,key,options=ClientOptions(auto_refresh_token=False,persist_session=False))
            session=client.auth.sign_in_with_password({'email':email,'password':password}).session
            users[-1].update(repo=SupabaseRepository(config,UserSession(user.id,session.access_token,{},time.time())),token=session.access_token)
        yield config,admin,users
    finally:
        for user in users:
            # Uses the same verified cleanup implementation, removing orphaned blobs too.
            remaining=admin.auth.admin.get_user_by_id(user['id'])
            if remaining.user:
                delete_account_data(admin,user['id'])


def raw_recording(day):
    return ('<gpx><trk><trkseg>'+''.join(f'<trkpt lat="0" lon="{i*0.0004}"><ele>{100+i}</ele><time>2026-01-{day:02d}T00:{i:02d}:00Z</time></trkpt>' for i in range(16))+'</trkseg></trk></gpx>').encode()


@pytest.fixture(scope='module')
def populated(live_users):
    config,admin,users=live_users
    for i,user in enumerate(users):
        repo=user['repo']
        # Verifies the auth.users trigger initialized a private profile.
        assert repo.profile()['id']==user['id']
        preview=preview_recording(raw_recording(i+1),'synthetic.gpx','day_hike')
        hike_id,created=save_recording(repo,preview,training_enabled=True)
        assert created
        user['hike_id']=hike_id
        codec=TrustedModelCodec('integration-only-signing-key-'*2)
        _,_,models=train_personal_models(repo,codec)
        user['models']=models
        prediction={'user_id':user['id'],'route_name':'Synthetic plan','distance_miles':1,
            'elevation_gain_ft':10,'elevation_loss_ft':0,'hike_type':'day_hike','route_direction':'one-way',
            'predicted_moving_time_min':21,'predicted_break_time_min':2,'predicted_total_time_min':23,
            'moving_model_version_id':models[0]['id'],'break_model_version_id':models[1]['id']}
        result=repo.client.table('predictions').insert(prediction).execute().data
        user['prediction_id']=result[0]['id']
    return config,admin,users


def test_direct_table_rls_and_ownership_checks(populated):
    _,_,(a,b)=populated
    client=a['repo'].client
    for table,record_id in [('hikes',b['hike_id']),('predictions',b['prediction_id']),('model_versions',b['models'][0]['id'])]:
        assert client.table(table).select('*').eq('id',record_id).execute().data==[]
        assert client.table(table).update({'user_id':a['id']}).eq('id',record_id).execute().data==[]
        assert client.table(table).delete().eq('id',record_id).execute().data==[]
    assert b['repo'].list_hikes()[0]['id']==b['hike_id']
    with pytest.raises(Exception):
        client.table('hikes').update({'user_id':b['id']}).eq('id',a['hike_id']).execute()
    for table in ('hike_segments','break_events'):
        assert client.table(table).select('*').eq('user_id',b['id']).execute().data==[]
        other=b['repo'].client.table(table).select('*').limit(1).execute().data[0]
        forged={**other,'id':str(uuid4()),'user_id':a['id']}
        # Own user_id with another user's parent hike is blocked by composite FK.
        with pytest.raises(Exception):client.table(table).insert(forged).execute()


def test_direct_storage_rls_no_public_downloads(populated):
    config,_,(a,b)=populated
    for bucket,path in [('hike-gpx',f"{b['id']}/{b['hike_id']}.gpx"),('user-models',b['models'][0]['storage_path'])]:
        assert b['repo'].client.storage.from_(bucket).download(path)
        # Bypass the Python path guard and call Storage with A's actual JWT.
        with pytest.raises(Exception):a['repo'].client.storage.from_(bucket).download(path)
        with pytest.raises(Exception):a['repo'].client.storage.from_(bucket).upload(path,b'forged',{'content-type':'application/octet-stream'})
        a['repo'].client.storage.from_(bucket).remove([path])
        assert b['repo'].client.storage.from_(bucket).download(path)
        public_url=b['repo'].client.storage.from_(bucket).get_public_url(path)
        assert httpx.get(public_url,timeout=20).status_code != 200


def test_anonymous_clients_cannot_read_private_tables_or_files(populated):
    config,_,(_,b)=populated
    anon=create_client(config.supabase_url,config.publishable_key)
    for table in ('profiles','hikes','hike_segments','break_events','model_versions','predictions'):
        try:
            rows=anon.table(table).select('*').execute().data
        except APIError:
            continue  # Permission denied also satisfies the fail-closed contract.
        assert rows==[]
    with pytest.raises(Exception):anon.storage.from_('hike-gpx').download(f"{b['id']}/{b['hike_id']}.gpx")


def test_owner_cannot_reference_other_users_model_in_prediction(populated):
    _,_,(a,b)=populated
    own=a['repo'].client.table('predictions').select('*').eq('id',a['prediction_id']).execute().data[0]
    own.update(id=str(uuid4()),moving_model_version_id=b['models'][0]['id'])
    with pytest.raises(Exception):a['repo'].client.table('predictions').insert(own).execute()


def test_two_user_training_and_loaded_models_stay_separate(populated):
    _,_,(a,b)=populated
    codec=TrustedModelCodec('integration-only-signing-key-'*2)
    for user in (a,b):
        moving,_,rows=load_personal_models(user['repo'],codec)
        assert moving.evaluation['eligible_hikes']==1
        assert {r['user_id'] for r in rows}=={user['id']}
        assert {r['user_id'] for r in user['repo'].snapshot()['segments']}=={user['id']}
    row=b['models'][0]
    raw=b['repo'].download('user-models',row['storage_path'])
    with pytest.raises(ValueError):codec.decode(raw,row,a['id'])


def test_delete_selected_hike_and_cleanup_account_data(populated):
    _,admin,(a,b)=populated
    # Clean A's application data without deleting the Auth identity, then ensure B survives.
    delete_account_data(admin,a['id'],delete_identity=False)
    assert a['repo'].list_hikes()==[]
    assert a['repo'].models()==[]
    assert a['repo'].predictions()==[]
    assert not a['repo'].object_paths('hike-gpx') and not a['repo'].object_paths('user-models')
    assert a['repo'].profile()['id']==a['id']
    assert b['repo'].list_hikes()
    b['repo'].delete_hike(b['hike_id'])
    assert b['repo'].list_hikes()==[]
    assert not b['repo'].object_paths('hike-gpx')
