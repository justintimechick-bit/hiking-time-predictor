"""Cleanup/import state machines, independent of live authorization tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from hiking_time_predictor.account_cleanup import delete_account_data
from hiking_time_predictor.migrate_local import import_bundle,package_project
from hiking_time_predictor.personal_hiking import export_personal_data
from test_personal_hiking import FakeRepo,recording


class Query:
    def __init__(self,admin,table):self.admin=admin;self.table=table;self.filters=[];self.op='select';self.value=None
    def select(self,*args):return self
    def eq(self,k,v):self.filters.append((k,v));return self
    def limit(self,*args):return self
    def update(self,value):self.op='update';self.value=value;return self
    def delete(self):self.op='delete';return self
    def insert(self,value):self.op='insert';self.value=value;return self
    def upsert(self,value):self.op='upsert';self.value=value;return self
    def execute(self):
        rows=self.admin.tables.setdefault(self.table,[])
        matches=[r for r in rows if all(r.get(k)==v for k,v in self.filters)]
        if self.op=='insert':rows.append(deepcopy(self.value));matches=[rows[-1]]
        if self.op=='upsert':
            existing=next((r for r in rows if r.get('user_id')==self.value['user_id']),None)
            if existing:existing.update(self.value)
            else:rows.append(deepcopy(self.value))
        if self.op=='update':
            for r in matches:r.update(self.value)
        if self.op=='delete':
            for r in matches:rows.remove(r)
            if self.table=='hikes':
                ids={r['id'] for r in matches}
                for child in ('hike_segments','break_events'):
                    self.admin.tables[child]=[r for r in self.admin.tables.get(child,[]) if r['hike_id'] not in ids]
        return SimpleNamespace(data=deepcopy(matches))


class Admin:
    def __init__(self):
        self.tables={};self.objects={};self.events=[];self.fail_remove=False
        self.storage=SimpleNamespace(from_=lambda bucket:Bucket(self,bucket))
        self.auth=SimpleNamespace(admin=SimpleNamespace(delete_user=self.delete_user))
    def table(self,name):return Query(self,name)
    def delete_user(self,uid):
        self.events.append('identity')
        self.tables['profiles']=[r for r in self.tables.get('profiles',[]) if r['id']!=uid]
    def rpc(self,name,params):
        assert name=='finish_hiking_data_reset'
        uid=params['p_user']
        for row in self.tables['profiles']:
            if row['id']==uid:row.update(deletion_requested_at=None,display_name='',preferred_hike_type='day_hike')
        for row in self.tables['account_deletion_jobs']:
            if row['user_id']==uid:row['status']='complete'
        return SimpleNamespace(execute=lambda:None)


class Bucket:
    def __init__(self,admin,bucket):self.admin=admin;self.bucket=bucket
    def list(self,prefix,options):return [{'id':'object','name':p.split('/')[1]} for b,p in self.admin.objects if b==self.bucket and p.startswith(prefix+'/')][:options.get('limit',100)]
    def remove(self,paths):
        if self.admin.fail_remove:raise ValueError('transient storage failure')
        for path in paths:self.admin.objects.pop((self.bucket,path),None)
        self.admin.events.append('storage')


def setup_admin():
    admin=Admin();a,b=str(uuid4()),str(uuid4());hid=str(uuid4())
    admin.tables={'profiles':[{'id':a},{'id':b}],'hikes':[{'id':hid,'user_id':a},{'id':str(uuid4()),'user_id':b}],
                  'hike_segments':[{'hike_id':hid,'user_id':a}],'break_events':[{'hike_id':hid,'user_id':a}]}
    admin.objects={('hike-gpx',a+'/'+hid+'.gpx'):b'a',('user-models',a+'/'+str(uuid4())+'.joblib'):b'model',
                   ('hike-gpx',b+'/'+str(uuid4())+'.gpx'):b'b'}
    return admin,a,b


def test_full_cleanup_removes_objects_before_identity_and_preserves_other_user():
    admin,a,b=setup_admin()
    delete_account_data(admin,a)
    assert admin.events.index('storage')<admin.events.index('identity')
    assert not any(p.startswith(a+'/') for _,p in admin.objects)
    assert any(p.startswith(b+'/') for _,p in admin.objects)
    assert admin.tables['account_deletion_jobs'][0]['status']=='complete'
    assert not any(r['id']==a for r in admin.tables['profiles'])
    assert admin.tables['hike_segments']==[] and admin.tables['break_events']==[]


def test_cleanup_failure_keeps_locked_pending_job_and_retry_finishes():
    admin,a,_=setup_admin();admin.fail_remove=True
    with pytest.raises(ValueError,match='storage failure'):delete_account_data(admin,a)
    assert admin.tables['account_deletion_jobs'][0]['status']=='pending'
    assert next(r for r in admin.tables['profiles'] if r['id']==a)['deletion_requested_at']
    assert 'identity' not in admin.events
    admin.fail_remove=False
    delete_account_data(admin,a)
    assert admin.tables['account_deletion_jobs'][0]['status']=='complete'


def test_data_reset_retains_auth_identity():
    admin,a,_=setup_admin()
    delete_account_data(admin,a,delete_identity=False)
    assert 'identity' not in admin.events
    assert next(r for r in admin.tables['profiles'] if r['id']==a)['deletion_requested_at'] is None
    assert admin.tables['account_deletion_jobs'][0]['status']=='complete'


def test_explicit_migration_import_verifies_backup_and_never_loads_legacy_pickle(tmp_path,monkeypatch):
    (tmp_path/'data/raw').mkdir(parents=True);(tmp_path/'models').mkdir()
    raw=recording();(tmp_path/'data/raw/original.gpx').write_bytes(raw)
    (tmp_path/'models/legacy.joblib').write_bytes(b'not a trusted pickle')
    path=tmp_path/'legacy.zip';package_project(tmp_path,path)
    repo=FakeRepo();repo.client=Admin()
    monkeypatch.setattr('joblib.load',lambda *_:pytest.fail('legacy pickle deserialized'))
    assert import_bundle(repo,path.read_bytes())==1
    assert repo.client.tables['legacy_imports'][0]['status']=='complete'
    assert len(repo.hikes)==1
    assert not repo.versions
    assert (tmp_path/'data/raw/original.gpx').read_bytes()==raw
    assert any(bucket=='user-models' and value==path.read_bytes() for (bucket,_),value in repo.objects.items())
    with pytest.raises(ValueError,match='already has a migration'):import_bundle(repo,path.read_bytes())
    import io,zipfile
    export=zipfile.ZipFile(io.BytesIO(export_personal_data(repo)))
    assert any(n.startswith('hike-gpx/') for n in export.namelist())
    assert 'profiles.json' in export.namelist()
