"""Synthetic per-user data only; no private recordings or local model files."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from math import degrees
from pathlib import Path
from uuid import uuid4
import zipfile

import pytest

from hiking_time_predictor.gpx import EARTH_RADIUS_M
from hiking_time_predictor.personal_hiking import (preview_recording,save_recording,clean,
    training_frames,train_personal_models,load_personal_models,export_personal_data)
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor.trusted_models import TrustedModelCodec
from hiking_time_predictor.web_upload import predict_upload,trip_statistics
from hiking_time_predictor.migrate_local import inspect_bundle,package_project


def recording(day=1,minutes=1):
    points=[]
    for i in range(16):
        points.append(f'<trkpt lat="0" lon="{degrees(i*45/EARTH_RADIUS_M)}"><ele>{100+i*2}</ele><time>2026-01-{day:02d}T00:{i*minutes:02d}:00Z</time></trkpt>')
    return ('<gpx version="1.1"><trk><trkseg>'+''.join(points)+'</trkseg></trk></gpx>').encode()


def recording_with_pauses(day=1):
    """Two minutes moving, a one-minute pause, and a ten-minute recording gap."""
    samples = [(0, 0), (100, 60), (100, 120), (100, 720), (200, 780)]
    points = ''.join(
        f'<trkpt lat="0" lon="{degrees(meters/EARTH_RADIUS_M)}"><ele>100</ele>'
        f'<time>2026-01-{day:02d}T00:{seconds//60:02d}:{seconds%60:02d}Z</time></trkpt>'
        for meters, seconds in samples)
    return ('<gpx version="1.1"><trk><trkseg>' + points + '</trkseg></trk></gpx>').encode()


class FakeRepo:
    def __init__(self,user_id=None):
        self.user_id=user_id or str(uuid4()); self.hikes=[];self.segments=[];self.events=[];self.versions=[];self.objects={};self.fail_import=False;self.fail_activation=False
    def list_hikes(self):return deepcopy(self.hikes)
    def set_training(self,hike_id,enabled):
        next(h for h in self.hikes if h['id']==hike_id)['training_enabled']=bool(enabled)
    def upload(self,bucket,path,contents,content_type):
        assert path.startswith(self.user_id+'/')
        if (bucket,path) in self.objects:raise ValueError('duplicate blob')
        self.objects[bucket,path]=contents
    def download(self,bucket,path):return self.objects[bucket,path]
    def remove_object(self,bucket,path):self.objects.pop((bucket,path),None)
    def object_paths(self,bucket):return [path for b,path in self.objects if b==bucket]
    def import_hike(self,row,segments,events):
        if self.fail_import:raise ValueError('import failed')
        self.hikes.append({'deleting':False,**deepcopy(row)})
        self.segments.extend({**s,'data':deepcopy(s),'hike_id':row['id'],'user_id':self.user_id} for s in segments)
        self.events.extend({**e,'id':str(uuid4()),'data':deepcopy(e),'hike_id':row['id'],'user_id':self.user_id,
            'classification':e['break_classification'],'quality_flags':e['quality_flag']} for e in events)
    def snapshot(self):return deepcopy({'hikes':self.hikes,'segments':self.segments,'breaks':self.events})
    def insert_model(self,row):self.versions.append(deepcopy(row))
    def models(self):return deepcopy(self.versions)
    def activate_models(self,a,b,snapshot):
        if self.fail_activation:raise ValueError('conflict')
        assert self.snapshot()==snapshot
        for row in self.versions:row['is_active']=row['id'] in (a,b)
    def rows(self,table):return []


def test_upload_duplicate_detection_and_two_user_isolation():
    a,b=FakeRepo(),FakeRepo()
    raw=recording()
    p=preview_recording(raw,'../../private.gpx','day_hike')
    assert p.filename=='private.gpx'
    assert save_recording(a,p,training_enabled=True)[1]
    assert not save_recording(a,preview_recording(raw,'again.gpx'))[1]
    assert len(a.hikes)==1 and not b.hikes
    assert save_recording(b,preview_recording(raw,'same.gpx'),training_enabled=True)[1]
    assert a.hikes[0]['user_id']!=b.hikes[0]['user_id']
    assert set(a.objects).isdisjoint(set(b.objects))


def test_failed_import_cleans_blob_and_creates_no_partial_records():
    repo=FakeRepo();repo.fail_import=True
    with pytest.raises(ValueError,match='import failed'):save_recording(repo,preview_recording(recording(),'test.gpx'))
    assert not repo.hikes and not repo.objects and not repo.segments


def test_untimed_route_and_xml_entities_cannot_be_history():
    with pytest.raises(ValueError,match='timestamped'):
        preview_recording(b'<gpx><trk><trkseg><trkpt lat="0" lon="0"><ele>1</ele></trkpt><trkpt lat="0" lon="0.01"><ele>2</ele></trkpt></trkseg></trk></gpx>','planned.gpx')
    with pytest.raises(ValueError,match='DTD'):
        preview_recording(b'<!DOCTYPE gpx [<!ENTITY x SYSTEM "file:///etc/passwd">]><gpx>&x;</gpx>','evil.gpx')


def test_training_and_prediction_use_only_owner_models_and_version_atomically():
    repo=FakeRepo();codec=TrustedModelCodec('test-signing-secret-'*3)
    for day in range(1,4):save_recording(repo,preview_recording(recording(day),'history.gpx','day_hike'),training_enabled=True)
    moving,breaks,rows=train_personal_models(repo,codec)
    assert moving.evaluation['eligible_hikes']==3
    assert breaks.training_hikes==3
    loaded_moving,loaded_breaks,active=load_personal_models(repo,codec)
    assert len(active)==2 and loaded_moving.model_name==moving.model_name
    planned=b'<gpx><rte><rtept lat="0" lon="0"><ele>100</ele></rtept><rtept lat="0" lon="0.01"><ele>120</ele></rtept></rte></gpx>'
    one=predict_upload(planned,'plan.gpx','day_hike',artifact=loaded_moving,break_artifact=loaded_breaks)
    stats=trip_statistics(one,include_predicted_breaks=True)
    assert stats['total_minutes']==stats['moving_minutes']+stats['break_minutes']
    other=FakeRepo()
    baseline,empty,models=load_personal_models(other,codec)
    assert baseline.estimator is None and empty.minutes_per_mile is None and not models
    mixed=repo.snapshot();mixed['segments'][0]['user_id']=other.user_id
    with pytest.raises(ValueError,match='different user'):training_frames(mixed,repo.user_id)
    previous={r['id'] for r in repo.models() if r['is_active']}
    repo.fail_activation=True
    with pytest.raises(ValueError,match='conflict'):train_personal_models(repo,codec)
    assert {r['id'] for r in repo.models() if r['is_active']}==previous


@pytest.mark.parametrize('tamper',['bytes','owner','signature','runtime','dataset'])
def test_model_checks_run_before_deserialization(monkeypatch,tamper):
    codec=TrustedModelCodec('secure-model-key-'*3);user_id=str(uuid4());model_id=str(uuid4());version=str(uuid4())
    contents,provenance=codec.encode(ModelArtifact(),user_id=user_id,model_id=model_id,kind='moving',dataset_fingerprint='a'*64,version=version)
    row={'id':model_id,'model_type':'moving','version':version,'dataset_fingerprint':'a'*64,'artifact_hash':provenance['sha256'],'provenance':provenance}
    if tamper=='bytes':contents+=b'x'
    elif tamper=='owner':user_id=str(uuid4())
    elif tamper=='signature':row['provenance']['signature']='0'*64
    elif tamper=='runtime':row['provenance']['runtime']['sklearn']='0.0'
    else:row['dataset_fingerprint']='b'*64
    def forbidden(*a,**k):pytest.fail('Untrusted pickle was deserialized')
    monkeypatch.setattr('hiking_time_predictor.trusted_models.joblib.load',forbidden)
    with pytest.raises(ValueError):codec.decode(contents,row,user_id)


def test_explicit_legacy_package_preserves_files_and_never_unpickles(tmp_path,monkeypatch):
    (tmp_path/'data/raw').mkdir(parents=True);(tmp_path/'models').mkdir()
    raw=recording();(tmp_path/'data/raw/old.gpx').write_bytes(raw)
    (tmp_path/'models/old.joblib').write_bytes(b'arbitrary legacy bytes, never load')
    bundle=tmp_path/'migration.zip'
    assert package_project(tmp_path,bundle)==2
    monkeypatch.setattr('joblib.load',lambda *_:pytest.fail('legacy artifact loaded'))
    data=inspect_bundle(bundle.read_bytes())
    assert data['data/raw/old.gpx']==raw
    assert (tmp_path/'data/raw/old.gpx').read_bytes()==raw
    assert 'models/old.joblib' in data
    with pytest.raises(ValueError,match='never overwritten'):package_project(tmp_path,bundle)
    malicious=io.BytesIO()
    with zipfile.ZipFile(malicious,'w') as archive:archive.writestr('../escape.gpx',raw)
    with pytest.raises(ValueError):inspect_bundle(malicious.getvalue())


def test_partial_timestamp_recording_preserves_quality_flags_and_training_filters():
    raw=recording().replace(b'<time>2026-01-01T00:03:00Z</time>',b'')
    preview=preview_recording(raw,'partial-recording.gpx','day_hike')
    assert preview.summary['missing_timestamp_count']==1
    assert 'missing_timestamp' in preview.summary['quality_flags']
    repo=FakeRepo();save_recording(repo,preview,training_enabled=True)
    moving,breaks,_=train_personal_models(repo,TrustedModelCodec('partial-recording-test-key-'*2))
    assert moving.estimator is None
    assert breaks.minutes_per_mile is None


def test_existing_uploads_without_ordinary_break_coverage_can_train_and_predict():
    import pandas as pd
    from hiking_time_predictor.breaks import NON_MOVING_TARGET, summarize_breaks

    repo = FakeRepo()
    for day in range(1, 6):
        raw = recording_with_pauses(day)
        preview = preview_recording(raw, 'history.gpx', 'day_hike')
        ordinary = summarize_breaks(pd.DataFrame([preview.summary]), preview.events)
        assert not ordinary.training_eligible.iloc[0]
        save_recording(repo, preview, training_enabled=True)

    before = repo.snapshot()
    codec = TrustedModelCodec('non-moving-training-test-key-' * 3)
    _, breaks, _ = train_personal_models(repo, codec)
    assert breaks.training_hikes == 5
    assert breaks.evaluation['target_definition'] == NON_MOVING_TARGET
    assert repo.snapshot() == before
    # No ordinary breaks were detected; their average is not fabricated from gaps.
    assert breaks.mean_duration_minutes is None
    assert breaks.breaks_per_mile == 0

    moving, loaded, _ = load_personal_models(repo, codec)
    planned = (f'<gpx><rte><rtept lat="0" lon="0"><ele>100</ele></rtept>'
               f'<rtept lat="0" lon="{degrees(200/EARTH_RADIUS_M)}"><ele>100</ele></rtept></rte></gpx>').encode()
    prediction = predict_upload(planned, 'plan.gpx', 'day_hike', artifact=moving, break_artifact=loaded)
    stats = trip_statistics(prediction, include_predicted_breaks=True)
    assert stats['break_minutes'] == pytest.approx(11)
    assert stats['total_minutes'] == pytest.approx(stats['moving_minutes'] + 11)
    assert trip_statistics(prediction)['total_minutes'] == stats['moving_minutes']
    assert any('unmeasured movement' in warning for warning in prediction.break_forecast.warnings)
