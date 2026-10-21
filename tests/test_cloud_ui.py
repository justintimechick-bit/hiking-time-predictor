"""Offline landing protection; no credentials, private files or network required."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def history_app(repo):
    from hiking_time_predictor.cloud_ui import history_page
    history_page(repo)


@pytest.fixture
def history_repo():
    from test_personal_hiking import FakeRepo, recording
    from hiking_time_predictor.personal_hiking import preview_recording, save_recording
    repo = FakeRepo()
    for day in (1, 2, 3):
        save_recording(repo, preview_recording(recording(day), 'Morning_hike.gpx', 'day_hike'),
                       training_enabled=day == 1)
    return repo


def test_history_lists_all_hikes_and_synchronizes_select_all(history_repo):
    app = AppTest.from_function(history_app, args=(history_repo,), default_timeout=15).run()
    keys = ['training_' + h['id'] for h in history_repo.hikes]
    assert not app.exception
    assert not any(s.label == 'Activity details' for s in app.selectbox)
    assert not any(c.label == 'Include in training' for c in app.checkbox)
    assert not any(b.label in ('Save training preference', 'Save training selections') for b in app.button)
    assert [app.checkbox(key=k).value for k in keys] == [True, False, False]
    assert all(app.checkbox(key=k).label == 'Morning_hike.gpx' for k in keys)
    assert app.checkbox(key='training_select_all').label == 'Use all uploaded data to train the models'
    assert not app.checkbox(key='training_select_all').value

    app.checkbox(key='training_select_all').check().run()
    assert not app.exception
    assert all(app.checkbox(key=k).value for k in keys)
    assert app.checkbox(key='training_select_all').value
    assert all(h['training_enabled'] for h in history_repo.hikes)
    app.checkbox(key=keys[1]).uncheck().run()
    assert [app.checkbox(key=k).value for k in keys] == [True, False, True]
    assert not app.checkbox(key='training_select_all').value
    assert [h['training_enabled'] for h in history_repo.hikes] == [True, False, True]
    app.checkbox(key=keys[1]).check().run()
    assert app.checkbox(key='training_select_all').value
    assert all(h['training_enabled'] for h in history_repo.hikes)
    app.checkbox(key='training_select_all').uncheck().run()
    assert not any(app.checkbox(key=k).value for k in keys)
    assert not any(h['training_enabled'] for h in history_repo.hikes)
    assert not history_repo.models()


def test_checklist_persists_immediately_and_controls_model_training(history_repo):
    from hiking_time_predictor.personal_hiking import train_personal_models
    from hiking_time_predictor.trusted_models import TrustedModelCodec
    app = AppTest.from_function(history_app, args=(history_repo,), default_timeout=15).run()
    app.checkbox(key='training_select_all').check().run()
    app.checkbox(key='training_' + history_repo.hikes[0]['id']).uncheck().run()
    assert not app.exception
    assert [h['training_enabled'] for h in history_repo.hikes] == [False, True, True]
    assert not history_repo.models()

    reloaded = AppTest.from_function(history_app, args=(history_repo,), default_timeout=15).run()
    assert not reloaded.exception
    assert [reloaded.checkbox(key='training_' + h['id']).value for h in history_repo.hikes] == [False, True, True]
    moving, breaks, _ = train_personal_models(history_repo, TrustedModelCodec('checklist-test-signing-key-' * 3))
    assert moving.evaluation['eligible_hikes'] == 2
    assert breaks.training_hikes == 2


def test_select_all_excludes_hikes_pending_deletion_and_preserves_details(history_repo):
    history_repo.hikes[2].update(deleting=True, training_enabled=False)
    app = AppTest.from_function(history_app, args=(history_repo,), default_timeout=15).run()
    pending_key = 'training_' + history_repo.hikes[2]['id']
    assert app.checkbox(key=pending_key).disabled
    app.checkbox(key='training_select_all').check().run()
    assert not app.exception
    assert not app.checkbox(key=pending_key).value
    assert app.checkbox(key='training_select_all').value
    assert [h['training_enabled'] for h in history_repo.hikes] == [True, True, False]

    hike = history_repo.hikes[1]
    app.button(key='details_' + hike['id']).click().run()
    next(b for b in app.button if b.label == 'Load activity charts and original GPX').click().run()
    assert not app.exception
    assert app.session_state['activity_detail'][0] == hike['id']
    assert len(app.get('plotly_chart')) == 2


def test_imported_hikes_appear_in_checklist_with_select_all(history_repo):
    from test_personal_hiking import recording
    app = AppTest.from_function(history_app, args=(history_repo,), default_timeout=15).run()
    app.checkbox(key='training_select_all').check().run()
    app.file_uploader(key='historical_files').set_value([
        ('Afternoon_hike.gpx', recording(4), 'application/gpx+xml'),
        ('Evening_hike.gpx', recording(5), 'application/gpx+xml'),
    ]).run()
    next(b for b in app.button if b.label == 'Preview recordings').click().run()
    next(b for b in app.button if b.label == 'Confirm import to my account').click().run()
    assert not app.exception
    assert len(history_repo.hikes) == 5
    assert [app.checkbox(key='training_' + h['id']).value for h in history_repo.hikes] == [True, True, True, False, False]
    assert [h['training_enabled'] for h in history_repo.hikes] == [True, True, True, False, False]
    assert not app.checkbox(key='training_select_all').value
    app.checkbox(key='training_select_all').check().run()
    assert not app.exception
    assert all(h['training_enabled'] for h in history_repo.hikes)
    assert not history_repo.models()


def test_history_selections_are_used_when_training_after_page_navigation(history_repo, monkeypatch):
    from types import SimpleNamespace
    from hiking_time_predictor.auth_sessions import UserSession
    from hiking_time_predictor import cloud_ui

    config = SimpleNamespace(model_key='history-navigation-signing-key-' * 3,
                             auth_url='https://hiking.test', session_cookie='opaque')
    monkeypatch.setattr(cloud_ui.CloudConfig, 'from_env', lambda: config)
    monkeypatch.setattr(cloud_ui, 'AuthManager', lambda _: SimpleNamespace(restore=lambda sid:
        UserSession(history_repo.user_id, 'test-access', {'email': 'synthetic@example.invalid'}, 0)))
    monkeypatch.setattr(cloud_ui, 'SupabaseRepository', lambda cfg, session: history_repo)
    history_repo.profile = lambda: {'id': history_repo.user_id, 'created_at': '2026-01-01',
                                   'display_name': 'Synthetic', 'preferred_hike_type': 'day_hike'}
    history_repo.predictions = lambda: []

    app = AppTest.from_file(str(Path(__file__).parents[1] / 'streamlit_app.py'), default_timeout=15).run()
    app.radio(key='account_page').set_value('My Hiking History').run()
    app.checkbox(key='training_select_all').check().run()
    app.checkbox(key='training_' + history_repo.hikes[0]['id']).uncheck().run()
    assert not app.exception and not app.error
    assert [h['training_enabled'] for h in history_repo.hikes] == [False, True, True]
    assert not history_repo.models()

    app.radio(key='account_page').set_value('My Models').run()
    next(b for b in app.button if b.label == 'Train Models').click().run()
    assert not app.exception and not app.error
    active = [r for r in history_repo.models() if r['is_active']]
    assert {r['model_type'] for r in active} == {'moving', 'break'}
    assert all(r['training_hike_count'] == 2 for r in active)

    app.radio(key='account_page').set_value('My Hiking History').run()
    assert not app.exception and not app.error
    assert [app.checkbox(key='training_' + h['id']).value for h in history_repo.hikes] == [False, True, True]


def test_missing_configuration_shows_landing_without_reading_owner_data(monkeypatch):
    def missing():raise ValueError('Configure Supabase for multi-user access.')
    monkeypatch.setattr('hiking_time_predictor.cloud_ui.CloudConfig.from_env',missing)
    def forbidden(*args,**kwargs):raise AssertionError('Local data must never be read by the cloud entrypoint')
    monkeypatch.setattr('hiking_time_predictor.web_history.load_history',forbidden)
    monkeypatch.setattr('hiking_time_predictor.predictor.load_model',forbidden)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py')).run()
    assert not app.exception
    assert app.title[0].value=='Personal Hiking Time Predictor'
    assert app.button[0].disabled
    assert not app.radio


def test_account_boundaries_clear_all_private_state(monkeypatch):
    from hiking_time_predictor import cloud_ui
    from types import SimpleNamespace
    state={'_bound_user':'user-a','prediction':'private-a','_models':'private-a','account_export':b'private-a'}
    monkeypatch.setattr(cloud_ui,'st',SimpleNamespace(session_state=state))
    cloud_ui.clear_private_state('user-b')
    assert state=={'_bound_user':'user-b'}
    cloud_ui.clear_private_state()
    assert not state


def test_authenticated_navigation_training_prediction_and_account_switch(monkeypatch):
    from types import SimpleNamespace
    from test_personal_hiking import FakeRepo,recording_with_pauses
    from hiking_time_predictor.personal_hiking import preview_recording,save_recording
    from hiking_time_predictor.auth_sessions import UserSession
    from hiking_time_predictor import cloud_ui
    a,b=FakeRepo(),FakeRepo()
    current=[a]
    config=SimpleNamespace(model_key='test-model-signing-key-'*3,auth_url='https://hiking.test',session_cookie='opaque')
    monkeypatch.setattr(cloud_ui.CloudConfig,'from_env',lambda:config)
    monkeypatch.setattr(cloud_ui,'AuthManager',lambda _:SimpleNamespace(restore=lambda sid:UserSession(current[0].user_id,'test-access',{'email':'synthetic@example.invalid'},0)))
    monkeypatch.setattr(cloud_ui,'SupabaseRepository',lambda cfg,session:current[0])
    # Profile/preferences are synthetic fixtures; no real private files are read.
    for repo in (a,b):
        repo.profile=lambda r=repo:{'id':r.user_id,'created_at':'2026-01-01','display_name':'Synthetic','preferred_hike_type':'day_hike'}
        repo.predictions=lambda:[]
    def no_local(*args,**kwargs):raise AssertionError('Cloud UI tried to read a local model')
    monkeypatch.setattr('hiking_time_predictor.predictor.load_model',no_local)
    monkeypatch.setattr('hiking_time_predictor.planned.load_break_model',no_local)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=15).run()
    assert not app.exception and app.title[0].value=='My Dashboard'
    for page in ('My Hiking History','My Models','Account Settings','Trail Predictor'):
        app.radio(key='account_page').set_value(page).run()
        assert not app.exception and not app.error
    planned=b'<gpx><rte><rtept lat="0" lon="0"><ele>100</ele></rtept><rtept lat="0" lon="0.01"><ele>120</ele></rtept></rte></gpx>'
    app.file_uploader(key='planned_gpx').set_value(('plan.gpx',planned,'application/gpx+xml')).run()
    next(x for x in app.button if x.label=='Predict Hiking Time').click().run()
    assert not app.exception and not app.error
    assert app.session_state['prediction'].artifact.estimator is None
    assert app.session_state['prediction'].break_forecast.predicted_minutes is None
    for day in (1,2):
        save_recording(a,preview_recording(recording_with_pauses(day),'synthetic.gpx','day_hike'),training_enabled=True)
    app.radio(key='account_page').set_value('My Models').run()
    next(x for x in app.button if x.label=='Train Models').click().run()
    assert not app.exception and not app.error
    assert len([r for r in a.models() if r['is_active']])==2
    assert not b.models()
    app.radio(key='account_page').set_value('Trail Predictor').run()
    app.file_uploader(key='planned_gpx').set_value(('plan.gpx',planned,'application/gpx+xml')).run()
    next(x for x in app.button if x.label=='Predict Hiking Time').click().run()
    assert not app.exception and not app.error
    prediction = app.session_state['prediction']
    assert prediction.break_forecast.predicted_minutes > 0
    assert next(m.value for m in app.metric if m.label == 'Original predicted break time') != 'Unavailable'
    from hiking_time_predictor.web_components import duration
    assert next(m.value for m in app.metric if m.label == 'Estimated total trip') == duration(
        prediction.forecast.moving_minutes + prediction.break_forecast.predicted_minutes)
    current[0]=b
    app.run()
    assert not app.exception and not app.error
    assert app.radio(key='account_page').value=='My Dashboard'
    assert 'prediction' not in app.session_state
    assert app.session_state['_bound_user']==b.user_id
