"""Redesigned cloud UI: pure presenters plus AppTest flows. Synthetic data only."""
from datetime import date, time, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from streamlit.testing.v1 import AppTest

from hiking_time_predictor.break_model import BreakArtifact
from hiking_time_predictor.model_status import break_status, is_stale, moving_status
from hiking_time_predictor.modeling import MODEL_LABELS, NAISMITH_LABEL
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor.solar import sun_times
from hiking_time_predictor.ui_format import (flag_labels, humanize_filename, humanize_note, md_escape, method_name,
                                             pace, relative_day, route_extremes, safe_filename)

APP = str(Path(__file__).parents[1] / 'streamlit_app.py')
PLANNED = (b'<gpx><rte><name>Ridge *bold* [link](x)</name><rtept lat="37.8" lon="-122.5"><ele>100</ele></rtept>'
           b'<rtept lat="37.8" lon="-122.49"><ele>160</ele></rtept><rtept lat="37.81" lon="-122.48"><ele>140</ele></rtept></rte></gpx>')


# --- Pure helpers ------------------------------------------------------------

def test_markdown_escape_neutralises_user_text():
    escaped = md_escape('**x** [a](javascript:1) :red[y] $z$ <b>')
    assert '\\*\\*x\\*\\*' in escaped and '\\[a\\]' in escaped and '\\:red' in escaped and '\\$' in escaped
    assert md_escape(None) == ''


def test_formatting_helpers():
    assert pace(24.41) == '24:25 /mi' and pace(None) == '—' and pace(0) == '—'
    assert humanize_filename('Mt_Diablo_summit-loop.GPX') == 'Mt Diablo summit loop'
    assert safe_filename('../My hike: day 1', '.gpx') == 'My_hike_day_1.gpx'
    assert flag_labels('gps_heuristic;missing_hike_type;recording_gap') == ['Hike type not set', 'GPS gaps']
    assert method_name('baseline estimate: Naismith') == 'Standard hiking rule'
    assert method_name(MODEL_LABELS['B']) == 'Your pace model (distance and climbing)'
    assert relative_day('2026-10-20T10:00:00Z', now='2026-10-21T09:00:00Z') == 'yesterday'
    assert relative_day('2026-09-01T10:00:00Z', now='2026-10-21T09:00:00Z') == 'Sep 1, 2026'


def test_backend_notes_become_plain_language():
    note = humanize_note('Route total elevation_gain_ft outside training range [831.161, 2404.058] ft '
                         '(1 input(s)); extrapolation may be unreliable.')
    assert 'climbing' in note and '831 ft' in note and '2,404 ft' in note and 'input(s)' not in note
    assert humanize_note('Outbound: Route total distance_miles outside training range [4.093, 9.581] miles (1 input(s)); x').startswith('Outbound leg:')
    assert 'ignored' in humanize_note('GPX timestamps are ignored; actual hiking time is unavailable.')
    assert humanize_note("Hike type 'backpacking' was not observed in training metadata.") == \
        "You haven't trained on any backpacking trips yet."
    assert humanize_note('Something new.') == 'Something new.'
    assert route_extremes({'elevation_gain_ft': 2606, 'distance_miles': 8.8},
                          {'elevation_gain_ft': (831, 2404), 'distance_miles': (4.1, 9.6)}) == [
        "Bigger climb than any hike you've trained on (2,606 ft vs. your longest 2,404 ft)"]


def test_sunset_matches_almanac_values():
    rise, down = sun_times(37.7749, -122.4194, date(2026, 6, 21))
    local = down.astimezone(ZoneInfo('America/Los_Angeles'))
    assert (local.hour, local.minute) in ((20, 34), (20, 35))
    assert rise.astimezone(ZoneInfo('America/Los_Angeles')).hour == 5
    winter = sun_times(37.7749, -122.4194, date(2026, 12, 21))[1].astimezone(ZoneInfo('America/Los_Angeles'))
    assert abs(winter.hour * 60 + winter.minute - (16 * 60 + 54)) <= 2
    assert sun_times(69.65, 18.96, date(2026, 6, 21)) == (None, None)  # midnight sun
    assert sun_times(float('nan'), 0, date(2026, 1, 1)) == (None, None)


def _artifact(model_name=None, selected=False, verified=8):
    rows = [{'model': NAISMITH_LABEL, 'hikes': verified, 'MAE_min': 16.8, 'bias_pred_minus_actual_min': 16.8}]
    if model_name:
        rows.append({'model': MODEL_LABELS[model_name], 'hikes': verified, 'MAE_min': 11.6})
    return ModelArtifact(model_name=model_name, selected=selected, estimator=object() if model_name else None,
                         evaluation={'eligible_hikes': verified, 'verified_held_out_hikes': verified,
                                     'minimum_validation_hikes': 10, 'whole_hike_metrics': rows})


def test_moving_status_states_follow_saved_policy():
    rows = [{'dataset_fingerprint': 'a'}]
    none = moving_status(ModelArtifact(), [], [])
    assert none.state == 'none' and 'standard hiking rule' in none.headline
    ready = moving_status(ModelArtifact(), [], [{'training_enabled': True}, {'training_enabled': True, 'deleting': True}])
    assert '1 hike is ready' in ready.headline
    learning = moving_status(_artifact('B'), rows)
    assert learning.state == 'learning' and learning.progress == pytest.approx(0.8) and '2 to go' in learning.headline
    assert '12 min' in learning.detail and '17 min' in learning.detail
    assert moving_status(_artifact('B', verified=11), rows).state == 'rule_wins'
    personal = moving_status(_artifact('B', selected=True, verified=12), rows)
    assert personal.state == 'personal' and personal.color == 'green'
    assert moving_status(ModelArtifact(evaluation={'eligible_hikes': 1}), rows).state == 'too_few'
    assert is_stale(rows, 'b') and not is_stale(rows, 'a') and not is_stale([], 'a')


def test_break_status_states():
    assert break_status(BreakArtifact(), []).state == 'none'
    assert break_status(BreakArtifact(), [{}]).state == 'unavailable'
    early = break_status(BreakArtifact(minutes_per_mile=2.5, training_hikes=3,
                                       evaluation={'minimum_evaluation_hikes': 5}), [{}])
    assert early.state == 'early' and '2 more hikes' in early.detail
    rate = break_status(BreakArtifact(minutes_per_mile=2.5, breaks_per_mile=0.25, mean_duration_minutes=8, training_hikes=8,
                                      evaluation={'metrics': [{'model': 'personal_minutes_per_mile', 'MAE_minutes': 4.9}]}), [{}])
    assert rate.state == 'rate' and 'every 4 mi' in rate.headline and '5 min' in rate.detail


# --- AppTest flows -----------------------------------------------------------

def goto(app, title):
    app._page_hash = next(h for h, i in app._registered_pages.items() if i.get('page_name') == title)
    try:
        return app.run()
    finally:
        app._page_hash = ''


@pytest.fixture
def signed_in(monkeypatch):
    from test_personal_hiking import FakeRepo
    from hiking_time_predictor.auth_sessions import UserSession
    from hiking_time_predictor import cloud_ui
    repo = FakeRepo()
    saved = []
    repo.profile = lambda: {'id': repo.user_id, 'created_at': '2026-01-01T00:00:00Z', 'display_name': '**Sam**',
                            'preferred_hike_type': 'backpacking'}
    repo.predictions = lambda: list(saved)
    repo.save_prediction = lambda row: saved.append({**row, 'created_at': '2026-10-21T12:00:00Z'})
    config = SimpleNamespace(model_key='redesign-test-signing-key-' * 3, auth_url='https://hiking.test',
                             session_cookie='opaque')
    monkeypatch.setattr(cloud_ui.CloudConfig, 'from_env', lambda: config)
    monkeypatch.setattr(cloud_ui, 'AuthManager', lambda _: SimpleNamespace(
        restore=lambda sid: UserSession(repo.user_id, 'test-access', {'email': 'synthetic@example.invalid'}, 0)))
    monkeypatch.setattr(cloud_ui, 'SupabaseRepository', lambda cfg, session: repo)
    repo.saved = saved
    return repo


def _metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


def test_new_account_dashboard_shows_onboarding_and_ctas(signed_in):
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception and not app.error
    assert app.title[0].value == 'Dashboard'
    assert "Let's learn your hiking pace" in [s.value for s in app.subheader]
    assert 'Welcome back, \\*\\*Sam\\*\\*.' in [c.value for c in app.caption]  # display name is escaped
    assert not app.metric  # no wall of zeros
    app.button(key='home_plan_now').click().run()
    assert not app.exception and app.title[0].value == 'Plan a hike'


def test_plan_predicts_on_upload_with_break_modes_start_time_and_save(signed_in):
    from hiking_time_predictor.web_components import duration
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'Plan a hike')
    # Preferred hike type from the profile is the default.
    assert app.segmented_control(key='trail_hike_type').value == 'backpacking'
    app.file_uploader(key='planned_gpx').set_value(('ridge.gpx', PLANNED, 'application/gpx+xml')).run()
    assert not app.exception and not app.error
    prediction = app.session_state['prediction']
    moving = prediction.forecast.moving_minutes
    assert _metric(app, 'Estimated total trip') == duration(moving)  # no break history: moving only
    assert 'Ridge \\*bold\\* \\[link\\]\\(x\\)' in [s.value for s in app.subheader]

    app.segmented_control(key='trail_break_mode').set_value('custom').run()
    assert app.number_input(key='trail_break_minutes').value == 30  # prefilled, never a silent 0
    app.number_input(key='trail_break_minutes').set_value(45).run()
    assert app.session_state['prediction'] is prediction  # break changes never rerun inference
    assert _metric(app, 'Estimated total trip') == duration(moving + 45)
    app.segmented_control(key='trail_break_mode').set_value('none').run()
    assert _metric(app, 'Breaks') == 'None' and _metric(app, 'Estimated total trip') == duration(moving)

    app.time_input(key='trail_start').set_value(time(8, 0)).run()
    assert not app.exception
    assert _metric(app, 'Back around').endswith('AM')

    app.segmented_control(key='trail_direction').set_value('out-and-back').run()
    assert app.session_state['prediction'] is not prediction  # route settings re-predict automatically
    assert app.session_state['prediction'].route.direction == 'out-and-back'

    app.button(key='save_plan').click().run()
    assert not app.exception and len(signed_in.saved) == 1
    row = signed_in.saved[0]
    assert row['route_direction'] == 'out-and-back' and row['hike_type'] == 'backpacking'
    assert row['predicted_break_time_min'] == 0  # breaks explicitly excluded
    assert row['details']['planned_start'].startswith(date.today().isoformat())
    assert app.button(key='save_plan').disabled and app.button(key='save_plan').label == 'Saved'


def test_plan_remembers_route_across_pages_and_clears(signed_in):
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'Plan a hike')
    app.file_uploader(key='planned_gpx').set_value(('ridge.gpx', PLANNED, 'application/gpx+xml')).run()
    first = app.session_state['prediction']
    goto(app, 'My pace')
    goto(app, 'Plan a hike')
    assert not app.exception and app.session_state['prediction'] is first
    assert any(m.label == 'Estimated total trip' for m in app.metric)
    app.button(key='trail_clear').click().run()
    assert 'prediction' not in app.session_state and not any(m.label == 'Estimated total trip' for m in app.metric)


def test_invalid_route_shows_inline_error_and_keeps_page(signed_in):
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'Plan a hike')
    app.file_uploader(key='planned_gpx').set_value(('bad.gpx', b'<gpx', 'application/gpx+xml')).run()
    assert not app.exception
    assert any("couldn't read this GPX" in e.value for e in app.error)
    assert app.title[0].value == 'Plan a hike' and 'prediction' not in app.session_state


def test_hike_detail_stops_and_delete_dialog(signed_in):
    from test_personal_hiking import recording_with_pauses
    from hiking_time_predictor.personal_hiking import preview_recording, save_recording
    deleted = []
    signed_in.delete_hike = lambda hike_id: deleted.append(hike_id) or signed_in.hikes.clear()
    save_recording(signed_in, preview_recording(recording_with_pauses(1), 'Lunch_loop.gpx', 'day_hike'),
                   training_enabled=True)
    hike_id = signed_in.hikes[0]['id']
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'My hikes')
    app.button(key='details_' + hike_id).click().run()
    assert not app.exception and not app.error
    assert app.session_state['activity_detail'][0] == hike_id
    assert app.button(key='save_breaks').disabled  # nothing to save yet
    app.button(key='delete_hike').click().run()
    assert app.button(key='delete_confirm')
    app.button(key='delete_cancel').click().run()
    assert not deleted and 'history_delete_hike' not in app.session_state
    app.button(key='delete_hike').click().run()
    app.button(key='delete_confirm').click().run()
    assert deleted == [hike_id] and not app.exception
    assert 'history_detail_hike_id' not in app.session_state
    assert 'No hikes yet' in [s.value for s in app.subheader]


def test_account_page_preferences_and_export(signed_in):
    saved = []
    signed_in.save_profile = lambda name, kind: saved.append((name, kind))
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'Account')
    assert not app.exception and not app.error
    assert 'synthetic@example.invalid' in [t.value for t in app.text]
    next(b for b in app.button if b.label == 'Save preferences').click().run()
    assert saved == [('**Sam**', 'backpacking')]
    next(b for b in app.button if b.label == 'Prepare my download').click().run()
    assert app.session_state['account_export'][:2] == b'PK'
    assert app.get('download_button')[0].proto.label == 'Download my data (.zip)'


def test_trained_dashboard_and_pace_page(signed_in):
    from test_personal_hiking import recording_with_pauses
    from hiking_time_predictor.personal_hiking import preview_recording, save_recording
    for day in (1, 2, 3):
        save_recording(signed_in, preview_recording(recording_with_pauses(day), f'h{day}.gpx', 'day_hike'),
                       training_enabled=True)
    app = AppTest.from_file(APP, default_timeout=15).run()
    goto(app, 'My pace')
    assert app.button(key='train_models').label == 'Train my pace model'
    app.button(key='train_models').click().run()
    assert not app.exception and not app.error
    assert app.button(key='train_models').label == 'Retrain my pace model'
    assert any('Trained on' in t.value for t in app.toast)
    goto(app, 'Dashboard')
    assert not app.exception and not app.error
    assert _metric(app, 'Hikes recorded') == '3'
    assert 'You vs. the standard hiking rule' in [s.value for s in app.subheader]
    signed_in.set_training(signed_in.hikes[0]['id'], False)
    app.run()
    assert any('changed since your pace model' in w.value for w in app.warning)  # stale model is surfaced
