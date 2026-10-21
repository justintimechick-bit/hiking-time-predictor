"""Local Streamlit navigation and real upload workflows using AppTest."""

from dataclasses import replace
from math import degrees
from pathlib import Path
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from streamlit.testing.v1 import AppTest

from hiking_time_predictor.gpx import EARTH_RADIUS_M, METERS_PER_MILE
from hiking_time_predictor.modeling import GEOMETRY, MODEL_LABELS
from hiking_time_predictor.predictor import ModelArtifact


PROJECT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT / "local_streamlit_app.py"
PAGES = ("Hiking Time Predictor", "My Hiking Data", "Model Performance")


def _app_with_local_model(model_path):
    """Run the actual entry point with a test-local artifact path."""
    from pathlib import Path

    import local_streamlit_app as streamlit_app

    streamlit_app.MODEL_PATH = Path(model_path)
    streamlit_app.main()


@pytest.fixture
def untimed_gpx():
    return ('<gpx version="1.1"><rte><name>Test ridge trail</name>' + "".join(
        f'<rtept lat="0" lon="{degrees(meters / EARTH_RADIUS_M)}"><ele>{elevation}</ele></rtept>'
        for meters, elevation in [(0, 100), (400, 120), (1200, 160), (1600, 130), (2200, 180)]
    ) + '</rte></gpx>').encode("utf-8")


@pytest.fixture
def synthetic_artifact():
    features = pd.DataFrame([[0.1, 0, 0], [0.5, 0, 0], [0.5, 100, 0], [0.5, 0, 100],
                             [0.3, 200, 40], [0.4, 50, 150]], columns=GEOMETRY)
    estimator = LinearRegression().fit(features, 1 + features.to_numpy() @ np.array([20, 0.03, 0.01]))
    return ModelArtifact(estimator=estimator, model_name="B", selected=True,
                         reason="Synthetic fixture for UI testing only.",
                         evaluation={"method": "synthetic fixture", "eligible_segments": 6,
                                     "eligible_hikes": 3, "verified_held_out_hikes": 0,
                                     "segment_metrics": [{"model": MODEL_LABELS["B"],
                                                          "observations": 6, "hikes": 3,
                                                          "MAE_min": 1.5, "RMSE_min": 2.0}]})


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    # The wrapper imports the real module on each rerun. Restore the shared
    # module constant when the test ends so other tests see the project path.
    import local_streamlit_app as streamlit_app

    monkeypatch.setattr(streamlit_app, "MODEL_PATH", streamlit_app.MODEL_PATH)
    monkeypatch.setattr(streamlit_app, "BREAK_MODEL_PATH", tmp_path / "absent-break-model.joblib")

    def create(artifact=None, *, contents=None):
        path = tmp_path / "fixture.joblib"
        if artifact is not None:
            joblib.dump(artifact, path)
        elif contents is not None:
            path.write_bytes(contents)
        return AppTest.from_function(_app_with_local_model, args=(str(path),), default_timeout=15).run()

    return create


def _upload(app, contents):
    return app.file_uploader(key="gpx_upload").set_value(
        ("untimed-trail.gpx", contents, "application/gpx+xml")
    ).run()


def _metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


def _captions(app):
    return "\n".join(item.value for item in app.caption)


def test_initial_page_and_navigation_have_no_exceptions():
    app = AppTest.from_file(str(APP_FILE), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == PAGES[0]
    assert app.button(key="predict").disabled
    assert "Upload a GPX" in app.info[0].value
    for page in PAGES[1:] + PAGES[:1]:
        app.radio(key="page").set_value(page).run()
        assert not app.exception
        assert not app.error
        assert app.title[0].value == page


def test_missing_model_is_labeled_baseline_and_metrics_are_unavailable(app_factory, untimed_gpx):
    app = app_factory()
    assert not app.exception
    assert "Predictor: Naismith baseline" in _captions(app)
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    assert not app.exception
    prediction = app.session_state["prediction"]
    assert prediction.forecast.moving_minutes == pytest.approx(prediction.forecast.naismith_minutes)
    assert prediction.forecast.experimental_minutes is None
    assert "Primary estimate: baseline estimate: Naismith" in _captions(app)
    app.radio(key="page").set_value(PAGES[2]).run()
    assert not app.exception
    assert _metric(app, "Eligible training segments") == "Unavailable"
    assert _metric(app, "Independent eligible hikes") == "Unavailable"
    assert "Evaluation metrics are unavailable" in "\n".join(item.value for item in app.info)


def test_invalid_local_artifact_is_visible_and_disables_prediction(app_factory, untimed_gpx):
    app = app_factory(contents=b"invalid joblib artifact")
    assert not app.exception
    assert "Model loading failed" in "\n".join(item.value for item in app.error)
    _upload(app, untimed_gpx)
    assert not app.exception
    assert app.button(key="predict").disabled
    assert "prediction" not in app.session_state
    app.radio(key="page").set_value(PAGES[2]).run()
    assert not app.exception
    assert "Saved model performance is unavailable" in "\n".join(item.value for item in app.info)


@pytest.mark.parametrize("direction,multiplier", [("One Way", 1), ("Out and Back", 2)])
@pytest.mark.parametrize("selected", [True, False])
def test_real_untimed_upload_predicts_and_labels_saved_model_policy(
    app_factory, untimed_gpx, synthetic_artifact, direction, multiplier, selected
):
    app = app_factory(replace(synthetic_artifact, selected=selected))
    app.selectbox(key="route_direction").set_value(direction).run()
    _upload(app, untimed_gpx)
    assert not app.exception
    assert "prediction" not in app.session_state
    assert "Test ridge trail" in [item.value for item in app.subheader]
    assert len(app.get("plotly_chart")) == 1
    app.button(key="predict").click().run()
    assert not app.exception
    assert not app.error
    prediction = app.session_state["prediction"]
    assert prediction.route.source == "planned route geometry"
    assert prediction.segments.distance_miles.sum() == pytest.approx(multiplier * 2200 / METERS_PER_MILE)
    assert prediction.forecast.moving_minutes == pytest.approx(prediction.segments.predicted_time_min.sum())
    assert prediction.segments.cumulative_predicted_time_min.iloc[-1] == pytest.approx(prediction.forecast.moving_minutes)
    assert len(app.get("plotly_chart")) == 2
    assert app.download_button(key="download").label == "Download segment predictions CSV"
    assert app.dataframe[0].value.equals(prediction.segments)
    assert _metric(app, "Planned breaks") == "0 min"
    assert _metric(app, "Estimated total trip") == _metric(app, "Estimated moving time")
    if selected:
        assert "Primary estimate: " + MODEL_LABELS["B"] in _captions(app)
        assert prediction.forecast.experimental_minutes is None
        assert "Experimental personalized estimate" not in [item.value for item in app.subheader]
    else:
        assert "Primary estimate: baseline estimate: Naismith" in _captions(app)
        assert prediction.forecast.experimental_minutes is not None
        assert "Experimental personalized estimate" in [item.value for item in app.subheader]
        assert "Saved regression moving time" in [item.label for item in app.metric]
    assert not prediction.input_path.parent.exists()


def test_planned_break_changes_update_cards_without_new_predictions(
    app_factory, untimed_gpx, synthetic_artifact, monkeypatch
):
    import local_streamlit_app as streamlit_app

    app = app_factory(synthetic_artifact)
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    prediction = app.session_state["prediction"]
    moving = _metric(app, "Estimated moving time")

    def forbidden(*args, **kwargs):
        pytest.fail("Changing breaks must not rerun model inference")

    monkeypatch.setattr(streamlit_app, "predict_upload", forbidden)
    app.checkbox(key="override_breaks").set_value(True).run()
    app.number_input(key="break_minutes").set_value(30).run()
    assert not app.exception
    assert app.session_state["prediction"] is prediction
    assert _metric(app, "Estimated moving time") == moving
    assert _metric(app, "Planned breaks") == "30 min"
    assert _metric(app, "Estimated total trip") == streamlit_app.duration(prediction.forecast.moving_minutes + 30)


def test_predicted_breaks_toggle_and_override_match_python_functions(app_factory, untimed_gpx, monkeypatch):
    import local_streamlit_app as streamlit_app
    from hiking_time_predictor.break_model import BreakArtifact
    from hiking_time_predictor.web_upload import trip_statistics

    synthetic_breaks = BreakArtifact(minutes_per_mile=5, breaks_per_mile=1, mean_duration_minutes=5,
                                    training_hikes=6, training_ranges={f: [0, 10000] for f in GEOMETRY},
                                    reason="Synthetic fixture for UI testing only.")
    monkeypatch.setattr(streamlit_app, "break_model_state", lambda: synthetic_breaks)
    app = app_factory()
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    prediction = app.session_state["prediction"]
    expected = trip_statistics(prediction, include_predicted_breaks=True)
    original = _metric(app, "Original predicted break time")
    assert original == streamlit_app.duration(expected["predicted_break_minutes"])
    assert _metric(app, "Planned breaks") == streamlit_app.duration(expected["break_minutes"])
    assert _metric(app, "Estimated total trip") == streamlit_app.duration(expected["total_minutes"])

    def forbidden(*args, **kwargs):
        pytest.fail("Changing break controls must not rerun moving inference")

    monkeypatch.setattr(streamlit_app, "predict_upload", forbidden)
    app.checkbox(key="override_breaks").set_value(True).run()
    app.number_input(key="break_minutes").set_value(20).run()
    assert _metric(app, "Original predicted break time") == original
    assert _metric(app, "Estimated total trip") == streamlit_app.duration(prediction.forecast.moving_minutes + 20)
    app.checkbox(key="include_breaks").set_value(False).run()
    assert not app.exception
    assert _metric(app, "Planned breaks") == "0 min"
    assert _metric(app, "Estimated total trip") == _metric(app, "Estimated moving time")
    assert app.session_state["prediction"] is prediction


def test_break_history_dashboard_shows_real_exports_without_errors():
    app = AppTest.from_file(str(APP_FILE), default_timeout=15).run()
    app.radio(key="page").set_value(PAGES[1]).run()
    assert not app.exception
    assert "MY BREAKING HABITS" in [item.value for item in app.subheader]
    # The app also works in fresh checkouts without generated personal data.
    if (PROJECT / "data/processed/break_events.csv").exists():
        assert "Historical breaks" in [item.label for item in app.metric]


@pytest.mark.parametrize("widget,value", [("route_direction", "Out and Back"), ("hike_type", "Backpacking")])
def test_route_choices_invalidate_prediction_until_predict_is_clicked(
    app_factory, untimed_gpx, synthetic_artifact, widget, value
):
    app = app_factory(synthetic_artifact)
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    first = app.session_state["prediction"]
    app.selectbox(key=widget).set_value(value).run()
    assert not app.exception
    assert "prediction" not in app.session_state
    assert "Estimated moving time" not in [item.label for item in app.metric]
    assert not app.download_button
    app.button(key="predict").click().run()
    assert not app.exception
    second = app.session_state["prediction"]
    assert second is not first
    if widget == "route_direction":
        assert second.route.direction == "out-and-back"
        assert len(second.segments) == 2 * len(first.segments)
    else:
        assert second.segments.hike_type.eq("backpacking").all()
    app.file_uploader(key="gpx_upload").clear().run()
    assert not app.exception
    assert "prediction" not in app.session_state
    assert "route_preview" not in app.session_state
    assert app.button(key="predict").disabled


@pytest.mark.parametrize("contents,message", [(b"<gpx", "Cannot parse GPX"), (b"\xff", "UTF-8")])
def test_invalid_upload_shows_recoverable_error(app_factory, contents, message, untimed_gpx):
    app = app_factory()
    _upload(app, contents)
    assert not app.exception
    assert message in "\n".join(item.value for item in app.error)
    assert "prediction" not in app.session_state
    _upload(app, untimed_gpx)
    assert not app.exception
    assert not app.error
    app.button(key="predict").click().run()
    assert not app.exception
    assert "prediction" in app.session_state


def test_valid_upload_recovers_after_invalid_file_then_same_original(app_factory, untimed_gpx):
    app = app_factory()
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    original_key = app.session_state["route_key"]
    _upload(app, b"<gpx")
    assert not app.exception
    assert app.error
    assert "prediction" not in app.session_state
    assert "route_key" not in app.session_state
    _upload(app, untimed_gpx)
    assert not app.exception
    assert not app.error
    assert app.session_state["route_key"] == original_key
    app.button(key="predict").click().run()
    assert not app.exception
    assert "prediction" in app.session_state


def test_changed_model_file_invalidates_existing_prediction(
    app_factory, untimed_gpx, synthetic_artifact
):
    import local_streamlit_app as streamlit_app

    app = app_factory(synthetic_artifact)
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    original_key = app.session_state["prediction_key"]
    assert app.session_state["prediction"].artifact.selected
    joblib.dump(replace(synthetic_artifact, selected=False,
                        reason="Updated synthetic test artifact; experimental status."), streamlit_app.MODEL_PATH)
    app.run()
    assert not app.exception
    assert "prediction" not in app.session_state
    assert app.session_state["route_key"] != original_key
    app.button(key="predict").click().run()
    assert not app.exception
    assert not app.session_state["prediction"].artifact.selected
    assert app.session_state["prediction"].forecast.experimental_minutes is not None


@pytest.mark.parametrize("initial_page", PAGES[1:])
def test_predict_button_returns_from_other_pages_and_generates_results(
    app_factory, untimed_gpx, synthetic_artifact, initial_page
):
    app = app_factory(synthetic_artifact)
    app.radio(key="page").set_value(initial_page).run()
    _upload(app, untimed_gpx)
    assert app.title[0].value == initial_page
    app.button(key="predict").click().run()
    assert not app.exception
    assert not app.error
    assert app.radio(key="page").value == PAGES[0]
    assert app.title[0].value == PAGES[0]
    assert "prediction" in app.session_state
    assert "Predicted hiking statistics" in [item.value for item in app.subheader]


def test_app_uploads_never_train_models(app_factory, untimed_gpx, synthetic_artifact, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The web app must never fit or train during upload or prediction")

    app = app_factory(synthetic_artifact)
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    monkeypatch.setattr("hiking_time_predictor.predictor.fit_model", forbidden)
    monkeypatch.setattr("hiking_time_predictor.predictor.train_predictor", forbidden)
    monkeypatch.setattr("hiking_time_predictor.modeling.fit_model", forbidden)
    _upload(app, untimed_gpx)
    app.button(key="predict").click().run()
    assert not app.exception
    app.selectbox(key="route_direction").set_value("Out and Back").run()
    app.button(key="predict").click().run()
    assert not app.exception
    assert app.session_state["prediction"].route.direction == "out-and-back"


def test_source_reload_refreshes_cached_model_and_existing_upload(tmp_path):
    # Match Streamlit's module eviction in a separate process so the other
    # tests retain their imported classes and the user's artifacts stay intact.
    script = textwrap.dedent('''
        from pathlib import Path
        import sys
        import joblib
        from streamlit.testing.v1 import AppTest
        import local_streamlit_app as streamlit_app
        from hiking_time_predictor.predictor import ModelArtifact

        directory = Path(sys.argv[1])
        model = directory / "synthetic-model.joblib"
        joblib.dump(ModelArtifact(), model)
        original_bytes = model.read_bytes()
        source = (
            "from pathlib import Path\\nimport local_streamlit_app as streamlit_app\\n"
            f"streamlit_app.MODEL_PATH = Path({str(model)!r})\\n"
            f"streamlit_app.BREAK_MODEL_PATH = Path({str(directory / 'missing-break-model.joblib')!r})\\n"
            "streamlit_app.main()\\n"
        )
        app = AppTest.from_string(source, default_timeout=15).run()
        assert not app.exception
        contents = (b'<gpx version="1.1"><rte><rtept lat="0" lon="0"><ele>100</ele></rtept>'
                    b'<rtept lat="0" lon="0.01"><ele>120</ele></rtept></rte></gpx>')
        app.file_uploader(key="gpx_upload").set_value(("synthetic.gpx", contents, "application/gpx+xml")).run()
        app.button(key="predict").click().run()
        assert not app.exception and not app.error
        old_artifact, error = streamlit_app.model_state()
        assert error is None
        old_prediction = app.session_state["prediction"]
        old_key = app.session_state["prediction_key"]

        for name in list(sys.modules):
            if name == "local_streamlit_app" or name == "hiking_time_predictor" or name.startswith("hiking_time_predictor."):
                sys.modules.pop(name)

        app.run()
        assert not app.exception and not app.error
        import local_streamlit_app as reloaded_app
        from hiking_time_predictor.predictor import ModelArtifact as ReloadedArtifact
        assert not isinstance(old_artifact, ReloadedArtifact)
        refreshed, error = reloaded_app.model_state()
        assert error is None and isinstance(refreshed, ReloadedArtifact)
        assert refreshed is not old_artifact
        assert reloaded_app.model_state()[0] is refreshed
        assert app.session_state["route_key"] != old_key
        assert "prediction" not in app.session_state

        app.button(key="predict").click().run()
        assert not app.exception and not app.error
        prediction = app.session_state["prediction"]
        assert prediction.artifact is refreshed
        assert prediction.forecast.moving_minutes == old_prediction.forecast.moving_minutes
        assert model.read_bytes() == original_bytes
        print("Reloaded upload prediction succeeded without retraining.")
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=PROJECT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reloaded upload prediction succeeded" in result.stdout
