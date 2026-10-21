"""Upload validation and inference consistency without Streamlit or user files."""

from dataclasses import replace
from io import StringIO
from math import degrees
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from hiking_time_predictor.gpx import EARTH_RADIUS_M, METERS_PER_MILE
from hiking_time_predictor.modeling import GEOMETRY, MODEL_LABELS
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor import web_upload
from hiking_time_predictor.web_upload import MAX_UPLOAD_BYTES, predict_upload, preview_upload, trip_statistics


@pytest.fixture
def untimed_gpx():
    return ('<gpx version="1.1"><rte>' + "".join(
        f'<rtept lat="0" lon="{degrees(meters / EARTH_RADIUS_M)}"><ele>{elevation}</ele></rtept>'
        for meters, elevation in [(0, 100), (400, 120), (1200, 160), (1600, 130), (2200, 180)]
    ) + '</rte></gpx>').encode("utf-8")


@pytest.fixture
def artifact():
    features = pd.DataFrame([[0.1, 0, 0], [0.5, 0, 0], [0.5, 100, 0], [0.5, 0, 100],
                             [0.3, 200, 40], [0.4, 50, 150]], columns=GEOMETRY)
    estimator = LinearRegression().fit(features, 1 + features.to_numpy() @ np.array([20, 0.03, 0.01]))
    return ModelArtifact(estimator=estimator, model_name="B", selected=True,
                         reason="Synthetic test fixture only.",
                         evaluation={"segment_metrics": [{"model": MODEL_LABELS["B"],
                                                          "observations": 6, "hikes": 3}]})


@pytest.mark.parametrize("direction,multiplier", [("one-way", 1), ("out-and-back", 2)])
def test_untimed_upload_geometry_and_prediction_are_consistent(untimed_gpx, artifact, direction, multiplier):
    preview = preview_upload(untimed_gpx, "planned.gpx", "day_hike", direction, config=artifact.config)
    result = predict_upload(untimed_gpx, "planned.gpx", "day_hike", direction, artifact=artifact)
    pd.testing.assert_frame_equal(preview.segments, result.route.segments)
    pd.testing.assert_frame_equal(preview.profile, result.route.profile)
    assert result.route.source == "planned route geometry"
    assert result.route.name == "planned"
    assert result.segments.distance_miles.sum() == pytest.approx(multiplier * 2200 / METERS_PER_MILE)
    assert result.forecast.moving_minutes == pytest.approx(result.segments.predicted_time_min.sum())
    assert result.segments.cumulative_predicted_time_min.iloc[-1] == pytest.approx(result.forecast.moving_minutes)
    assert result.route.profile.distance_miles.iloc[-1] == pytest.approx(result.segments.distance_miles.sum())
    assert not result.input_path.exists()
    assert not result.input_path.parent.exists()
    assert "moving_time_min" not in result.segments
    if direction == "out-and-back":
        assert set(result.route.profile.leg) == {"outbound", "return"}
        outbound = result.route.profile.loc[result.route.profile.leg.eq("outbound")]
        inbound = result.route.profile.loc[result.route.profile.leg.eq("return")]
        np.testing.assert_allclose(inbound.longitude, outbound.longitude.iloc[::-1])


def test_upload_filename_is_safe_and_preserves_fallback_name(untimed_gpx, monkeypatch):
    original = web_upload.gpx_route
    observed = []

    def inspect(path, *args, **kwargs):
        observed.append(path)
        assert path.exists()
        assert path.name == "Unnamed trail.gpx"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(web_upload, "gpx_route", inspect)
    result = preview_upload(untimed_gpx, r"../../..\Unnamed trail.GPX", "day_hike")
    assert result.name == "Unnamed trail"
    assert not observed[0].parent.exists()


def test_named_gpx_retains_its_route_name(untimed_gpx):
    named = untimed_gpx.replace(b"<rte>", b"<rte><name>The ridge trail</name>")
    assert preview_upload(named, "irrelevant.gpx", "backpacking").name == "The ridge trail"


@pytest.mark.parametrize("contents,filename,message", [
    (b"", "trail.gpx", "empty"),
    (b"   ", "trail.gpx", "empty"),
    (b"x" * (MAX_UPLOAD_BYTES + 1), "trail.gpx", "20 MiB"),
    (b"\xff", "trail.gpx", "UTF-8"),
    (b"<gpx", "trail.gpx", "Cannot parse GPX"),
    (b"<gpx/>", "trail.csv", r"\.gpx"),
    (b"<gpx/>", "", "filename"),
    ("<gpx/>", "trail.gpx", "bytes"),
])
def test_invalid_uploads_raise_clear_errors(contents, filename, message):
    with pytest.raises(ValueError, match=message):
        preview_upload(contents, filename, "day_hike")


@pytest.mark.parametrize("operation", ["preview", "predict"])
def test_temporary_upload_cleanup_even_when_backend_raises(untimed_gpx, artifact, monkeypatch, operation):
    observed = []

    def fail(path, *args, **kwargs):
        observed.append(Path(path))
        assert observed[-1].exists()
        raise ValueError("Synthetic backend failure")

    target = "gpx_route" if operation == "preview" else "predict_planned_route"
    monkeypatch.setattr(web_upload, target, fail)
    with pytest.raises(ValueError, match="Synthetic backend failure"):
        if operation == "preview":
            preview_upload(untimed_gpx, "trail.gpx", "day_hike")
        else:
            predict_upload(untimed_gpx, "trail.gpx", "day_hike", artifact=artifact)
    assert len(observed) == 1
    assert not observed[0].exists()
    assert not observed[0].parent.exists()


def test_upload_inference_never_fits_or_loads_models(untimed_gpx, artifact, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Uploads must never train or load a model")

    monkeypatch.setattr(artifact.estimator, "fit", forbidden)
    monkeypatch.setattr("hiking_time_predictor.predictor.fit_model", forbidden)
    monkeypatch.setattr("hiking_time_predictor.predictor.train_predictor", forbidden)
    monkeypatch.setattr("hiking_time_predictor.planned.load_model", forbidden)
    result = predict_upload(untimed_gpx, "trail.gpx", "day_hike", artifact=artifact)
    assert result.forecast.label == MODEL_LABELS["B"]


def test_trip_statistics_preserve_experimental_model_policy_and_csv_totals(untimed_gpx, artifact):
    result = predict_upload(untimed_gpx, "trail.gpx", "backpacking", "out-and-back",
                            artifact=replace(artifact, selected=False))
    stats = trip_statistics(result, 35)
    exported = pd.read_csv(StringIO(result.segments.to_csv(index=False)))
    assert stats["segment_count"] == len(exported)
    for column in GEOMETRY:
        assert stats[column] == pytest.approx(exported[column].sum())
    assert stats["moving_minutes"] == pytest.approx(exported.predicted_time_min.sum())
    assert stats["moving_minutes"] == pytest.approx(stats["naismith_minutes"])
    assert stats["experimental_minutes"] == pytest.approx(exported.regression_time_min.sum())
    assert stats["experimental_minutes"] != pytest.approx(stats["moving_minutes"])
    assert stats["total_minutes"] == pytest.approx(stats["moving_minutes"] + 35)
    assert stats["break_minutes"] == 35
    assert stats["average_speed_mph"] == pytest.approx(stats["distance_miles"] / stats["moving_minutes"] * 60)
    assert trip_statistics(result)["break_minutes"] == 0


@pytest.mark.parametrize("breaks", [-1, float("nan"), float("inf"), "bad", None, True])
def test_break_allowance_must_be_finite_and_nonnegative(untimed_gpx, breaks):
    result = predict_upload(untimed_gpx, "trail.gpx", "day_hike", artifact=ModelArtifact())
    with pytest.raises(ValueError, match="finite nonnegative"):
        trip_statistics(result, breaks)
