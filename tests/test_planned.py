"""End-to-end planned inference, using synthetic geometry and known model equations."""

from dataclasses import replace
import hashlib
from math import degrees
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from hiking_time_predictor.cli import main
from hiking_time_predictor.gpx import EARTH_RADIUS_M, METERS_PER_MILE, ImportConfig, process_hike
from hiking_time_predictor.modeling import GEOMETRY, MODEL_FEATURES, MODEL_LABELS, FeatureSchemaError, make_estimator
from hiking_time_predictor.planned import predict_planned_route
from hiking_time_predictor.predictor import ModelArtifact, load_model
from hiking_time_predictor.route_report import export_prediction, prediction_markdown
from hiking_time_predictor.routes import gpx_route, reverse_route


PROJECT = Path(__file__).resolve().parents[1]
REAL_GPX = PROJECT / "data/planned/(before hike)Hogpen_Gat_to_Poplar_Stomp.gpx"


@pytest.fixture
def route_file(tmp_path):
    path = tmp_path / "synthetic-planned.gpx"
    path.write_text('<gpx version="1.1"><rte><name>Synthetic test route</name>' + "".join(
        f'<rtept lat="0" lon="{degrees(meters / EARTH_RADIUS_M)}"><ele>{elevation}</ele></rtept>'
        for meters, elevation in [(0, 100), (400, 120), (1200, 160), (1600, 130), (2200, 180)]
    ) + '</rte></gpx>')
    return path


@pytest.fixture
def artifact():
    # Deliberately synthetic: independent terrain columns, known positive equation.
    features = pd.DataFrame([[0.1, 0, 0], [0.5, 0, 0], [0.5, 100, 0], [0.5, 0, 100],
                             [0.3, 200, 40], [0.4, 50, 150]], columns=GEOMETRY)
    estimator = LinearRegression().fit(features, 1 + features.to_numpy() @ np.array([20, 0.03, 0.01]))
    return ModelArtifact(estimator=estimator, model_name="B", selected=True,
                         reason="Synthetic test model only.",
                         segment_ranges={name: (0, 1000) for name in GEOMETRY},
                         route_ranges={name: (0, 10000) for name in GEOMETRY},
                         hike_types=["day_hike", "backpacking"],
                         evaluation={"method": "synthetic test fixture", "segment_metrics": [
                             {"model": MODEL_LABELS["B"], "observations": 6, "hikes": 3,
                              "MAE_min": 1.5, "RMSE_min": 2.0}]})


def test_geometry_matches_recorded_training_pipeline_for_connected_moving_edges(tmp_path):
    root = Element("gpx", version="1.1", creator="SYNTHETIC TEST")
    parent = SubElement(SubElement(root, "trk"), "trkseg")
    for index in range(31):
        point = SubElement(parent, "trkpt", lat="0", lon=str(degrees(80 * index / EARTH_RADIUS_M)))
        SubElement(point, "ele").text = str(100 + 30 * np.sin(index / 5))
        SubElement(point, "time").text = f"2026-01-01T00:{index:02d}:00Z"
    path = tmp_path / "synthetic-recording.gpx"
    path.write_bytes(tostring(root))
    config = ImportConfig(segment_miles=0.25, elevation_window_m=50)
    _, historical, _, intervals = process_hike(path, "synthetic", {"hike_type": "day_hike"}, config)
    assert intervals.state.eq("moving").all()
    planned = gpx_route(path, "day_hike", config)
    pd.testing.assert_frame_equal(planned.segments[GEOMETRY], historical[GEOMETRY], atol=1e-8, rtol=1e-8)
    assert not any("time" in name for name in planned.segments)


def test_independent_return_reverses_points_swaps_terrain_and_rebins(route_file, artifact):
    route = gpx_route(route_file, "backpacking", artifact.config)
    reverse = reverse_route(route)
    np.testing.assert_allclose(reverse.profile[["latitude", "longitude", "smoothed_elevation_m"]],
                               route.profile.iloc[::-1][["latitude", "longitude", "smoothed_elevation_m"]])
    assert reverse.segments.elevation_gain_ft.sum() == pytest.approx(route.segments.elevation_loss_ft.sum())
    assert reverse.segments.elevation_loss_ft.sum() == pytest.approx(route.segments.elevation_gain_ft.sum())
    # Re-bin from the return start; do not move the short tail to the beginning.
    assert reverse.segments.distance_miles.tolist() == pytest.approx(route.segments.distance_miles.tolist())
    assert reverse.segments.distance_miles.iloc[0] == pytest.approx(0.5)
    assert reverse.segments.distance_miles.iloc[-1] < 0.5
    result = predict_planned_route(route_file, "backpacking", "out-and-back", artifact=artifact)
    legs = result.leg_totals
    assert legs.loc["outbound", "predicted_time_min"] != pytest.approx(legs.loc["return", "predicted_time_min"])
    expected_difference = 0.02 * (route.segments.elevation_gain_ft.sum() - route.segments.elevation_loss_ft.sum())
    assert legs.loc["outbound", "predicted_time_min"] - legs.loc["return", "predicted_time_min"] == pytest.approx(expected_difference)
    assert result.forecast.moving_minutes == pytest.approx(legs.predicted_time_min.sum())
    assert result.segments.cumulative_predicted_time_min.iloc[-1] == pytest.approx(result.forecast.moving_minutes)
    assert result.route.profile.distance_miles.iloc[-1] == pytest.approx(4400 / METERS_PER_MILE)


def test_inference_exact_feature_order_and_no_training_or_labels(route_file, artifact, monkeypatch):
    def no_fit(*args, **kwargs):
        pytest.fail("Planned inference must never fit or retrain")
    predict = artifact.estimator.predict
    observed = []

    def inspect(features):
        observed.append(list(features.columns))
        assert "moving_time_min" not in features
        return predict(features)

    monkeypatch.setattr(artifact.estimator, "fit", no_fit)
    monkeypatch.setattr(artifact.estimator, "predict", inspect)
    result = predict_planned_route(route_file, "day_hike", artifact=artifact)
    assert observed == [MODEL_FEATURES["B"]]
    assert "moving_time_min" not in result.segments
    assert result.forecast.moving_minutes == pytest.approx(result.segments.predicted_time_min.sum())


def test_type_pipeline_receives_known_type_and_saved_order(route_file, artifact):
    geometry = pd.DataFrame([[0.1, 0, 0], [0.5, 0, 0], [0.5, 100, 0], [0.5, 0, 100]], columns=GEOMETRY)
    features = pd.concat([geometry.assign(hike_type="day_hike"), geometry.assign(hike_type="backpacking")], ignore_index=True)
    times = 2 + features[GEOMETRY].to_numpy() @ np.array([20, 0.03, 0.01]) + features.hike_type.eq("backpacking") * 3
    estimator = make_estimator("C").fit(features[MODEL_FEATURES["C"]], times)
    metrics = [{**artifact.evaluation["segment_metrics"][0], "model": MODEL_LABELS["C"]}]
    typed = replace(artifact, model_name="C", estimator=estimator, evaluation={"segment_metrics": metrics})
    day = predict_planned_route(route_file, "day_hike", artifact=typed)
    pack = predict_planned_route(route_file, "backpacking", artifact=typed)
    assert pack.forecast.moving_minutes - day.forecast.moving_minutes == pytest.approx(3 * len(day.segments))


def test_experimental_fit_stays_separate_from_primary_baseline(route_file, artifact):
    result = predict_planned_route(route_file, "backpacking", artifact=replace(artifact, selected=False))
    assert result.forecast.moving_minutes == result.forecast.naismith_minutes
    assert result.forecast.experimental_minutes == pytest.approx(result.segments.regression_time_min.sum())
    assert result.forecast.experimental_minutes != pytest.approx(result.forecast.moving_minutes)
    text = prediction_markdown(result)
    assert "Experimental saved-model moving time" in text
    assert "not prediction intervals" in text


@pytest.mark.parametrize("bad_intercept", [-100000, np.nan, np.inf])
def test_invalid_predictions_use_baseline_for_entire_route(route_file, artifact, bad_intercept):
    artifact.estimator.intercept_ = bad_intercept
    result = predict_planned_route(route_file, "day_hike", "out-and-back", artifact=artifact)
    assert result.forecast.moving_minutes == pytest.approx(result.forecast.naismith_minutes)
    assert result.segments.regression_time_min.isna().all()
    assert result.forecast.experimental_minutes is None
    assert any("Regression unavailable" in message for message in result.forecast.warnings)


def test_absent_default_and_unevaluated_models_use_labeled_baseline(route_file, artifact, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for supplied in (None, replace(artifact, evaluation={})):
        result = predict_planned_route(route_file, "day_hike", artifact=supplied)
        assert "baseline" in result.forecast.label
        assert result.forecast.experimental_minutes is None
        assert result.segments.regression_time_min.isna().all()


@pytest.mark.parametrize("contents", [None, b"invalid joblib"])
def test_explicit_missing_corrupt_models_raise_clear_errors(route_file, tmp_path, contents):
    path = tmp_path / "missing-or-invalid.joblib"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(ValueError, match="Could not load the model artifact"):
        predict_planned_route(route_file, "day_hike", model_path=path)


def test_incompatible_fitted_feature_schema_fails_without_silent_fallback(route_file, artifact, tmp_path):
    estimator = LinearRegression().fit(pd.DataFrame({"distance_miles": [0.1, 0.5], "actual_speed": [2, 3]}), [3, 10])
    bad = replace(artifact, estimator=estimator)
    with pytest.raises(FeatureSchemaError, match="Unsupported fitted feature schema"):
        predict_planned_route(route_file, "day_hike", artifact=bad)
    path = tmp_path / "incompatible.joblib"
    joblib.dump(bad, path)
    with pytest.raises(FeatureSchemaError, match="actual_speed"):
        load_model(path)


@pytest.mark.parametrize("kwargs,message", [
    ({"direction": "loop"}, "direction must"),
    ({"missing_elevation": "zero"}, "missing_elevation must"),
    ({"hike_type": "running"}, "hike_type"),
])
def test_unsupported_input_combinations_fail_clearly(route_file, kwargs, message):
    values = {"hike_type": "day_hike", **kwargs}
    with pytest.raises(ValueError, match=message):
        gpx_route(route_file, **values)


def test_feature_order_mismatch_and_conflicting_model_sources_fail(route_file, artifact):
    with pytest.raises(ValueError, match="not both"):
        predict_planned_route(route_file, "day_hike", artifact=artifact, model_path="model.joblib")
    artifact.estimator.feature_names_in_ = np.array(GEOMETRY[::-1], dtype=object)
    with pytest.raises(FeatureSchemaError, match="Unsupported fitted feature schema"):
        predict_planned_route(route_file, "day_hike", artifact=artifact)


def test_missing_elevation_interpolation_is_explicit_and_bounded(route_file):
    route_file.write_text(route_file.read_text().replace("<ele>120</ele>", ""))
    with pytest.raises(ValueError, match="finite elevation"):
        gpx_route(route_file, "day_hike")
    route = gpx_route(route_file, "day_hike", missing_elevation="interpolate", direction="out-and-back")
    assert any("Interpolated 1" in note for note in route.notes)
    assert route.profile.elevation_m.isna().sum() == 2  # Preserve raw missing values on both legs.
    assert np.isfinite(route.profile.smoothed_elevation_m).all()
    route_file.write_text(route_file.read_text().replace("<ele>100</ele>", ""))
    with pytest.raises(ValueError, match="section endpoint"):
        gpx_route(route_file, "day_hike", missing_elevation="interpolate")


def test_disconnected_reverse_sections_never_bridge_gaps(tmp_path):
    path = tmp_path / "disconnected.gpx"
    path.write_text('<gpx version="1.1">' + "".join(
        f'<trk><trkseg><trkpt lat="0" lon="{start}"><ele>{height}</ele></trkpt>'
        f'<trkpt lat="0" lon="{start + .001}"><ele>{height + 10}</ele></trkpt></trkseg></trk>'
        for start, height in [(0, 100), (10, 1000)]
    ) + '</gpx>')
    outbound = gpx_route(path, "day_hike")
    both = gpx_route(path, "day_hike", direction="out-and-back")
    assert both.segments.distance_miles.sum() == pytest.approx(2 * outbound.segments.distance_miles.sum())
    assert len(both.segments) == 4
    assert both.segments.elevation_gain_ft.sum() == pytest.approx(
        outbound.segments.elevation_gain_ft.sum() + outbound.segments.elevation_loss_ft.sum())
    assert both.profile.loc[both.profile.leg.eq("return"), "longitude"].iloc[0] == pytest.approx(10.001)


def test_segment_extrapolation_and_implausible_positive_speed_are_flagged(route_file, artifact):
    artifact.segment_ranges["elevation_gain_ft"] = (0, 1)
    result = predict_planned_route(route_file, "day_hike", artifact=artifact)
    assert result.segments.outside_training_range.any()
    assert any("Segment elevation_gain_ft outside" in note for note in result.forecast.warnings)
    artifact.estimator.coef_[:] = 0
    artifact.estimator.intercept_ = 0.01
    result = predict_planned_route(route_file, "day_hike", artifact=artifact)
    assert any("speed above" in note for note in result.forecast.warnings)


def test_exports_round_trip_segment_sums_and_write_all_figures(route_file, artifact, tmp_path):
    result = predict_planned_route(route_file, "backpacking", "out-and-back", artifact=artifact)
    original = route_file.read_bytes()
    paths = export_prediction(result, tmp_path / "report")
    frame = pd.read_csv(paths["predictions.csv"])
    assert frame.predicted_time_min.sum() == pytest.approx(result.forecast.moving_minutes)
    assert frame.regression_time_min.sum() == pytest.approx(result.forecast.moving_minutes)
    assert frame.primary_model.eq(MODEL_LABELS["B"]).all()
    assert frame.regression_status.eq("selected").all()
    for name in ("elevation_profile.png", "segment_times.png", "cumulative_time.png"):
        assert paths[name].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert paths[name].stat().st_size > 1000
    assert "Synthetic test route" in paths["report.md"].read_text()
    assert route_file.read_bytes() == original
    with pytest.raises(ValueError, match="historical inputs"):
        export_prediction(result, PROJECT / "data/processed")


def test_cli_predict_route_exports_without_timestamps(route_file, artifact, tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.joblib"
    joblib.dump(artifact, model)
    monkeypatch.chdir(tmp_path)
    main(["predict-route", "--gpx", str(route_file), "--type", "backpacking", "--direction", "out-and-back",
          "--model", str(model), "--output-dir", str(tmp_path / "output")])
    text = capsys.readouterr().out
    assert "outbound" in text and "return" in text
    assert "Synthetic test route" in text
    assert "Predicted moving time" in text
    assert (tmp_path / "output/predictions.csv").exists()


@pytest.mark.skipif(not REAL_GPX.exists(), reason="Personal planned GPX is not distributed with source")
def test_real_route_geometry_and_existing_inputs_unchanged(tmp_path):
    inputs = [*PROJECT.glob("models/*"), *PROJECT.glob("data/**/*")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.is_file()}
    result = predict_planned_route(REAL_GPX, "backpacking", "out-and-back", artifact=ModelArtifact())
    outbound = result.route.profile.loc[result.route.profile.leg.eq("outbound")]
    # Integration sanity checks only; implementation derives every value from geometry.
    assert len(outbound) == 628
    assert result.leg_totals.loc["outbound", "distance_miles"] == pytest.approx(5.76, abs=.01)
    assert len(result.segments) == 24
    assert result.segments.groupby("leg").distance_miles.count().eq(12).all()
    totals = result.leg_totals
    assert totals.loc["return", "elevation_gain_ft"] == pytest.approx(totals.loc["outbound", "elevation_loss_ft"])
    assert result.forecast.moving_minutes == pytest.approx(result.segments.predicted_time_min.sum())
    export_prediction(result, tmp_path / "report")
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
