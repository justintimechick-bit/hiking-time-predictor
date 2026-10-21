"""Independent synthetic hikes test selection, persistence, and saved inference."""

import joblib
import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.gpx import ImportConfig
from hiking_time_predictor.modeling import GEOMETRY, MODEL_LABELS, TARGET, naismith_minutes
from hiking_time_predictor.predictor import (
    MIN_VALIDATION_HIKES, ModelArtifact, forecast_route, load_model, range_warnings,
    save_model, train_predictor,
)
from hiking_time_predictor.routes import manual_route


@pytest.fixture
def training_data():
    rows = []
    for hike in range(12):
        for segment in range(3):
            distance = 0.5 if segment < 2 else 0.1 + 0.03 * (hike % 5)
            gain, loss = (hike * 31 + segment * 41) % 190, (hike * 47 + segment * 19) % 150
            rows.append({"hike_id": f"synthetic-{hike}", "segment_index": segment,
                         "distance_miles": distance, "elevation_gain_ft": gain, "elevation_loss_ft": loss,
                         TARGET: 2 + 12 * distance + 0.02 * gain + 0.01 * loss,
                         "quality_flags": "missing_hike_type", "unknown_time_min": 0, "hike_type": None})
    segments = pd.DataFrame(rows)
    summary = segments.groupby("hike_id")[[*GEOMETRY, TARGET]].sum().reset_index()
    summary["quality_flags"] = "missing_hike_type"
    summary["unknown_time_min"] = 0
    return segments, summary


def test_selects_best_only_after_independent_whole_hike_evaluation(training_data):
    segments, summary = training_data
    original = segments.copy(deep=True)
    artifact = train_predictor(segments, summary)
    assert artifact.selected
    assert artifact.model_name == "B"
    assert artifact.evaluation["verified_held_out_hikes"] >= MIN_VALIDATION_HIKES
    for fold in artifact.evaluation["folds"]:
        assert not set(fold["train_hikes"].split(";")) & set(fold["test_hikes"].split(";"))
    result = forecast_route(manual_route(1.2, 100, 50, "backpacking").segments, artifact)
    assert result.label == MODEL_LABELS["B"]
    assert result.moving_minutes == pytest.approx(3 * 2 + 12 * 1.2 + 0.02 * 100 + 0.01 * 50)
    assert result.experimental_minutes is None
    pd.testing.assert_frame_equal(segments, original)


def test_many_segments_from_one_hike_cannot_establish_generalization(training_data):
    segments, _ = training_data
    segments["hike_id"] = "just-one-hike"
    segments["segment_index"] = np.arange(len(segments))
    artifact = train_predictor(segments)
    assert not artifact.selected
    assert artifact.estimator is None
    assert "Fewer than two" in artifact.reason


@pytest.mark.parametrize("count,selected", [(9, False), (10, True)])
def test_minimum_whole_hike_selection_boundary(training_data, count, selected):
    segments, summary = training_data
    artifact = train_predictor(segments.iloc[:3 * count], summary)
    assert artifact.selected is selected


def test_insufficient_hikes_keep_regression_experimental(training_data):
    segments, summary = training_data
    segments = segments.iloc[:9]
    artifact = train_predictor(segments, summary)
    assert not artifact.selected
    assert artifact.estimator is not None
    result = forecast_route(manual_route(1.2, 100, 50, "day_hike").segments, artifact)
    assert result.moving_minutes == result.naismith_minutes
    assert "baseline" in result.label
    assert result.experimental_minutes is not None


def test_missing_summary_or_partial_hikes_cannot_promote_segment_winner(training_data):
    segments, summary = training_data
    assert not train_predictor(segments).selected
    segments.loc[segments.hike_id.isin([f"synthetic-{i}" for i in range(4)])
                 & segments.segment_index.eq(0), "quality_flags"] = "recording_gap"
    artifact = train_predictor(segments, summary)
    assert artifact.evaluation["eligible_hikes"] == 12
    assert artifact.evaluation["verified_held_out_hikes"] == 8
    assert not artifact.selected


def test_baseline_remains_primary_when_regression_does_not_improve_it(training_data):
    segments, _ = training_data
    segments[TARGET] = naismith_minutes(segments)
    summary = segments.groupby("hike_id")[["distance_miles", TARGET]].sum().reset_index()
    summary["quality_flags"], summary["unknown_time_min"] = "", 0
    artifact = train_predictor(segments, summary)
    assert not artifact.selected
    assert "does not beat" in artifact.reason


def test_type_model_is_compared_on_identical_known_type_rows(training_data):
    segments, _ = training_data
    segments["hike_type"] = np.where(segments.hike_id.isin([f"synthetic-{i}" for i in range(6)]),
                                      "day_hike", "backpacking")
    segments[TARGET] += np.where(segments.hike_type.eq("backpacking"), 7, 0)
    unknown = segments.iloc[:3].assign(hike_id="synthetic-unknown", hike_type=None)
    segments = pd.concat([segments, unknown], ignore_index=True)
    summary = segments.groupby("hike_id")[["distance_miles", TARGET]].sum().reset_index()
    summary["quality_flags"], summary["unknown_time_min"] = "", 0
    artifact = train_predictor(segments, summary)
    assert artifact.selected and artifact.model_name == "C"
    assert artifact.evaluation["eligible_hikes"] == 12
    metrics = artifact.evaluation["segment_metrics"]
    assert len(metrics) == 4
    assert {item["observations"] for item in metrics} == {36}
    route = manual_route(1.2, 100, 50, "backpacking").segments
    backpacking = forecast_route(route, artifact).moving_minutes
    day = forecast_route(route.assign(hike_type="day_hike"), artifact).moving_minutes
    assert backpacking - day == pytest.approx(7 * len(route))


@pytest.mark.parametrize("with_type", [False, True])
def test_joblib_round_trip_preserves_predictions_status_and_settings(tmp_path, training_data, with_type):
    segments, summary = training_data
    if with_type:
        segments["hike_type"] = np.where(segments.index // 3 < 6, "day_hike", "backpacking")
        segments[TARGET] += np.where(segments.hike_type.eq("backpacking"), 7, 0)
        summary = segments.groupby("hike_id")[["distance_miles", TARGET]].sum().reset_index()
        summary["quality_flags"], summary["unknown_time_min"] = "", 0
    config = ImportConfig(elevation_window_m=40)
    artifact = train_predictor(segments, summary, config)
    route = manual_route(1.2, 100, 50, "backpacking", config).segments
    expected = forecast_route(route, artifact)
    path = tmp_path / "nested" / "model.joblib"
    save_model(artifact, path)
    loaded = load_model(path)
    assert forecast_route(route, loaded) == expected
    assert loaded.config == config
    assert loaded.segment_ranges == artifact.segment_ranges
    assert loaded.evaluation == artifact.evaluation
    assert loaded.provenance["scikit_learn_version"]


def test_baseline_artifact_round_trip_and_no_data(tmp_path, training_data):
    segments, _ = training_data
    artifact = train_predictor(segments.iloc[:0])
    path = tmp_path / "baseline.joblib"
    save_model(artifact, path)
    loaded = load_model(path)
    result = forecast_route(manual_route(3, 2000, 1000, "day_hike").segments, loaded)
    assert result.moving_minutes == 120
    assert result.experimental_minutes is None
    assert any("Training range unavailable" in warning for warning in result.warnings)


@pytest.mark.parametrize("content", [b"", b"not a joblib file"])
def test_corrupt_artifact_has_clear_error(tmp_path, content):
    path = tmp_path / "bad.joblib"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Could not load"):
        load_model(path)


def test_unrelated_or_unknown_version_artifacts_rejected(tmp_path):
    path = tmp_path / "bad.joblib"
    for artifact in ({"model": "unrelated"}, ModelArtifact(format_version=200)):
        joblib.dump(artifact, path)
        with pytest.raises(ValueError, match="Unsupported"):
            load_model(path)


def test_ranges_check_local_terrain_total_and_unknown_type(training_data):
    artifact = train_predictor(*training_data)
    route = manual_route(10, 10000, 5000, "backpacking").segments
    messages = range_warnings(route, artifact)
    assert any("Segment elevation_gain_ft outside training range" in message for message in messages)
    assert any("Route total distance_miles outside training range" in message for message in messages)
    assert any("not observed" in message for message in messages)


def test_invalid_regression_falls_back_instead_of_clipping(training_data):
    artifact = train_predictor(*training_data)
    artifact.estimator.intercept_ = -10000
    route = manual_route(1, 0, 0, "day_hike").segments
    result = forecast_route(route, artifact)
    assert result.moving_minutes == result.naismith_minutes
    assert result.experimental_minutes is None
    assert any("Regression unavailable" in warning for warning in result.warnings)


def test_inference_uses_saved_fit_without_target_or_retraining(training_data, monkeypatch):
    import hiking_time_predictor.modeling as modeling
    import hiking_time_predictor.predictor as predictor

    artifact = train_predictor(*training_data)
    route = manual_route(1.2, 100, 50, "day_hike").segments
    expected = forecast_route(route, artifact)

    def fail_if_training(*args, **kwargs):
        raise AssertionError("Inference must not train")

    monkeypatch.setattr(modeling, "fit_model", fail_if_training)
    monkeypatch.setattr(predictor, "fit_model", fail_if_training)
    assert forecast_route(route, artifact) == expected
    assert forecast_route(route.assign(moving_time_min=99999, observed_speed=99999), artifact) == expected


def test_training_rejects_segment_size_mismatch(training_data):
    with pytest.raises(ValueError, match="segment size"):
        train_predictor(*training_data, config=ImportConfig(segment_miles=0.25))


def test_summary_requires_quality_evidence_and_unique_hike_ids(training_data):
    segments, summary = training_data
    with pytest.raises(ValueError, match="summary needs unique columns"):
        train_predictor(segments, summary.drop(columns="quality_flags"))
    with pytest.raises(ValueError, match="nonblank and unique"):
        train_predictor(segments, pd.concat([summary, summary.iloc[:1]]))
    summary.loc[0, "unknown_time_min"] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        train_predictor(segments, summary)


def test_negative_held_out_segment_disqualifies_candidate(training_data, monkeypatch):
    import hiking_time_predictor.predictor as predictor

    evaluate = predictor.evaluate_models

    def with_invalid_prediction(*args, **kwargs):
        evaluation = evaluate(*args, **kwargs)
        index = evaluation.predictions.index[evaluation.predictions.model.eq(MODEL_LABELS["B"])][0]
        evaluation.predictions.loc[index, "predicted_time_min"] = -1
        return evaluation

    monkeypatch.setattr(predictor, "evaluate_models", with_invalid_prediction)
    artifact = train_predictor(*training_data)
    assert artifact.model_name != "B"
