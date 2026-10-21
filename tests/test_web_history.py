"""Dashboard metrics use measured fixtures and verified evaluation, never fits."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.modeling import (
    aggregate_hike_predictions, evaluate_models, fit_model, prepare_segments,
)
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor.web_history import evaluation_frames, load_history


@pytest.fixture
def history_root(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    summary = pd.DataFrame({
        "hike_id": ["first", "second"],
        "raw_distance_miles": [2.2, 6.0],
        "distance_miles": [2.0, 5.0],
        "moving_time_min": [40.0, 150.0],
        "elevation_gain_ft": [100.0, 400.0],
        "quality_flags": ["", "missing_hike_type"],
        "unknown_time_min": [0.0, 0.0],
    })
    segments = pd.DataFrame({
        "hike_id": ["first", "second", "second"],
        "segment_index": [0, 0, 1],
        "distance_miles": [0.5, 0.5, 0.25],
        "elevation_gain_ft": [10.0, 20.0, 5.0],
        "elevation_loss_ft": [5.0, 0.0, 10.0],
        "moving_time_min": [10.0, 15.0, 8.0],
        "quality_flags": ["", "missing_hike_type", "recording_gap"],
        "unknown_time_min": [0.0, 0.0, 1.0],
        "hike_type": ["day_hike", None, None],
    })
    summary.to_csv(processed / "hiking_summary.csv", index=False)
    segments.to_csv(processed / "hiking_segments.csv", index=False)
    return tmp_path


def write_evaluation(root):
    processed = root / "data" / "processed"
    exports = processed / "regression"
    exports.mkdir()
    segments = prepare_segments(pd.read_csv(processed / "hiking_segments.csv", dtype={"hike_id": "string"}))
    summary = pd.read_csv(processed / "hiking_summary.csv", dtype={"hike_id": "string"})
    predictions = evaluate_models(segments.loc[segments.eligible_ab]).predictions
    totals = aggregate_hike_predictions(predictions, segments, summary)
    predictions.to_csv(exports / "held_out_segment_predictions.csv", index=False)
    totals.to_csv(exports / "hike_totals_and_coverage.csv", index=False)
    manifest = {
        "input_sha256": hashlib.sha256((processed / "hiking_segments.csv").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((processed / "hiking_summary.csv").read_bytes()).hexdigest(),
    }
    (exports / "run_summary.json").write_text(json.dumps(manifest))
    return predictions, totals


def test_missing_files_leave_metrics_unavailable(tmp_path):
    history = load_history(tmp_path)
    assert history.summary.empty
    assert history.segments.empty
    assert history.held_out_predictions.empty
    assert history.hike_totals.empty
    assert all(value is None for value in history.metrics.values())
    assert any("summary unavailable" in note for note in history.notes)
    assert any("segments unavailable" in note for note in history.notes)


def test_aggregates_use_measured_weighted_pace_and_quality_rules(history_root):
    history = load_history(history_root, "B")
    assert history.metrics == {
        "logged_miles": pytest.approx(8.2),
        "hike_count": 2,
        "usable_segments": 2,
        "average_pace_min_per_mile": pytest.approx(190 / 7),
        "elevation_gain_total_ft": 500,
        "elevation_gain_mean_ft": 250,
        "elevation_gain_max_ft": 400,
    }
    assert history.segments.eligible_ab.tolist() == [True, True, False]
    assert load_history(history_root, "C").metrics["usable_segments"] == 1
    assert any("raw_distance_miles" in note for note in history.notes)
    assert any("stationary and unknown" in note for note in history.notes)


def test_zero_moving_distance_has_no_average_pace(history_root):
    path = history_root / "data" / "processed" / "hiking_summary.csv"
    summary = pd.read_csv(path)
    summary["distance_miles"] = 0
    summary.to_csv(path, index=False)
    history = load_history(history_root)
    assert history.metrics["average_pace_min_per_mile"] is None
    assert any("distance is zero" in note for note in history.notes)


@pytest.mark.parametrize("value", ["not a number", np.nan, np.inf, -1])
def test_invalid_measurements_are_not_silently_omitted(history_root, value):
    path = history_root / "data" / "processed" / "hiking_summary.csv"
    summary = pd.read_csv(path).astype(object)
    summary.loc[0, "raw_distance_miles"] = value
    summary.to_csv(path, index=False)
    history = load_history(history_root)
    assert history.metrics["logged_miles"] is None
    assert history.metrics["hike_count"] == 2
    assert any("Historical statistic unavailable" in note for note in history.notes)


def test_missing_summary_columns_are_unavailable(history_root):
    path = history_root / "data" / "processed" / "hiking_summary.csv"
    pd.read_csv(path).drop(columns="elevation_gain_ft").to_csv(path, index=False)
    history = load_history(history_root)
    assert history.metrics["elevation_gain_total_ft"] is None
    assert history.metrics["elevation_gain_mean_ft"] is None
    assert history.metrics["elevation_gain_max_ft"] is None
    assert any("Missing column elevation_gain_ft" in note for note in history.notes)


def test_summary_duplicate_hikes_are_rejected(history_root):
    path = history_root / "data" / "processed" / "hiking_summary.csv"
    summary = pd.read_csv(path)
    summary.loc[1, "hike_id"] = "first"
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonblank and unique"):
        load_history(history_root)


def test_duplicate_csv_headers_are_rejected(history_root):
    path = history_root / "data" / "processed" / "hiking_summary.csv"
    path.write_text("hike_id,hike_id\nfirst,first\n")
    with pytest.raises(ValueError, match="Duplicate column names"):
        load_history(history_root)


def test_numeric_looking_hike_ids_preserve_leading_zeroes_through_evaluation(history_root):
    processed = history_root / "data" / "processed"
    for filename in ("hiking_summary.csv", "hiking_segments.csv"):
        path = processed / filename
        frame = pd.read_csv(path)
        frame["hike_id"] = frame.hike_id.map({"first": "001", "second": "1"})
        frame.to_csv(path, index=False)
    write_evaluation(history_root)
    history = load_history(history_root)
    assert history.summary.hike_id.tolist() == ["001", "1"]
    assert history.segments.hike_id.tolist() == ["001", "1", "1"]
    assert set(history.held_out_predictions.hike_id) == {"001", "1"}
    assert set(history.hike_totals.hike_id) == {"001", "1"}


def test_invalid_segment_schema_reuses_backend_validation(history_root):
    path = history_root / "data" / "processed" / "hiking_segments.csv"
    pd.read_csv(path).drop(columns="unknown_time_min").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing segment columns"):
        load_history(history_root)


def test_valid_provenance_loads_consistent_segment_and_hike_charts(history_root, monkeypatch):
    predictions, totals = write_evaluation(history_root)

    def no_training(*args, **kwargs):
        raise AssertionError("Dashboard history must never train a model.")

    monkeypatch.setattr("hiking_time_predictor.modeling.fit_model", no_training)
    monkeypatch.setattr("hiking_time_predictor.predictor.train_predictor", no_training)
    history = load_history(history_root)
    pd.testing.assert_frame_equal(history.held_out_predictions, predictions, check_dtype=False)
    pd.testing.assert_frame_equal(history.hike_totals, totals, check_dtype=False)
    assert not history.hike_totals.whole_hike_eligible.any()


@pytest.mark.parametrize("filename", ["hiking_segments.csv", "hiking_summary.csv"])
def test_stale_source_hash_disables_saved_historical_charts(history_root, filename):
    write_evaluation(history_root)
    source = history_root / "data" / "processed" / filename
    with source.open("a") as output:
        output.write("\n")
    history = load_history(history_root)
    assert history.held_out_predictions.empty
    assert history.hike_totals.empty
    assert any(f"does not match current {filename}" in note for note in history.notes)


@pytest.mark.parametrize("manifest", [None, {}, [], "invalid JSON"])
def test_absent_or_invalid_provenance_disables_saved_historical_charts(history_root, manifest):
    write_evaluation(history_root)
    path = history_root / "data" / "processed" / "regression" / "run_summary.json"
    if manifest is None:
        path.unlink()
    else:
        path.write_text(manifest if isinstance(manifest, str) else json.dumps(manifest))
    history = load_history(history_root)
    assert history.held_out_predictions.empty
    assert history.hike_totals.empty
    assert any("Historical held-out charts unavailable" in note for note in history.notes)


def test_saved_prediction_measurements_must_match_current_eligible_rows(history_root):
    write_evaluation(history_root)
    path = history_root / "data" / "processed" / "regression" / "held_out_segment_predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, "moving_time_min"] += 1
    predictions.to_csv(path, index=False)
    history = load_history(history_root)
    assert history.held_out_predictions.empty
    assert history.hike_totals.empty
    assert any("measurements do not match" in note for note in history.notes)


def test_saved_totals_must_match_segment_export_and_coverage(history_root):
    write_evaluation(history_root)
    path = history_root / "data" / "processed" / "regression" / "hike_totals_and_coverage.csv"
    totals = pd.read_csv(path)
    totals.loc[0, "whole_hike_eligible"] = True
    totals.to_csv(path, index=False)
    history = load_history(history_root)
    assert not history.held_out_predictions.empty
    assert history.hike_totals.empty
    assert any("coverage checks" in note for note in history.notes)


def test_absent_artifact_metrics_are_empty_without_synthetic_scores():
    frames = evaluation_frames(ModelArtifact())
    assert set(frames) == {"segment_metrics", "whole_hike_metrics", "folds", "coefficients"}
    assert all(frame.empty for frame in frames.values())


def test_evaluation_frames_use_saved_metrics_and_fitted_coefficients(history_root):
    segments = load_history(history_root).segments
    estimator = fit_model(segments.loc[segments.eligible_ab], "A")
    metrics = [{"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": 2.5, "RMSE_min": 3.0}]
    artifact = ModelArtifact(estimator=estimator, model_name="A", evaluation={"segment_metrics": metrics})
    frames = evaluation_frames(artifact)
    assert frames["segment_metrics"].to_dict("records") == metrics
    assert frames["whole_hike_metrics"].empty
    coefficients = frames["coefficients"].set_index("term")
    assert coefficients.loc["intercept", "value"] == estimator.intercept_
    assert coefficients.loc["distance_miles", "value"] == estimator.coef_[0]


def test_c_coefficients_come_from_actual_fitted_pipeline(history_root):
    segments = load_history(history_root).segments.loc[lambda data: data.eligible_ab].copy()
    segments["hike_type"] = ["day_hike", "backpacking"]
    estimator = fit_model(segments, "C")
    frames = evaluation_frames(ModelArtifact(estimator=estimator, model_name="C"))
    assert len(frames["coefficients"]) == 5
    assert any("type__hike_type_" in term for term in frames["coefficients"].term)


def test_invalid_saved_evaluation_is_rejected():
    with pytest.raises(ValueError, match="list of evaluation records"):
        evaluation_frames(ModelArtifact(evaluation={"segment_metrics": "not records"}))


@pytest.mark.parametrize("missing", ["model", "observations", "hikes", "MAE_min", "RMSE_min"])
@pytest.mark.parametrize("metric_name", ["segment_metrics", "whole_hike_metrics"])
def test_partial_saved_scores_are_unavailable_without_inventing_metrics(missing, metric_name):
    record = {"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": 2.5, "RMSE_min": 3.0}
    del record[missing]
    frames = evaluation_frames(ModelArtifact(evaluation={metric_name: [record]}))
    assert frames[metric_name].empty


def test_incomplete_record_does_not_silently_drop_unevaluated_models():
    complete = {"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": 2.5, "RMSE_min": 3.0}
    incomplete = {"model": "B: distance + gain + loss", "observations": 10, "hikes": 3}
    frames = evaluation_frames(ModelArtifact(evaluation={"segment_metrics": [complete, incomplete]}))
    assert frames["segment_metrics"].empty


@pytest.mark.parametrize(("column", "value"), [
    ("MAE_min", -1), ("RMSE_min", -1), ("MAE_min", np.inf),
    ("RMSE_min", "invalid"), ("MAE_min", True),
    ("observations", -1), ("hikes", -1), ("observations", 0),
    ("hikes", 1.5), ("hikes", 11), ("observations", np.inf),
    ("negative_predictions", 1.5), ("negative_predictions", -1),
    ("negative_predictions", 11), ("bias_pred_minus_actual_min", np.inf),
])
def test_invalid_present_evaluation_numbers_raise_clear_error(column, value):
    record = {"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": 2.5, "RMSE_min": 3.0}
    record[column] = value
    with pytest.raises(ValueError):
        evaluation_frames(ModelArtifact(evaluation={"segment_metrics": [record]}))


def test_missing_required_metric_value_is_unavailable_and_folds_remain_saved():
    record = {"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": np.nan, "RMSE_min": 3.0}
    folds = [{"fold": 1, "train_hikes": "001;002", "test_hikes": "003"}]
    frames = evaluation_frames(ModelArtifact(evaluation={"segment_metrics": [record], "folds": folds}))
    assert frames["segment_metrics"].empty
    assert frames["folds"].to_dict("records") == folds


def test_optional_evaluation_diagnostics_preserve_signed_bias_and_actual_counts():
    record = {"model": "A: distance", "observations": 10, "hikes": 3, "MAE_min": 2.5,
              "RMSE_min": 3.0, "bias_pred_minus_actual_min": -1.1, "negative_predictions": 2}
    frames = evaluation_frames(ModelArtifact(evaluation={"segment_metrics": [record]}))
    assert frames["segment_metrics"].to_dict("records") == [record]


def test_unsupported_model_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model name"):
        load_history(tmp_path, "pack_weight_model")
