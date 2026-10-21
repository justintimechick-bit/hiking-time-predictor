"""Synthetic measurements check leakage, quality rules, and route arithmetic.

These fixtures never enter the user's real CSVs or reported performance.
"""

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.modeling import (
    GEOMETRY, MODEL_FEATURES, TARGET, aggregate_hike_predictions,
    error_metrics, evaluate_models, fit_model, group_splits, model_c_status,
    naismith_minutes, predict_route, prepare_segments,
)


@pytest.fixture
def segments():
    rows = []
    for hike in range(4):
        for segment in range(3):
            distance = 0.25 + segment * 0.1
            gain = hike * 20 + segment * 10
            loss = (hike + segment) * 5
            rows.append({"hike_id": f"synthetic-{hike}", "segment_index": segment,
                         "distance_miles": distance, "elevation_gain_ft": gain,
                         "elevation_loss_ft": loss, TARGET: 2 + 20 * distance + 0.02 * gain,
                         "quality_flags": "missing_hike_type", "unknown_time_min": 0,
                         "hike_type": None})
    return pd.DataFrame(rows)


def test_quality_filter_does_not_require_optional_type_or_modify_source(segments):
    original = segments.copy(deep=True)
    selected = prepare_segments(segments)
    assert selected.eligible_ab.all()
    assert not selected.eligible_c.any()
    pd.testing.assert_frame_equal(segments, original)


@pytest.mark.parametrize("flag", ["recording_gap", "unknown_interval", "implausible_speed",
                                    "missing_elevation", "user_confirmed_rest", "future_new_flag"])
def test_measurement_flags_are_excluded(segments, flag):
    segments.loc[0, "quality_flags"] = "missing_hike_type;" + flag
    assert not prepare_segments(segments).loc[0, "eligible_ab"]


@pytest.mark.parametrize(("column", "value"), [
    ("distance_miles", 0), ("distance_miles", True), (TARGET, -1),
    ("elevation_gain_ft", np.inf), ("elevation_loss_ft", -1),
    ("unknown_time_min", 0.1), ("unknown_time_min", np.nan),
    ("segment_index", 0.5), ("hike_id", " "),
])
def test_invalid_or_unknown_measurements_are_ineligible(segments, column, value):
    segments = segments.astype(object)
    segments.loc[0, column] = value
    assert not prepare_segments(segments).loc[0, "eligible_ab"]


def test_missing_quality_schema_and_duplicate_segments_are_rejected(segments):
    with pytest.raises(ValueError, match="Missing segment columns"):
        prepare_segments(segments.drop(columns="quality_flags"))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_segments(pd.concat([segments, segments.iloc[:1]]))


def test_numeric_text_cannot_disguise_duplicate_segment_keys(segments):
    extra = segments.iloc[:1].copy().astype(object)
    extra.loc[0, "segment_index"] = "0"
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_segments(pd.concat([segments, extra], ignore_index=True))


def test_hike_splits_are_disjoint_and_each_segment_tested_once(segments):
    result = evaluate_models(segments)
    assert result.method == "LeaveOneGroupOut"
    for fold in result.folds.itertuples():
        assert not set(fold.train_hikes.split(";")) & set(fold.test_hikes.split(";"))
    for _, group in result.predictions.groupby("model"):
        assert len(group) == len(segments)
        assert not group.duplicated(["hike_id", "segment_index"]).any()


def test_single_hike_produces_no_test_results(segments):
    result = evaluate_models(segments.iloc[:3])
    assert result.predictions.empty
    assert "fewer than two" in result.method
    assert error_metrics(result.predictions).empty


def test_many_hikes_use_reproducible_group_shuffle(segments):
    many = pd.concat([segments.iloc[:1].assign(hike_id=f"test-{n}") for n in range(25)], ignore_index=True)
    first, method = group_splits(many)
    second, _ = group_splits(many)
    assert method.startswith("GroupShuffleSplit")
    np.testing.assert_array_equal(first[0][1], second[0][1])
    train, test = first[0]
    assert not set(many.iloc[train].hike_id) & set(many.iloc[test].hike_id)


def test_target_derived_columns_never_enter_predictors(segments):
    contaminated = segments.assign(actual_speed_mph=9999, elapsed_time_min=9999, observed_time=9999)
    model = fit_model(contaminated, "B")
    assert list(model.feature_names_in_) == GEOMETRY
    assert all(set(features) <= {*GEOMETRY, "hike_type"} for features in MODEL_FEATURES.values())
    clean_result = evaluate_models(segments).predictions
    dirty_result = evaluate_models(contaminated).predictions
    np.testing.assert_allclose(clean_result.predicted_time_min, dirty_result.predicted_time_min)


def test_evaluation_refuses_flagged_rows(segments):
    segments.loc[0, "quality_flags"] = "recording_gap"
    with pytest.raises(ValueError, match="ineligible"):
        evaluate_models(segments)


def test_c_requires_metadata_and_training_category_contrast(segments):
    assert not model_c_status(prepare_segments(segments))[0]
    types = {f"synthetic-{n}": HIKE_TYPE for n, HIKE_TYPE in enumerate(
        ["day_hike", "day_hike", "backpacking", "backpacking"])}
    segments["hike_type"] = segments.hike_id.map(types)
    segments[TARGET] += np.where(segments.hike_type.eq("backpacking"), 7.0, 0.0)
    checked = prepare_segments(segments)
    assert model_c_status(checked)[0]
    result = evaluate_models(checked, ("A", "B", "C"))
    assert result.predictions.model.nunique() == 4
    model = fit_model(checked, "C")
    assert len(model.named_steps["features"].named_transformers_["type"].categories_[0]) == 2
    same_geometry = segments.iloc[:1][GEOMETRY]
    backpacking = predict_route(model, "C", same_geometry.assign(hike_type="backpacking"))
    day_hike = predict_route(model, "C", same_geometry.assign(hike_type="day_hike"))
    assert (backpacking.predicted_time_min.iloc[0] - day_hike.predicted_time_min.iloc[0]) == pytest.approx(7.0)
    with pytest.raises(ValueError, match="known hike_type"):
        predict_route(model, "C", segments.iloc[:1].assign(hike_type=None))


def test_c_with_one_hike_of_a_category_is_skipped(segments):
    segments["hike_type"] = np.where(segments.hike_id.eq("synthetic-0"), "backpacking", "day_hike")
    assert not model_c_status(prepare_segments(segments))[0]


def test_conflicting_type_in_one_hike_is_rejected(segments):
    segments.loc[0, "hike_type"] = "day_hike"
    segments.loc[1, "hike_type"] = "backpacking"
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_segments(segments)


def test_naismith_units_and_additivity(segments):
    route = pd.DataFrame({"distance_miles": [3, 1.5], "elevation_gain_ft": [2000, 0]})
    np.testing.assert_allclose(naismith_minutes(route), [120, 30])
    assert naismith_minutes(route).sum() == naismith_minutes(route.sum().to_frame().T)[0]


def test_metrics_use_real_errors_and_hike_totals(segments):
    predicted = pd.DataFrame({"model": ["test", "test"], "hike_id": ["a", "b"],
                              TARGET: [10, 20], "predicted_time_min": [12, 16]})
    metrics = error_metrics(predicted).iloc[0]
    assert metrics.MAE_min == 3
    assert metrics.RMSE_min == pytest.approx(np.sqrt(10))
    assert metrics.bias_pred_minus_actual_min == -1


def test_partial_coverage_never_becomes_whole_hike(segments):
    summary = segments.groupby("hike_id").agg({TARGET: "sum", "distance_miles": "sum"}).reset_index()
    summary["quality_flags"] = "missing_hike_type"
    summary["unknown_time_min"] = 0
    selected = segments.drop(index=0)
    predictions = evaluate_models(selected).predictions
    totals = aggregate_hike_predictions(predictions, segments, summary)
    partial = totals.hike_id.eq("synthetic-0")
    assert not totals.loc[partial, "whole_hike_eligible"].any()
    assert totals.loc[~partial, "whole_hike_eligible"].all()
    assert (totals.loc[partial, "distance_coverage_fraction"] < 1).all()
    expected = selected.loc[selected.hike_id.eq("synthetic-0"), TARGET].sum()
    np.testing.assert_allclose(totals.loc[partial, TARGET], expected)
    # Even complete exported rows need clean recording-level evidence.
    summary["quality_flags"] = "incomplete_elapsed_time"
    assert not aggregate_hike_predictions(predictions, segments, summary).whole_hike_eligible.any()


def test_future_route_needs_no_target_and_sums_each_intercept(segments):
    model = fit_model(segments, "A")
    route = segments.iloc[:3][["distance_miles"]]
    prediction = predict_route(model, "A", route)
    expected = len(route) * model.intercept_ + model.coef_[0] * route.distance_miles.sum()
    assert prediction.predicted_time_min.sum() == pytest.approx(expected)
    assert TARGET not in prediction
    with pytest.raises(ValueError, match="Invalid route"):
        predict_route(model, "A", route.assign(distance_miles=-1))
    with pytest.raises(ValueError, match="Missing route features"):
        predict_route(model, "B", route)


def test_negative_future_predictions_are_rejected(segments):
    model = fit_model(segments, "A")
    model.intercept_ = -100
    with pytest.raises(ValueError, match="invalid moving time"):
        predict_route(model, "A", segments.iloc[:1][["distance_miles"]])
