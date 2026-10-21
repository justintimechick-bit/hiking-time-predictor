"""Whole-hike synthetic examples; these observations are not historical claims."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.break_model import BASELINE, FEATURES, predict_break_time, train_break_predictor
from hiking_time_predictor.breaks import BreakConfig
from hiking_time_predictor.planned import predict_planned_route
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor.web_upload import trip_statistics


def synthetic_hikes(n):
    return pd.DataFrame([dict(hike_id=f"synthetic-{i}", distance_miles=i + 1.0,
                              elevation_gain_ft=(i % 3) * 100., elevation_loss_ft=(i % 4) * 80.,
                              hike_type="day_hike" if i % 2 == 0 else "backpacking",
                              break_count=i + 1, total_break_minutes=5. * (i + 1), training_eligible=True) for i in range(n)],
                        columns=["hike_id", *FEATURES["C"], "break_count", "total_break_minutes", "training_eligible"])


@pytest.mark.parametrize("n", [0, 1, 4, 10])
def test_small_sample_baseline_and_unavailable_metrics(n):
    artifact, held_out = train_break_predictor(synthetic_hikes(n))
    assert artifact.selected_model == BASELINE
    if n:
        assert artifact.minutes_per_mile == 5
        forecast = predict_break_time(3, 100, 80, "day_hike", artifact)
        assert forecast.predicted_minutes == 15
    else:
        assert predict_break_time(3, 100, 80, "day_hike", artifact).predicted_minutes is None
    if n < 5:
        assert artifact.evaluation["metrics"] == []
        assert held_out.empty
    else:
        assert len(held_out) == n * 3
        assert artifact.evaluation["experimental"]


def test_folds_hold_out_entire_hikes_and_baseline_uses_training_only():
    data = synthetic_hikes(6)
    artifact, held_out = train_break_predictor(data)
    for fold in artifact.evaluation["folds"]:
        assert set(fold["training_hike_ids"]).isdisjoint(fold["test_hike_ids"])
        assert len(fold["test_hike_ids"]) == 1
    data.loc[0, "total_break_minutes"] = 9999
    _, changed = train_break_predictor(data)
    original = held_out.loc[held_out.hike_id.eq("synthetic-0")].set_index("model")
    withheld = changed.loc[changed.hike_id.eq("synthetic-0")].set_index("model")
    np.testing.assert_allclose(original.predicted_break_minutes, withheld.predicted_break_minutes)


def test_duplicate_observations_rejected_and_target_inputs_absent():
    with pytest.raises(ValueError, match="one observation"):
        train_break_predictor(pd.concat([synthetic_hikes(5), synthetic_hikes(1)]))
    artifact, _ = train_break_predictor(synthetic_hikes(6))
    for name, model in artifact.estimators.items():
        assert list(model.feature_names_in_) == FEATURES[name]
        assert "break_count" not in FEATURES[name]
        assert "moving_time_min" not in FEATURES[name]


def test_hike_type_requires_both_categories_in_every_training_fold():
    artifact, held_out = train_break_predictor(synthetic_hikes(12))
    assert "C" in artifact.estimators
    assert held_out.loc[held_out.model.eq("C")].hike_id.nunique() == 12
    data = synthetic_hikes(12)
    data["hike_type"] = None
    artifact, _ = train_break_predictor(data)
    assert "C" not in artifact.estimators
    assert "unavailable" in artifact.evaluation["model_c_status"]


def test_regression_can_be_selected_only_with_enough_hikes_and_lower_mae():
    data = synthetic_hikes(14)
    data["total_break_minutes"] = 15 + 2 * data.distance_miles
    artifact, _ = train_break_predictor(data)
    assert artifact.selected_model in ("A", "B", "C")
    assert not artifact.evaluation["experimental"]
    assert predict_break_time(0, 0, 0, "day_hike", artifact).predicted_minutes == 0
    small, _ = train_break_predictor(data.iloc[:10])
    assert small.selected_model == BASELINE


def test_negative_regression_output_is_clamped():
    data = synthetic_hikes(6)
    data["total_break_minutes"] = np.maximum(0, data.distance_miles * 10 - 20)
    artifact, _ = train_break_predictor(data)
    forecast = predict_break_time(0.1, 0, 0, "day_hike", artifact)
    assert forecast.predicted_minutes >= 0
    assert all(v >= 0 for v in forecast.experimental_minutes.values())


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_planned_geometry(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        predict_break_time(value, 0, 0, "day_hike")


def test_untimed_plan_breaks_manual_override_and_toggle_preserve_moving_time(tmp_path):
    path = tmp_path / "untimed.gpx"
    path.write_text('<gpx version="1.1"><rte><rtept lat="0" lon="0"><ele>100</ele></rtept>'
                    '<rtept lat="0" lon="0.01"><ele>120</ele></rtept></rte></gpx>')
    artifact, _ = train_break_predictor(synthetic_hikes(6))
    result = predict_planned_route(path, "day_hike", artifact=ModelArtifact(), break_artifact=artifact)
    moving = result.forecast.moving_minutes
    stats = trip_statistics(result, include_predicted_breaks=True)
    assert stats["break_minutes"] == pytest.approx(result.segments.distance_miles.sum() * 5)
    assert stats["total_minutes"] == pytest.approx(moving + result.break_forecast.predicted_minutes)
    overridden = trip_statistics(result, include_predicted_breaks=True, manual_break_minutes=20)
    assert overridden["total_minutes"] == moving + 20
    assert overridden["predicted_break_minutes"] == stats["predicted_break_minutes"]
    assert overridden["break_override_active"]
    assert trip_statistics(result, include_predicted_breaks=False, manual_break_minutes=20)["total_minutes"] == moving
    assert trip_statistics(result, include_predicted_breaks=True, manual_break_minutes=0)["total_minutes"] == moving
    assert result.forecast.moving_minutes == moving == result.segments.predicted_time_min.sum()
    out_back = predict_planned_route(path, "day_hike", "out-and-back", artifact=ModelArtifact(), break_artifact=artifact)
    assert out_back.break_forecast.predicted_minutes == pytest.approx(result.break_forecast.predicted_minutes * 2)
