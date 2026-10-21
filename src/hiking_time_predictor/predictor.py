"""Evaluate, persist, and apply a small segment-regression predictor."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from .gpx import ImportConfig
from .modeling import (
    GEOMETRY, MODEL_LABELS, NAISMITH_LABEL, TARGET, aggregate_hike_predictions,
    error_metrics, evaluate_models, fit_model, model_c_status, naismith_minutes,
    FeatureSchemaError, predict_route, prepare_segments, validate_estimator_schema,
)
from .routes import validate_route_features


# A conservative project policy, not a statistical guarantee of generalization.
MIN_VALIDATION_HIKES = 10
DEFAULT_MODEL = Path("models/hiking_predictor.joblib")


@dataclass
class ModelArtifact:
    estimator: object = None
    model_name: str | None = None
    selected: bool = False
    reason: str = "No trained model loaded; using the Naismith baseline."
    config: ImportConfig = field(default_factory=ImportConfig)
    segment_ranges: dict = field(default_factory=dict)
    route_ranges: dict = field(default_factory=dict)
    hike_types: list = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    format_version: int = 1


def _ranges(frame):
    return {name: (float(frame[name].min()), float(frame[name].max()))
            for name in GEOMETRY} if not frame.empty else {}


def train_predictor(segments, summary=None, config=None):
    """Use identical hike-grouped folds/cohorts for every candidate and baseline.

    Promote the best whole-hike MAE only with ten verified complete held-out
    hikes, a lower MAE than Naismith, and valid held-out segment predictions.
    Otherwise retain the best evaluated segment model as experimental only.
    """
    config = config or ImportConfig()
    if summary is not None:
        required = {"hike_id", "quality_flags", "unknown_time_min", "distance_miles", TARGET}
        if summary.columns.duplicated().any() or not required.issubset(summary.columns):
            raise ValueError(f"Recording summary needs unique columns including {sorted(required)}.")
        summary = summary.copy()
        summary["hike_id"] = summary.hike_id.astype("string").str.strip()
        if summary.hike_id.isna().any() or summary.hike_id.eq("").any() or summary.hike_id.duplicated().any():
            raise ValueError("Recording summary hike_id must be nonblank and unique.")
        for name in ("distance_miles", TARGET, "unknown_time_min"):
            booleans = summary[name].map(lambda value: isinstance(value, (bool, np.bool_)))
            summary[name] = pd.to_numeric(summary[name], errors="coerce")
            values = summary[name].to_numpy(dtype=float, na_value=np.nan)
            if booleans.any() or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Recording summary {name} must be finite and nonnegative.")
    checked = prepare_segments(segments)
    eligible = checked.loc[checked.eligible_ab].copy()
    known = checked.loc[checked.eligible_c].copy()
    c_ready, c_reason = model_c_status(known)
    names = ("A", "B", "C") if c_ready else ("A", "B")
    cohort = known if c_ready else eligible
    if cohort.distance_miles.gt(config.segment_miles + 1e-8).any():
        raise ValueError("Training segments exceed the configured segment size; use matching import settings.")
    artifact = ModelArtifact(config=config)
    artifact.segment_ranges = _ranges(cohort)
    artifact.hike_types = sorted(cohort.hike_type.dropna().unique().tolist())
    # Only verified complete hikes establish a route-total training range.
    artifact.evaluation = {"eligible_hikes": int(cohort.hike_id.nunique()),
                           "eligible_segments": len(cohort), "model_c_status": c_reason,
                           "minimum_validation_hikes": MIN_VALIDATION_HIKES}
    if cohort.hike_id.nunique() < 2:
        artifact.reason = "Fewer than two eligible independent hikes; no evaluated regression is available."
        return artifact

    evaluation = evaluate_models(cohort, names)
    segment_scores = error_metrics(evaluation.predictions)
    totals = aggregate_hike_predictions(evaluation.predictions, checked, summary)
    whole = totals.loc[totals.whole_hike_eligible]
    whole_scores = error_metrics(whole)
    whole_hikes = int(whole.hike_id.nunique())
    complete_ids = whole.hike_id.unique()
    artifact.route_ranges = _ranges(cohort.loc[cohort.hike_id.isin(complete_ids)]
                                    .groupby("hike_id")[GEOMETRY].sum())
    artifact.evaluation.update({"method": evaluation.method, "verified_held_out_hikes": whole_hikes,
                                "segment_metrics": segment_scores.to_dict("records"),
                                "whole_hike_metrics": whole_scores.to_dict("records"),
                                "folds": evaluation.folds.to_dict("records")})
    # Disqualify negative segment predictions even if their summed hike time is positive.
    valid_labels = set(segment_scores.loc[segment_scores.negative_predictions.eq(0), "model"])
    candidates = segment_scores.loc[segment_scores.model.isin(MODEL_LABELS.values())
                                    & segment_scores.model.isin(valid_labels)]
    if candidates.empty:
        artifact.reason = "All evaluated regressions produced invalid held-out times; using Naismith."
        return artifact
    ranking = candidates
    if whole_hikes >= MIN_VALIDATION_HIKES:
        ranking = whole_scores.loc[whole_scores.model.isin(candidates.model)]
    best = ranking.sort_values(["MAE_min", "RMSE_min", "model"]).iloc[0]
    name = next(name for name, label in MODEL_LABELS.items() if label == best.model)
    artifact.model_name = name
    artifact.estimator = fit_model(cohort, name)
    if whole_hikes < MIN_VALIDATION_HIKES:
        noun = "hike" if whole_hikes == 1 else "hikes"
        artifact.reason = (f"Only {whole_hikes} verified complete held-out {noun}; at least "
                           f"{MIN_VALIDATION_HIKES} are required to select a route predictor.")
    else:
        baseline = whole_scores.loc[whole_scores.model.eq(NAISMITH_LABEL)].iloc[0]
        artifact.selected = bool(best.MAE_min < baseline.MAE_min - 1e-9)
        artifact.reason = (f"{best.model} has the lowest held-out whole-hike MAE and beats Naismith "
                           f"across {whole_hikes} complete hikes." if artifact.selected else
                           "The best evaluated regression does not beat Naismith's whole-hike MAE.")
    return artifact


def save_model(artifact, path):
    """Save estimator, status, feature ranges, settings, and evaluation together."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact.provenance.update({"trained_at_utc": datetime.now(timezone.utc).isoformat(),
                                "scikit_learn_version": sklearn.__version__,
                                "joblib_version": joblib.__version__})
    joblib.dump(artifact, path)


def load_model(path):
    """Load only trusted local joblib artifacts; pickle can execute Python code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artifact = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Could not load the model artifact {path}: {error}. "
                         "Use a valid trusted artifact or retrain with hiking_predictor train.") from error
    if (not isinstance(artifact, ModelArtifact) or artifact.format_version != 1
            or not isinstance(artifact.config, ImportConfig)
            or artifact.model_name not in (None, *MODEL_LABELS)
            or (artifact.estimator is None) != (artifact.model_name is None)
            or (artifact.selected and artifact.estimator is None)):
        raise ValueError("Unsupported model artifact; retrain with hiking_predictor train.")
    if artifact.estimator is not None:
        validate_estimator_schema(artifact.estimator, artifact.model_name)
    return artifact


def range_warnings(segments, artifact):
    """Check segment geometry as well as totals; route totals alone hide steep sections."""
    messages = []
    if not artifact.segment_ranges:
        messages.append("Training range unavailable; train a model to check input limits.")
    for scope, frame, ranges in (
        ("Segment", segments, artifact.segment_ranges),
        ("Route total", segments[GEOMETRY].sum().to_frame().T, artifact.route_ranges),
    ):
        for name, (low, high) in ranges.items():
            values = frame[name].astype(float)
            outside = values.lt(low - 1e-8) | values.gt(high + 1e-8)
            if outside.any():
                units = "miles" if name == "distance_miles" else "ft"
                messages.append(f"{scope} {name} outside training range [{low:.3f}, {high:.3f}] {units} "
                                f"({int(outside.sum())} input(s)); extrapolation may be unreliable.")
    if artifact.segment_ranges and not artifact.route_ranges:
        messages.append("Whole-route training range unavailable: no verified complete held-out hikes.")
    hike_type = segments.hike_type.iloc[0]
    if artifact.segment_ranges and hike_type not in artifact.hike_types:
        messages.append(f"Hike type '{hike_type}' was not observed in training metadata.")
    return messages


@dataclass
class Forecast:
    moving_minutes: float
    naismith_minutes: float
    label: str
    experimental_minutes: float | None
    warnings: list[str]
    segment_predictions: pd.DataFrame = field(default_factory=pd.DataFrame, compare=False, repr=False)


def forecast_route(segments, artifact=None):
    """Run saved inference once per segment and sum; never fit during prediction."""
    artifact = artifact or ModelArtifact()
    validate_route_features(segments)
    baseline_segments = naismith_minutes(segments)
    baseline = float(baseline_segments.sum())
    if not np.isfinite(baseline):
        raise ValueError("Route totals are too large to produce a finite estimate.")
    messages = range_warnings(segments, artifact)
    learned = None
    learned_segments = None
    if artifact.estimator is not None:
        try:
            learned_segments = predict_route(artifact.estimator, artifact.model_name, segments).predicted_time_min.to_numpy()
            learned = float(learned_segments.sum())
            if not np.isfinite(learned):
                raise ValueError("The summed regression prediction is not finite.")
        except FeatureSchemaError:
            raise
        except ValueError as error:
            learned = None
            learned_segments = None
            messages.append(f"Regression unavailable for these inputs: {error} Using the Naismith baseline.")
    selected = artifact.selected and learned is not None
    table = segments.copy().reset_index(drop=True)
    table.insert(0, "segment_number", np.arange(1, len(table) + 1))
    table["naismith_time_min"] = baseline_segments
    table["regression_time_min"] = learned_segments if learned_segments is not None else np.nan
    table["predicted_time_min"] = learned_segments if selected else baseline_segments
    table["cumulative_distance_miles"] = table.distance_miles.astype(float).cumsum()
    table["cumulative_predicted_time_min"] = table.predicted_time_min.cumsum()
    table["cumulative_regression_time_min"] = table.regression_time_min.cumsum()
    table["predicted_pace_min_per_mile"] = table.predicted_time_min / table.distance_miles.astype(float)
    table["regression_pace_min_per_mile"] = table.regression_time_min / table.distance_miles.astype(float)
    table["outside_training_range"] = False
    for name, (low, high) in artifact.segment_ranges.items():
        values = table[name].astype(float)
        table["outside_training_range"] |= values.lt(low - 1e-8) | values.gt(high + 1e-8)
    if learned_segments is not None:
        speeds = table.distance_miles.astype(float).to_numpy() / learned_segments * 60
        if (speeds > artifact.config.max_speed_mph).any():
            messages.append(f"Regression implies speed above {artifact.config.max_speed_mph:g} mph in "
                            f"{int((speeds > artifact.config.max_speed_mph).sum())} segment(s); "
                            "positive predictions can still be implausible, especially for short final segments.")
    return Forecast(learned if selected else baseline, baseline,
                    MODEL_LABELS[artifact.model_name] if selected else "baseline estimate: Naismith",
                    learned if not selected else None, messages, table)
