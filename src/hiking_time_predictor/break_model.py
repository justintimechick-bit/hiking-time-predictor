"""Whole-hike break baseline and honest, grouped regression comparisons."""

from dataclasses import dataclass, field
import hashlib
from math import isfinite
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .breaks import BreakConfig, NON_MOVING_TARGET, ORDINARY_BREAK_TARGET
from .data import HIKE_TYPES


DEFAULT_BREAK_MODEL = Path(__file__).resolve().parents[2] / "models" / "break_predictor.joblib"
PREDICTION_COLUMNS = ["fold", "hike_id", "model", "actual_break_minutes", "predicted_break_minutes"]
FEATURES = {
    "A": ["distance_miles"],
    "B": ["distance_miles", "elevation_gain_ft", "elevation_loss_ft"],
    "C": ["distance_miles", "elevation_gain_ft", "elevation_loss_ft", "hike_type"],
}
BASELINE = "personal_minutes_per_mile"


@dataclass
class BreakArtifact:
    minutes_per_mile: float | None = None
    breaks_per_mile: float | None = None
    mean_duration_minutes: float | None = None
    selected_model: str = BASELINE
    estimators: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    training_hikes: int = 0
    training_ranges: dict = field(default_factory=dict)
    reason: str = "No historical break model is available. Run analyze-breaks first."
    config: BreakConfig = field(default_factory=BreakConfig)
    provenance: dict = field(default_factory=dict)


@dataclass
class BreakForecast:
    predicted_minutes: float | None
    baseline_minutes: float | None
    expected_break_count: float | None
    expected_average_minutes: float | None
    label: str
    warnings: list = field(default_factory=list)
    experimental_minutes: dict = field(default_factory=dict)


def _fit(name, data):
    features = FEATURES[name]
    if name == "C":
        transform = ColumnTransformer([
            ("geometry", "passthrough", FEATURES["B"]),
            ("type", OneHotEncoder(drop="first", handle_unknown="error"), ["hike_type"]),
        ])
        model = make_pipeline(transform, LinearRegression())
    else:
        model = LinearRegression()
    return model.fit(data[features], data.total_break_minutes)


def _type_supported(data, minimum):
    counts = data.hike_type.value_counts()
    return data.hike_type.isin(HIKE_TYPES).all() and all(counts.get(t, 0) >= minimum for t in HIKE_TYPES)


def train_break_predictor(hikes, config=None):
    """Fit rates on hike totals; refit every candidate inside each held-out fold.

    Selection requires twelve hikes by default and a 10% MAE improvement over
    the baseline. With fewer hikes the comparisons are explicitly experimental.
    No GPS points, segments, elapsed times, or break counts become predictors.
    """
    config = config or BreakConfig()
    if hikes.hike_id.duplicated().any():
        raise ValueError("Break training requires exactly one observation per hike_id.")
    targets = hikes.target_definition.unique().tolist() if "target_definition" in hikes else [ORDINARY_BREAK_TARGET]
    if not targets:
        targets = [ORDINARY_BREAK_TARGET]
    if len(targets) != 1 or targets[0] not in (ORDINARY_BREAK_TARGET, NON_MOVING_TARGET):
        raise ValueError("Break observations must use the same supported training target.")
    target = targets[0]
    data = hikes.loc[hikes.training_eligible.eq(True)].copy().reset_index(drop=True)
    if not data.empty:
        numeric = data[[*FEATURES["B"], "total_break_minutes", "break_count"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or data.distance_miles.le(0).any():
            raise ValueError("Eligible break observations need finite nonnegative geometry and targets, with positive distance.")
    artifact = BreakArtifact(config=config, training_hikes=len(data))
    status_c = "Model C unavailable: known hike types and enough hikes in both categories are required."
    names = ["A", "B"]
    if _type_supported(data, config.minimum_type_hikes + 1):
        names.append("C")
        status_c = "Model C evaluated on hikes with both known types represented in every training fold."
    artifact.evaluation = {"method": "leave-one-hike-out", "metrics": [], "folds": [], "target_definition": target,
                           "model_c_status": status_c, "experimental": len(data) < config.minimum_selection_hikes,
                           "minimum_selection_hikes": config.minimum_selection_hikes,
                           "minimum_evaluation_hikes": config.minimum_evaluation_hikes}
    if data.empty:
        artifact.reason = ("No selected recordings have complete, valid timing and usable hiking geometry outside camping/non-hiking activity; no personal break allowance can be estimated."
                           if target == NON_MOVING_TARGET else
                           "No recordings have adequate break-label coverage; no personal rate can be estimated.")
        return artifact, pd.DataFrame(columns=PREDICTION_COLUMNS)
    distance = float(data.distance_miles.sum())
    minutes = float(data.total_break_minutes.sum())
    count = int(data.break_count.sum())
    artifact.minutes_per_mile = minutes / distance
    artifact.breaks_per_mile = count / distance
    ordinary_minutes = float(data.ordinary_break_minutes.sum()) if "ordinary_break_minutes" in data else minutes
    artifact.mean_duration_minutes = ordinary_minutes / count if count else None
    artifact.training_ranges = {f: [float(data[f].min()), float(data[f].max())] for f in FEATURES["B"]}
    predictions, folds = [], []
    if len(data) >= config.minimum_evaluation_hikes:
        for fold, (train_indices, test_indices) in enumerate(LeaveOneGroupOut().split(data, groups=data.hike_id)):
            train, test = data.iloc[train_indices], data.iloc[test_indices]
            folds.append({"fold": fold, "training_hike_ids": train.hike_id.tolist(),
                          "test_hike_ids": test.hike_id.tolist()})
            # Calculate the baseline rate using training hikes only.
            outputs = {BASELINE: test.distance_miles.to_numpy() * train.total_break_minutes.sum() / train.distance_miles.sum()}
            for name in names:
                outputs[name] = np.maximum(0, _fit(name, train).predict(test[FEATURES[name]]))
            for name, values in outputs.items():
                for (_, hike), prediction in zip(test.iterrows(), values):
                    predictions.append({"fold": fold, "hike_id": hike.hike_id, "model": name,
                                        "actual_break_minutes": float(hike.total_break_minutes),
                                        "predicted_break_minutes": float(prediction)})
    held_out = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    metrics = []
    for name, part in held_out.groupby("model", sort=False):
        metrics.append({"model": name, "hikes": len(part),
                        "MAE_minutes": float(mean_absolute_error(part.actual_break_minutes, part.predicted_break_minutes))})
    artifact.evaluation.update(metrics=metrics, folds=folds)
    if len(data) >= config.minimum_evaluation_hikes:
        artifact.estimators = {name: _fit(name, data) for name in names}
    artifact.reason = (f"Selected the personal distance baseline using {len(data)} independent recordings. "
                       f"Regression selection requires at least {config.minimum_selection_hikes} hikes and 10% lower held-out MAE. "
                       "GPS points are not independent examples.")
    if config.minimum_evaluation_hikes <= len(data) < config.minimum_selection_hikes:
        artifact.reason += " Available regression comparisons are experimental."
    if len(data) >= config.minimum_selection_hikes:
        errors = {row["model"]: row["MAE_minutes"] for row in metrics}
        best = min(names, key=lambda name: errors[name])
        if errors[best] < 0.9 * errors[BASELINE]:
            artifact.selected_model = best
            artifact.reason = f"Selected break regression {best}: at least 10% lower leave-one-hike-out MAE than the personal baseline."
        else:
            artifact.reason = "Selected the personal baseline: regression did not improve held-out MAE by at least 10%."
    if len(data) < config.minimum_evaluation_hikes:
        artifact.reason += " Too few hikes for MAE evaluation; no accuracy score is reported and regression is not fitted."
    if target == NON_MOVING_TARGET:
        artifact.reason += (" Break allowance uses elapsed time minus moving time, including short pauses. "
                            "Unresolved GPS intervals are included in this allowance and may contain unmeasured movement.")
    return artifact, held_out


def break_coefficients(artifact):
    rows = []
    for name, model in artifact.estimators.items():
        estimator = model[-1] if name == "C" else model
        features = model[0].get_feature_names_out() if name == "C" else FEATURES[name]
        rows.append({"model": name, "feature": "intercept", "coefficient": float(estimator.intercept_)})
        rows.extend({"model": name, "feature": feature, "coefficient": float(coefficient)}
                    for feature, coefficient in zip(features, estimator.coef_))
    return pd.DataFrame(rows, columns=["model", "feature", "coefficient"])


def predict_break_time(distance_miles, elevation_gain_ft, elevation_loss_ft, hike_type, artifact=None):
    artifact = artifact or BreakArtifact()
    geometry = [distance_miles, elevation_gain_ft, elevation_loss_ft]
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not isfinite(v) or v < 0 for v in geometry):
        raise ValueError("Break prediction geometry must be finite and nonnegative.")
    if hike_type not in HIKE_TYPES:
        raise ValueError("hike_type must be day_hike or backpacking.")
    warnings = [artifact.reason]
    if artifact.minutes_per_mile is None:
        return BreakForecast(None, None, None, None, "unavailable", warnings)
    baseline = float(distance_miles * artifact.minutes_per_mile)
    row = pd.DataFrame([[*geometry, hike_type]], columns=FEATURES["C"])
    estimates = {name: max(0.0, float(model.predict(row[FEATURES[name]])[0])) for name, model in artifact.estimators.items()}
    selected = baseline if artifact.selected_model == BASELINE else estimates[artifact.selected_model]
    if distance_miles == 0:
        selected, baseline = 0.0, 0.0
        estimates = {name: 0.0 for name in estimates}
    if not np.isfinite([selected, baseline, *estimates.values()]).all():
        raise ValueError("Break estimate overflowed; use finite, realistic route totals.")
    for name, value in zip(FEATURES["B"], geometry):
        low, high = artifact.training_ranges[name]
        if value < low or value > high:
            warnings.append(f"Planned {name} is outside the historical range ({low:.2f}–{high:.2f}).")
    if artifact.evaluation.get("target_definition") == NON_MOVING_TARGET:
        warnings.append("Break allowance estimates total non-moving time, including short pauses and unresolved GPS intervals. "
                        "GPS gaps may contain unmeasured movement. Break count and average duration describe only detected ordinary breaks.")
    else:
        warnings.append("Labels describe observed meaningful rests, not guaranteed intentional breaks. Unresolved time and short stops are excluded.")
    if artifact.selected_model != "C":
        warnings.append("Hike type does not adjust this break estimate; known day-hike/backpacking history is insufficient or Model C was not selected.")
    return BreakForecast(selected, baseline, distance_miles * artifact.breaks_per_mile,
                         artifact.mean_duration_minutes,
                         "personal break baseline" if artifact.selected_model == BASELINE else f"personal break regression {artifact.selected_model}",
                         warnings, {name: value for name, value in estimates.items() if name != artifact.selected_model})


def file_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_break_model(path=DEFAULT_BREAK_MODEL, *, project=None):
    """Load only trusted local artifacts; decline stale labels instead of retraining."""
    path = Path(path)
    if not path.exists():
        return BreakArtifact()
    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Cannot load the local break model: {error}") from error
    if not isinstance(artifact, BreakArtifact):
        raise ValueError("Unsupported break model artifact.")
    if project is not None:
        from .import_gpx import find_gpx_files
        if artifact.evaluation.get("source_files") != [p.name for p in find_gpx_files(Path(project) / "data/raw")]:
            return BreakArtifact(reason="Recorded GPX files changed. Run analyze-breaks to refresh the personal break model.")
        for relative, expected in artifact.provenance.items():
            if file_digest(Path(project) / relative) != expected:
                return BreakArtifact(reason="Historical break inputs or corrections changed. Run analyze-breaks to refresh labels and predictions.")
    return artifact
