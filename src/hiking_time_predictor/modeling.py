"""Hike-grouped regression and auditable evaluation of recorded moving time.

Only route geometry and explicitly supplied hike type can enter X. Quality
diagnostics and identifiers are used for selection/evaluation, never prediction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hiking_time_predictor.data import HIKE_TYPES


TARGET = "moving_time_min"
GEOMETRY = ["distance_miles", "elevation_gain_ft", "elevation_loss_ft"]
MODEL_FEATURES = {
    "A": ["distance_miles"],
    "B": GEOMETRY,
    "C": [*GEOMETRY, "hike_type"],
}
MODEL_LABELS = {
    "A": "A: distance",
    "B": "B: distance + gain + loss",
    "C": "C: geometry + hike type",
}
NAISMITH_LABEL = "Naismith (3 mph + 2000 ft/hr)"
# Missing optional metadata does not invalidate geometry or measured time.
# Every other flag, including an unfamiliar future flag, requires review.
ALLOWED_FLAGS = {"missing_hike_type"}


class FeatureSchemaError(ValueError):
    """The supplied model/features do not implement a supported training schema."""


def validate_estimator_schema(estimator, model_name):
    """Check the fitted input names before passing ordered columns to sklearn."""
    if model_name not in MODEL_FEATURES:
        raise FeatureSchemaError(f"Unsupported model feature combination: {model_name!r}.")
    expected = MODEL_FEATURES[model_name]
    actual = getattr(estimator, "feature_names_in_", None)
    if actual is None or list(actual) != expected:
        raise FeatureSchemaError(f"Unsupported fitted feature schema: expected {expected}, "
                                 f"got {None if actual is None else list(actual)}.")


def _measurement_flags(value):
    return set(str(value).split(";")) - {""} if pd.notna(value) else set()


def prepare_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Return all rows with numeric measurements and explicit eligibility reasons.

    A/B do not require hike type. C requires known, consistent metadata. Unknown
    time must be exactly zero even if a producer accidentally omits its flag.
    Short final segments are retained; no rows are removed based on model errors.
    """
    required = ["hike_id", "segment_index", *GEOMETRY, TARGET,
                "quality_flags", "unknown_time_min"]
    missing = [column for column in required if column not in segments]
    if missing:
        raise ValueError(f"Missing segment columns: {missing}.")
    if segments.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    data = segments.copy().reset_index(drop=True)
    data["hike_id"] = data["hike_id"].astype("string").str.strip()
    if "hike_type" not in data:
        data["hike_type"] = pd.NA
    data["hike_type"] = data["hike_type"].astype("string").str.strip().replace("", pd.NA)
    if data.duplicated(["hike_id", "segment_index"]).any():
        raise ValueError("Duplicate (hike_id, segment_index) rows; do not count recordings twice.")
    known = data.loc[data["hike_type"].notna()]
    if known.groupby("hike_id")["hike_type"].nunique().gt(1).any():
        raise ValueError("Conflicting hike_type metadata within a hike.")

    reasons = [[] for _ in range(len(data))]

    def mark(mask, reason):
        for position in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[position].append(reason)

    mark(data["hike_id"].isna() | data["hike_id"].eq("").fillna(False), "missing_hike_id")
    for column in [*GEOMETRY, TARGET, "unknown_time_min", "segment_index"]:
        booleans = data[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        data[column] = pd.to_numeric(data[column], errors="coerce")
        finite = np.isfinite(data[column].to_numpy(dtype=float, na_value=np.nan))
        invalid = ~finite | booleans
        if column in ("distance_miles", TARGET):
            invalid |= data[column].le(0).fillna(True)
        else:
            invalid |= data[column].lt(0).fillna(True)
        if column == "segment_index":
            invalid |= data[column].mod(1).ne(0).fillna(True)
        mark(invalid, f"invalid_{column}")
    if data.duplicated(["hike_id", "segment_index"]).any():
        raise ValueError("Duplicate (hike_id, segment_index) rows after numeric conversion.")
    mark(data["unknown_time_min"].gt(0).fillna(False), "unknown_time_present")
    for position, value in enumerate(data["quality_flags"]):
        for flag in sorted(_measurement_flags(value) - ALLOWED_FLAGS):
            reasons[position].append(f"flag:{flag}")
    data["exclusion_reasons_ab"] = [";".join(items) for items in reasons]
    data["eligible_ab"] = data["exclusion_reasons_ab"].eq("")
    data["eligible_c"] = data["eligible_ab"] & data["hike_type"].isin(HIKE_TYPES)
    return data


def group_splits(data: pd.DataFrame, random_state: int = 42):
    """Use LOGO for 2–20 hikes, a fixed 25% group holdout for larger collections.

    The threshold is a transparent teaching choice, not a statistical guarantee.
    A single hike cannot supply a between-hike test set.
    """
    groups = data["hike_id"]
    count = groups.nunique()
    if count < 2:
        return [], "Unavailable: fewer than two independent hikes"
    if count <= 20:
        splitter = LeaveOneGroupOut()
        method = "LeaveOneGroupOut"
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        method = f"GroupShuffleSplit (25% of hikes, seed {random_state})"
    splits = list(splitter.split(data, groups=groups))
    for train, test in splits:
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise AssertionError("A hike cannot occur in both training and testing.")
    return splits, method


def model_c_status(data: pd.DataFrame) -> tuple[bool, str]:
    """Require a category contrast in every training fold of the known-type cohort."""
    if data.empty:
        return False, "Model C skipped: no eligible segments have known hike_type."
    splits, _ = group_splits(data)
    if not splits:
        return False, "Model C skipped: known metadata covers fewer than two hikes."
    if any(set(data.iloc[train]["hike_type"]) != set(HIKE_TYPES) for train, _ in splits):
        return False, "Model C skipped: both hike types must occur in every training fold."
    return True, "Model C available: compare A, B, C and Naismith on the same known-type rows/folds."


def make_estimator(model_name: str):
    """Build ordinary least squares; encode C's category inside each training fold."""
    if model_name not in MODEL_FEATURES:
        raise ValueError(f"Unknown model: {model_name}.")
    if model_name != "C":
        return LinearRegression()
    # Learned categories are fit on training data only. Unknown types are rejected,
    # rather than silently predicting them as the reference category.
    features = ColumnTransformer([
        ("geometry", "passthrough", GEOMETRY),
        ("type", OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False),
         ["hike_type"]),
    ])
    return Pipeline([("features", features), ("regression", LinearRegression())])


def fit_model(data: pd.DataFrame, model_name: str):
    """Fit using an allowlist of features; audit columns cannot leak into X."""
    estimator = make_estimator(model_name)
    return estimator.fit(data[MODEL_FEATURES[model_name]], data[TARGET])


def coefficient_table(estimator, model_name: str) -> pd.DataFrame:
    """Expose intercept, coefficients, and rank for teaching and instability checks."""
    if model_name == "C":
        regression = estimator.named_steps["regression"]
        names = estimator.named_steps["features"].get_feature_names_out()
    else:
        regression = estimator
        names = MODEL_FEATURES[model_name]
    return pd.DataFrame({
        "term": ["intercept", *names],
        "value": [float(regression.intercept_), *regression.coef_],
        "design_rank": regression.rank_,
        "feature_count": len(names),
    })


def naismith_minutes(geometry: pd.DataFrame) -> np.ndarray:
    """Original imperial variant: 60 min/3 miles + 60 min/2000 feet ascent.

    No breaks, descent adjustment, learned calibration, or observed timing inputs.
    Evaluate against moving time on the same accepted distance/ascent.
    """
    values = geometry[["distance_miles", "elevation_gain_ft"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Naismith requires finite, nonnegative distance and ascent.")
    return 20.0 * values[:, 0] + 0.03 * values[:, 1]


@dataclass
class Evaluation:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    coefficients: pd.DataFrame
    method: str


def evaluate_models(data: pd.DataFrame, model_names=("A", "B")) -> Evaluation:
    """Predict only held-out hikes, using the exact same rows/folds for each model."""
    checked = prepare_segments(data)
    eligibility = "eligible_c" if "C" in model_names else "eligible_ab"
    if not checked[eligibility].all():
        raise ValueError("Evaluation input contains ineligible segments; inspect the quality audit.")
    if "C" in model_names:
        ready, reason = model_c_status(checked)
        if not ready:
            raise ValueError(reason)
    splits, method = group_splits(checked)
    predictions, folds, coefficients = [], [], []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train, test = checked.iloc[train_index], checked.iloc[test_index]
        folds.append({"fold": fold, "train_hikes": ";".join(sorted(train["hike_id"].unique())),
                      "test_hikes": ";".join(sorted(test["hike_id"].unique())),
                      "train_segments": len(train), "test_segments": len(test)})
        base = test[["hike_id", "segment_index", *GEOMETRY, TARGET]].copy()
        base["fold"] = fold
        for name in model_names:
            estimator = fit_model(train, name)
            predicted = estimator.predict(test[MODEL_FEATURES[name]])
            predictions.append(base.assign(model=MODEL_LABELS[name], predicted_time_min=predicted))
            table = coefficient_table(estimator, name)
            coefficients.append(table.assign(fold=fold, model=MODEL_LABELS[name]))
        predictions.append(base.assign(model=NAISMITH_LABEL, predicted_time_min=naismith_minutes(test)))
    prediction_columns = ["hike_id", "segment_index", *GEOMETRY, TARGET, "fold", "model", "predicted_time_min"]
    return Evaluation(
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=prediction_columns),
        pd.DataFrame(folds),
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
        method,
    )


def error_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Each row has equal weight: segments for segment metrics, hikes for totals."""
    rows = []
    for name, group in predictions.groupby("model", sort=False):
        actual, predicted = group[TARGET], group["predicted_time_min"]
        rows.append({"model": name, "observations": len(group), "hikes": group["hike_id"].nunique(),
                     "MAE_min": mean_absolute_error(actual, predicted),
                     "RMSE_min": np.sqrt(mean_squared_error(actual, predicted)),
                     "bias_pred_minus_actual_min": (predicted - actual).mean(),
                     "negative_predictions": int(predicted.lt(0).sum())})
    return pd.DataFrame(rows, columns=["model", "observations", "hikes", "MAE_min", "RMSE_min",
                                       "bias_pred_minus_actual_min", "negative_predictions"])


def aggregate_hike_predictions(predictions, all_segments, summary=None):
    """Sum matched held-out segments; only label verified complete recordings whole.

    Filtering often leaves only part of a hike. Comparing that sum against an
    entire hike's observed time would be invalid. Full-recording eligibility also
    requires a clean summary and matching exported distance/time totals.
    """
    rows = []
    summaries = {} if summary is None else summary.set_index("hike_id").to_dict("index")
    for (fold, name, hike_id), group in predictions.groupby(["fold", "model", "hike_id"], sort=False):
        source = all_segments.loc[all_segments["hike_id"].eq(hike_id)]
        complete = (not group["segment_index"].duplicated().any()
                    and set(group["segment_index"]) == set(source["segment_index"]))
        info = summaries.get(hike_id, {})
        summary_ok = (bool(info) and not (_measurement_flags(info.get("quality_flags")) - ALLOWED_FLAGS)
                      and info.get("unknown_time_min", np.nan) == 0
                      and np.isclose(info.get(TARGET, np.nan), group[TARGET].sum())
                      and np.isclose(info.get("distance_miles", np.nan), group["distance_miles"].sum()))
        rows.append({"fold": fold, "model": name, "hike_id": hike_id,
                     TARGET: group[TARGET].sum(), "predicted_time_min": group["predicted_time_min"].sum(),
                     "retained_segments": len(group), "exported_segments": len(source),
                     "distance_coverage_fraction": group["distance_miles"].sum() / source["distance_miles"].sum(),
                     "complete_segment_coverage": complete,
                     "whole_hike_eligible": complete and summary_ok})
    return pd.DataFrame(rows, columns=["fold", "model", "hike_id", TARGET, "predicted_time_min",
                                      "retained_segments", "exported_segments", "distance_coverage_fraction",
                                      "complete_segment_coverage", "whole_hike_eligible"])


def predict_route(estimator, model_name: str, route_segments: pd.DataFrame) -> pd.DataFrame:
    """Predict geometry-only future segments, including the short final segment.

    Call .sum() on predicted_time_min for route moving time. No observed time or
    speed is needed. Invalid negative OLS predictions raise rather than being
    silently clipped. Route segmentation must match the training convention.
    """
    if model_name not in MODEL_FEATURES:
        raise FeatureSchemaError(f"Unsupported model feature combination: {model_name!r}.")
    required = MODEL_FEATURES[model_name]
    missing = [column for column in required if column not in route_segments]
    if missing:
        raise ValueError(f"Missing route features: {missing}.")
    validate_estimator_schema(estimator, model_name)
    if route_segments.empty:
        raise ValueError("At least one route segment is required.")
    route = route_segments.copy()
    for column in [name for name in required if name != "hike_type"]:
        booleans = route[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        route[column] = pd.to_numeric(route[column], errors="coerce")
        values = route[column].to_numpy(dtype=float, na_value=np.nan)
        invalid = ~np.isfinite(values) | booleans.to_numpy()
        invalid |= values <= 0 if column == "distance_miles" else values < 0
        if invalid.any():
            raise ValueError(f"Invalid route feature: {column}.")
    if model_name == "C":
        if not route["hike_type"].isin(HIKE_TYPES).all() or route["hike_type"].nunique() != 1:
            raise ValueError("Supply one known hike_type for the entire route.")
    prediction = estimator.predict(route[required])
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(route),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("The linear model produced an invalid moving time; inspect route/model limits.")
    return route.assign(predicted_time_min=prediction)
