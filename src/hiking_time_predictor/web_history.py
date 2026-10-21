"""Read measured history and saved evaluation for dashboards without training.

This module deliberately has no Streamlit dependency. Historical prediction
exports are usable only when their run manifest matches both current inputs.
Saved artifact evaluation describes that artifact's training run, independently
of the current historical CSV files.
"""

from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import (
    GEOMETRY, MODEL_FEATURES, TARGET, aggregate_hike_predictions,
    coefficient_table, prepare_segments, validate_estimator_schema,
)


@dataclass
class HistoryData:
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    held_out_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    hike_totals: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


def _read_csv(path: Path) -> pd.DataFrame:
    """Reject ambiguous headers rather than letting pandas rename duplicates."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            header = next(csv.reader(source), [])
        if not header:
            raise ValueError("The file has no CSV header.")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names are not allowed.")
        # Hike IDs are labels, not numbers: "001" and "1" are distinct hikes.
        return pd.read_csv(path, dtype={"hike_id": "string"} if "hike_id" in header else None)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error


def _numbers(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Missing column {column}.")
    values = pd.to_numeric(frame[column], errors="coerce")
    boolean = frame[column].map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean.any() or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"{column} contains missing or invalid measurements.")
    return values


def _nonnegative(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numbers(frame, column)
    if values.lt(0).any():
        raise ValueError(f"{column} contains negative measurements.")
    return values


def _summary_metrics(history: HistoryData) -> None:
    summary = history.summary
    if "hike_id" not in summary:
        raise ValueError("Historical summary needs a hike_id column.")
    summary["hike_id"] = summary.hike_id.astype("string").str.strip()
    if summary.hike_id.isna().any() or summary.hike_id.eq("").any() or summary.hike_id.duplicated().any():
        raise ValueError("Historical summary hike_id values must be nonblank and unique.")
    history.metrics["hike_count"] = len(summary)
    history.notes.append("Hike count counts unique recordings in hiking_summary.csv.")

    def measured_columns(*names):
        try:
            return [_nonnegative(summary, name) for name in names]
        except ValueError as error:
            history.notes.append(f"Historical statistic unavailable: {error}")
            return None

    distance = measured_columns("raw_distance_miles")
    if distance is not None:
        history.metrics["logged_miles"] = float(distance[0].sum())
        history.notes.append("Logged miles sum raw_distance_miles, including recorded nonmoving GPS distance.")
    pace = measured_columns(TARGET, "distance_miles")
    if pace is not None and pace[1].sum() > 0:
        history.metrics["average_pace_min_per_mile"] = float(pace[0].sum() / pace[1].sum())
        history.notes.append("Average moving pace is total recorded moving minutes / accepted moving miles; stationary and unknown time are excluded.")
    elif pace is not None:
        history.notes.append("Average moving pace is unavailable because accepted moving distance is zero.")
    gain = measured_columns("elevation_gain_ft")
    if gain is not None:
        history.metrics["elevation_gain_total_ft"] = float(gain[0].sum())
        if len(gain[0]):
            history.metrics["elevation_gain_mean_ft"] = float(gain[0].mean())
            history.metrics["elevation_gain_max_ft"] = float(gain[0].max())
        history.notes.append("Elevation statistics use smoothed, accepted moving elevation_gain_ft from historical summaries.")


def _matching_provenance(processed: Path, notes: list[str]) -> bool:
    manifest = processed / "regression" / "run_summary.json"
    try:
        run = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        notes.append(f"Historical held-out charts unavailable: evaluation provenance is missing or unreadable ({error}).")
        return False
    if not isinstance(run, dict):
        notes.append("Historical held-out charts unavailable: evaluation provenance must be a JSON object.")
        return False
    for key, filename in (("input_sha256", "hiking_segments.csv"), ("summary_sha256", "hiking_summary.csv")):
        source = processed / filename
        if not source.is_file() or run.get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            notes.append(f"Historical held-out charts unavailable: evaluation provenance does not match current {filename}. Re-run evaluation explicitly to refresh exports.")
            return False
    return True


def _validated_predictions(frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    required = {"hike_id", "segment_index", *GEOMETRY, TARGET, "fold", "model", "predicted_time_min"}
    if not required.issubset(frame):
        raise ValueError(f"Held-out predictions need columns {sorted(required)}.")
    if frame.empty:
        return frame
    for name in ("segment_index", *GEOMETRY, TARGET, "fold"):
        frame[name] = _nonnegative(frame, name)
    frame["predicted_time_min"] = _numbers(frame, "predicted_time_min")
    for name in ("fold", "segment_index"):
        if frame[name].mod(1).ne(0).any():
            raise ValueError(f"Held-out {name} must be an integer.")
    for name in ("hike_id", "model"):
        frame[name] = frame[name].astype("string").str.strip()
        if frame[name].isna().any() or frame[name].eq("").any():
            raise ValueError(f"Held-out {name} must be nonblank.")
    if frame.duplicated(["model", "hike_id", "segment_index"]).any():
        raise ValueError("Duplicate held-out model/hike/segment rows.")
    accepted = segments.loc[segments.eligible_ab, ["hike_id", "segment_index", *GEOMETRY, TARGET]]
    joined = frame.merge(accepted, on=["hike_id", "segment_index"], how="left", suffixes=("", "_source"), validate="many_to_one")
    for name in (*GEOMETRY, TARGET):
        if not np.isclose(joined[name], joined[f"{name}_source"], rtol=1e-7, atol=1e-8).all():
            raise ValueError("Held-out measurements do not match eligible current historical segments.")
    return frame


def _validated_totals(frame: pd.DataFrame, history: HistoryData) -> pd.DataFrame:
    # Whole-hike eligibility requires measured totals and known timing coverage.
    # A valid segment export cannot repair incomplete summary measurements.
    for name in ("distance_miles", TARGET, "unknown_time_min"):
        _nonnegative(history.summary, name)
    expected = aggregate_hike_predictions(history.held_out_predictions, history.segments, history.summary)
    if not set(expected.columns).issubset(frame):
        raise ValueError(f"Hike totals need columns {list(expected.columns)}.")
    keys = ["fold", "model", "hike_id"]
    if frame.duplicated(keys).any() or len(frame) != len(expected):
        raise ValueError("Saved hike totals do not match held-out prediction groups.")
    joined = expected.merge(frame, on=keys, how="outer", suffixes=("_expected", "_saved"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Saved hike total identifiers do not match held-out predictions.")
    for name in set(expected.columns) - set(keys):
        saved, actual = joined[f"{name}_saved"], joined[f"{name}_expected"]
        if name in ("whole_hike_eligible", "complete_segment_coverage"):
            if not saved.map(lambda value: isinstance(value, (bool, np.bool_))).all() or not saved.eq(actual).all():
                raise ValueError(f"Saved hike total {name} disagrees with current coverage checks.")
        elif not np.isclose(pd.to_numeric(saved, errors="coerce"), actual, rtol=1e-7, atol=1e-8).all():
            raise ValueError(f"Saved hike total {name} disagrees with held-out segments.")
    return frame


def load_history(project_root: Path, model_name: str | None = None) -> HistoryData:
    """Load recorded measurements, eligibility, and provenance-checked exports.

    Missing inputs produce unavailable metrics (None), never synthetic values.
    Pace is weighted by accepted moving distance, not an average of hike paces.
    C's usable count additionally requires valid hike-type metadata.
    """
    if model_name is not None and model_name not in MODEL_FEATURES:
        raise ValueError(f"Unsupported model name: {model_name!r}.")
    history = HistoryData(metrics={name: None for name in (
        "logged_miles", "hike_count", "usable_segments", "average_pace_min_per_mile",
        "elevation_gain_total_ft", "elevation_gain_mean_ft", "elevation_gain_max_ft",
    )})
    processed = Path(project_root) / "data" / "processed"
    summary_path, segments_path = processed / "hiking_summary.csv", processed / "hiking_segments.csv"
    if summary_path.is_file():
        history.summary = _read_csv(summary_path)
        _summary_metrics(history)
    else:
        history.notes.append("Historical summary unavailable: data/processed/hiking_summary.csv is missing.")
    if segments_path.is_file():
        history.segments = prepare_segments(_read_csv(segments_path))
        eligibility = "eligible_c" if model_name == "C" else "eligible_ab"
        history.metrics["usable_segments"] = int(history.segments[eligibility].sum())
        history.notes.append(f"Usable segments follow modeling.prepare_segments {eligibility} quality rules; they are measured moving-time targets.")
    else:
        history.notes.append("Historical training segments unavailable: data/processed/hiking_segments.csv is missing.")
    if _matching_provenance(processed, history.notes):
        predictions_path = processed / "regression" / "held_out_segment_predictions.csv"
        totals_path = processed / "regression" / "hike_totals_and_coverage.csv"
        if predictions_path.is_file():
            try:
                history.held_out_predictions = _validated_predictions(_read_csv(predictions_path), history.segments)
            except ValueError as error:
                history.notes.append(f"Historical held-out charts unavailable: {error}")
        else:
            history.notes.append("Historical held-out segment predictions are unavailable; no predictions were generated.")
        if totals_path.is_file() and not history.held_out_predictions.empty:
            try:
                history.hike_totals = _validated_totals(_read_csv(totals_path), history)
            except ValueError as error:
                history.notes.append(f"Historical hike comparison unavailable: {error}")
        else:
            history.notes.append("Historical hike comparisons are unavailable; matching held-out predictions and saved totals are required.")
    return history


def evaluation_frames(artifact) -> dict[str, pd.DataFrame]:
    """Expose only stored evaluation and fitted coefficients; never fit a model."""
    evaluation = getattr(artifact, "evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Saved model evaluation must be a dictionary.")
    frames = {}
    for name in ("segment_metrics", "whole_hike_metrics", "folds"):
        records = evaluation.get(name, [])
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError(f"Saved {name} must be a list of evaluation records.")
        frame = pd.DataFrame(records)
        if name != "folds":
            required = {"model", "observations", "hikes", "MAE_min", "RMSE_min"}
            # A partial record cannot establish evaluation scores. Do not fill
            # missing measurements or invent zero counts to make a chart work.
            if not required.issubset(frame) or frame[list(required)].isna().any().any():
                frame = pd.DataFrame()
            elif not frame.empty:
                frame["model"] = frame.model.astype("string").str.strip()
                if frame.model.eq("").any() or frame.model.duplicated().any():
                    raise ValueError(f"Saved {name} model labels must be nonblank and unique.")
                for column in ("observations", "hikes", "MAE_min", "RMSE_min"):
                    frame[column] = _nonnegative(frame, column)
                for column in ("observations", "hikes"):
                    if frame[column].le(0).any() or frame[column].mod(1).ne(0).any():
                        raise ValueError(f"Saved {name} {column} must be positive integer counts.")
                if frame.hikes.gt(frame.observations).any():
                    raise ValueError(f"Saved {name} hike counts cannot exceed observation counts.")
                if "bias_pred_minus_actual_min" in frame:
                    frame["bias_pred_minus_actual_min"] = _numbers(frame, "bias_pred_minus_actual_min")
                if "negative_predictions" in frame:
                    frame["negative_predictions"] = _nonnegative(frame, "negative_predictions")
                    if frame.negative_predictions.mod(1).ne(0).any() or frame.negative_predictions.gt(frame.observations).any():
                        raise ValueError(f"Saved {name} negative_predictions must be integer counts within observations.")
        frames[name] = frame
    frames["coefficients"] = pd.DataFrame()
    estimator = getattr(artifact, "estimator", None)
    if estimator is not None:
        model_name = getattr(artifact, "model_name", None)
        validate_estimator_schema(estimator, model_name)
        frames["coefficients"] = coefficient_table(estimator, model_name)
    return frames
