"""Load and validate hiking observations without training a model."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_COLUMNS = [
    "hike_id",
    "distance_miles",
    "elevation_gain_ft",
    "elevation_loss_ft",
    "hike_type",
    "moving_time_min",
]

NUMERIC_COLUMNS = [
    "distance_miles",
    "elevation_gain_ft",
    "elevation_loss_ft",
    "moving_time_min",
]

HIKE_TYPES = ("day_hike", "backpacking")


def validate_hiking_data(hikes: pd.DataFrame, allow_empty: bool = True) -> None:
    """Raise ValueError for invalid observations; leave the DataFrame unchanged.

    An empty table with the correct headers is a valid starting template.
    Use allow_empty=False when a later step requires actual observations.
    Numeric text is accepted if it can be converted to a finite number.
    """
    if hikes.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    missing = [column for column in DATA_COLUMNS if column not in hikes.columns]
    unexpected = [column for column in hikes.columns if column not in DATA_COLUMNS]
    if missing or unexpected:
        raise ValueError(
            f"Columns do not match the dataset schema. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    if hikes.empty:
        if not allow_empty:
            raise ValueError("At least one hike is required.")
        return

    if hikes.isna().any().any():
        raise ValueError("Missing values are not allowed; fill in every column.")

    # Strip surrounding spaces when checking IDs so 'hike-1' and ' hike-1 '
    # cannot accidentally identify the same observation twice.
    hike_ids = hikes["hike_id"].astype("string").str.strip()
    if hike_ids.eq("").any():
        raise ValueError("hike_id must not be blank.")
    if hike_ids.duplicated().any():
        raise ValueError("hike_id must be unique for each hike.")

    if not hikes["hike_type"].isin(HIKE_TYPES).all():
        raise ValueError("hike_type must be 'day_hike' or 'backpacking'.")

    for column in NUMERIC_COLUMNS:
        values = hikes[column]
        numbers = pd.to_numeric(values, errors="coerce")
        # True/False are not measurements, even though Python treats them as 1/0.
        contains_boolean = values.map(lambda value: isinstance(value, (bool, np.bool_))).any()
        if contains_boolean or not np.isfinite(numbers.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values.")

        if column in ("distance_miles", "moving_time_min"):
            if (numbers <= 0).any():
                raise ValueError(f"{column} must be greater than zero.")
        elif (numbers < 0).any():
            raise ValueError(f"{column} must be zero or greater.")


def load_hiking_data(csv_path: str | Path) -> pd.DataFrame:
    """Read a CSV, validate it, and return consistently typed observations."""
    # Read identifiers as text so an ID such as '001' keeps its leading zeros.
    hikes = pd.read_csv(csv_path, dtype={"hike_id": "string", "hike_type": "string"})
    validate_hiking_data(hikes)

    hikes["hike_id"] = hikes["hike_id"].str.strip()
    for column in NUMERIC_COLUMNS:
        hikes[column] = pd.to_numeric(hikes[column])

    return hikes[DATA_COLUMNS]


def select_model_ready_segments(segments: pd.DataFrame, require_pack_weight: bool = False) -> pd.DataFrame:
    """Select complete, unflagged GPX rows for a future model; train nothing.

    The original six-column manual-hike loader above keeps its existing rules.
    Segment IDs repeat within a hike; (hike_id, segment_index) identifies a row.
    Unknown categories remain missing in the source and are excluded here.
    A future evaluation must split by hike_id, since adjacent segments are related.
    """
    required = [*NUMERIC_COLUMNS, "hike_type", "quality_flags"]
    if require_pack_weight:
        required.append("pack_weight_lbs")
    missing = [column for column in required if column not in segments]
    if missing:
        raise ValueError(f"Missing segment columns: {missing}.")
    keep = segments["hike_type"].isin(HIKE_TYPES) & segments["quality_flags"].fillna("").eq("")
    for column in NUMERIC_COLUMNS + (["pack_weight_lbs"] if require_pack_weight else []):
        numbers = pd.to_numeric(segments[column], errors="coerce")
        keep &= np.isfinite(numbers.to_numpy(dtype=float))
        keep &= numbers.gt(0) if column in ("distance_miles", "moving_time_min") else numbers.ge(0)
    return segments.loc[keep].copy()
