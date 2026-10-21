"""Small synthetic fixtures test the rules; they are not personal hiking history."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.data import DATA_COLUMNS, load_hiking_data, validate_hiking_data


@pytest.fixture
def valid_hikes():
    # Artificial test inputs only. Never add these rows to the real dataset.
    return pd.DataFrame(
        [
            ["unit-test-001", 2.0, 300.0, 250.0, "day_hike", 60.0],
            ["unit-test-002", 4.0, 0.0, 0.0, "backpacking", 150.0],
        ],
        columns=DATA_COLUMNS,
    )


def test_valid_hikes_pass_without_modifying_input(valid_hikes):
    original = valid_hikes.copy(deep=True)
    validate_hiking_data(valid_hikes)
    pd.testing.assert_frame_equal(valid_hikes, original)


def test_real_template_contains_headers_only():
    template = Path(__file__).resolve().parents[1] / "data" / "raw" / "hikes_template.csv"
    hikes = load_hiking_data(template)
    assert hikes.empty
    assert list(hikes.columns) == DATA_COLUMNS


def test_empty_data_can_be_rejected_when_observations_are_required():
    with pytest.raises(ValueError, match="At least one hike"):
        validate_hiking_data(pd.DataFrame(columns=DATA_COLUMNS), allow_empty=False)


def test_missing_column_is_rejected(valid_hikes):
    with pytest.raises(ValueError, match="Missing.*elevation_loss_ft"):
        validate_hiking_data(valid_hikes.drop(columns="elevation_loss_ft"))


def test_unexpected_column_is_rejected(valid_hikes):
    valid_hikes["extra_column"] = "unexpected"
    with pytest.raises(ValueError, match="Unexpected.*extra_column"):
        validate_hiking_data(valid_hikes)


def test_duplicate_columns_are_rejected(valid_hikes):
    valid_hikes.columns = ["hike_id", *DATA_COLUMNS[1:-1], "hike_id"]
    with pytest.raises(ValueError, match="Duplicate column"):
        validate_hiking_data(valid_hikes)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("hike_id", "  ", "hike_id must not be blank"),
        ("hike_id", " unit-test-002 ", "hike_id must be unique"),
        ("hike_type", "Day Hiking", "hike_type must be"),
        ("hike_type", None, "Missing values"),
        ("distance_miles", 0, "distance_miles must be greater than zero"),
        ("distance_miles", -1, "distance_miles must be greater than zero"),
        ("moving_time_min", 0, "moving_time_min must be greater than zero"),
        ("moving_time_min", -1, "moving_time_min must be greater than zero"),
        ("elevation_gain_ft", -1, "elevation_gain_ft must be zero or greater"),
        ("elevation_loss_ft", -1, "elevation_loss_ft must be zero or greater"),
        ("moving_time_min", "unknown", "moving_time_min must contain finite numeric"),
        ("distance_miles", np.nan, "Missing values"),
        ("elevation_gain_ft", np.inf, "elevation_gain_ft must contain finite numeric"),
        ("elevation_loss_ft", -np.inf, "elevation_loss_ft must contain finite numeric"),
        ("distance_miles", True, "distance_miles must contain finite numeric"),
    ],
)
def test_invalid_values_are_rejected(valid_hikes, column, value, message):
    # Object dtype lets these tests insert bad text into an otherwise numeric column.
    invalid_hikes = valid_hikes.astype(object)
    invalid_hikes.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_hiking_data(invalid_hikes)


def test_csv_round_trip_preserves_text_ids_and_numeric_values(valid_hikes, tmp_path):
    valid_hikes.loc[0, "hike_id"] = "001"
    csv_path = tmp_path / "synthetic_hikes.csv"
    valid_hikes.to_csv(csv_path, index=False)

    loaded = load_hiking_data(csv_path)

    assert loaded.loc[0, "hike_id"] == "001"
    assert loaded.loc[1, "hike_type"] == "backpacking"
    np.testing.assert_allclose(loaded[DATA_COLUMNS[1:4]], valid_hikes[DATA_COLUMNS[1:4]])
    np.testing.assert_allclose(loaded["moving_time_min"], valid_hikes["moving_time_min"])


def test_csv_with_bad_data_is_rejected(valid_hikes, tmp_path):
    valid_hikes.loc[0, "moving_time_min"] = -5
    csv_path = tmp_path / "invalid_synthetic_hikes.csv"
    valid_hikes.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="moving_time_min must be greater than zero"):
        load_hiking_data(csv_path)
