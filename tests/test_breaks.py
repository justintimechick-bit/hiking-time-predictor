"""Synthetic recordings verify break labels, persistence, and time accounting."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import degrees
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.breaks import (
    BreakConfig, CORRECTION_COLUMNS, NON_MOVING_TARGET, apply_break_corrections, extract_break_events,
    load_corrections, save_corrections, summarize_breaks,
)
from hiking_time_predictor.gpx import EARTH_RADIUS_M, process_hike
from hiking_time_predictor.import_gpx import REST_COLUMNS


@pytest.fixture
def recording(tmp_path):
    def write(samples):
        root = Element("gpx", version="1.1", creator="synthetic-break-test")
        segment = SubElement(SubElement(root, "trk"), "trkseg")
        for meters, seconds in samples:
            point = SubElement(segment, "trkpt", lat="0", lon=str(degrees(meters / EARTH_RADIUS_M)))
            SubElement(point, "ele").text = "100"
            if seconds is not None:
                SubElement(point, "time").text = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()
        path = tmp_path / "synthetic.gpx"
        path.write_bytes(tostring(root))
        return path
    return write


def extract(path, rests=None, config=None):
    summary, segments, points, intervals = process_hike(path, "synthetic", confirmed_rests=rests)
    events = extract_break_events(points, intervals, config)
    hikes = summarize_breaks(pd.DataFrame([summary]), events, config)
    return summary, segments, events, hikes


def test_rest_duration_position_and_time_accounting(recording):
    path = recording([(0, 0), (100, 60), (100, 120), (100, 180), (100, 240), (200, 300)])
    summary, segments, events, hikes = extract(path)
    rests = events.loc[events.break_classification.eq("stationary_break")]
    assert len(rests) == 1
    assert rests.duration_minutes.iloc[0] == 3
    assert rests.distance_into_hike_miles.iloc[0] == pytest.approx(100 / 1609.344)
    assert rests.cumulative_elevation_gain_ft.iloc[0] == 0
    assert hikes.total_break_minutes.iloc[0] == 3
    assert summary["moving_time_min"] == 2
    assert events.duration_minutes.sum() == summary["elapsed_time_min"]
    assert segments.moving_time_min.sum() + rests.duration_minutes.sum() == summary["elapsed_time_min"]
    assert events.loc[events.break_classification.eq("stationary_break"), "moving_minutes"].eq(0).all()


@pytest.mark.parametrize("samples", [
    [(0, 0), (100, 60), (100, 2760), (200, 2820)],  # 45-minute recording gap
    [(0, 0), (100, 60), (100, None), (100, 240), (200, 300)],
])
def test_unconfirmed_gaps_and_missing_timestamps_are_unknown(recording, samples):
    _, _, events, hikes = extract(recording(samples))
    assert not events.break_classification.eq("stationary_break").any()
    assert events.break_classification.eq("unknown").any()
    assert hikes.total_break_minutes.iloc[0] == 0
    assert not hikes.training_eligible.iloc[0]


def test_existing_hiker_confirmation_identifies_rest_in_gap(recording):
    path = recording([(0, 0), (100, 60), (100, 660), (200, 720)])
    _, _, _, intervals = process_hike(path, "synthetic")
    gap = intervals.loc[intervals.state.eq("unknown")].iloc[0]
    rests = pd.DataFrame([[path.name, gap.start_time_utc, gap.end_time_utc, "explicit synthetic confirmation"]], columns=REST_COLUMNS)
    _, _, events, hikes = extract(path, rests)
    identified = events.loc[events.break_classification.eq("stationary_break")]
    assert identified.duration_minutes.sum() == 10
    assert "hiker_confirmed" in identified.quality_flag.iloc[0]
    assert hikes.unresolved_minutes.iloc[0] == 0


def test_confirmed_gap_does_not_vouch_for_neighboring_drift_or_lose_its_label(recording):
    path = recording([(0, 0), (100, 60)] + [(100 + t, 60 + t * 60) for t in range(1, 25)]
                     + [(124, 2100), (224, 2160)])
    _, _, _, intervals = process_hike(path, "synthetic")
    gap = intervals.loc[intervals.state.eq("unknown")].iloc[0]
    rests = pd.DataFrame([[path.name, gap.start_time_utc, gap.end_time_utc, "confirmed synthetic rest"]], columns=REST_COLUMNS)
    _, _, events, _ = extract(path, rests)
    assert events.break_classification.eq("unknown").any()
    assert events.loc[events.break_classification.eq("stationary_break"), "duration_minutes"].sum() == 10


def test_overnight_and_extended_stationary_periods_are_excluded(recording):
    samples = [(0, 0), (100, 60)] + [(100, t) for t in range(120, 60 + 8 * 3600 + 1, 60)] + [(200, 60 + 8 * 3600 + 60)]
    _, _, events, hikes = extract(recording(samples))
    assert events.break_classification.eq("camping").sum() == 1
    assert hikes.total_break_minutes.iloc[0] == 0
    assert hikes.excluded_minutes.iloc[0] == 480
    assert not hikes.training_eligible.iloc[0]
    samples = [(0, 0), (100, 60)] + [(100, t) for t in range(120, 60 + 90 * 60 + 1, 60)] + [(200, 5520)]
    _, _, events, hikes = extract(recording(samples))
    assert events.break_classification.eq("unknown").any()
    assert hikes.total_break_minutes.iloc[0] == 0


def test_short_stops_are_separate_and_thresholds_adjustable(recording):
    path = recording([(0, 0), (100, 60), (100, 120), (200, 180)])
    _, _, events, hikes = extract(path)
    assert events.break_classification.eq("short_stop").sum() == 1
    assert hikes.short_stop_minutes.iloc[0] == 1
    _, _, _, hikes = extract(path, config=replace(BreakConfig(), min_break_minutes=0.5))
    assert hikes.total_break_minutes.iloc[0] == 1


def test_gradual_geographic_drift_is_ambiguous(recording):
    path = recording([(0, 0)] + [(t, t * 60) for t in range(1, 25)])
    _, _, events, hikes = extract(path)
    assert events.break_classification.eq("unknown").all()
    assert events.quality_flag.str.contains("stationary_cluster_drift").all()
    assert hikes.total_break_minutes.iloc[0] == 0


def test_corrections_survive_reprocessing_and_stale_endpoints_are_not_dropped(recording, tmp_path):
    path = recording([(0, 0), (100, 60), (100, 660), (200, 720)])
    _, _, events, _ = extract(path)
    edited = events.loc[events.break_classification.eq("unknown")].copy()
    edited["break_classification"] = "stationary_break"
    edited["correction_notes"] = "confirmed lunch stop in synthetic test"
    correction_path = tmp_path / "break_corrections.csv"
    assert save_corrections(events, edited, correction_path) == 1
    before = correction_path.read_bytes()
    _, _, regenerated, _ = extract(path)
    corrected = apply_break_corrections(regenerated, load_corrections(correction_path))
    assert corrected.loc[corrected.break_classification.eq("stationary_break"), "duration_minutes"].sum() == 10
    assert correction_path.read_bytes() == before
    assert corrected.break_id.tolist() == events.break_id.tolist()
    stale = load_corrections(correction_path)
    stale["end_time"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Stale"):
        apply_break_corrections(regenerated, stale)


def test_correction_cannot_double_count_moving_time(recording):
    _, _, events, _ = extract(recording([(0, 0), (100, 60)]))
    first = events.iloc[0]
    corrections = pd.DataFrame([[first.hike_id, first.start_time, first.end_time, "stationary_break", "test"]], columns=CORRECTION_COLUMNS)
    with pytest.raises(ValueError, match="cannot also count"):
        apply_break_corrections(events, corrections)


def test_per_hike_summary_includes_zero_break_hikes(recording):
    _, _, _, hikes = extract(recording([(0, 0), (100, 60), (200, 120)]))
    assert hikes.break_count.iloc[0] == 0
    assert hikes.total_break_minutes.iloc[0] == 0
    assert hikes.training_eligible.iloc[0]


@pytest.mark.parametrize("samples,non_moving", [
    ([(0, 0), (100, 60), (100, 120), (200, 180)], 1),
    ([(0, 0), (100, 60)] + [(100 + t, 60 + t * 60) for t in range(1, 25)]
     + [(224, 1560)], 24),
    ([(0, 0), (100, 60), (100, 660), (200, 720)], 10),
])
def test_non_moving_allowance_includes_short_pauses_drift_and_gaps(recording, samples, non_moving):
    summary, _, points, intervals = process_hike(recording(samples), "synthetic")
    events = extract_break_events(points, intervals)
    original = events.copy(deep=True)
    hikes = summarize_breaks(pd.DataFrame([summary]), events, target=NON_MOVING_TARGET)
    assert hikes.total_break_minutes.iloc[0] == pytest.approx(non_moving)
    assert hikes.training_eligible.iloc[0]
    assert hikes.target_definition.iloc[0] == NON_MOVING_TARGET
    # An estimated allowance never rewrites unknown intervals as confirmed rests.
    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize("elapsed,moving", [(np.nan, 2), (np.inf, 2), (1, 2), (0, 2), (3, np.nan)])
def test_non_moving_allowance_rejects_invalid_time_totals(recording, elapsed, moving):
    summary, _, points, intervals = process_hike(recording([(0, 0), (100, 60), (200, 120)]), "synthetic")
    summary.update(elapsed_time_min=elapsed, moving_time_min=moving)
    hikes = summarize_breaks(pd.DataFrame([summary]), extract_break_events(points, intervals), target=NON_MOVING_TARGET)
    assert not hikes.training_eligible.iloc[0]


def test_non_moving_allowance_still_excludes_incomplete_and_non_hiking_recordings(recording):
    path = recording([(0, 0), (100, 60), (100, 120), (100, 180), (100, 240), (200, 300)])
    summary, _, points, intervals = process_hike(path, "synthetic")
    events = extract_break_events(points, intervals)
    reviewed = events.copy()
    reviewed.loc[reviewed.break_classification.eq("stationary_break"), "break_classification"] = "non_hiking"
    excluded = summarize_breaks(pd.DataFrame([summary]), reviewed, target=NON_MOVING_TARGET)
    assert not excluded.training_eligible.iloc[0]
    assert "camping_or_non_hiking_activity" in excluded.eligibility_notes.iloc[0]
    summary["quality_flags"] = "missing_timestamp;incomplete_elapsed_time"
    incomplete = summarize_breaks(pd.DataFrame([summary]), events, target=NON_MOVING_TARGET)
    assert not incomplete.training_eligible.iloc[0]
    assert "incomplete_timestamps" in incomplete.eligibility_notes.iloc[0]


@pytest.mark.parametrize("values", [{"min_break_minutes": -1}, {"max_break_minutes": 1}, {"camping_minutes": 30},
                                         {"geographic_jitter_m": np.nan}, {"minimum_evaluation_hikes": 1},
                                         {"minimum_selection_hikes": 2}, {"max_unresolved_fraction": 2}])
def test_invalid_break_config(values):
    with pytest.raises(ValueError):
        BreakConfig(**values)
