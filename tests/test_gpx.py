"""Tiny synthetic GPX recordings only. No real hikes or invented training rows."""

from datetime import datetime, timedelta, timezone
import hashlib
from math import degrees
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.data import select_model_ready_segments
from hiking_time_predictor.gpx import (
    EARTH_RADIUS_M, FEET_PER_METER, METERS_PER_MILE, ImportConfig,
    distance_meters, process_hike, read_trackpoints, smooth_elevation,
)
from hiking_time_predictor.import_gpx import (
    find_gpx_files, import_recordings, load_confirmed_rests, load_metadata, main,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def synthetic_gpx(tmp_path):
    def write(tracks, name="synthetic-unit-test.gpx", waypoint=False, route=False):
        """tracks -> segments -> (distance along equator in m, seconds, elevation m)."""
        root = Element("gpx", version="1.1", creator="SYNTHETIC TEST ONLY",
                       xmlns="http://www.topografix.com/GPX/1/1")
        if waypoint:
            SubElement(root, "wpt", lat="40", lon="-100")
        if route:
            SubElement(SubElement(root, "rte"), "rtept", lat="30", lon="-80")
        for segments in tracks:
            track = SubElement(root, "trk")
            for samples in segments:
                segment = SubElement(track, "trkseg")
                for meters, seconds, elevation in samples:
                    point = SubElement(segment, "trkpt", lat="0", lon=str(degrees(meters / EARTH_RADIUS_M)))
                    if elevation is not None:
                        SubElement(point, "ele").text = str(elevation)
                    if seconds is not None:
                        SubElement(point, "time").text = (
                            (START + timedelta(seconds=seconds)).isoformat() if isinstance(seconds, (int, float)) else seconds
                        )
        path = tmp_path / name
        path.write_bytes(tostring(root, encoding="utf-8", xml_declaration=True))
        return path
    return write


def test_parse_tracks_utc_elevation_and_ignore_waypoint_route(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, 105)]]], waypoint=True, route=True)
    points, info = read_trackpoints(path)
    assert len(points) == 2
    assert info["gpx_version"] == "1.1"
    assert info["waypoint_count"] == info["route_count"] == 1
    assert info["track_count"] == info["track_segment_count"] == 1
    assert points.iloc[0].time_utc == pd.Timestamp(START)
    assert points["elevation_m"].tolist() == [100, 105]


def test_waypoints_routes_are_not_recorded_hikes(synthetic_gpx):
    path = synthetic_gpx([], waypoint=True, route=True)
    with pytest.raises(ValueError, match="no recorded trackpoints"):
        read_trackpoints(path)


def test_coordinate_distance_known_values_and_antimeridian():
    assert distance_meters(0, 0, 0, 0) == 0
    assert distance_meters(0, 0, 0, 1) == pytest.approx(111_195.0802, abs=0.001)
    assert distance_meters(0, 179.9, 0, -179.9) == pytest.approx(22_239.016, abs=0.01)
    assert distance_meters(35, -80, 36, -81) == distance_meters(36, -81, 35, -80)


@pytest.mark.parametrize("coordinates", [(91, 0, 0, 0), (0, 181, 0, 0), (0, 0, np.nan, 0)])
def test_invalid_distance_coordinates_rejected(coordinates):
    with pytest.raises(ValueError):
        distance_meters(*coordinates)


def test_elapsed_and_moving_time_conserve_minutes(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, 110), (200, 120, 105)]]])
    summary, segments, points, intervals = process_hike(path, "synthetic")
    assert summary["elapsed_time_min"] == summary["moving_time_min"] == 2
    assert summary["stationary_time_min"] == summary["unknown_time_min"] == 0
    assert summary["distance_miles"] == pytest.approx(200 / METERS_PER_MILE)
    assert summary["raw_elevation_gain_ft"] == pytest.approx(10 * FEET_PER_METER)
    assert summary["raw_elevation_loss_ft"] == pytest.approx(5 * FEET_PER_METER)
    assert segments["elapsed_time_min"].sum() == 2
    assert points.iloc[-1].distance_miles == pytest.approx(200 / METERS_PER_MILE)
    assert set(intervals.state) == {"moving"}


def test_timezone_offsets_normalized_to_utc(synthetic_gpx):
    path = synthetic_gpx([[[(0, "2026-01-01T01:00:00+01:00", 100), (100, 60, 100)]]])
    points, _ = read_trackpoints(path)
    assert points.iloc[0].time_utc == pd.Timestamp(START)


def test_naive_timestamp_is_missing_rather_than_assumed_utc(synthetic_gpx):
    path = synthetic_gpx([[[(0, "2026-01-01T00:00:00", 100), (100, 60, 100)]]])
    points, info = read_trackpoints(path)
    assert pd.isna(points.iloc[0].time_utc)
    assert "naive_timestamp" in points.iloc[0].quality_flags
    assert info["missing_timestamp_count"] == 1


def test_nonfinite_elevation_is_missing(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, float("inf"))]]])
    summary, _, points, _ = process_hike(path, "synthetic")
    assert summary["missing_elevation_count"] == 1
    assert np.isnan(summary["elevation_gain_ft"])
    assert np.isnan(points.iloc[1].elevation_m)


def test_reversed_timestamps_rejected(synthetic_gpx):
    path = synthetic_gpx([[[(0, 60, 100), (100, 0, 100)]]])
    with pytest.raises(ValueError, match="timestamps must not reverse"):
        process_hike(path, "synthetic")


def test_duplicate_timestamp_retained_flagged_and_not_given_speed(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (1, 0, 100), (101, 60, 100)]]])
    summary, segments, points, intervals = process_hike(path, "synthetic")
    assert len(points) == 3 and summary["duplicate_timestamp_count"] == 1
    assert intervals.iloc[0].state == "unknown"
    assert np.isnan(intervals.iloc[0].endpoint_speed_mph)
    assert intervals.iloc[0].elapsed_seconds == 0
    assert summary["distance_miles"] == pytest.approx(100 / METERS_PER_MILE)
    assert "duplicate_timestamp" in segments.iloc[0].quality_flags


def test_missing_timestamp_bracket_is_unknown_and_not_connected(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (50, None, 110), (100, 60, 120)]]])
    summary, segments, points, intervals = process_hike(path, "synthetic")
    assert summary["missing_timestamp_count"] == 1
    assert summary["raw_distance_miles"] == pytest.approx(100 / METERS_PER_MILE)
    assert summary["distance_miles"] == summary["moving_time_min"] == 0
    assert summary["unknown_time_min"] == summary["elapsed_time_min"] == 1
    assert len(intervals) == 1 and intervals.iloc[0].state == "unknown"
    assert "missing_timestamp" in segments.iloc[0].quality_flags
    assert pd.isna(points.iloc[1].time_utc)


def test_missing_edge_timestamps_have_no_invented_duration(synthetic_gpx):
    path = synthetic_gpx([[[(0, None, 100), (50, 60, 100), (100, 120, 100), (150, None, 100)]]])
    summary, segments, _, _ = process_hike(path, "synthetic")
    assert summary["elapsed_time_min"] == segments.elapsed_time_min.sum() == 1
    assert "incomplete_elapsed_time" in summary["quality_flags"]


def test_all_missing_timestamps_export_no_training_segments(synthetic_gpx):
    path = synthetic_gpx([[[(0, None, 100), (100, None, 100)]]])
    summary, segments, _, intervals = process_hike(path, "synthetic")
    assert np.isnan(summary["elapsed_time_min"])
    assert segments.empty and intervals.empty
    assert "insufficient_timestamps" in summary["quality_flags"]


def test_missing_elevation_is_not_zero_or_interpolated(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, None), (200, 120, 110)]]])
    summary, segments, points, _ = process_hike(path, "synthetic")
    assert summary["missing_elevation_count"] == 1
    assert np.isnan(summary["raw_elevation_gain_ft"])
    assert np.isnan(summary["elevation_gain_ft"])
    assert np.isnan(segments.iloc[0].elevation_gain_ft)
    assert np.isnan(points.iloc[1].elevation_m)
    assert np.isnan(points.iloc[1].smoothed_elevation_m)
    assert points.iloc[0].profile_block != points.iloc[2].profile_block


def test_invalid_coordinates_flagged_and_not_bridged(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, 100), (200, 120, 100)]]])
    path.write_text(path.read_text().replace('lat="0"', 'lat="91"', 1))
    summary, _, points, intervals = process_hike(path, "synthetic")
    assert summary["invalid_coordinate_count"] == 1
    assert np.isnan(summary["raw_distance_miles"])
    assert intervals.iloc[0].state == "unknown"
    assert not points.iloc[0].coordinate_valid


def test_stationary_interval_and_gps_drift_cluster(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (0, 30, 100), (0, 60, 100), (100, 120, 100)]]])
    summary, _, _, _ = process_hike(path, "synthetic")
    assert summary["stationary_time_min"] == summary["moving_time_min"] == 1
    jitter = synthetic_gpx([[[(x, i * 10, 100) for i, x in enumerate([0, 7, -7, 7, -7, 7, 0])]]], "synthetic-jitter.gpx")
    summary, _, _, intervals = process_hike(jitter, "synthetic")
    assert summary["stationary_time_min"] == 1
    assert summary["distance_miles"] == 0
    assert summary["raw_distance_miles"] > 0
    assert intervals.state.eq("stationary").all()


@pytest.mark.parametrize("displacement, possible_rest", [(1, True), (500, False)])
def test_long_gaps_remain_unknown_even_when_endpoints_nearby(synthetic_gpx, displacement, possible_rest):
    path = synthetic_gpx([[[(0, 0, 100), (displacement, 600, 120)]]])
    summary, segments, points, intervals = process_hike(path, "synthetic")
    assert summary["unknown_time_min"] == summary["elapsed_time_min"] == 10
    assert summary["moving_time_min"] == summary["stationary_time_min"] == 0
    assert summary["distance_miles"] == 0
    assert summary["recording_gap_count"] == 1
    assert ("possible_rest_during_gap" in intervals.iloc[0].quality_flags) == possible_rest
    assert len(segments) == 1  # Auditable zero-distance unknown-time row.
    assert points.iloc[0].profile_block != points.iloc[1].profile_block


def test_thresholds_change_decision_explicitly(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 180, 100)]]])
    assert process_hike(path, "synthetic")[0]["unknown_time_min"] == 3
    assert process_hike(path, "synthetic", config=ImportConfig(max_gap_seconds=200))[0]["moving_time_min"] == 3


def test_very_slow_progress_is_ambiguous(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (12, 120, 100)]]])
    summary, _, _, intervals = process_hike(path, "synthetic")
    assert summary["unknown_time_min"] == 2
    assert "ambiguous_slow_movement" in intervals.iloc[0].quality_flags


def test_implausible_speed_excludes_gps_jump(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (1000, 1, 100)]]])
    summary, _, _, intervals = process_hike(path, "synthetic")
    assert summary["distance_miles"] == 0
    assert "implausible_speed" in intervals.iloc[0].quality_flags


def test_elevation_median_removes_spike_keeps_input():
    x = np.arange(0, 105, 5, dtype=float)
    y = np.full(len(x), 100.0)
    y[10] = 150
    original = y.copy()
    smoothed = smooth_elevation(x, y, ImportConfig())
    np.testing.assert_allclose(smoothed, 100)
    np.testing.assert_array_equal(y, original)


def test_elevation_trend_and_duplicate_distances():
    x = np.arange(0, 105, 5, dtype=float)
    y = x * 0.1 + 100
    smoothed = smooth_elevation(x, y, ImportConfig())
    np.testing.assert_allclose(smoothed[3:-3], y[3:-3])
    assert (np.diff(smoothed) >= 0).all()
    np.testing.assert_allclose(smooth_elevation([0, 0, 0], [100, 110, 105], ImportConfig()), 105)


def test_half_mile_boundaries_interpolate_and_conserve(synthetic_gpx):
    distance = 1.2 * METERS_PER_MILE
    path = synthetic_gpx([[[(0, 0, 100), (distance, 1200, 200)]]])
    summary, segments, _, _ = process_hike(path, "synthetic", config=ImportConfig(max_gap_seconds=1500))
    np.testing.assert_allclose(segments.distance_miles, [0.5, 0.5, 0.2])
    np.testing.assert_allclose(segments.moving_time_min, [20 * 0.5 / 1.2, 20 * 0.5 / 1.2, 20 * 0.2 / 1.2])
    assert segments.segment_index.tolist() == [0, 1, 2]
    assert segments.is_partial_segment.tolist() == [False, False, True]
    assert segments.iloc[0].end_time_utc == START + timedelta(seconds=500)
    assert segments.iloc[1].start_time_utc == segments.iloc[0].end_time_utc
    assert segments.elevation_gain_ft.sum() == pytest.approx(summary["elevation_gain_ft"])
    assert segments.elapsed_time_min.sum() == summary["elapsed_time_min"]


def test_gap_is_never_split_or_given_interpolated_movement(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (700, 600, 110), (1700, 2000, 200), (1900, 2200, 210)]]])
    summary, segments, _, intervals = process_hike(path, "synthetic", config=ImportConfig(max_gap_seconds=900))
    assert summary["distance_miles"] == pytest.approx(900 / METERS_PER_MILE)
    assert segments.unknown_time_min.tolist() == [1400 / 60, 0]
    assert intervals.iloc[1].state == "unknown"
    assert not any(START + timedelta(seconds=600) < t < START + timedelta(seconds=2000)
                   for t in segments.end_time_utc)
    assert segments.elapsed_time_min.sum() == pytest.approx(summary["elapsed_time_min"])


@pytest.mark.parametrize("stop_duration", [60, 1800])
def test_exact_half_mile_with_trailing_stop_has_no_empty_final_segment(synthetic_gpx, stop_duration):
    distance = 0.5 * METERS_PER_MILE
    path = synthetic_gpx([[[(0, 0, 100), (distance, 600, 110), (distance, 600 + stop_duration, 110)]]])
    summary, segments, _, _ = process_hike(path, "synthetic", config=ImportConfig(max_gap_seconds=900))
    assert len(segments) == 1
    assert not segments.iloc[0].is_partial_segment
    assert segments.iloc[0].distance_miles == pytest.approx(0.5)
    assert summary["elapsed_time_min"] == segments.elapsed_time_min.sum() == (600 + stop_duration) / 60


@pytest.mark.parametrize("tracks", [
    [[[(0, 0, 100), (100, 60, 110)], [(10000, 120, 500), (10100, 180, 510)]]],
    [[[(0, 0, 100), (100, 60, 110)]], [[(10000, 120, 500), (10100, 180, 510)]]],
])
def test_multiple_tracks_segments_never_connect_disconnected_points(synthetic_gpx, tracks):
    path = synthetic_gpx(tracks)
    summary, segments, points, intervals = process_hike(path, "synthetic")
    assert summary["raw_distance_miles"] == pytest.approx(200 / METERS_PER_MILE)
    assert summary["raw_elevation_gain_ft"] == pytest.approx(20 * FEET_PER_METER)
    assert summary["distance_miles"] == pytest.approx(200 / METERS_PER_MILE)
    assert summary["unknown_time_min"] == 1
    assert "track_boundary" in segments.iloc[0].quality_flags
    assert intervals.iloc[1].state == "unknown"
    assert points.iloc[1].profile_block != points.iloc[2].profile_block


def test_multiple_files_metadata_unique_ids_and_raw_files_preserved(synthetic_gpx, tmp_path):
    a = synthetic_gpx([[[(0, 0, 100), (100, 60, 100)]]], "Synthetic A.gpx")
    b = synthetic_gpx([[[(0, 0, 100), (100, 60, 100)]]], "Synthetic_A.gpx-2")
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a, b]}
    metadata_path = tmp_path / "metadata.csv"
    summary, segments = import_recordings([a, b], metadata_path, tmp_path.parent / (tmp_path.name + "-processed"))
    assert summary.hike_id.is_unique and len(summary) == 2
    assert len(segments) == 2 and segments.hike_type.isna().all()
    table = pd.read_csv(metadata_path, dtype={"hike_type": "string"})
    assert table.hike_type.isna().all() and table.pack_weight_lbs.isna().all()
    assert select_model_ready_segments(segments).empty
    for path, digest in original.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    table.loc[0, "hike_type"] = "backpacking"
    table.loc[0, "pack_weight_lbs"] = 20
    table.to_csv(metadata_path, index=False)
    updated = load_metadata(metadata_path, [a, b])
    assert updated.hike_id.tolist() == summary.hike_id.tolist()
    assert updated.iloc[0].hike_type == "backpacking"
    assert updated.iloc[0].pack_weight_lbs == "20.0"


def test_filename_id_collision_gets_unique_stable_suffix(synthetic_gpx, tmp_path):
    a = synthetic_gpx([[[(0, 0, 100)]]], "Synthetic A.gpx")
    b = synthetic_gpx([[[(0, 0, 100)]]], "Synthetic_A.gpx")
    metadata = load_metadata(tmp_path / "metadata.csv", [a, b])
    assert metadata.hike_id.is_unique


@pytest.mark.parametrize("column, value, match", [
    ("hike_id", "", "hike_id must"), ("hike_type", "guessed", "hike_type must"),
    ("pack_weight_lbs", "heavy", "pack_weight_lbs must"), ("pack_weight_lbs", "-1", "pack_weight_lbs must"),
])
def test_invalid_metadata_rejected(synthetic_gpx, tmp_path, column, value, match):
    file = synthetic_gpx([[[(0, 0, 100)]]])
    table = load_metadata(tmp_path / "metadata.csv", [file])
    table.loc[0, column] = value
    table.to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError, match=match):
        load_metadata(tmp_path / "metadata.csv", [file])


def test_duplicate_metadata_ids_rejected(synthetic_gpx, tmp_path):
    a = synthetic_gpx([[[(0, 0, 100)]]], "synthetic-a.gpx")
    b = synthetic_gpx([[[(0, 0, 100)]]], "synthetic-b.gpx")
    table = load_metadata(tmp_path / "metadata.csv", [a, b])
    table.loc[1, "hike_id"] = table.loc[0, "hike_id"]
    table.to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="hike_id must"):
        load_metadata(tmp_path / "metadata.csv", [a, b])


def test_discovery_includes_numbered_downloads_only(synthetic_gpx, tmp_path):
    a = synthetic_gpx([[[(0, 0, 100)]]], "synthetic-a.gpx")
    b = synthetic_gpx([[[(0, 0, 100)]]], "synthetic-b.gpx-2")
    (tmp_path / "other.csv").write_text("not a GPX recording")
    assert find_gpx_files(tmp_path) == [a, b]


def test_optional_pack_weight_filter_and_flagged_rows(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, 110)]]])
    _, segments, _, _ = process_hike(path, "synthetic", {"hike_type": "day_hike"})
    assert len(select_model_ready_segments(segments)) == 1
    assert select_model_ready_segments(segments, require_pack_weight=True).empty
    segments["pack_weight_lbs"] = 0
    assert len(select_model_ready_segments(segments, require_pack_weight=True)) == 1
    segments["quality_flags"] = "recording_gap"
    assert select_model_ready_segments(segments).empty


def test_failed_batch_preserves_existing_outputs(synthetic_gpx, tmp_path):
    good = synthetic_gpx([[[(0, 0, 100), (100, 60, 100)]]])
    bad = tmp_path / "synthetic-invalid.gpx"
    bad.write_text("invalid XML")
    out = tmp_path.parent / (tmp_path.name + "-processed")
    out.mkdir()
    existing = out / "hiking_segments.csv"
    existing.write_text("existing user data\n")
    with pytest.raises(ValueError, match="Could not import"):
        import_recordings([good, bad], tmp_path / "metadata.csv", out)
    assert existing.read_text() == "existing user data\n"
    assert not (tmp_path / "metadata.csv").exists()


def test_output_cannot_overwrite_raw_folder(synthetic_gpx, tmp_path):
    file = synthetic_gpx([[[(0, 0, 100)]]])
    with pytest.raises(ValueError, match="outside the raw"):
        import_recordings([file], tmp_path / "metadata.csv", tmp_path)


@pytest.mark.parametrize("options", [{"segment_miles": 0}, {"max_gap_seconds": -1},
                                      {"elevation_window_m": np.nan}, {"max_speed_mph": 0.1}])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        ImportConfig(**options)


def test_cli_help_documents_entrypoint(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_confirmed_rest_moves_unknown_time_without_reconnecting_gap(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (100, 60, 110), (500, 660, 200), (600, 720, 210)]]])
    baseline, old_segments, old_points, old_intervals = process_hike(path, "synthetic")
    rests = pd.DataFrame([{
        "start_time_utc": pd.Timestamp(START + timedelta(seconds=60)),
        "end_time_utc": pd.Timestamp(START + timedelta(seconds=660)),
        "notes": "Synthetic hiker confirmation for test only.",
    }])
    summary, segments, points, intervals = process_hike(path, "synthetic", confirmed_rests=rests)
    assert summary["unknown_time_min"] == 0
    assert summary["stationary_time_min"] == 10
    assert summary["confirmed_rest_time_min"] == summary["detected_unknown_time_min"] == 10
    assert summary["confirmed_rest_count"] == 1
    for column in ["distance_miles", "moving_time_min", "elapsed_time_min", "elevation_gain_ft", "elevation_loss_ft"]:
        assert summary[column] == baseline[column]
    pd.testing.assert_series_equal(points.profile_block, old_points.profile_block)
    pd.testing.assert_series_equal(points.smoothed_elevation_m, old_points.smoothed_elevation_m)
    rest = intervals.iloc[1]
    assert rest.state == "stationary" and rest.detected_state == "unknown"
    assert rest.state_source == "user_confirmed" and rest.distance_m == 0
    assert rest.review_notes == rests.iloc[0].notes
    assert "recording_gap" in rest.quality_flags and "user_confirmed_rest" in rest.quality_flags
    assert "unknown_interval" not in segments.iloc[0].quality_flags
    assert old_intervals.iloc[1].state == "unknown"
    assert old_segments.iloc[0].unknown_time_min == 10


def test_rest_confirmation_only_applies_to_matching_file(synthetic_gpx, tmp_path):
    a = synthetic_gpx([[[(0, 0, 100), (1, 600, 100)]]], "synthetic-a.gpx")
    b = synthetic_gpx([[[(0, 0, 100), (1, 600, 100)]]], "synthetic-b.gpx")
    rests_path = tmp_path / "confirmed_rests.csv"
    pd.DataFrame([{
        "source_file": a.name, "start_time_utc": START.isoformat(),
        "end_time_utc": (START + timedelta(seconds=600)).isoformat(), "notes": "Synthetic confirmation.",
    }]).to_csv(rests_path, index=False)
    summary, _ = import_recordings([a, b], tmp_path / "metadata.csv", tmp_path.parent / (tmp_path.name + "-processed"))
    a_row = summary.loc[summary.source_file.eq(a.name)].iloc[0]
    b_row = summary.loc[summary.source_file.eq(b.name)].iloc[0]
    assert a_row.stationary_time_min == 10 and a_row.unknown_time_min == 0
    assert b_row.unknown_time_min == 10 and b_row.stationary_time_min == 0
    assert process_hike(b, "future-synthetic")[0]["unknown_time_min"] == 10


def test_stale_rest_endpoints_rejected(synthetic_gpx):
    path = synthetic_gpx([[[(0, 0, 100), (1, 600, 100)]]])
    rests = pd.DataFrame([{
        "start_time_utc": pd.Timestamp(START), "end_time_utc": pd.Timestamp(START + timedelta(seconds=601)),
        "notes": "Synthetic stale endpoint.",
    }])
    with pytest.raises(ValueError, match="match exactly one"):
        process_hike(path, "synthetic", confirmed_rests=rests)


@pytest.mark.parametrize("change", ["naive_time", "reversed_time", "duplicate"])
def test_invalid_rest_confirmation_table_rejected(tmp_path, change):
    row = {"source_file": "synthetic.gpx", "start_time_utc": START.isoformat(),
           "end_time_utc": (START + timedelta(seconds=60)).isoformat(), "notes": "Synthetic test only."}
    if change == "naive_time":
        row["start_time_utc"] = "2026-01-01T00:00:00"
    elif change == "reversed_time":
        row["end_time_utc"] = (START - timedelta(seconds=1)).isoformat()
    rows = [row, row] if change == "duplicate" else [row]
    path = tmp_path / "confirmed_rests.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_confirmed_rests(path)
