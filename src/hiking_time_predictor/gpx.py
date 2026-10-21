"""Turn recorded trackpoints into auditable measurements, without training a model.

The tables deliberately keep raw values, interval decisions, and quality flags.
See the README for the assumptions behind the configurable heuristic.
"""

from dataclasses import asdict, dataclass
from datetime import timezone
from math import atan2, cos, isfinite, radians, sin, sqrt
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd


METERS_PER_MILE = 1609.344
FEET_PER_METER = 1 / 0.3048
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class ImportConfig:
    """Units are in the names. Defaults are estimates to review, not ground truth."""

    segment_miles: float = 0.5
    max_gap_seconds: float = 120.0
    stationary_speed_mph: float = 0.3
    stationary_radius_m: float = 10.0
    rest_window_seconds: float = 60.0
    max_speed_mph: float = 8.0
    elevation_window_m: float = 30.0
    elevation_grid_m: float = 5.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than zero.")
        if self.max_speed_mph <= self.stationary_speed_mph:
            raise ValueError("max_speed_mph must exceed stationary_speed_mph.")
        if self.elevation_grid_m > self.elevation_window_m:
            raise ValueError("elevation_grid_m must not exceed elevation_window_m.")


def distance_meters(lat1, lon1, lat2, lon2):
    """Horizontal great-circle distance using the haversine formula.

    Elevation is excluded so noisy altitude does not inflate trail distance.
    This spherical-earth approximation is adequate for short hiking intervals.
    """
    for latitude in (lat1, lat2):
        if not isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("Latitude must be finite and between -90 and 90.")
    for longitude in (lon1, lon2):
        if not isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("Longitude must be finite and between -180 and 180.")
    a1, a2 = radians(lat1), radians(lat2)
    dlat, dlon = a2 - a1, radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(a1) * cos(a2) * sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))  # Protect sqrt from floating-point roundoff.
    return 2 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(1 - a))


def combine_flags(*values):
    """Store flags as sorted semicolon-separated text, readable in a CSV."""
    return ";".join(sorted({flag for value in values if value for flag in value.split(";")}))


def read_trackpoints(path: str | Path):
    """Read tracks only; never use route points or waypoints as hiking history.

    Missing/invalid measurements remain missing. Reversed known timestamps are
    rejected. Duplicate timestamps are flagged as zero-duration unknown edges.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as recording:
        gpx = gpxpy.parse(recording)
    rows = []
    for track_index, track in enumerate(gpx.tracks):
        for track_segment_index, segment in enumerate(track.segments):
            for point in segment.points:
                flags = []
                lat, lon = point.latitude, point.longitude
                coordinate_valid = (
                    lat is not None and lon is not None
                    and isfinite(lat) and isfinite(lon)
                    and -90 <= lat <= 90 and -180 <= lon <= 180
                )
                if not coordinate_valid:
                    flags.append("invalid_coordinate")
                elevation = point.elevation
                if elevation is None or not isfinite(elevation):
                    elevation = np.nan
                    flags.append("missing_elevation")
                timestamp = point.time
                if timestamp is None or timestamp.utcoffset() is None:
                    if timestamp is not None:
                        flags.append("naive_timestamp")
                    timestamp = None
                    flags.append("missing_timestamp")
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                rows.append({
                    "point_index": len(rows),
                    "track_index": track_index,
                    "track_segment_index": track_segment_index,
                    "latitude": lat,
                    "longitude": lon,
                    "coordinate_valid": coordinate_valid,
                    "elevation_m": elevation,
                    "time_utc": timestamp,
                    "quality_flags": ";".join(flags),
                })
    if not rows:
        raise ValueError(f"{path.name}: no recorded trackpoints (routes/waypoints are ignored).")
    points = pd.DataFrame(rows)
    points["time_utc"] = pd.to_datetime(points["time_utc"], utc=True)
    known_times = points["time_utc"].dropna()
    differences = known_times.diff().dt.total_seconds()
    if differences.lt(0).any():
        raise ValueError(f"{path.name}: timestamps must not reverse in recording order.")
    for index in differences.index[differences.eq(0)]:
        points.at[index, "quality_flags"] = combine_flags(points.at[index, "quality_flags"], "duplicate_timestamp")
    info = {
        "source_file": path.name,
        "gpx_version": gpx.version,
        "track_count": len(gpx.tracks),
        "track_segment_count": sum(len(t.segments) for t in gpx.tracks),
        "trackpoint_count": len(points),
        "waypoint_count": len(gpx.waypoints),
        "route_count": len(gpx.routes),
        "missing_timestamp_count": int(points["time_utc"].isna().sum()),
        "duplicate_timestamp_count": int(differences.eq(0).sum()),
        "missing_elevation_count": int(points["elevation_m"].isna().sum()),
        "invalid_coordinate_count": int((~points["coordinate_valid"]).sum()),
    }
    return points, info


def same_recorded_segment(a, b):
    return (a.track_index, a.track_segment_index) == (b.track_index, b.track_segment_index)


def classify_intervals(points: pd.DataFrame, config: ImportConfig):
    """Partition known elapsed time. A gap is never automatically a rest stop.

    Consecutive *known* timestamps bracket missing timestamps as one unknown
    interval. No geographic connection is made across that bracket. Untimed
    beginning/end points are retained but have no invented durations.
    """
    rows = []
    known_indices = points.index[points["time_utc"].notna()].tolist()
    for start, end in zip(known_indices, known_indices[1:]):
        a, b = points.loc[start], points.loc[end]
        seconds = (b.time_utc - a.time_utc).total_seconds()
        flags = combine_flags(*points.loc[start:end, "quality_flags"])
        distance = np.nan
        consecutive = end == start + 1
        connected = consecutive and same_recorded_segment(a, b)
        if not consecutive:
            flags = combine_flags(flags, "missing_timestamp")
        if not same_recorded_segment(a, b):
            flags = combine_flags(flags, "track_boundary")
        if connected and a.coordinate_valid and b.coordinate_valid:
            distance = distance_meters(a.latitude, a.longitude, b.latitude, b.longitude)
        speed = distance / METERS_PER_MILE / (seconds / 3600) if seconds > 0 else np.nan
        state = "unknown"
        if seconds == 0:
            flags = combine_flags(flags, "duplicate_timestamp")
        elif seconds > config.max_gap_seconds:
            flags = combine_flags(flags, "recording_gap")
            if isfinite(distance) and distance <= config.stationary_radius_m:
                flags = combine_flags(flags, "possible_rest_during_gap")
        elif not isfinite(distance):
            pass  # Missing coordinates/time or a track boundary: no connection.
        elif speed > config.max_speed_mph:
            flags = combine_flags(flags, "implausible_speed")
        elif speed <= config.stationary_speed_mph:
            if distance <= config.stationary_radius_m:
                state = "stationary"
            else:
                flags = combine_flags(flags, "ambiguous_slow_movement")
        else:
            state = "moving"
        rows.append({
            "start_point_index": start,
            "end_point_index": end,
            "start_time_utc": a.time_utc,
            "end_time_utc": b.time_utc,
            "elapsed_seconds": seconds,
            "endpoint_distance_m": distance,
            "endpoint_speed_mph": speed,
            "state": state,
            "quality_flags": flags,
        })
    columns = ["start_point_index", "end_point_index", "start_time_utc", "end_time_utc",
               "elapsed_seconds", "endpoint_distance_m", "endpoint_speed_mph", "state", "quality_flags"]
    intervals = pd.DataFrame(rows, columns=columns)

    # A walker resting may produce short, fast-looking GPS jitter. Also detect
    # sustained clusters: all points within a radius of the first point for at
    # least rest_window_seconds. Never look through an unknown interval.
    i = 0
    while i < len(intervals):
        if intervals.at[i, "state"] == "unknown":
            i += 1
            continue
        anchor = points.loc[intervals.at[i, "start_point_index"]]
        j = i
        duration = 0.0
        while j < len(intervals) and intervals.at[j, "state"] != "unknown":
            p = points.loc[intervals.at[j, "end_point_index"]]
            if distance_meters(anchor.latitude, anchor.longitude, p.latitude, p.longitude) > config.stationary_radius_m:
                break
            duration += intervals.at[j, "elapsed_seconds"]
            j += 1
        if duration >= config.rest_window_seconds:
            intervals.loc[i:j - 1, "state"] = "stationary"
            i = j
        else:
            i += 1
    intervals["distance_m"] = intervals["endpoint_distance_m"].where(intervals["state"].eq("moving"), 0.0)
    intervals["detected_state"] = intervals["state"]
    intervals["state_source"] = "heuristic"
    intervals["review_notes"] = ""
    return intervals


def apply_confirmed_rests(intervals, confirmed_rests):
    """Use the hiker's confirmation while retaining the original GPS decision.

    Confirmations match whole intervals by both UTC endpoints. They never add
    distance or reconnect a gap, and do not change other intervals or future hikes.
    """
    intervals = intervals.copy()
    if confirmed_rests is None or confirmed_rests.empty:
        return intervals
    for rest in confirmed_rests.itertuples():
        match = (intervals["start_time_utc"].eq(rest.start_time_utc)
                 & intervals["end_time_utc"].eq(rest.end_time_utc))
        if match.sum() != 1:
            raise ValueError("A confirmed rest must match exactly one recorded interval; "
                             "review the UTC endpoints in confirmed_rests.csv.")
        index = intervals.index[match][0]
        intervals.at[index, "state"] = "stationary"
        intervals.at[index, "state_source"] = "user_confirmed"
        intervals.at[index, "review_notes"] = rest.notes
        intervals.at[index, "distance_m"] = 0.0
        intervals.at[index, "quality_flags"] = combine_flags(
            intervals.at[index, "quality_flags"], "user_confirmed_rest"
        )
    return intervals


def smooth_elevation(distances_m, elevations_m, config: ImportConfig):
    """Centered rolling median on an evenly spaced distance grid.

    Equal-distance sampling prevents dense GPS samples from dominating the
    filter. Equal-distance elevations (a stop) are first reduced to their median.
    Call this on one valid continuous block at a time, never across a data gap.
    """
    x, y = np.asarray(distances_m, dtype=float), np.asarray(elevations_m, dtype=float)
    if not len(x):
        return y.copy()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x) < 0).any():
        raise ValueError("Smoothing needs finite elevation and nondecreasing distance within a block.")
    profile = pd.DataFrame({"x": x, "y": y}).groupby("x", sort=True)["y"].median()
    if len(profile) == 1:
        return np.full(len(x), profile.iloc[0])
    grid = np.arange(profile.index[0], profile.index[-1], config.elevation_grid_m)
    grid = np.append(grid, profile.index[-1])
    raw_grid = np.interp(grid, profile.index, profile.to_numpy())
    # Window centers are actual distances, including the possibly short last
    # grid interval. searchsorted avoids assuming the last step is exactly 5 m.
    half_window = config.elevation_window_m / 2
    left = np.searchsorted(grid, grid - half_window, side="left")
    right = np.searchsorted(grid, grid + half_window, side="right")
    filtered = np.array([np.median(raw_grid[a:b]) for a, b in zip(left, right)])
    return np.interp(x, grid, filtered)


def add_elevation_profile(points, intervals, config):
    """Keep raw elevations; split filtering at gaps, missing values, and boundaries."""
    points = points.copy()
    points["distance_m"] = 0.0
    points["smoothed_elevation_m"] = np.nan
    points["profile_block"] = -1
    connected_edges = {}
    increments = np.zeros(len(points))
    for row in intervals.itertuples():
        increments[row.end_point_index] = row.distance_m
        if row.detected_state != "unknown":
            connected_edges[row.end_point_index] = row.start_point_index
    points["distance_m"] = increments.cumsum()
    block = []
    block_number = 0

    def finish_block(indices, number):
        if indices:
            part = points.loc[indices]
            points.loc[indices, "smoothed_elevation_m"] = smooth_elevation(
                part["distance_m"], part["elevation_m"], config
            )
            points.loc[indices, "profile_block"] = number

    for i, point in points.iterrows():
        usable = point.coordinate_valid and pd.notna(point.time_utc) and isfinite(point.elevation_m)
        if not usable or (block and connected_edges.get(i) != block[-1]):
            finish_block(block, block_number)
            block_number += 1
            block = []
        if usable:
            block.append(i)
    finish_block(block, block_number)
    points["distance_miles"] = points["distance_m"] / METERS_PER_MILE
    points["elevation_ft"] = points["elevation_m"] * FEET_PER_METER
    points["smoothed_elevation_ft"] = points["smoothed_elevation_m"] * FEET_PER_METER
    return points


SEGMENT_COLUMNS = [
    "hike_id", "segment_index", "distance_miles", "elevation_gain_ft", "elevation_loss_ft",
    "elapsed_time_min", "moving_time_min", "stopped_or_unknown_time_min",
    "start_time_utc", "end_time_utc", "quality_flags", "hike_type",
    "stationary_time_min", "unknown_time_min", "pack_weight_lbs", "is_partial_segment",
]


def build_segments(points, intervals, hike_id, metadata, config):
    """Split accepted distance; allocate time/elevation by the same edge fraction.

    Unknown intervals contribute observed time and flags, but zero distance and
    zero inferred elevation. They stay intact, with their actual endpoint times.
    A bin may span disconnected blocks; its flags identify this for review.
    """
    rows = []
    target_m = config.segment_miles * METERS_PER_MILE

    def new_segment():
        return {
            "hike_id": hike_id, "segment_index": len(rows), "distance_miles": 0.0,
            "elevation_gain_ft": 0.0, "elevation_loss_ft": 0.0,
            "elapsed_time_min": 0.0, "moving_time_min": 0.0,
            "stationary_time_min": 0.0, "unknown_time_min": 0.0,
            "start_time_utc": pd.NaT, "end_time_utc": pd.NaT,
            "quality_flags": "" if metadata.get("hike_type") else "missing_hike_type",
            "hike_type": metadata.get("hike_type") or None,
            "pack_weight_lbs": metadata.get("pack_weight_lbs"),
        }

    current = new_segment()
    for interval in intervals.itertuples():
        # Wait for more movement before closing a full bin. Trailing stops/gaps
        # belong to that bin instead of creating a spurious zero-distance row.
        if interval.distance_m > 0 and current["distance_miles"] * METERS_PER_MILE >= target_m - 1e-8:
            rows.append(current)
            current = new_segment()
        a, b = points.loc[interval.start_point_index], points.loc[interval.end_point_index]
        gain = loss = 0.0
        flags = interval.quality_flags
        if interval.state == "moving":
            if pd.isna(a.smoothed_elevation_m) or pd.isna(b.smoothed_elevation_m):
                flags = combine_flags(flags, "missing_elevation")
                gain = loss = np.nan
            else:
                change = (b.smoothed_elevation_m - a.smoothed_elevation_m) * FEET_PER_METER
                gain, loss = max(change, 0.0), max(-change, 0.0)
        if interval.state == "unknown":
            flags = combine_flags(flags, "unknown_interval")
        fraction_done = 0.0
        while fraction_done < 1 - 1e-12:
            remaining_m = target_m - current["distance_miles"] * METERS_PER_MILE
            if interval.distance_m > 0:
                fraction = min(1 - fraction_done, remaining_m / interval.distance_m)
            else:
                fraction = 1.0
            # This interpolation is only reached for a valid, connected moving
            # edge. For an unknown or stationary interval fraction is always 1.
            if interval.distance_m == 0:
                start_time, end_time = interval.start_time_utc, interval.end_time_utc
            else:
                start_time = interval.start_time_utc + pd.Timedelta(seconds=interval.elapsed_seconds * fraction_done)
                end_time = interval.start_time_utc + pd.Timedelta(seconds=interval.elapsed_seconds * (fraction_done + fraction))
            if pd.isna(current["start_time_utc"]):
                current["start_time_utc"] = start_time
            current["end_time_utc"] = end_time
            minutes = interval.elapsed_seconds / 60 * fraction
            current["distance_miles"] += interval.distance_m / METERS_PER_MILE * fraction
            current["elapsed_time_min"] += minutes
            current[f"{interval.state}_time_min"] += minutes
            current["elevation_gain_ft"] += gain * fraction
            current["elevation_loss_ft"] += loss * fraction
            current["quality_flags"] = combine_flags(current["quality_flags"], flags)
            fraction_done += fraction
            if fraction_done < 1 - 1e-12:
                rows.append(current)
                current = new_segment()
    if current["elapsed_time_min"] > 0:
        rows.append(current)
    for row in rows:
        row["stopped_or_unknown_time_min"] = row["stationary_time_min"] + row["unknown_time_min"]
        row["is_partial_segment"] = row["distance_miles"] < config.segment_miles - 1e-9
        if row["distance_miles"] == 0:
            row["quality_flags"] = combine_flags(row["quality_flags"], "no_measured_movement")
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def process_hike(path, hike_id, metadata=None, config=None, confirmed_rests=None):
    """Return summary, segments, original+smoothed points, and interval decisions."""
    config = config or ImportConfig()
    metadata = metadata or {}
    points, info = read_trackpoints(path)
    intervals = classify_intervals(points, config)
    intervals = apply_confirmed_rests(intervals, confirmed_rests)
    points = add_elevation_profile(points, intervals, config)
    segments = build_segments(points, intervals, hike_id, metadata, config)

    # Raw totals use adjacent recorded trackpoints within a GPX segment even
    # when a time gap exists. They are diagnostics, not gap-filled route length.
    raw_distance = raw_gain = raw_loss = 0.0
    raw_increments = np.zeros(len(points))
    raw_distance_complete = raw_elevation_complete = True
    for a, b in zip(points.itertuples(), points.iloc[1:].itertuples()):
        if not same_recorded_segment(a, b):
            continue
        if a.coordinate_valid and b.coordinate_valid:
            distance = distance_meters(a.latitude, a.longitude, b.latitude, b.longitude)
            raw_distance += distance
            raw_increments[b.point_index] = distance
        else:
            raw_distance_complete = False
        if isfinite(a.elevation_m) and isfinite(b.elevation_m):
            change = b.elevation_m - a.elevation_m
            raw_gain += max(change, 0.0)
            raw_loss += max(-change, 0.0)
        else:
            raw_elevation_complete = False
    points["raw_distance_miles"] = raw_increments.cumsum() / METERS_PER_MILE
    times = points["time_utc"].dropna()
    flags = combine_flags(*points["quality_flags"], *intervals["quality_flags"])
    if not metadata.get("hike_type"):
        flags = combine_flags(flags, "missing_hike_type")
    if len(times) < 2:
        flags = combine_flags(flags, "insufficient_timestamps")
    if len(times) and (pd.isna(points.iloc[0].time_utc) or pd.isna(points.iloc[-1].time_utc)):
        flags = combine_flags(flags, "incomplete_elapsed_time")
    def time_for(state):
        return intervals.loc[intervals["state"].eq(state), "elapsed_seconds"].sum() / 60
    # This baseline uses the same accepted moving intervals as cleaned gain;
    # comparing it with cleaned gain isolates the effect of the median filter.
    moving_changes = [
        points.at[row.end_point_index, "elevation_m"] - points.at[row.start_point_index, "elevation_m"]
        for row in intervals.itertuples() if row.state == "moving"
    ]
    unsmoothed_gain = sum(max(change, 0.0) for change in moving_changes)
    unsmoothed_loss = sum(max(-change, 0.0) for change in moving_changes)
    summary = {
        "hike_id": hike_id, **info,
        "hike_type": metadata.get("hike_type"), "pack_weight_lbs": metadata.get("pack_weight_lbs"),
        "start_time_utc": times.iloc[0] if len(times) else pd.NaT,
        "end_time_utc": times.iloc[-1] if len(times) else pd.NaT,
        "raw_distance_miles": raw_distance / METERS_PER_MILE if raw_distance_complete else np.nan,
        "observed_raw_distance_miles": raw_distance / METERS_PER_MILE,
        "distance_miles": intervals["distance_m"].sum() / METERS_PER_MILE,
        "raw_elevation_gain_ft": raw_gain * FEET_PER_METER if raw_elevation_complete else np.nan,
        "raw_elevation_loss_ft": raw_loss * FEET_PER_METER if raw_elevation_complete else np.nan,
        "unsmoothed_moving_elevation_gain_ft": unsmoothed_gain * FEET_PER_METER,
        "unsmoothed_moving_elevation_loss_ft": unsmoothed_loss * FEET_PER_METER,
        "elevation_gain_ft": segments["elevation_gain_ft"].sum(skipna=False) if len(segments) else np.nan,
        "elevation_loss_ft": segments["elevation_loss_ft"].sum(skipna=False) if len(segments) else np.nan,
        "elapsed_time_min": (times.iloc[-1] - times.iloc[0]).total_seconds() / 60 if len(times) >= 2 else np.nan,
        "moving_time_min": time_for("moving"),
        "stationary_time_min": time_for("stationary"),
        "unknown_time_min": time_for("unknown"),
        "detected_unknown_time_min": intervals.loc[intervals["detected_state"].eq("unknown"), "elapsed_seconds"].sum() / 60,
        "confirmed_rest_count": int(intervals["state_source"].eq("user_confirmed").sum()),
        "confirmed_rest_time_min": intervals.loc[intervals["state_source"].eq("user_confirmed"), "elapsed_seconds"].sum() / 60,
        "recording_gap_count": int(intervals["quality_flags"].str.contains("recording_gap").sum()),
        "segment_count": len(segments),
        "full_segment_count": int((~segments["is_partial_segment"]).sum()),
        "quality_flags": flags,
    }
    points.insert(0, "hike_id", hike_id)
    intervals.insert(0, "hike_id", hike_id)
    return summary, segments, points, intervals
