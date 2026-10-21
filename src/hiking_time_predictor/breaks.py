"""Meaningful break events layered on the existing moving/stationary/unknown audit.

This module never changes moving-time measurements. An unknown recording interval
can become a confirmed break only through an explicit, persistent human correction.
Account training can separately estimate an allowance from elapsed minus moving
time without relabelling those intervals.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .gpx import FEET_PER_METER, METERS_PER_MILE, combine_flags, distance_meters


CLASSIFICATIONS = ("moving", "stationary_break", "unknown", "short_stop", "camping", "non_hiking")
ORDINARY_BREAK_TARGET = "ordinary_breaks"
NON_MOVING_TARGET = "elapsed_minus_moving"
CORRECTION_COLUMNS = ["hike_id", "start_time", "end_time", "break_classification", "notes"]
EVENT_COLUMNS = [
    "hike_id", "break_id", "start_time", "end_time", "duration_minutes",
    "distance_into_hike_miles", "cumulative_elevation_gain_ft", "break_classification",
    "quality_flag", "detected_classification", "correction_notes", "moving_minutes",
]


@dataclass(frozen=True)
class BreakConfig:
    min_break_minutes: float = 2.0
    max_break_minutes: float = 60.0
    camping_minutes: float = 240.0
    geographic_jitter_m: float = 10.0
    max_unresolved_fraction: float = 0.05
    minimum_evaluation_hikes: int = 5
    minimum_selection_hikes: int = 12
    minimum_type_hikes: int = 5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if not self.min_break_minutes < self.max_break_minutes < self.camping_minutes:
            raise ValueError("Require min_break_minutes < max_break_minutes < camping_minutes.")
        if self.max_unresolved_fraction > 1:
            raise ValueError("max_unresolved_fraction must not exceed 1.")
        for name in ("minimum_evaluation_hikes", "minimum_selection_hikes", "minimum_type_hikes"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2.")
        if self.minimum_selection_hikes < self.minimum_evaluation_hikes:
            raise ValueError("Selection requires at least as many hikes as evaluation.")


def load_corrections(path):
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=CORRECTION_COLUMNS)
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return validate_corrections(frame)


def validate_corrections(frame):
    if frame.columns.duplicated().any() or set(frame.columns) != set(CORRECTION_COLUMNS):
        raise ValueError(f"Break correction columns must be {CORRECTION_COLUMNS}.")
    frame = frame[CORRECTION_COLUMNS].copy()
    if frame.hike_id.str.strip().eq("").any() or not frame.break_classification.isin(CLASSIFICATIONS).all():
        raise ValueError("Corrections need a hike_id and a supported break classification.")
    for column in ("start_time", "end_time"):
        timestamps = [pd.Timestamp(value) for value in frame[column]]
        if any(pd.isna(t) or t.tzinfo is None for t in timestamps):
            raise ValueError("Break correction timestamps must include a timezone.")
        frame[column] = pd.to_datetime(timestamps, utc=True)
    if (frame.end_time <= frame.start_time).any():
        raise ValueError("Break corrections require a positive duration.")
    if frame.duplicated(["hike_id", "start_time", "end_time"]).any():
        raise ValueError("Duplicate break corrections are not allowed.")
    return frame


def extract_break_events(points, intervals, config=None):
    """Collapse contiguous interval states, checking duration and spatial drift.

    Moving runs are retained for review and accounting; only stationary_break
    rows are labels. Fast GPS jitter is already handled by gpx.classify_intervals.
    Hiker-confirmed rests retain that provenance even when GPS evidence is weak.
    """
    config = config or BreakConfig()
    rows = []

    def drifting(part, track):
        indices = sorted(set(part.start_point_index) | set(part.end_point_index))
        cluster = track.loc[indices]
        anchor = cluster.iloc[0]
        return any(distance_meters(anchor.latitude, anchor.longitude, p.latitude, p.longitude)
                   > config.geographic_jitter_m for p in cluster.itertuples())

    for hike_id, edges in intervals.groupby("hike_id", sort=False):
        edges = edges.reset_index(drop=True)
        track = points.loc[points.hike_id.eq(hike_id)].set_index("point_index")
        gains = {}
        cumulative_gain = 0.0
        for edge in edges.itertuples():
            gains[edge.start_point_index] = cumulative_gain
            if edge.state == "moving":
                a = track.loc[edge.start_point_index, "smoothed_elevation_m"]
                b = track.loc[edge.end_point_index, "smoothed_elevation_m"]
                if np.isfinite([a, b]).all():
                    cumulative_gain += max(b - a, 0) * FEET_PER_METER
            gains[edge.end_point_index] = cumulative_gain
        # Zero-duration duplicate points have no break label or fabricated time.
        edges = edges.loc[edges.elapsed_seconds.gt(0)].copy()
        disconnected = edges.start_point_index.ne(edges.end_point_index.shift())
        # A discarded zero-duration edge may lie between two otherwise contiguous
        # runs; treat it conservatively as a boundary instead of joining through it.
        run = (edges.state.ne(edges.state.shift()) | disconnected).cumsum()
        parts = []
        for _, part in edges.groupby(run, sort=False):
            confirmed = part.state_source.eq("user_confirmed")
            if (part.iloc[0].state == "stationary" and confirmed.any() and not confirmed.all()
                    and part.elapsed_seconds.sum() / 60 <= config.max_break_minutes):
                blocks = [block for _, block in part.groupby(confirmed.ne(confirmed.shift()).cumsum(), sort=False)]
                if any(not block.state_source.eq("user_confirmed").all() and drifting(block, track) for block in blocks):
                    # Retain a confirmed rest even if neighboring GPS-only time
                    # is ambiguous. Those uncertain minutes get their own event.
                    parts.extend(blocks)
                    continue
            parts.append(part)
        for part in parts:
            first, last = part.iloc[0], part.iloc[-1]
            minutes = float(part.elapsed_seconds.sum() / 60)
            classification = first.state
            flags = combine_flags(*part.quality_flags.fillna(""))
            confirmed = part.state_source.eq("user_confirmed").any()
            if classification == "stationary":
                classification = "short_stop" if minutes < config.min_break_minutes else "stationary_break"
                if minutes >= config.camping_minutes:
                    classification = "camping"
                    flags = combine_flags(flags, "extended_stationary_camping_candidate")
                elif minutes > config.max_break_minutes:
                    classification = "unknown"
                    flags = combine_flags(flags, "exceeds_ordinary_break_duration")
                else:
                    # Confirmation of one edge does not vouch for adjacent,
                    # unconfirmed drifting clusters. Check those blocks too.
                    sources = part.state_source.eq("user_confirmed")
                    blocks = sources.ne(sources.shift()).cumsum()
                    for _, block in part.groupby(blocks, sort=False):
                        if block.state_source.eq("user_confirmed").all():
                            continue
                        if drifting(block, track):
                            classification = "unknown"
                            flags = combine_flags(flags, "stationary_cluster_drift")
                flags = combine_flags(flags, "hiker_confirmed" if confirmed else "gps_heuristic")
            elif classification == "unknown":
                flags = combine_flags(flags, "unresolved_recording_interval")
            start, end = pd.Timestamp(first.start_time_utc), pd.Timestamp(last.end_time_utc)
            identity = f"{hike_id}|{start.isoformat()}|{end.isoformat()}"
            rows.append({
                "hike_id": hike_id, "break_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                "start_time": start, "end_time": end, "duration_minutes": minutes,
                "distance_into_hike_miles": float(track.loc[first.start_point_index, "distance_m"]) / METERS_PER_MILE,
                "cumulative_elevation_gain_ft": gains[first.start_point_index],
                "break_classification": classification, "quality_flag": flags,
                "detected_classification": classification, "correction_notes": "",
                "moving_minutes": minutes if first.state == "moving" else 0.0,
            })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def apply_break_corrections(events, corrections, config=None):
    config = config or BreakConfig()
    corrected = events.copy()
    if corrections.empty:
        return corrected
    corrections = validate_corrections(corrections)
    for correction in corrections.itertuples():
        match = (corrected.hike_id.eq(correction.hike_id)
                 & corrected.start_time.eq(correction.start_time)
                 & corrected.end_time.eq(correction.end_time))
        if match.sum() != 1:
            raise ValueError(f"Stale break correction for {correction.hike_id}: endpoints no longer match. "
                             "Review the correction CSV; no corrections were discarded.")
        index = corrected.index[match][0]
        if correction.break_classification == "stationary_break":
            if corrected.at[index, "moving_minutes"] > 0:
                raise ValueError("A moving interval cannot also count as a break. Use confirmed_rests.csv and reimport first.")
            if corrected.at[index, "duration_minutes"] > config.max_break_minutes:
                raise ValueError("A break exceeds max_break_minutes; review as camping/non_hiking or adjust the threshold.")
        corrected.at[index, "break_classification"] = correction.break_classification
        corrected.at[index, "correction_notes"] = correction.notes
        corrected.at[index, "quality_flag"] = combine_flags(corrected.at[index, "quality_flag"], "manual_review")
    return corrected


def save_corrections(events, edited, path, config=None):
    """Upsert only edited classifications; preserve all other human corrections."""
    if events.break_id.duplicated().any() or edited.break_id.duplicated().any():
        raise ValueError("Event IDs must be unique.")
    originals = events.set_index("break_id")
    existing = load_corrections(path)
    updates = []
    for row in edited.itertuples():
        old = originals.loc[row.break_id]
        if row.break_classification != old.break_classification or row.correction_notes != old.correction_notes:
            updates.append([old.hike_id, old.start_time, old.end_time, row.break_classification, row.correction_notes])
    if not updates:
        return 0
    new = validate_corrections(pd.DataFrame(updates, columns=CORRECTION_COLUMNS))
    merged = pd.concat([existing, new], ignore_index=True).drop_duplicates(
        ["hike_id", "start_time", "end_time"], keep="last")
    apply_break_corrections(events, merged, config)  # Validate before any write.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(path)
    return len(new)


def summarize_breaks(summary, events, config=None, *, target=ORDINARY_BREAK_TARGET):
    """Summarize event labels or a whole-recording non-moving allowance.

    The allowance includes short pauses and unresolved GPS time without changing
    any event labels. It does not require reliable locations of individual stops.
    """
    if target not in (ORDINARY_BREAK_TARGET, NON_MOVING_TARGET):
        raise ValueError("Unsupported break training target.")
    config = config or BreakConfig()
    rows = []
    for hike in summary.itertuples():
        part = events.loc[events.hike_id.eq(hike.hike_id)]
        breaks = part.loc[part.break_classification.eq("stationary_break")]
        unresolved = float(part.loc[part.break_classification.eq("unknown"), "duration_minutes"].sum())
        excluded = part.break_classification.isin(["camping", "non_hiking"])
        flags = str(hike.quality_flags or "")
        reasons = []
        if not np.isfinite([hike.elevation_gain_ft, hike.elevation_loss_ft]).all():
            reasons.append("incomplete_route_geometry")
        if (not np.isfinite([hike.distance_miles, hike.moving_time_min]).all()
                or hike.distance_miles <= 0 or hike.moving_time_min <= 0):
            reasons.append("no_measured_hiking")
        if any(flag in flags for flag in ("insufficient_timestamps", "incomplete_elapsed_time", "missing_timestamp")):
            reasons.append("incomplete_timestamps")
        denominator = float(part.duration_minutes.sum())
        if target == ORDINARY_BREAK_TARGET and (denominator <= 0 or unresolved / denominator > config.max_unresolved_fraction):
            reasons.append("too_much_unresolved_time")
        # A whole recording with camping/non-hiking events has no reliable
        # day-hike target. Keep it in EDA but do not fit an ordinary-break rate.
        if excluded.any():
            reasons.append("camping_or_non_hiking_activity")
        durations = breaks.duration_minutes
        ordinary_minutes = float(durations.sum())
        total = ordinary_minutes
        if target == NON_MOVING_TARGET:
            elapsed = getattr(hike, "elapsed_time_min", np.nan)
            total = float(elapsed - hike.moving_time_min)
            if not isfinite(elapsed) or elapsed <= 0 or not isfinite(total) or total < -1e-8:
                reasons.append("invalid_elapsed_or_moving_time")
            elif total < 0:
                total = 0.0  # Floating-point roundoff on an all-moving recording.
        count = len(breaks)
        rows.append({
            "hike_id": hike.hike_id, "hike_type": hike.hike_type,
            "distance_miles": hike.distance_miles, "elevation_gain_ft": hike.elevation_gain_ft,
            "elevation_loss_ft": hike.elevation_loss_ft, "moving_time_min": hike.moving_time_min,
            "break_count": count, "total_break_minutes": total,
            "ordinary_break_minutes": ordinary_minutes, "target_definition": target,
            "average_break_minutes": float(durations.mean()) if count else np.nan,
            "median_break_minutes": float(durations.median()) if count else np.nan,
            "breaks_per_mile": count / hike.distance_miles if hike.distance_miles > 0 else np.nan,
            "break_minutes_per_mile": total / hike.distance_miles if hike.distance_miles > 0 else np.nan,
            "mean_distance_between_breaks_miles": float(breaks.distance_into_hike_miles.diff().mean()) if count > 1 else np.nan,
            "unresolved_minutes": unresolved,
            "short_stop_minutes": float(part.loc[part.break_classification.eq("short_stop"), "duration_minutes"].sum()),
            "excluded_minutes": float(part.loc[excluded, "duration_minutes"].sum()),
            "training_eligible": not reasons, "eligibility_notes": ";".join(reasons),
        })
    return pd.DataFrame(rows)
