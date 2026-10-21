"""Prediction features from manual totals or GPX geometry, never observed time."""

from dataclasses import dataclass, field
from math import ceil, isfinite
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd

from .data import HIKE_TYPES
from .gpx import FEET_PER_METER, METERS_PER_MILE, ImportConfig, distance_meters, smooth_elevation
from .modeling import GEOMETRY


@dataclass
class Route:
    segments: pd.DataFrame
    source: str
    notes: list[str]
    name: str = "Route"
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: ImportConfig = field(default_factory=ImportConfig)
    direction: str = "one-way"


def validate_route_features(segments):
    """Require positive miles, nonnegative feet, and one explicitly supplied type."""
    if segments.empty or segments.columns.duplicated().any():
        raise ValueError("Supply at least one route segment with unique columns.")
    if any(name not in segments for name in [*GEOMETRY, "hike_type"]):
        raise ValueError("Route segments require distance, gain, loss, and hike_type.")
    for name in GEOMETRY:
        values = pd.to_numeric(segments[name], errors="coerce").to_numpy(dtype=float)
        booleans = segments[name].map(lambda value: isinstance(value, (bool, np.bool_)))
        invalid = ~np.isfinite(values) | booleans.to_numpy()
        invalid |= values <= 0 if name == "distance_miles" else values < 0
        if invalid.any():
            raise ValueError(f"{name} must be finite and {'positive' if name == 'distance_miles' else 'nonnegative'}.")
    if not segments.hike_type.isin(HIKE_TYPES).all() or segments.hike_type.nunique() != 1:
        raise ValueError("Supply one hike_type: day_hike or backpacking.")


def manual_route(distance, gain, loss, hike_type, config=None):
    """Approximate the missing profile by distributing ascent/descent by distance."""
    config = config or ImportConfig()
    totals = pd.DataFrame([[distance, gain, loss, hike_type]], columns=[*GEOMETRY, "hike_type"])
    validate_route_features(totals)
    distance, gain, loss = totals[GEOMETRY].iloc[0].astype(float)
    # Avoid a tiny extra bin when a floating-point total is almost an exact multiple.
    count = max(1, ceil(distance / config.segment_miles - 1e-10))
    miles = np.full(count, config.segment_miles)
    miles[-1] = distance - config.segment_miles * (count - 1)
    segments = pd.DataFrame({"distance_miles": miles,
                             "elevation_gain_ft": gain * miles / distance,
                             "elevation_loss_ft": loss * miles / distance,
                             "hike_type": hike_type})
    return Route(segments, "manual totals", [
        "Manual gain/loss are distributed by distance; the local terrain profile is unknown."
    ], config=config)


def _profile_segments(distances, elevations, config):
    """Split each spatial edge at distance-bin boundaries and conserve gain/loss."""
    distances = np.asarray(distances, dtype=float)
    changes = np.diff(elevations) * FEET_PER_METER
    total = distances[-1] / METERS_PER_MILE
    if total <= 0:
        raise ValueError("Each GPX section needs positive horizontal distance.")
    count = max(1, ceil(total / config.segment_miles - 1e-10))
    rows = np.zeros((count, 3))
    for start_m, end_m, change in zip(distances[:-1], distances[1:], changes):
        start, end = start_m / METERS_PER_MILE, end_m / METERS_PER_MILE
        if end <= start:
            continue  # Repeated coordinates have identical smoothed elevation.
        first = min(int(start / config.segment_miles), count - 1)
        last = min(int(end / config.segment_miles), count - 1)
        for index in range(first, last + 1):
            overlap = max(0.0, min(end, (index + 1) * config.segment_miles)
                          - max(start, index * config.segment_miles))
            fraction = overlap / (end - start)
            rows[index] += [overlap, max(change, 0) * fraction, max(-change, 0) * fraction]
    return pd.DataFrame(rows, columns=GEOMETRY)


def _geometry_route(sections, hike_type, config, name, source, notes, leg="outbound"):
    """Bin connected sections independently; retain points for plots and reversal."""
    frames, profiles = [], []
    offset = 0.0
    for section_index, section in enumerate(sections):
        section = section.copy().reset_index(drop=True)
        distances = np.r_[0.0, np.cumsum([
            distance_meters(a.latitude, a.longitude, b.latitude, b.longitude)
            for a, b in zip(section.itertuples(), section.iloc[1:].itertuples())
        ])]
        frames.append(_profile_segments(distances, section.smoothed_elevation_m, config)
                      .assign(hike_type=hike_type, leg=leg, section_index=section_index))
        section["distance_miles"] = distances / METERS_PER_MILE + offset
        section["section_index"] = section_index
        section["leg"] = leg
        profiles.append(section)
        offset = float(section.distance_miles.iloc[-1])
    segments = pd.concat(frames, ignore_index=True)
    validate_route_features(segments)
    return Route(segments, source, list(notes), name, pd.concat(profiles, ignore_index=True), config)


def reverse_route(route: Route) -> Route:
    """Reverse geographic points, then recalculate edges and distance bins.

    Reverse the already filtered elevation profile with the coordinates. Filtering
    twice, or anchoring the 5-m grid at the other end, would change the terrain
    slightly. This preserves exactly the same terrain in both directions while
    swapping ascent/descent, including within the newly positioned return bins.
    """
    if route.profile.empty or route.direction != "one-way":
        raise ValueError("Reversal requires a one-way GPX route with an elevation profile.")
    sections = [section.iloc[::-1] for _, section in route.profile.groupby("section_index", sort=False)]
    return _geometry_route(sections[::-1], route.segments.hike_type.iloc[0], route.config,
                           route.name, route.source, route.notes, leg="return")


def gpx_route(path: str | Path, hike_type: str, config=None, *, direction="one-way",
              missing_elevation="error"):
    """Read <rtept> or <trkpt> geometry, including untimed planned tracks.

    Timestamped tracks can be replayed as geometry but supply no actual moving
    time here. The separate recorded-activity importer produces training targets.
    Disconnected sections are smoothed and binned separately, never joined.
    """
    config = config or ImportConfig()
    if direction not in ("one-way", "out-and-back"):
        raise ValueError("direction must be one-way or out-and-back.")
    if missing_elevation not in ("error", "interpolate"):
        raise ValueError("missing_elevation must be error or interpolate.")
    if hike_type not in HIKE_TYPES:
        raise ValueError("Supply one hike_type: day_hike or backpacking.")
    with Path(path).open(encoding="utf-8-sig") as stream:
        try:
            gpx = gpxpy.parse(stream)
        except (gpxpy.gpx.GPXException, ValueError, TypeError) as error:
            raise ValueError(f"Cannot parse GPX: {error}") from error
    if gpx.tracks and gpx.routes:
        raise ValueError("GPX contains both tracks and routes; export only the intended route geometry.")
    if len(gpx.routes) > 1:
        raise ValueError("GPX contains multiple routes; export one intended route.")
    groups = ([segment.points for track in gpx.tracks for segment in track.segments]
              if gpx.tracks else [route.points for route in gpx.routes])
    if not groups:
        raise ValueError("GPX needs trackpoints or routepoints; waypoints alone do not define a trail.")
    sections = []
    notes = ["GPX timestamps are ignored; actual hiking time is unavailable in this prediction workflow."]
    timestamped_track = False
    for points in groups:
        if len(points) < 2:
            raise ValueError("Each GPX section needs at least two points.")
        for point in points:
            if (point.latitude is None or point.longitude is None
                    or not isfinite(point.latitude) or not -90 <= point.latitude <= 90
                    or not isfinite(point.longitude) or not -180 <= point.longitude <= 180):
                raise ValueError("GPX coordinates must be finite, with latitude within ±90 and longitude within ±180.")
            if point.elevation is not None and not isfinite(point.elevation):
                raise ValueError("GPX needs finite elevation at every point to estimate gain/loss.")
        distances = np.r_[0.0, np.cumsum([
            distance_meters(a.latitude, a.longitude, b.latitude, b.longitude)
            for a, b in zip(points, points[1:])
        ])]
        raw = np.array([np.nan if p.elevation is None else p.elevation for p in points])
        missing = np.isnan(raw)
        elevations = raw.copy()
        if missing.any():
            if missing_elevation == "error":
                raise ValueError("GPX needs finite elevation at every point to estimate gain/loss; "
                                 "missing_elevation='interpolate' can fill interior gaps explicitly.")
            if missing[0] or missing[-1]:
                raise ValueError("Cannot interpolate missing elevation at a section endpoint; supply measured elevations.")
            elevations[missing] = np.interp(distances[missing], distances[~missing], raw[~missing])
            notes.append(f"Interpolated {int(missing.sum())} missing elevation value(s) by distance within a section; "
                         "terrain in these gaps is uncertain.")
        sections.append(pd.DataFrame({"latitude": [p.latitude for p in points],
                                      "longitude": [p.longitude for p in points],
                                      "elevation_m": raw,
                                      "smoothed_elevation_m": smooth_elevation(distances, elevations, config)}))
        timestamped_track |= bool(gpx.tracks) and any(p.time is not None for p in points)
    source = "timestamped track geometry" if timestamped_track else "planned route geometry"
    if len(groups) > 1:
        notes.append("Disconnected GPX sections are summed separately; no distance or elevation is added across gaps.")
    name = (gpx.name or next((item.name for item in [*gpx.tracks, *gpx.routes] if item.name), None)
            or Path(path).stem)
    outbound = _geometry_route(sections, hike_type, config, name, source, notes)
    if direction == "one-way":
        return outbound
    inbound = reverse_route(outbound)
    inbound.profile["distance_miles"] += outbound.segments.distance_miles.sum()
    return Route(pd.concat([outbound.segments, inbound.segments], ignore_index=True), source, notes,
                 name, pd.concat([outbound.profile, inbound.profile], ignore_index=True), config, direction)
