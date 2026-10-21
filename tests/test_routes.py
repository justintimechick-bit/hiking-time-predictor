"""Planned-route geometry has no actual time and conserves distance and ascent."""

from math import degrees
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pandas as pd
import pytest

from hiking_time_predictor.gpx import EARTH_RADIUS_M, FEET_PER_METER, METERS_PER_MILE, ImportConfig
from hiking_time_predictor.modeling import TARGET, naismith_minutes
from hiking_time_predictor.routes import gpx_route, manual_route


@pytest.fixture
def planned_gpx(tmp_path):
    def write(groups, kind="track", timestamps=False, version="1.1"):
        root = Element("gpx", version=version, creator="SYNTHETIC UNIT TEST",
                       xmlns=f"http://www.topografix.com/GPX/1/{version[-1]}")
        for group in groups:
            if kind == "track":
                parent = SubElement(SubElement(root, "trk"), "trkseg")
                tag = "trkpt"
            else:
                parent = SubElement(root, "rte")
                tag = "rtept"
            for index, (meters, elevation) in enumerate(group):
                point = SubElement(parent, tag, lat="0", lon=str(degrees(meters / EARTH_RADIUS_M)))
                if elevation is not None:
                    SubElement(point, "ele").text = str(elevation)
                if timestamps:
                    SubElement(point, "time").text = f"2026-01-01T00:00:{index:02d}Z"
        path = tmp_path / "synthetic-planned-route.gpx"
        path.write_bytes(tostring(root, encoding="utf-8", xml_declaration=True))
        return path
    return write


@pytest.mark.parametrize("kind,version", [("track", "1.1"), ("route", "1.1"), ("track", "1.0"), ("route", "1.0")])
def test_untimed_gpx_provides_features_not_actual_time(planned_gpx, kind, version):
    route = gpx_route(planned_gpx([[(0, 100), (400, 120), (1000, 110)]], kind, version=version), "backpacking")
    assert route.source == "planned route geometry"
    assert TARGET not in route.segments
    assert not any("time" in column for column in route.segments)
    assert route.segments.hike_type.eq("backpacking").all()
    assert route.segments.distance_miles.sum() == pytest.approx(1000 / METERS_PER_MILE)


def test_segment_crossings_and_partial_tail_conserve_geometry(planned_gpx):
    config = ImportConfig(segment_miles=0.5, elevation_window_m=1, elevation_grid_m=1)
    # Integer-meter endpoints make this centered 1-m window preserve endpoint elevations.
    route = gpx_route(planned_gpx([[(0, 100), (1000, 120), (2000, 110)]]), "day_hike", config)
    assert route.segments.distance_miles.tolist() == pytest.approx([0.5, 0.5, 2000 / METERS_PER_MILE - 1])
    assert route.segments.elevation_gain_ft.sum() == pytest.approx(20 * FEET_PER_METER)
    assert route.segments.elevation_loss_ft.sum() == pytest.approx(10 * FEET_PER_METER)
    assert naismith_minutes(route.segments).sum() == pytest.approx(20 * 2000 / METERS_PER_MILE + 0.03 * 20 * FEET_PER_METER)


def test_disconnected_sections_do_not_invent_distance_or_ascent(planned_gpx):
    route = gpx_route(planned_gpx([[(0, 0), (100, 0)], [(100000, 1000), (100100, 1000)]]), "day_hike")
    assert len(route.segments) == 2
    assert route.segments.distance_miles.sum() == pytest.approx(200 / METERS_PER_MILE)
    assert route.segments.elevation_gain_ft.sum() == 0
    assert any("Disconnected" in note for note in route.notes)


def test_timestamps_do_not_change_features_or_supply_actual_time(planned_gpx):
    points = [[(0, 100), (500, 120), (1000, 110)]]
    untimed = gpx_route(planned_gpx(points), "day_hike")
    timed = gpx_route(planned_gpx(points, timestamps=True), "day_hike")
    assert timed.source == "timestamped track geometry"
    pd.testing.assert_frame_equal(timed.segments, untimed.segments)
    assert TARGET not in timed.segments
    assert any("timestamps are ignored" in note for note in timed.notes)


@pytest.mark.parametrize("points,match", [
    ([(0, None), (100, 20)], "finite elevation"),
    ([(0, 10), (100, np.nan)], "finite elevation"),
    ([(0, 10), (100, np.inf)], "finite elevation"),
    ([(0, 10)], "at least two"),
    ([(0, 10), (0, 20)], "positive horizontal"),
])
def test_missing_invalid_or_degenerate_profiles_are_rejected(planned_gpx, points, match):
    with pytest.raises(ValueError, match=match):
        gpx_route(planned_gpx([points]), "day_hike")


@pytest.mark.parametrize("latitude,longitude", [(91, 0), (0, 181), ("nan", 0), (0, "inf")])
def test_invalid_coordinates_are_rejected(tmp_path, latitude, longitude):
    path = tmp_path / "invalid.gpx"
    path.write_text(f'<gpx version="1.1"><rte><rtept lat="{latitude}" lon="{longitude}"><ele>0</ele></rtept>'
                    '<rtept lat="0" lon="0.1"><ele>0</ele></rtept></rte></gpx>')
    with pytest.raises(ValueError, match="coordinates"):
        gpx_route(path, "day_hike")


def test_waypoints_and_ambiguous_alternatives_are_rejected(tmp_path, planned_gpx):
    path = tmp_path / "waypoints.gpx"
    path.write_text('<gpx version="1.1"><wpt lat="0" lon="0"/></gpx>')
    with pytest.raises(ValueError, match="waypoints alone"):
        gpx_route(path, "day_hike")
    path = planned_gpx([[(0, 0), (100, 10)], [(100, 10), (200, 0)]], kind="route")
    with pytest.raises(ValueError, match="multiple routes"):
        gpx_route(path, "day_hike")
    path = planned_gpx([[(0, 0), (100, 10)]])
    path.write_text(path.read_text().replace("</gpx>", '<rte><rtept lat="0" lon="0"/></rte></gpx>'))
    with pytest.raises(ValueError, match="both tracks and routes"):
        gpx_route(path, "day_hike")


def test_malformed_gpx_is_reported(tmp_path):
    path = tmp_path / "broken.gpx"
    path.write_text("<gpx><broken")
    with pytest.raises(ValueError, match="Cannot parse GPX"):
        gpx_route(path, "day_hike")


@pytest.mark.parametrize("distance", [0.1, 0.5, 0.500000000000001, 1, 1.2, 7.5])
def test_manual_segmentation_preserves_totals_and_short_tail(distance):
    route = manual_route(distance, 1800, 1200, "backpacking")
    assert route.segments.distance_miles.sum() == pytest.approx(distance)
    assert route.segments.elevation_gain_ft.sum() == pytest.approx(1800)
    assert route.segments.elevation_loss_ft.sum() == pytest.approx(1200)
    assert route.segments.distance_miles.le(0.5 + 1e-10).all()
    assert route.segments.distance_miles.gt(0).all()
    assert any("profile is unknown" in note for note in route.notes)


@pytest.mark.parametrize("distance,gain,loss,hike_type", [
    (0, 0, 0, "day_hike"), (-1, 0, 0, "day_hike"), (1, -1, 0, "day_hike"),
    (1, 0, -1, "day_hike"), (np.nan, 0, 0, "day_hike"), (1, np.inf, 0, "day_hike"),
    (1, 0, np.nan, "day_hike"), (True, 0, 0, "day_hike"), (1, 0, 0, "unknown"),
])
def test_invalid_manual_measurements_rejected(distance, gain, loss, hike_type):
    with pytest.raises(ValueError):
        manual_route(distance, gain, loss, hike_type)
