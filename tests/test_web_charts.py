"""Chart/map/export consistency using synthetic GPX and historical observations only."""

from dataclasses import replace
from io import BytesIO
from math import degrees

import folium
import numpy as np
import pandas as pd
import pytest
from folium.map import FitBounds
from sklearn.linear_model import LinearRegression

from hiking_time_predictor.gpx import EARTH_RADIUS_M, FEET_PER_METER
from hiking_time_predictor.modeling import GEOMETRY, MODEL_LABELS
from hiking_time_predictor.predictor import ModelArtifact
from hiking_time_predictor.web_charts import (
    actual_predicted_figure,
    elevation_figure,
    error_figure,
    history_distance_figure,
    segment_csv,
    segment_figure,
    trail_map,
)
from hiking_time_predictor.web_upload import predict_upload, preview_upload


@pytest.fixture
def untimed_gpx():
    return ('<gpx version="1.1"><rte><name>Synthetic chart trail</name>' + "".join(
        f'<rtept lat="0" lon="{degrees(meters / EARTH_RADIUS_M)}">'
        f'<ele>{elevation}</ele></rtept>'
        for meters, elevation in [(0, 100), (400, 120), (1200, 160), (1600, 130), (2200, 180)]
    ) + '</rte></gpx>').encode("utf-8")


@pytest.fixture
def disconnected_gpx():
    # Two short sections separated by hundreds of miles, with no joining edge.
    return ('<gpx version="1.1"><trk>' + "".join(
        f'<trkseg><trkpt lat="{latitude}" lon="{longitude}"><ele>100</ele></trkpt>'
        f'<trkpt lat="{latitude}" lon="{longitude + .002}"><ele>120</ele></trkpt></trkseg>'
        for latitude, longitude in [(0, 0), (5, 10)]
    ) + '</trk></gpx>').encode("utf-8")


@pytest.fixture
def artifact():
    features = pd.DataFrame(
        [[0.1, 0, 0], [0.5, 0, 0], [0.5, 100, 0], [0.5, 0, 100],
         [0.3, 200, 40], [0.4, 50, 150]], columns=GEOMETRY,
    )
    estimator = LinearRegression().fit(features, 1 + features.to_numpy() @ np.array([20, 0.03, 0.01]))
    return ModelArtifact(
        estimator=estimator, model_name="B", selected=True,
        reason="Synthetic chart test fixture only.",
        evaluation={"segment_metrics": [{"model": MODEL_LABELS["B"], "observations": 6, "hikes": 3}]},
    )


@pytest.mark.parametrize("direction", ["one-way", "out-and-back"])
@pytest.mark.parametrize("fixture_name", ["untimed_gpx", "disconnected_gpx"])
def test_elevation_traces_match_backend_profile_without_joining_sections(request, fixture_name, direction):
    route = preview_upload(request.getfixturevalue(fixture_name), "synthetic.gpx", "day_hike", direction)
    before = route.profile.copy(deep=True)
    figure = elevation_figure(route)
    sections = list(route.profile.groupby(["leg", "section_index"], sort=False))
    assert len(figure.data) == len(sections)
    for trace, ((leg, _), section) in zip(figure.data, sections):
        np.testing.assert_allclose(trace.x, section.distance_miles)
        np.testing.assert_allclose(trace.y, section.smoothed_elevation_m * FEET_PER_METER)
        assert trace.name == ("Return" if leg == "return" else "Outbound")
    assert "miles" in figure.layout.xaxis.title.text
    assert "feet" in figure.layout.yaxis.title.text
    if direction == "out-and-back":
        outbound_end = route.profile.loc[route.profile.leg.eq("outbound"), "distance_miles"].max()
        assert len(figure.layout.shapes) == 1
        assert figure.layout.shapes[0].x0 == pytest.approx(outbound_end)
        assert figure.layout.shapes[0].x1 == pytest.approx(outbound_end)
        assert any(annotation.text == "Turnaround" for annotation in figure.layout.annotations)
    else:
        assert not figure.layout.shapes
    pd.testing.assert_frame_equal(route.profile, before)


def test_return_elevation_uses_reversed_terrain_and_continuous_trip_distance(untimed_gpx):
    route = preview_upload(untimed_gpx, "synthetic.gpx", "day_hike", "out-and-back")
    outbound, returning = elevation_figure(route).data
    np.testing.assert_allclose(returning.y, np.asarray(outbound.y)[::-1])
    assert returning.x[0] == pytest.approx(outbound.x[-1])
    assert returning.x[-1] == pytest.approx(2 * outbound.x[-1])
    assert np.all(np.diff(returning.x) >= 0)


@pytest.mark.parametrize("direction", ["one-way", "out-and-back"])
def test_map_keeps_disconnected_sections_separate_and_fits_all_points(disconnected_gpx, direction):
    route = preview_upload(disconnected_gpx, "synthetic.gpx", "day_hike", direction)
    result = trail_map(route)
    polylines = [child for child in result._children.values() if isinstance(child, folium.PolyLine)]
    sections = list(route.profile.groupby(["leg", "section_index"], sort=False))
    assert len(polylines) == len(sections) == (4 if direction == "out-and-back" else 2)
    for polyline, (_, section) in zip(polylines, sections):
        np.testing.assert_allclose(polyline.locations, section[["latitude", "longitude"]])
        # A chart regression connecting sections would create a multi-degree edge.
        assert np.max(np.abs(np.diff(np.asarray(polyline.locations), axis=0))) < .003
    bounds = [child for child in result._children.values() if isinstance(child, FitBounds)]
    assert len(bounds) == 1
    coordinates = route.profile[["latitude", "longitude"]]
    np.testing.assert_allclose(bounds[0].bounds, [coordinates.min(), coordinates.max()])
    tiles = [child for child in result._children.values() if isinstance(child, folium.TileLayer)]
    assert len(tiles) == 1
    assert "openstreetmap.org" in tiles[0].tiles


@pytest.mark.parametrize("direction", ["one-way", "out-and-back"])
def test_map_markers_match_route_endpoints_and_turnaround(untimed_gpx, direction):
    route = preview_upload(untimed_gpx, "synthetic.gpx", "day_hike", direction)
    result = trail_map(route)
    markers = [child for child in result._children.values() if isinstance(child, folium.Marker)]
    assert len(markers) == (3 if direction == "out-and-back" else 2)
    labels = {
        next(child.text for child in marker._children.values() if isinstance(child, folium.Tooltip)): marker
        for marker in markers
    }
    np.testing.assert_allclose(labels["Start"].location, route.profile[["latitude", "longitude"]].iloc[0])
    finish_label = "Finish (back at start)" if direction == "out-and-back" else "Finish"
    np.testing.assert_allclose(labels[finish_label].location, route.profile[["latitude", "longitude"]].iloc[-1])
    if direction == "out-and-back":
        turnaround = route.profile.loc[route.profile.leg.eq("outbound"), ["latitude", "longitude"]].iloc[-1]
        np.testing.assert_allclose(labels["Turnaround"].location, turnaround)
        np.testing.assert_allclose(labels[finish_label].location, labels["Start"].location)


def test_map_rejects_route_without_a_profile(untimed_gpx):
    route = preview_upload(untimed_gpx, "synthetic.gpx", "day_hike")
    with pytest.raises(ValueError, match="GPX profile"):
        trail_map(replace(route, profile=pd.DataFrame()))


@pytest.mark.parametrize("direction", ["one-way", "out-and-back"])
@pytest.mark.parametrize("model_mode", ["baseline", "selected", "experimental"])
def test_segment_chart_and_csv_match_inference_rows(untimed_gpx, artifact, direction, model_mode):
    supplied = ModelArtifact() if model_mode == "baseline" else replace(artifact, selected=model_mode == "selected")
    prediction = predict_upload(untimed_gpx, "synthetic.gpx", "day_hike", direction, artifact=supplied)
    table = prediction.segments.copy(deep=True)
    figure = segment_figure(prediction)
    traces = {trace.name: trace for trace in figure.data}
    expected = {
        "Primary estimate": table.predicted_time_min,
        "Gain": table.elevation_gain_ft,
        "Loss": -table.elevation_loss_ft,
        "Cumulative primary estimate": table.cumulative_predicted_time_min,
    }
    if model_mode == "experimental":
        expected["Experimental regression"] = table.regression_time_min
        expected["Cumulative experimental regression"] = table.cumulative_regression_time_min
        assert not np.allclose(table.predicted_time_min, table.regression_time_min)
    assert set(traces) == set(expected)
    for name, values in expected.items():
        np.testing.assert_allclose(traces[name].x, table.segment_number)
        np.testing.assert_allclose(traces[name].y, values)
    assert traces["Cumulative primary estimate"].y[-1] == pytest.approx(prediction.forecast.moving_minutes)
    assert list(traces["Primary estimate"].customdata[:, 0]) == list(table.leg)
    np.testing.assert_allclose(traces["Primary estimate"].customdata[:, 1].astype(float), table.distance_miles)
    exported_bytes = segment_csv(prediction)
    assert isinstance(exported_bytes, bytes)
    exported = pd.read_csv(BytesIO(exported_bytes))
    pd.testing.assert_frame_equal(exported, table, check_exact=False, rtol=1e-12, atol=1e-12)
    assert len(exported) == len(traces["Primary estimate"].x)
    np.testing.assert_allclose(exported.predicted_time_min, traces["Primary estimate"].y)
    assert exported.cumulative_predicted_time_min.iloc[-1] == pytest.approx(prediction.forecast.moving_minutes)
    pd.testing.assert_frame_equal(prediction.segments, table)


def test_historical_distance_figure_uses_supplied_measured_rows():
    summary = pd.DataFrame({"hike_id": ["synthetic-b", "synthetic-a"],
                            "distance_miles": [2.125, 6.75], "moving_time_min": [68.4, 190.2]})
    before = summary.copy(deep=True)
    figure = history_distance_figure(summary)
    assert len(figure.data) == 1
    np.testing.assert_allclose(figure.data[0].x, summary.distance_miles)
    np.testing.assert_allclose(figure.data[0].y, summary.moving_time_min)
    assert list(figure.data[0].customdata) == list(summary.hike_id)
    pd.testing.assert_frame_equal(summary, before)


def test_actual_predicted_figure_uses_supplied_held_out_rows_per_model():
    predictions = pd.DataFrame({
        "model": ["Synthetic B", "Synthetic A", "Synthetic B", "Synthetic A"],
        "hike_id": ["synthetic-1", "synthetic-1", "synthetic-2", "synthetic-2"],
        "segment_index": [0, 0, 1, 1], "moving_time_min": [7.1, 7.1, 12.3, 12.3],
        "predicted_time_min": [6.8, 8.2, 14.1, 10.9],
    })
    before = predictions.copy(deep=True)
    figure = actual_predicted_figure(predictions)
    assert [trace.name for trace in figure.data] == ["Synthetic B", "Synthetic A", "Perfect prediction"]
    for trace, (_, group) in zip(figure.data, predictions.groupby("model", sort=False)):
        np.testing.assert_allclose(trace.x, group.moving_time_min)
        np.testing.assert_allclose(trace.y, group.predicted_time_min)
        np.testing.assert_array_equal(trace.customdata, group[["hike_id", "segment_index"]])
    np.testing.assert_allclose(figure.data[-1].x, [0, 14.1])
    np.testing.assert_allclose(figure.data[-1].y, figure.data[-1].x)
    pd.testing.assert_frame_equal(predictions, before)


def test_empty_historical_predictions_do_not_fabricate_points():
    predictions = pd.DataFrame(columns=["model", "hike_id", "segment_index", "moving_time_min", "predicted_time_min"])
    assert len(actual_predicted_figure(predictions).data) == 0


def test_error_figure_uses_supplied_evaluation_metrics():
    metrics = pd.DataFrame({"model": ["Synthetic B", "Naismith"],
                            "MAE_min": [1.123456789, 2.234567891], "RMSE_min": [1.9, 3.6]})
    before = metrics.copy(deep=True)
    figure = error_figure(metrics)
    assert [trace.name for trace in figure.data] == ["MAE", "RMSE"]
    for trace, column in zip(figure.data, ["MAE_min", "RMSE_min"]):
        assert list(trace.x) == list(metrics.model)
        np.testing.assert_allclose(trace.y, metrics[column], rtol=0, atol=0)
    pd.testing.assert_frame_equal(metrics, before)
