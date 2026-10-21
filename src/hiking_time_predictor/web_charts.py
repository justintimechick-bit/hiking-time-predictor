"""Presentation-only Plotly/Folium builders; all measurements come from the backend."""

import folium
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .gpx import FEET_PER_METER


FOREST = "#24533f"
MOSS = "#819857"
SLATE = "#647685"
GOLD = "#ba852b"


def _style(figure, height=350):
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Arial, sans-serif", size=13, color="#243b32"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=18, t=25, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor="#e7ece8", zeroline=False)
    figure.update_yaxes(gridcolor="#e7ece8", zeroline=False)
    return figure


def trail_map(route):
    """Draw each connected section separately, including both out-and-back legs."""
    profile = route.profile
    if profile.empty:
        raise ValueError("A GPX profile is required to draw the trail map.")
    coordinates = profile[["latitude", "longitude"]]
    result = folium.Map(location=coordinates.mean().tolist(), tiles="OpenStreetMap",
                        control_scale=True, prefer_canvas=True)
    for (leg, _), section in profile.groupby(["leg", "section_index"], sort=False):
        returning = leg == "return"
        folium.PolyLine(
            section[["latitude", "longitude"]].to_numpy().tolist(),
            color=MOSS if returning else FOREST,
            weight=3 if returning else 5, opacity=0.9,
            dash_array="7 7" if returning else None,
            tooltip="Return leg" if returning else "Outbound leg",
        ).add_to(result)
    start = coordinates.iloc[0].tolist()
    finish = coordinates.iloc[-1].tolist()
    if route.direction == "out-and-back":
        outbound = profile.loc[profile.leg.eq("outbound")]
        turnaround = outbound[["latitude", "longitude"]].iloc[-1].tolist()
        folium.Marker(start, tooltip="Start", popup="Start",
                      icon=folium.Icon(color="darkgreen", icon="play", prefix="fa")).add_to(result)
        # Offset the overlapping marker icon so both start and finish can be selected.
        folium.Marker(finish, tooltip="Finish (back at start)", popup="Finish (back at start)",
                      icon=folium.DivIcon(icon_size=(32, 32), icon_anchor=(-12, 16), html=
                          '<div style="background:#647685;color:white;border:2px solid white;'
                          'border-radius:50%;width:26px;height:26px;text-align:center;'
                          'line-height:22px;font-size:15px" title="Finish">&#9873;</div>')).add_to(result)
        folium.Marker(turnaround, tooltip="Turnaround", popup="Turnaround",
                      icon=folium.Icon(color="orange", icon="repeat", prefix="fa")).add_to(result)
    else:
        folium.Marker(start, tooltip="Start", popup="Start",
                      icon=folium.Icon(color="darkgreen", icon="play", prefix="fa")).add_to(result)
        folium.Marker(finish, tooltip="Finish", popup="Finish",
                      icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(result)
    result.fit_bounds([coordinates.min().tolist(), coordinates.max().tolist()], padding=(28, 28))
    return result


def elevation_figure(route):
    """Use the backend's smoothed, distance-ordered profile, including reversal."""
    figure = go.Figure()
    seen = set()
    for (leg, _), section in route.profile.groupby(["leg", "section_index"], sort=False):
        figure.add_trace(go.Scatter(
            x=section.distance_miles.to_numpy(),
            y=section.smoothed_elevation_m.to_numpy() * FEET_PER_METER,
            name="Return" if leg == "return" else "Outbound", legendgroup=leg,
            showlegend=leg not in seen, mode="lines",
            line=dict(color=MOSS if leg == "return" else FOREST, width=2.5,
                      dash="dot" if leg == "return" else "solid"),
            hovertemplate="%{x:.2f} mi · %{y:,.0f} ft<extra>%{fullData.name}</extra>",
        ))
        seen.add(leg)
    if route.direction == "out-and-back":
        distance = float(route.profile.loc[route.profile.leg.eq("outbound"), "distance_miles"].max())
        figure.add_vline(x=distance, line_dash="dash", line_color=GOLD,
                         annotation_text="Turnaround", annotation_position="top left")
    figure.update_xaxes(title_text="Distance (miles)")
    figure.update_yaxes(title_text="Elevation (feet)")
    return _style(figure, 320)


def segment_figure(prediction):
    """Plot the exact segment table returned by saved inference, without recalculation."""
    table = prediction.segments
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.09,
                           subplot_titles=("Moving time per segment", "Ascent and descent", "Time along the route"))
    figure.add_trace(go.Bar(
        x=table.segment_number.to_numpy(), y=table.predicted_time_min.to_numpy(),
        name="Primary estimate", marker_color=FOREST,
        customdata=table[["leg", "distance_miles"]].to_numpy(),
        hovertemplate="Segment %{x} · %{customdata[0]}<br>%{customdata[1]:.2f} mi · %{y:.1f} min<extra>Primary estimate</extra>",
    ), row=1, col=1)
    experimental = prediction.forecast.experimental_minutes is not None
    if experimental:
        figure.add_trace(go.Scatter(
            x=table.segment_number.to_numpy(), y=table.regression_time_min.to_numpy(),
            mode="lines+markers", name="Experimental regression", line=dict(color=GOLD),
        ), row=1, col=1)
    for column, name, color, sign in (("elevation_gain_ft", "Gain", MOSS, 1),
                                     ("elevation_loss_ft", "Loss", SLATE, -1)):
        figure.add_trace(go.Bar(x=table.segment_number.to_numpy(), y=table[column].to_numpy() * sign,
                                 name=name, marker_color=color), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=table.segment_number.to_numpy(), y=table.cumulative_predicted_time_min.to_numpy(),
        name="Cumulative primary estimate", mode="lines+markers", line=dict(color=FOREST),
    ), row=3, col=1)
    if experimental:
        figure.add_trace(go.Scatter(
            x=table.segment_number.to_numpy(), y=table.cumulative_regression_time_min.to_numpy(),
            name="Cumulative experimental regression", mode="lines", line=dict(color=GOLD, dash="dot"),
        ), row=3, col=1)
    figure.update_yaxes(title_text="Minutes", row=1, col=1)
    figure.update_yaxes(title_text="Feet", row=2, col=1)
    figure.update_yaxes(title_text="Minutes", row=3, col=1)
    figure.update_xaxes(title_text="Segment number (travel order)", row=3, col=1)
    figure.update_layout(barmode="relative")
    _style(figure, 700)
    figure.update_layout(legend=dict(y=-0.13, yanchor="top"), margin=dict(b=105))
    return figure


def segment_csv(prediction):
    """Export the same ordered rows used by the charts; no rounded intermediate data."""
    return prediction.segments.to_csv(index=False).encode("utf-8")


def history_distance_figure(summary):
    figure = go.Figure(go.Scatter(
        x=summary.distance_miles.to_numpy(), y=summary.moving_time_min.to_numpy(),
        customdata=summary.hike_id.astype(str).to_numpy(), mode="markers",
        marker=dict(color=FOREST, size=12, opacity=0.85, line=dict(color="white", width=1)),
        hovertemplate="%{customdata}<br>%{x:.2f} moving mi · %{y:.1f} moving min<extra></extra>",
    ))
    figure.update_xaxes(title_text="Measured moving distance (miles)")
    figure.update_yaxes(title_text="Measured moving time (minutes)")
    return _style(figure)


def actual_predicted_figure(predictions):
    """Held-out segment comparisons only; never generate training-fit predictions."""
    figure = go.Figure()
    for (name, group), color in zip(predictions.groupby("model", sort=False), [FOREST, MOSS, GOLD, SLATE]):
        figure.add_trace(go.Scatter(
            x=group.moving_time_min.to_numpy(), y=group.predicted_time_min.to_numpy(),
            customdata=group[["hike_id", "segment_index"]].to_numpy(),
            mode="markers", name=name, marker=dict(color=color, size=8, opacity=0.8),
            hovertemplate="%{customdata[0]} · segment %{customdata[1]}<br>Actual %{x:.2f} min<br>Predicted %{y:.2f} min<extra>%{fullData.name}</extra>",
        ))
    if not predictions.empty:
        values = predictions[["moving_time_min", "predicted_time_min"]].to_numpy(dtype=float)
        low, high = min(0.0, float(np.min(values))), float(np.max(values))
        figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction",
                                   line=dict(color=SLATE, dash="dash"), hoverinfo="skip"))
    figure.update_xaxes(title_text="Actual segment moving time (minutes)")
    figure.update_yaxes(title_text="Held-out predicted time (minutes)")
    figure.update_layout(hovermode="closest")
    result = _style(figure, 410)
    result.update_layout(hovermode="closest")
    return result


def error_figure(metrics):
    figure = go.Figure()
    for column, name, color in (("MAE_min", "MAE", FOREST), ("RMSE_min", "RMSE", MOSS)):
        figure.add_trace(go.Bar(x=metrics.model.to_numpy(), y=metrics[column].to_numpy(), name=name,
                                 marker_color=color, hovertemplate="%{x}<br>%{y:.3f} min<extra>%{fullData.name}</extra>"))
    figure.update_layout(barmode="group")
    figure.update_yaxes(title_text="Error (minutes)")
    return _style(figure)
