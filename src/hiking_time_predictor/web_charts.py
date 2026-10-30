"""Presentation-only Plotly/Folium builders; all measurements come from the backend."""

from html import escape

import folium
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .gpx import FEET_PER_METER


FOREST = "#24533f"
MOSS = "#819857"
SLATE = "#647685"
GOLD = "#ba852b"
FONT = '"Source Sans", "Source Sans 3", "Source Sans Pro", sans-serif'


def _hours_label(minutes):
    if minutes is None or not np.isfinite(float(minutes)):
        return "—"
    minutes = int(round(float(minutes)))
    return f"{minutes // 60}:{minutes % 60:02d}"


def _style(figure, height=350):
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family=FONT, size=13, color="#243b32"),
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


# Cloud UI charts. Hike and route names are user text: they are HTML-escaped
# before reaching Plotly hover labels, which render a subset of HTML.

def pace_vs_rule_figure(hikes):
    """Each recorded hike's moving time against the standard hiking rule's estimate.

    ``hikes`` needs name, date_label, distance_miles, elevation_gain_ft and
    moving_time_min columns. Dots below the dashed line were faster than the rule.
    """
    rule = (hikes.distance_miles.astype(float) * 20 + hikes.elevation_gain_ft.astype(float) * 0.03) / 60
    actual = hikes.moving_time_min.astype(float) / 60
    customdata = np.column_stack([
        hikes.name.map(lambda n: escape(str(n))), hikes.date_label.astype(str),
        hikes.distance_miles.map(lambda v: f"{v:.1f}"), hikes.elevation_gain_ft.map(lambda v: f"{v:,.0f}"),
        hikes.moving_time_min.map(_hours_label), (rule * 60).map(_hours_label)])
    top = float(max(rule.max(), actual.max()) * 1.08) if len(hikes) else 1.0
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, top], y=[0, top], mode="lines", name="Standard hiking rule",
        line=dict(color=SLATE, dash="dash", width=1.5), hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=rule.to_numpy(), y=actual.to_numpy(), mode="markers", name="Your hikes", customdata=customdata,
        marker=dict(color=FOREST, size=11, opacity=0.9, line=dict(color="white", width=2)),
        hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} mi · %{customdata[3]} ft climb"
                       "<br>You: %{customdata[4]} moving · Rule: %{customdata[5]}<extra></extra>")))
    figure.add_annotation(x=top, y=top, text="Standard hiking rule", showarrow=False, xanchor="right",
                          yanchor="bottom", font=dict(color=SLATE, size=12))
    figure.update_xaxes(title_text="Standard hiking rule estimate (hours)", range=[0, top], rangemode="tozero")
    figure.update_yaxes(title_text="Your moving time (hours)", range=[0, top], rangemode="tozero")
    _style(figure, 340)
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def history_elevation_figure(points, stops=None):
    """Recorded elevation in feet by distance, with rest stops marked."""
    data = points[["raw_distance_miles", "smoothed_elevation_m"]].dropna()
    x = data.raw_distance_miles.to_numpy(dtype=float)
    y = data.smoothed_elevation_m.to_numpy(dtype=float) * FEET_PER_METER
    figure = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", name="Elevation", fill="tozeroy", fillcolor="rgba(129,152,87,0.18)",
        line=dict(color=FOREST, width=2.5), hovertemplate="%{x:.2f} mi · %{y:,.0f} ft<extra></extra>"))
    if stops is not None and len(stops) and len(x):
        at = stops.distance_into_hike_miles.to_numpy(dtype=float)
        figure.add_trace(go.Scatter(
            x=at, y=np.interp(at, x, y), mode="markers", name="Stops",
            marker=dict(color=GOLD, size=10, symbol="circle", line=dict(color="white", width=2)),
            customdata=np.column_stack([stops.label.astype(str), stops.duration_minutes.map(lambda v: f"{v:.0f}")]),
            hovertemplate="%{customdata[0]} · %{customdata[1]} min<br>mile %{x:.1f}<extra></extra>"))
    if len(y):
        pad = max((y.max() - y.min()) * 0.15, 30)
        figure.update_yaxes(range=[y.min() - pad, y.max() + pad])
    figure.update_xaxes(title_text="Distance (miles)")
    figure.update_yaxes(title_text="Elevation (feet)")
    _style(figure, 300)
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def history_pace_figure(segments):
    """Moving pace for each recorded distance segment, placed at its mile marker."""
    table = segments.loc[segments.distance_miles.astype(float) > 0].copy()
    start = table.distance_miles.astype(float).cumsum().shift(fill_value=0)
    middle = start + table.distance_miles.astype(float) / 2
    pace = table.moving_time_min.astype(float) / table.distance_miles.astype(float)
    labels = pace.map(lambda p: f"{int(round(p * 60)) // 60}:{int(round(p * 60)) % 60:02d}" if np.isfinite(p) else "—")
    figure = go.Figure(go.Bar(
        x=middle.to_numpy(), y=pace.to_numpy(), width=(table.distance_miles.astype(float) * 0.9).to_numpy(),
        marker_color=MOSS, name="Pace",
        customdata=np.column_stack([labels, table.elevation_gain_ft.astype(float).map(lambda v: f"{v:,.0f}")]),
        hovertemplate="Mile %{x:.1f} · %{customdata[0]} /mi<br>+%{customdata[1]} ft climb<extra></extra>"))
    figure.update_xaxes(title_text="Distance (miles)")
    figure.update_yaxes(title_text="Moving pace (min per mile)")
    _style(figure, 260)
    figure.update_layout(showlegend=False, hovermode="closest", bargap=0.05)
    return figure


def time_along_route_figure(prediction, start=None):
    """Estimated moving time (or clock time from a planned start) by distance."""
    table = prediction.segments
    distance = np.concatenate([[0.0], table.cumulative_distance_miles.to_numpy(dtype=float)])
    minutes = np.concatenate([[0.0], table.cumulative_predicted_time_min.to_numpy(dtype=float)])
    if start is not None:
        from datetime import timedelta
        labels = [(start + timedelta(minutes=float(m))).strftime("%I:%M %p").lstrip("0") for m in minutes]
    else:
        labels = [_hours_label(m) + " in" for m in minutes]
    if start is not None:
        from datetime import timedelta
        y = [(start + timedelta(minutes=float(m))).replace(tzinfo=None) for m in minutes]
    else:
        y = minutes / 60
    figure = go.Figure(go.Scatter(
        x=distance, y=y, mode="lines", name="Moving time", line=dict(color=FOREST, width=3),
        fill="tozeroy" if start is None else None, fillcolor="rgba(36,83,63,0.08)", customdata=np.array(labels),
        hovertemplate="Mile %{x:.1f} · %{customdata}<extra></extra>"))
    if getattr(prediction.route, "direction", "one-way") == "out-and-back":
        outbound = table.loc[table.leg.eq("outbound"), "cumulative_distance_miles"]
        if len(outbound):
            figure.add_vline(x=float(outbound.iloc[-1]), line_dash="dash", line_color=GOLD,
                             annotation_text="Turnaround", annotation_position="top left")
    figure.update_xaxes(title_text="Distance (miles)", rangemode="tozero")
    if start is not None:
        figure.update_yaxes(title_text="Clock time (moving only)", tickformat="%-I:%M %p")
    else:
        figure.update_yaxes(title_text="Moving time (hours)", rangemode="tozero")
    _style(figure, 260)
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def recording_map(points, stops=None):
    """A recorded hike's track on OpenStreetMap, split at track segments, with start/finish and stops."""
    valid = points.loc[points.coordinate_valid.fillna(False).astype(bool)] if "coordinate_valid" in points else points
    coordinates = valid[["latitude", "longitude"]].dropna()
    if coordinates.empty:
        raise ValueError("A GPX track with coordinates is required to draw the map.")
    result = folium.Map(location=coordinates.mean().tolist(), tiles="OpenStreetMap",
                        control_scale=True, prefer_canvas=True)
    groups = valid.groupby(["track_index", "track_segment_index"], sort=False) \
        if {"track_index", "track_segment_index"} <= set(valid) else [(None, valid)]
    for _, section in groups:
        line = section[["latitude", "longitude"]].dropna().to_numpy().tolist()
        if len(line) > 1:
            folium.PolyLine(line, color=FOREST, weight=4, opacity=0.9, tooltip="Recorded track").add_to(result)
    folium.Marker(coordinates.iloc[0].tolist(), tooltip="Start",
                  icon=folium.Icon(color="darkgreen", icon="play", prefix="fa")).add_to(result)
    folium.Marker(coordinates.iloc[-1].tolist(), tooltip="Finish",
                  icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(result)
    if stops is not None and len(stops) and "distance_m" in valid:
        track = valid.dropna(subset=["latitude", "longitude", "distance_m"])
        for stop in stops.itertuples():
            meters = float(stop.distance_into_hike_miles) * 1609.344
            nearest = track.iloc[int(np.argmin(np.abs(track.distance_m.to_numpy(dtype=float) - meters)))]
            folium.CircleMarker([nearest.latitude, nearest.longitude], radius=6, color="white", weight=2,
                                fill=True, fill_color=GOLD, fill_opacity=1,
                                tooltip=escape(f"{stop.label} · {float(stop.duration_minutes):.0f} min")).add_to(result)
    result.fit_bounds([coordinates.min().tolist(), coordinates.max().tolist()], padding=(24, 24))
    return result
