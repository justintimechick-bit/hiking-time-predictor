"""Markdown/CSV and standalone plots for planned predictions; no CLI dependency."""

from pathlib import Path

import numpy as np

from .gpx import FEET_PER_METER
from .modeling import GEOMETRY, MODEL_FEATURES, MODEL_LABELS


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _duration(minutes):
    return f"{minutes:.2f} min ({int(minutes // 60)} h {minutes % 60:.1f} min)"


def prediction_markdown(result):
    """Render the same report in the terminal or a saved Markdown document."""
    route, forecast, artifact = result.route, result.forecast, result.artifact
    totals = result.segments[GEOMETRY].sum()
    regression = result.segments.regression_time_min.sum(min_count=1)
    has_regression = np.isfinite(regression)
    lines = [f"# {_text(route.name)}", "", f"Source: {_text(result.input_path.name)}",
             f"Direction: **{route.direction}** | Hike type: **{route.segments.hike_type.iloc[0]}**",
             f"Distance: **{totals.distance_miles:.3f} miles** | Elevation gain: **{totals.elevation_gain_ft:.1f} ft** "
             f"| Elevation loss: **{totals.elevation_loss_ft:.1f} ft**", "",
             f"Model used for primary estimate: **{forecast.label}**",
             f"Predicted moving time: **{_duration(forecast.moving_minutes)}**",
             f"Average predicted moving speed: **{60 * totals.distance_miles / forecast.moving_minutes:.2f} mph**",
             f"Naismith's Rule baseline: **{_duration(forecast.naismith_minutes)}**", "",
             f"Model status: {_text(artifact.reason)}",
             f"Preprocessing: {route.config.segment_miles:g}-mile bins; centered "
             f"{route.config.elevation_window_m:g}-meter median window on a "
             f"{route.config.elevation_grid_m:g}-meter elevation grid."]
    if artifact.model_name:
        status = "selected" if artifact.selected else "experimental"
        lines.extend([f"Saved regression: **{MODEL_LABELS[artifact.model_name]} ({status})**",
                      f"Ordered input schema: `{MODEL_FEATURES[artifact.model_name]}`"])
    if has_regression:
        label = "Saved-model" if artifact.selected else "Experimental saved-model"
        lines.extend([f"{label} moving time: **{_duration(regression)}**",
                      f"{label} average moving speed: **{60 * totals.distance_miles / regression:.2f} mph**"])
    if result.break_forecast is not None:
        breaks = result.break_forecast
        lines.extend(["", "## Break time estimation", "", f"Approach: **{breaks.label}**"])
        if breaks.predicted_minutes is not None:
            lines.extend([f"Predicted break time: **{_duration(breaks.predicted_minutes)}**",
                          f"Predicted total trail time: **{_duration(forecast.moving_minutes + breaks.predicted_minutes)}**",
                          f"Personal distance baseline: **{_duration(breaks.baseline_minutes)}**",
                          f"Expected number of breaks (historical rate): **{breaks.expected_break_count:.2f}**"])
            if breaks.expected_average_minutes is not None:
                lines.append(f"Historical average break duration: **{breaks.expected_average_minutes:.2f} min**")
            if has_regression and not artifact.selected:
                lines.append(f"Experimental saved-model moving + breaks: **{_duration(regression + breaks.predicted_minutes)}**")
        else:
            lines.append("Break estimate unavailable; the reported moving time remains usable.")
        lines.extend(f"- {_text(warning)}" for warning in breaks.warnings)
    lines.extend(["", "## Leg estimates", "",
                  "| Leg | Miles | Gain ft | Loss ft | Primary min | Saved regression min | Naismith min |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for leg, row in result.leg_totals.iterrows():
        learned = f"{row.regression_time_min:.2f}" if np.isfinite(row.regression_time_min) else "unavailable"
        lines.append(f"| {leg} | {row.distance_miles:.3f} | {row.elevation_gain_ft:.1f} | "
                     f"{row.elevation_loss_ft:.1f} | {row.predicted_time_min:.2f} | {learned} | {row.naismith_time_min:.2f} |")
    lines.extend(["", "## Segments", "",
                  "Primary time/pace follow the model status above. Regression columns show the saved fit separately. "
                  "Cumulative values continue across the turn. An asterisk marks segment geometry outside the training range.", "",
                  "| Segment | Leg | Miles | Gain ft | Loss ft | Primary min | Cumulative min | Pace min/mi | Regression min | Regression cumulative min | Regression pace min/mi |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in result.segments.itertuples():
        number = f"{row.segment_number}{'*' if row.outside_training_range else ''}"
        learned = (f"{row.regression_time_min:.2f} | {row.cumulative_regression_time_min:.2f} | {row.regression_pace_min_per_mile:.2f}"
                   if has_regression else "unavailable | unavailable | unavailable")
        lines.append(f"| {number} | {row.leg} | {row.distance_miles:.3f} | {row.elevation_gain_ft:.1f} | "
                     f"{row.elevation_loss_ft:.1f} | {row.predicted_time_min:.2f} | "
                     f"{row.cumulative_predicted_time_min:.2f} | {row.predicted_pace_min_per_mile:.2f} | {learned} |")
    lines.extend(["", "## Saved model evaluation", ""])
    if artifact.evaluation.get("segment_metrics"):
        lines.extend([f"Method: **{_text(artifact.evaluation.get('method', 'not recorded'))}**. "
                      f"Eligible observations: {artifact.evaluation.get('eligible_segments', 'unknown')} segments from "
                      f"{artifact.evaluation.get('eligible_hikes', 'unknown')} hikes. "
                      f"Verified complete held-out hikes: {artifact.evaluation.get('verified_held_out_hikes', 0)}.", "",
                      "| Evaluation scope | Model | Observations | Hikes | MAE min | RMSE min |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"])
        for key, scope in (("segment_metrics", "Segment"), ("whole_hike_metrics", "Whole hike")):
            for row in artifact.evaluation.get(key, []):
                lines.append(f"| {scope} | {_text(row['model'])} | {row['observations']} | {row['hikes']} | "
                             f"{row['MAE_min']:.3f} | {row['RMSE_min']:.3f} |")
    else:
        lines.append("No evaluated regression metrics are available.")
    lines.extend(["", "MAE is the average absolute error on held-out observations. RMSE gives larger errors more weight. "
                  "These historical errors are **not prediction intervals** or a guaranteed error bound for this route.", "",
                  "## Limitations and warnings", ""])
    limitations = [*route.notes, *forecast.warnings,
                  "Moving time excludes breaks, camping, and other stops. Weather, trail surface, fatigue, and pack weight are not modeled.",
                  "Planned geometry uses the training haversine miles, filtered elevation feet, edge allocation, and distance bins. "
                  "Historical training geometry additionally excludes detected stops/unknown intervals using recorded timestamps; "
                  "that movement classification cannot be applied to an untimed plan.",
                  "Sparse route points can miss bends and elevation changes. Disconnected sections remain separate; any connecting travel is unestimated.",
                  "A regression intercept is applied to every segment, so short final segments and section count can affect the sum.",
                  "Naismith uses 20 min/mile + 0.03 min/ascent ft, with no descent or hike-type adjustment."]
    if artifact.estimator is not None and artifact.model_name != "C":
        limitations.append("This regression uses geometry only; supplying backpacking does not add a pack/type adjustment. "
                           + artifact.evaluation.get("model_c_status", ""))
    lines.extend(f"- {_text(note)}" for note in dict.fromkeys(limitations))
    lines.extend(["", "## Charts", "", "![Elevation profile](elevation_profile.png)", "",
                  "![Predicted time by segment](segment_times.png)", "",
                  "![Cumulative distance versus predicted time](cumulative_time.png)", ""])
    return "\n".join(lines)


def _save_charts(result, output_dir):
    # Figure + Agg canvas avoids GUI state and works in a future web server.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def figure(ylabel, xlabel):
        fig = Figure(figsize=(10, 4.5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set(title=result.route.name, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        return fig, ax

    segments = result.segments
    experimental = result.forecast.experimental_minutes is not None
    fig, ax = figure("Elevation (ft)", "Distance along hike (miles)")
    # Separate lines keep plots from drawing invented connections across gaps.
    for (leg, section), profile in result.route.profile.groupby(["leg", "section_index"], sort=False):
        ax.plot(profile.distance_miles, profile.smoothed_elevation_m * FEET_PER_METER,
                label=f"{leg}, section {section + 1}")
    ax.legend()
    fig.savefig(output_dir / "elevation_profile.png", dpi=150)

    fig, ax = figure("Moving time (min)", "Segment number")
    numbers = segments.segment_number
    ax.bar(numbers, segments.predicted_time_min, color="#28677b", label=result.forecast.label)
    if experimental:
        ax.plot(numbers, segments.regression_time_min, "o--", color="#b45d27", label="Experimental saved regression")
    elif result.artifact.selected:
        ax.plot(numbers, segments.naismith_time_min, "o--", color="#b45d27", label="Naismith baseline")
    ax.legend()
    fig.savefig(output_dir / "segment_times.png", dpi=150)

    fig, ax = figure("Cumulative distance (miles)", "Cumulative predicted moving time (min)")
    distance = np.r_[0, segments.cumulative_distance_miles]
    ax.plot(np.r_[0, segments.cumulative_predicted_time_min], distance, "o-", label=result.forecast.label)
    if experimental:
        ax.plot(np.r_[0, segments.cumulative_regression_time_min], distance, "o--", label="Experimental saved regression")
    elif result.artifact.selected:
        ax.plot(np.r_[0, segments.naismith_time_min.cumsum()], distance, "o--", label="Naismith baseline")
    ax.legend()
    fig.savefig(output_dir / "cumulative_time.png", dpi=150)


def export_prediction(result, output_dir):
    """Write report, segment CSV, profile CSV and three PNGs outside source data."""
    output_dir = Path(output_dir).resolve()
    project = Path(__file__).resolve().parents[2]
    protected = [project / "data", project / "models"]
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise ValueError("Export predictions outside data/ and models/; historical inputs must remain separate.")
    names = ["report.md", "predictions.csv", "elevation_profile.csv", "elevation_profile.png",
             "segment_times.png", "cumulative_time.png"]
    paths = {name: output_dir / name for name in names}
    if any(path.resolve() in (result.input_path, result.model_path) for path in paths.values()):
        raise ValueError("Export would overwrite the input GPX or saved model.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_charts(result, output_dir)
    has_regression = result.segments.regression_time_min.notna().all()
    status = ("selected" if result.artifact.selected else "experimental") if has_regression else "unavailable"
    result.segments.assign(primary_model=result.forecast.label,
                           regression_model=MODEL_LABELS.get(result.artifact.model_name, "unavailable"),
                           regression_status=status,
                           hike_direction=result.route.direction).to_csv(paths["predictions.csv"], index=False)
    result.route.profile.to_csv(paths["elevation_profile.csv"], index=False)
    paths["report.md"].write_text(prediction_markdown(result), encoding="utf-8")
    return paths
