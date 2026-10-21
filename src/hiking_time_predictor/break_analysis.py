"""Reproducible analysis of all recorded hikes; no changes to moving-time exports."""

from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .break_model import break_coefficients, file_digest, train_break_predictor
from .breaks import (BreakConfig, CORRECTION_COLUMNS, apply_break_corrections,
                     extract_break_events, load_corrections, summarize_breaks)
from .gpx import ImportConfig, process_hike
from .import_gpx import find_gpx_files, load_confirmed_rests, load_metadata


def break_figures(hikes, events, held_out=None):
    """Standalone Matplotlib figures also reused by the analysis notebook."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figures = {}

    def make(name, title, xlabel, ylabel):
        fig = Figure(figsize=(9, 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        figures[name] = fig
        return ax

    for name, column, title, label in (
        ("distance_break_time", "distance_miles", "Hiking distance and observed meaningful breaks", "Measured moving distance (miles)"),
        ("elevation_break_time", "elevation_gain_ft", "Elevation gain and observed meaningful breaks", "Cleaned elevation gain (ft)"),
        ("moving_break_time", "moving_time_min", "Moving time and observed meaningful breaks", "Measured moving time (min)"),
    ):
        ax = make(name, title, label, "Total ordinary break time (min)")
        ax.scatter(hikes[column], hikes.total_break_minutes, color="#28674c")
    breaks = events.loc[events.break_classification.eq("stationary_break")]
    ax = make("break_durations", "Individual meaningful break durations", "Break duration (min)", "Events")
    ax.hist(breaks.duration_minutes, bins=min(20, max(1, len(breaks))), color="#28674c")
    ax = make("break_frequency", "Break frequency by recording", "Breaks per measured mile", "Recordings")
    ax.hist(hikes.breaks_per_mile, bins=min(10, max(1, len(hikes))), color="#28674c")
    ax = make("break_locations", "Break locations along cumulative hiking distance", "Measured distance into hike (miles)", "Recording")
    for index, hike in enumerate(hikes.itertuples()):
        part = breaks.loc[breaks.hike_id.eq(hike.hike_id)]
        ax.plot([0, hike.distance_miles], [index, index], color="#ccd6ca")
        ax.scatter(part.distance_into_hike_miles, np.full(len(part), index),
                   s=12 + part.duration_minutes * 3, color="#28674c", alpha=0.65)
    ax.set_yticks(range(len(hikes)), [str(i + 1) for i in range(len(hikes))])
    if held_out is not None and not held_out.empty:
        ax = make("actual_predicted", "Held-out whole-hike break predictions", "Observed break minutes", "Predicted break minutes")
        for name, part in held_out.groupby("model"):
            ax.scatter(part.actual_break_minutes, part.predicted_break_minutes, label=name, alpha=0.7)
        high = max(held_out.actual_break_minutes.max(), held_out.predicted_break_minutes.max())
        ax.plot([0, high], [0, high], "--", color="#777777")
        ax.legend()
    return figures


def analysis_markdown(hikes, events, artifact, import_config, example=None):
    breaks = events.loc[events.break_classification.eq("stationary_break")]
    eligible = hikes.loc[hikes.training_eligible]
    minutes = float(breaks.duration_minutes.sum())
    miles = float(hikes.distance_miles.sum())
    mean = float(breaks.duration_minutes.mean()) if len(breaks) else None
    median = float(breaks.duration_minutes.median()) if len(breaks) else None
    between = breaks.groupby("hike_id").distance_into_hike_miles.diff().dropna()

    def number(value):
        return "unavailable" if value is None or pd.isna(value) else f"{value:.2f}"

    lines = ["# Personal break analysis", "",
             "Generated from the actual local GPX recordings. Probable stationary breaks are heuristic labels, "
             "with existing hiker confirmations preserved; they do not establish the reason for stopping.", "",
             f"- Recordings reviewed: **{len(hikes)}**; measured moving distance: **{miles:.2f} miles**.",
             f"- Ordinary breaks identified: **{len(breaks)}**.",
             f"- Total ordinary break minutes: **{minutes:.2f}**.",
             f"- Average individual duration: **{number(mean)} min**; median: **{number(median)} min**.",
             f"- Pooled ordinary break time per mile: **{number(minutes / miles if miles else None)} min/mi**.",
             f"- Mean of per-hike break rates: **{number(hikes.break_minutes_per_mile.mean())} min/mi**.",
             f"- Pooled break frequency: **{number(len(breaks) / miles if miles else None)} breaks/mi**.",
             f"- Mean measured distance between consecutive breaks: **{number(between.mean())} miles** "
             f"({len(between)} within-hike pairs).", "",
             "## Detection and review", "",
             f"Minimum ordinary break: {artifact.config.min_break_minutes:g} minutes; maximum: {artifact.config.max_break_minutes:g} minutes. "
             f"Continuous stationary periods of at least {artifact.config.camping_minutes:g} minutes are camping candidates. "
             f"Heuristic clusters must remain within {artifact.config.geographic_jitter_m:g} meters of their first point.",
             f"The existing importer uses a {import_config.stationary_speed_mph:g} mph stationary threshold, "
             f"{import_config.stationary_radius_m:g} m jitter radius, {import_config.rest_window_seconds:g} s rest window, "
             f"{import_config.max_gap_seconds:g} s recording-gap threshold and {import_config.max_speed_mph:g} mph speed ceiling.", "",
             "Long gaps, missing timestamps, geographic drift and ambiguous intervals remain unknown unless explicitly reviewed. "
             "A hiker confirmation can identify a rest during a recording gap; matching endpoints alone cannot. "
             "Camping and non-hiking events are excluded, as are short stops. Ordinary breaks never include measured moving intervals.", "",
             f"Excluded from ordinary labels: **{hikes.short_stop_minutes.sum():.2f} short-stop minutes**, "
             f"**{hikes.unresolved_minutes.sum():.2f} unresolved minutes**, "
             f"**{hikes.excluded_minutes.sum():.2f} camping/non-hiking minutes**.",
             "Review `data/processed/break_events.csv` in the My Breaking Habits dashboard. Save edits to "
             "`data/break_corrections.csv`, then rerun `python -m hiking_predictor analyze-breaks`. "
             "The corrections table is never overwritten by reprocessing; stale endpoints raise an error. "
             "Existing `data/confirmed_rests.csv` is read and preserved separately.", "",
             "## Per-recording measurements", "",
             "| Hike ID | Miles | Breaks | Break min | Mean min | Median min | Breaks/mi | Min/mi | Mean spacing mi | Eligible |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for h in hikes.itertuples():
        lines.append(f"| {h.hike_id} | {h.distance_miles:.2f} | {h.break_count} | {h.total_break_minutes:.2f} | "
                     f"{number(h.average_break_minutes)} | {number(h.median_break_minutes)} | {number(h.breaks_per_mile)} | "
                     f"{number(h.break_minutes_per_mile)} | {number(h.mean_distance_between_breaks_miles)} | "
                     f"{'yes' if h.training_eligible else h.eligibility_notes} |")
    lines.extend(["", "## Relationships and data limitations", ""])
    for column, label in (("distance_miles", "distance"), ("elevation_gain_ft", "elevation gain")):
        correlation = hikes[[column, "total_break_minutes"]].corr().iloc[0, 1]
        lines.append(f"Pearson correlation of {label} with observed ordinary break time: **{number(correlation)}**. "
                     "This is a descriptive association in a small sample, not evidence of causation.")
    known = hikes.hike_type.isin(["day_hike", "backpacking"])
    lines.append(f"Known hike types: **{int(known.sum())}/{len(hikes)}**. " + artifact.evaluation["model_c_status"])
    if artifact.evaluation["model_c_status"].startswith("Model C evaluated"):
        for kind, part in hikes.loc[known].groupby("hike_type"):
            lines.append(f"{kind}: {len(part)} recordings, {part.total_break_minutes.sum() / part.distance_miles.sum():.2f} observed break min/mi.")
    lines.extend(["", "The recordings provide fewer independent examples than the GPS point count suggests. "
                  "Accepted moving distance omits gaps and stationary jitter; it is smaller than raw GPX geometry and the user's approximate 55-mile history. "
                  "Future untimed geometry includes no movement classification, so mileage may differ. "
                  f"The eligible subset omits recordings with more than {100 * artifact.config.max_unresolved_fraction:g}% unresolved time; its rate can be biased toward better-recorded or longer-rest trips. "
                  "Long rests rely partly on existing user confirmations. Short pauses are omitted; trailhead stops may still need manual review. "
                  "Recordings are treated as independent trips; split files from the same outing must be consolidated before evaluation. "
                  "No weather, pack-weight, fatigue, or intentional-break labels are inferred.", "",
                  "## Prediction and evaluation", "", artifact.reason,
                  f"Eligible whole-hike observations: **{len(eligible)}**. Personal rate: **{number(artifact.minutes_per_mile)} min/mi**.",
                  "The baseline is total eligible break minutes divided by total eligible measured miles. "
                  "A future prediction multiplies that rate by the full planned mileage, including both legs of an out-and-back hike.", "",
                  "Model A fits a straight line using distance. Model B also uses elevation gain and loss. "
                  "Model C adds a one-hot encoded hike type only when both categories have sufficient coverage. "
                  "Each row is a whole recording. Every validation fold holds out one hike and calculates all rates and fitted coefficients from the remaining hikes. "
                  "Elapsed time, observed break count, and break duration are never predictive inputs.", ""])
    if artifact.evaluation["metrics"]:
        lines.extend(["| Approach | Held-out hikes | MAE minutes |", "| --- | ---: | ---: |"])
        for m in artifact.evaluation["metrics"]:
            lines.append(f"| {m['model']} | {m['hikes']} | {m['MAE_minutes']:.2f} |")
        lines.extend(["", "MAE is the average absolute difference between a held-out hike's predicted and observed break minutes. "
                      "These measured errors describe this small, labeled cohort; they are not guaranteed future error bounds. "
                      "With fewer than twelve eligible hikes, regression remains experimental even if its MAE is lower."])
    else:
        lines.append("Too few eligible independent hikes for evaluation. MAE is unavailable; no accuracy score is invented.")
    if example:
        lines.extend(["", "## Existing Hogpen Gap to Poplar Stomp plan", "",
                      "Hike type is supplied as day_hike for the example; blank historical categories remain blank.", "",
                      "| Direction | Miles | Primary moving min | Break min | Total trail min | Experimental regression moving + breaks min |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in example:
            lines.append(f"| {row['direction']} | {row['distance_miles']:.3f} | {row['moving_minutes']:.2f} | "
                         f"{number(row['break_minutes'])} | {number(row['total_minutes'])} | {number(row['experimental_total_minutes'])} |")
        lines.append("\nMoving time retains the existing model-selection policy. Breaks are added exactly once after segment predictions; "
                     "turning breaks off leaves moving time unchanged. The manual override replaces the break allowance and keeps the original prediction visible.")
    lines.extend(["", "## Exploratory visualizations", ""])
    for name, caption in (("distance_break_time", "Distance vs. break time"), ("elevation_break_time", "Elevation gain vs. break time"),
                          ("break_durations", "Break duration histogram"), ("break_frequency", "Breaks per mile"),
                          ("break_locations", "Cumulative distance with break locations; marker size represents duration"),
                          ("moving_break_time", "Moving time vs. break time")):
        lines.extend([f"![{caption}](data/processed/break_analysis/{name}.png)", ""])
    return "\n".join(lines)


def analyze_breaks(project, *, break_config_path=None, import_config_path=None):
    """Review every raw recording with existing preprocessing and confirmations."""
    project = Path(project).resolve()
    break_path = Path(break_config_path) if break_config_path else project / "config/breaks.json"
    import_path = Path(import_config_path) if import_config_path else project / "config/gpx_import.json"
    for explicit, path in ((break_config_path, break_path), (import_config_path, import_path)):
        if explicit is not None and not path.exists():
            raise ValueError(f"Configuration file does not exist: {path}")
    config = BreakConfig(**json.loads(break_path.read_text())) if break_path.exists() else BreakConfig()
    import_config = ImportConfig(**json.loads(import_path.read_text())) if import_path.exists() else ImportConfig()
    files = find_gpx_files(project / "data/raw")
    if not files:
        raise ValueError("No recorded GPX files found in data/raw.")
    metadata_path = project / "data/hike_metadata.csv"
    rests_path = project / "data/confirmed_rests.csv"
    correction_path = project / "data/break_corrections.csv"
    metadata = load_metadata(metadata_path, files)
    rests = load_confirmed_rests(rests_path)
    summaries, points, intervals = [], [], []
    for file in files:
        row = metadata.loc[metadata.source_file.eq(file.name)].iloc[0]
        details = {"hike_type": row.hike_type or None,
                   "pack_weight_lbs": float(row.pack_weight_lbs) if row.pack_weight_lbs else None}
        summary, _, track, edges = process_hike(file, row.hike_id, details, import_config,
                                              rests.loc[rests.source_file.eq(file.name)])
        summaries.append(summary)
        points.append(track)
        intervals.append(edges)
    events = extract_break_events(pd.concat(points, ignore_index=True), pd.concat(intervals, ignore_index=True), config)
    events = apply_break_corrections(events, load_corrections(correction_path), config)
    hikes = summarize_breaks(pd.DataFrame(summaries), events, config)
    artifact, held_out = train_break_predictor(hikes, config)
    artifact.evaluation["source_files"] = [p.name for p in files]
    output = project / "data/processed/break_analysis"
    output.mkdir(parents=True, exist_ok=True)
    # Create only absent correction scaffolds; never erase human review.
    if not correction_path.exists():
        pd.DataFrame(columns=CORRECTION_COLUMNS).to_csv(correction_path, index=False)
    events_path = output.parent / "break_events.csv"
    hikes_path = output.parent / "break_hike_summary.csv"
    events.to_csv(events_path, index=False)
    hikes.to_csv(hikes_path, index=False)
    held_out.to_csv(output / "held_out_predictions.csv", index=False)
    pd.DataFrame(artifact.evaluation["metrics"], columns=["model", "hikes", "MAE_minutes"]).to_csv(output / "model_metrics.csv", index=False)
    break_coefficients(artifact).to_csv(output / "coefficients.csv", index=False)
    (output / "evaluation.json").write_text(json.dumps(artifact.evaluation, indent=2) + "\n")
    (output / "settings.json").write_text(json.dumps({"break_config": asdict(config), "import_config": asdict(import_config)}, indent=2) + "\n")
    dependencies = [*files, metadata_path, rests_path, correction_path, break_path, import_path, events_path, hikes_path,
                    *[output / name for name in ("held_out_predictions.csv", "model_metrics.csv", "evaluation.json", "settings.json", "coefficients.csv")]]
    artifact.provenance = {str(p.resolve().relative_to(project) if p.resolve().is_relative_to(project) else p.resolve()): file_digest(p)
                           for p in dependencies}
    (project / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, project / "models/break_predictor.joblib")
    for name, figure in break_figures(hikes, events, held_out).items():
        figure.savefig(output / f"{name}.png", dpi=150)
    if held_out.empty:
        (output / "actual_predicted.png").unlink(missing_ok=True)
    example = []
    planned_files = list((project / "data/planned").glob("*Hogpen*Poplar*.gpx"))
    if len(planned_files) == 1:
        from .planned import predict_planned_route
        from .predictor import ModelArtifact, load_model
        from .route_report import export_prediction
        from .web_upload import trip_statistics
        moving_path = project / "models/hiking_predictor.joblib"
        moving_artifact = load_model(moving_path) if moving_path.exists() else ModelArtifact()
        for direction in ("one-way", "out-and-back"):
            result = predict_planned_route(planned_files[0], "day_hike", direction,
                                           artifact=moving_artifact,
                                           break_artifact=artifact)
            stats = trip_statistics(result, include_predicted_breaks=True)
            stats["direction"] = direction
            stats["experimental_total_minutes"] = (result.forecast.experimental_minutes + stats["break_minutes"]
                                                    if result.forecast.experimental_minutes is not None else None)
            example.append(stats)
            export_prediction(result, project / "reports/planned/hogpen-break-estimates" / direction)
        pd.DataFrame(example).to_csv(output / "hogpen_example.csv", index=False)
    report = analysis_markdown(hikes, events, artifact, import_config, example)
    (project / "BREAK_ANALYSIS.md").write_text(report)
    return hikes, events, artifact
