"""Small terminal interface for training and future-route moving-time estimates."""

import argparse
from dataclasses import asdict
import hashlib
import json
from math import floor
from pathlib import Path
import re

import pandas as pd

from .gpx import ImportConfig
from .modeling import GEOMETRY, MODEL_LABELS
from .predictor import DEFAULT_MODEL, ModelArtifact, forecast_route, load_model, save_model, train_predictor
from .planned import predict_planned_route
from .route_report import export_prediction, prediction_markdown
from .routes import gpx_route, manual_route


def format_duration(minutes):
    rounded = floor(minutes + 0.5)
    hours, remainder = divmod(rounded, 60)
    return f"{hours} h {remainder:02d} min"


def _train(args):
    settings_path = args.settings or Path("data/processed/import_settings.json")
    if args.settings is not None or settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict) or not isinstance(settings.get("config"), dict):
            raise ValueError("Import settings must contain a 'config' object from the recorded GPX importer.")
        config = ImportConfig(**settings["config"])
    else:
        config = ImportConfig()
        print("Warning: no import settings found; assuming default 0.5-mile segments and 30-meter elevation smoothing.")
    segments = pd.read_csv(args.segments, dtype={"hike_id": "string"})
    summary_path = args.summary or args.segments.parent / "hiking_summary.csv"
    summary = (pd.read_csv(summary_path, dtype={"hike_id": "string"})
               if args.summary is not None or summary_path.exists() else None)
    if args.model.resolve() in {args.segments.resolve(), summary_path.resolve(), settings_path.resolve()}:
        raise ValueError("Save the model to a separate file; do not overwrite training inputs.")
    artifact = train_predictor(segments, summary, config)
    artifact.provenance.update({"segments_sha256": hashlib.sha256(args.segments.read_bytes()).hexdigest(),
                                "summary_sha256": (hashlib.sha256(summary_path.read_bytes()).hexdigest()
                                                   if summary is not None else None),
                                "import_config": asdict(config)})
    save_model(artifact, args.model)
    info = artifact.evaluation
    print(f"Eligible data: {info['eligible_segments']} segments from {info['eligible_hikes']} hikes.")
    print(f"Verified complete held-out hikes: {info.get('verified_held_out_hikes', 0)}")
    if artifact.model_name:
        status = "selected" if artifact.selected else "experimental"
        print(f"Saved {MODEL_LABELS[artifact.model_name]} ({status}) to {args.model}")
    else:
        print(f"Saved baseline settings to {args.model}")
    print(artifact.reason)
    print(info["model_c_status"])


def _predict(args):
    path = args.model or DEFAULT_MODEL
    # Missing default artifacts allow a useful baseline without a training step.
    # An explicitly requested file must load successfully.
    artifact = load_model(path) if args.model is not None or path.exists() else ModelArtifact()
    if args.gpx is not None:
        if args.gain is not None or args.loss is not None:
            raise ValueError("--gpx cannot be combined with --gain or --loss; geometry comes from the file.")
        route = gpx_route(args.gpx, args.hike_type, artifact.config)
    else:
        if args.gain is None or args.loss is None:
            raise ValueError("Manual input requires --distance (miles), --gain (feet), and --loss (feet).")
        route = manual_route(args.distance, args.gain, args.loss, args.hike_type, artifact.config)
    forecast = forecast_route(route.segments, artifact)
    totals = route.segments[GEOMETRY].sum()
    print(f"Input: {route.source}; {args.hike_type}; {len(route.segments)} segment(s)")
    print(f"Distance: {totals.distance_miles:.2f} miles | Gain: {totals.elevation_gain_ft:.0f} ft "
          f"| Loss: {totals.elevation_loss_ft:.0f} ft")
    print(f"Predicted moving time: {format_duration(forecast.moving_minutes)} ({forecast.label})")
    from .break_model import load_break_model, predict_break_time
    breaks = predict_break_time(totals.distance_miles, totals.elevation_gain_ft, totals.elevation_loss_ft,
                                args.hike_type, load_break_model(project=Path(__file__).resolve().parents[2]))
    if breaks.predicted_minutes is None:
        print(f"Predicted break time: unavailable. {' '.join(breaks.warnings)}")
    else:
        print(f"Predicted break time: {format_duration(breaks.predicted_minutes)} ({breaks.label})")
        print(f"Predicted total trail time: {format_duration(forecast.moving_minutes + breaks.predicted_minutes)}")
        print(f"Personal break baseline comparison: {breaks.baseline_minutes:.2f} min")
        for warning in breaks.warnings:
            print(f"Break estimate note: {warning}")
    print(f"Naismith's Rule estimated time: {format_duration(forecast.naismith_minutes)}")
    difference = forecast.moving_minutes - forecast.naismith_minutes
    print(f"Difference (prediction - Naismith): {difference:+.1f} min")
    print(artifact.reason)
    if forecast.experimental_minutes is not None:
        print(f"Experimental {MODEL_LABELS[artifact.model_name]}: {format_duration(forecast.experimental_minutes)}")
        print(f"Experimental - Naismith: {forecast.experimental_minutes - forecast.naismith_minutes:+.1f} min")
    if artifact.estimator is not None and artifact.model_name != "C":
        print("This regression uses geometry only; it does not adjust for hike type.")
    for note in route.notes:
        print(f"Note: {note}")
    for warning in forecast.warnings:
        print(f"Warning: {warning}")
    print("Moving time excludes breaks. Naismith uses 20 min/mile + 0.03 min/ascent ft.")


def _predict_planned(args):
    result = predict_planned_route(args.gpx, args.hike_type, args.direction,
                                   model_path=args.model, missing_elevation=args.missing_elevation)
    slug = re.sub(r"[^a-z0-9]+", "-", result.route.name.lower()).strip("-") or "route"
    output_dir = args.output_dir or Path("reports/planned") / slug / args.direction
    paths = export_prediction(result, output_dir)
    print(prediction_markdown(result))
    print(f"Report and charts: {paths['report.md']}")
    print(f"Segment CSV: {paths['predictions.csv']}")


def _analyze_breaks(args):
    from .break_analysis import analyze_breaks
    hikes, events, artifact = analyze_breaks(args.project, break_config_path=args.break_config,
                                           import_config_path=args.import_config)
    breaks = events.loc[events.break_classification.eq("stationary_break")]
    print(f"Reviewed {len(hikes)} recordings; identified {len(breaks)} ordinary breaks totaling {breaks.duration_minutes.sum():.2f} minutes.")
    print(artifact.reason)
    print(f"Report: {args.project.resolve() / 'BREAK_ANALYSIS.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hiking_predictor",
                                     description="Predict hiking moving time from manual totals or planned GPX geometry.")
    commands = parser.add_subparsers(dest="command", required=True)
    breaks = commands.add_parser("analyze-breaks", help="Review recorded GPX files, export break events, evaluate and save a personal break predictor.")
    breaks.add_argument("--project", type=Path, default=Path.cwd())
    breaks.add_argument("--break-config", type=Path, help="BreakConfig JSON; default config/breaks.json.")
    breaks.add_argument("--import-config", type=Path, help="Existing ImportConfig JSON; default config/gpx_import.json.")
    breaks.set_defaults(run=_analyze_breaks)
    train = commands.add_parser("train", help="Evaluate by hike, fit, and save a trusted local joblib artifact.")
    train.add_argument("--segments", type=Path, default=Path("data/processed/hiking_segments.csv"))
    train.add_argument("--summary", type=Path, help="Recorded summary CSV; defaults beside the segments CSV.")
    train.add_argument("--settings", type=Path, help="Importer import_settings.json; defaults to data/processed/import_settings.json.")
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Output joblib path.")
    train.set_defaults(run=_train)
    predict = commands.add_parser("predict", help="Load the saved model and estimate future moving time.")
    geometry = predict.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--distance", type=float, help="Horizontal distance in miles.")
    geometry.add_argument("--gpx", type=Path, help="GPX route or track; timestamps are not required.")
    predict.add_argument("--gain", type=float, help="Elevation gain in feet (manual input only).")
    predict.add_argument("--loss", type=float, help="Elevation loss in feet, as a positive magnitude (manual input only).")
    predict.add_argument("--type", dest="hike_type", required=True, choices=("day_hike", "backpacking"))
    predict.add_argument("--model", type=Path, help=f"Trusted joblib artifact; default: {DEFAULT_MODEL}.")
    predict.set_defaults(run=_predict)
    planned = commands.add_parser("predict-route", help="Predict one-way or out-and-back GPX geometry and export a report.")
    planned.add_argument("--gpx", type=Path, required=True, help="Planned GPX track or route; no timestamps needed.")
    planned.add_argument("--type", dest="hike_type", required=True, choices=("day_hike", "backpacking"))
    planned.add_argument("--direction", choices=("one-way", "out-and-back"), default="one-way")
    planned.add_argument("--model", type=Path, help=f"Trusted joblib artifact; default: {DEFAULT_MODEL}.")
    planned.add_argument("--missing-elevation", choices=("error", "interpolate"), default="error",
                         help="Reject missing elevation, or explicitly interpolate interior gaps within sections.")
    planned.add_argument("--output-dir", type=Path, help="Report directory; defaults to reports/planned/<route>/<direction>.")
    planned.set_defaults(run=_predict_planned)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
