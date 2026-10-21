"""Command line: python -m hiking_time_predictor.import_gpx [--file PATH]."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import HIKE_TYPES
from .gpx import ImportConfig, process_hike


METADATA_COLUMNS = ["source_file", "hike_id", "hike_type", "pack_weight_lbs", "notes"]
REST_COLUMNS = ["source_file", "start_time_utc", "end_time_utc", "notes"]


def load_confirmed_rests(path):
    """Read explicit interval confirmations; an absent table means no overrides."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=REST_COLUMNS)
    rests = pd.read_csv(path, dtype="string", keep_default_na=False)
    if set(rests.columns) != set(REST_COLUMNS) or rests.columns.duplicated().any():
        raise ValueError(f"Confirmed rest columns must be {REST_COLUMNS}.")
    rests = rests[REST_COLUMNS].copy()
    rests["source_file"] = rests["source_file"].str.strip()
    if rests["source_file"].eq("").any():
        raise ValueError("Confirmed rests need a nonblank source_file.")
    for column in ("start_time_utc", "end_time_utc"):
        timestamps = []
        for value in rests[column]:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Confirmed rest timestamps must include a timezone, such as Z for UTC.")
            timestamps.append(timestamp.tz_convert("UTC"))
        rests[column] = pd.to_datetime(timestamps, utc=True)
    if rests["end_time_utc"].lt(rests["start_time_utc"]).any():
        raise ValueError("A confirmed rest cannot end before it starts.")
    if rests.duplicated(["source_file", "start_time_utc", "end_time_utc"]).any():
        raise ValueError("Duplicate confirmed rest intervals are not allowed.")
    return rests


def find_gpx_files(raw_dir):
    """Include normal .gpx files and numbered downloads such as .gpx-2."""
    return sorted(p for p in Path(raw_dir).iterdir()
                  if p.is_file() and re.search(r"\.gpx(?:-\d+)?$", p.name, re.IGNORECASE))


def default_hike_id(filename):
    """An identifier from the filename, not an inferred trip category."""
    stem = re.sub(r"\.gpx(?=-\d+$|$)", "", filename, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "hike"


def load_metadata(path, files):
    """Preserve supplied metadata; scaffold blank categories/weights/notes.

    IDs are assigned only for new rows. Existing blank IDs and duplicate IDs
    require correction instead of silently changing the user's identity mapping.
    The caller writes this table only after all recordings import successfully.
    """
    path = Path(path)
    if path.exists():
        table = pd.read_csv(path, dtype="string", keep_default_na=False)
        if set(table.columns) != set(METADATA_COLUMNS) or table.columns.duplicated().any():
            raise ValueError(f"Metadata columns must be {METADATA_COLUMNS}.")
        table = table[METADATA_COLUMNS].copy()
    else:
        table = pd.DataFrame(columns=METADATA_COLUMNS)
    for name in METADATA_COLUMNS[:-1]:
        table[name] = table[name].astype("string").str.strip()
    if table["source_file"].eq("").any() or table["source_file"].duplicated().any():
        raise ValueError("Metadata source_file must be nonblank and unique.")
    if table["hike_id"].eq("").any() or table["hike_id"].duplicated().any():
        raise ValueError("Metadata hike_id must be nonblank and unique across all hikes.")
    if not table["hike_type"].isin(["", *HIKE_TYPES]).all():
        raise ValueError("Metadata hike_type must be blank, day_hike, or backpacking.")
    weights = table["pack_weight_lbs"]
    numeric = pd.to_numeric(weights.where(weights.ne("")), errors="coerce")
    invalid = weights.ne("") & (~np.isfinite(numeric.to_numpy(dtype=float)) | numeric.lt(0))
    if invalid.any():
        raise ValueError("pack_weight_lbs must be blank or a finite number >= 0.")
    used = set(table["hike_id"])
    for file in files:
        if file.name not in set(table["source_file"]):
            hike_id = default_hike_id(file.name)
            if hike_id in used:
                # A stable suffix handles filenames that normalize to the same ID.
                hike_id += "-" + hashlib.sha256(file.name.encode()).hexdigest()[:8]
            if hike_id in used:
                raise ValueError(f"ID collision for {file.name}; add an explicit metadata row.")
            used.add(hike_id)
            table.loc[len(table)] = [file.name, hike_id, "", "", ""]
    return table.astype("string").copy(deep=True)


def import_recordings(files, metadata_path, output_dir, config=None, confirmed_rests_path=None):
    """Build all tables before writing; original recordings are only read."""
    config = config or ImportConfig()
    files = [Path(p).resolve() for p in files]
    if not files:
        raise ValueError("No GPX recordings found.")
    if len({p.name for p in files}) != len(files):
        raise ValueError("Source filenames must be unique (metadata is keyed by filename).")
    metadata_path, output_dir = Path(metadata_path).resolve(), Path(output_dir).resolve()
    raw_parents = {p.parent for p in files}
    if any(output_dir == p or p in output_dir.parents for p in raw_parents):
        raise ValueError("Write processed outputs outside the raw recording folder.")
    if metadata_path in files:
        raise ValueError("The metadata path cannot be an original GPX recording.")
    table = load_metadata(metadata_path, files)
    if confirmed_rests_path is None:
        confirmed_rests_path = metadata_path.parent / "confirmed_rests.csv"
    rests = load_confirmed_rests(confirmed_rests_path)
    summaries, segment_tables, point_tables, interval_tables, hashes = [], [], [], [], {}
    for file in files:
        row = table.loc[table["source_file"].eq(file.name)].iloc[0]
        metadata = {"hike_type": row["hike_type"] or None,
                    "pack_weight_lbs": float(row["pack_weight_lbs"]) if row["pack_weight_lbs"] else None}
        try:
            file_rests = rests.loc[rests["source_file"].eq(file.name)]
            summary, segments, points, intervals = process_hike(file, row["hike_id"], metadata, config, file_rests)
        except Exception as error:
            raise ValueError(f"Could not import {file.name}: {error}") from error
        summaries.append(summary)
        segment_tables.append(segments)
        point_tables.append(points)
        interval_tables.append(intervals)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    summary_table = pd.DataFrame(summaries)
    segments = pd.concat(segment_tables, ignore_index=True)
    points = pd.concat(point_tables, ignore_index=True)
    intervals = pd.concat(interval_tables, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(metadata_path, index=False)
    for name, frame in (("hiking_summary", summary_table), ("hiking_segments", segments),
                        ("hiking_trackpoints", points), ("hiking_intervals", intervals)):
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    used_rests = rests.loc[rests["source_file"].isin(hashes)].copy()
    for column in ("start_time_utc", "end_time_utc"):
        used_rests[column] = used_rests[column].map(lambda timestamp: timestamp.isoformat())
    (output_dir / "import_settings.json").write_text(json.dumps({
        "config": asdict(config), "source_sha256": hashes,
        "confirmed_rests": used_rests.to_dict(orient="records"),
    }, indent=2) + "\n", encoding="utf-8")
    return summary_table, segments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import recorded GPX tracks without training a model.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--file", type=Path, action="append", help="Import only this recording (repeatable).")
    parser.add_argument("--metadata", type=Path, default=Path("data/hike_metadata.csv"))
    parser.add_argument("--confirmed-rests", type=Path, help="CSV with confirmed rest intervals; defaults beside metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, help="JSON object with ImportConfig threshold values.")
    parser.add_argument("--segment-miles", type=float, help="Override configured segment size.")
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if args.segment_miles is not None:
            values["segment_miles"] = args.segment_miles
        config = ImportConfig(**values)
        files = args.file or find_gpx_files(args.raw_dir)
        summary, segments = import_recordings(files, args.metadata, args.output_dir, config, args.confirmed_rests)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    columns = ["hike_id", "trackpoint_count", "raw_distance_miles", "distance_miles",
               "raw_elevation_gain_ft", "elevation_gain_ft", "elapsed_time_min",
               "moving_time_min", "stationary_time_min", "unknown_time_min", "segment_count"]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {len(segments)} segments for {len(summary)} recordings to {args.output_dir}.")
    print(f"Review quality_flags and fill known metadata in {args.metadata}.")
    print("Outputs contain only the files selected in this run; rerun without --file for the full dataset.")


if __name__ == "__main__":
    main()
