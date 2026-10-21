"""Ephemeral GPX upload adapters for web interfaces, without UI dependencies.

Uploaded tracks are written only to a temporary directory so the existing
file-based geometry and prediction workflows can be reused unchanged.
"""

from contextlib import contextmanager
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .gpx import ImportConfig
from .planned import PlannedPrediction, predict_planned_route
from .predictor import ModelArtifact
from .routes import Route, gpx_route


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _safe_upload_name(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Upload a file with a .gpx filename.")
    # Browser filenames are normally basenames; handle both path separators
    # explicitly so supplied paths can never escape the temporary directory.
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if Path(basename).suffix.lower() != ".gpx":
        raise ValueError("Upload a .gpx file.")
    stem = re.sub(r"[^\w .()-]", "_", Path(basename).stem).strip(" ._")[:100]
    return f"{stem or 'uploaded-trail'}.gpx"


@contextmanager
def _temporary_upload(contents: bytes, filename: str):
    safe_name = _safe_upload_name(filename)
    if not isinstance(contents, bytes):
        raise ValueError("GPX upload contents must be bytes.")
    if not contents or not contents.strip():
        raise ValueError("The GPX file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("The GPX file exceeds the 20 MiB upload limit.")
    from .safe_xml import validate_gpx_xml
    validate_gpx_xml(contents)
    try:
        contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The GPX file must contain UTF-8 XML text.") from error
    with TemporaryDirectory(prefix="hiking-time-upload-") as directory:
        path = Path(directory) / safe_name
        path.write_bytes(contents)
        yield path


def preview_upload(contents: bytes, filename: str, hike_type: str,
                   direction: str = "one-way", *, config: ImportConfig | None = None) -> Route:
    """Validate an upload and return existing GPX geometry, including return legs.

    An unnamed GPX uses the sanitized original filename stem as its trail name.
    No uploaded file remains on disk after this call, even if parsing fails.
    """
    with _temporary_upload(contents, filename) as path:
        return gpx_route(path, hike_type, config, direction=direction)


def predict_upload(contents: bytes, filename: str, hike_type: str,
                   direction: str = "one-way", *, artifact: ModelArtifact, break_artifact=None) -> PlannedPrediction:
    """Predict with a supplied saved artifact; never load, fit, or store a model.

    The result retains the temporary input path as provenance only; that path
    has been deleted by the time this function returns. Tables stay in memory.
    The existing forecast policy chooses the primary and experimental estimates.
    """
    with _temporary_upload(contents, filename) as path:
        return predict_planned_route(path, hike_type, direction, artifact=artifact, break_artifact=break_artifact)


def trip_statistics(prediction: PlannedPrediction, break_minutes: float = 0, *,
                    include_predicted_breaks=False, manual_break_minutes=None) -> dict:
    """Summarize the same segment predictions used by charts and CSV exports.

    Breaks are a separately supplied planning allowance, never a model feature.
    """
    original = prediction.break_forecast.predicted_minutes if prediction.break_forecast else None
    allowance = break_minutes
    if include_predicted_breaks:
        allowance = (manual_break_minutes if manual_break_minutes is not None
                     else original if original is not None else 0)
    if isinstance(allowance, bool):
        raise ValueError("Planned break time must be a finite nonnegative number of minutes.")
    try:
        breaks = float(allowance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Planned break time must be a finite nonnegative number of minutes.") from error
    if not isfinite(breaks) or breaks < 0:
        raise ValueError("Planned break time must be a finite nonnegative number of minutes.")
    segments = prediction.segments
    distance = float(segments.distance_miles.sum())
    moving = float(prediction.forecast.moving_minutes)
    total = moving + breaks
    if not isfinite(total):
        raise ValueError("Planned total trip time must be finite.")
    return {
        "distance_miles": distance,
        "elevation_gain_ft": float(segments.elevation_gain_ft.sum()),
        "elevation_loss_ft": float(segments.elevation_loss_ft.sum()),
        "segment_count": len(segments),
        "moving_minutes": moving,
        "break_minutes": breaks,
        "total_minutes": total,
        "average_speed_mph": distance / moving * 60,
        "naismith_minutes": float(prediction.forecast.naismith_minutes),
        "experimental_minutes": prediction.forecast.experimental_minutes,
        "predicted_break_minutes": original,
        "break_baseline_minutes": prediction.break_forecast.baseline_minutes if prediction.break_forecast else None,
        "break_override_active": include_predicted_breaks and manual_break_minutes is not None,
        "break_estimate_available": original is not None,
    }
