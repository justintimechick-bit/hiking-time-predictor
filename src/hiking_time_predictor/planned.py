"""Reusable planned-hike workflow, independent of terminal or web interfaces."""

from dataclasses import dataclass, replace
from pathlib import Path

from .break_model import BreakForecast, load_break_model, predict_break_time
from .modeling import GEOMETRY, MODEL_LABELS, validate_estimator_schema
from .predictor import DEFAULT_MODEL, Forecast, ModelArtifact, forecast_route, load_model, range_warnings
from .routes import Route, gpx_route


@dataclass
class PlannedPrediction:
    route: Route
    forecast: Forecast
    artifact: ModelArtifact
    input_path: Path
    model_path: Path | None = None
    break_forecast: BreakForecast | None = None

    @property
    def segments(self):
        return self.forecast.segment_predictions

    @property
    def leg_totals(self):
        """Both estimates are sums of independent, ordered segment predictions."""
        return self.segments.groupby("leg", sort=False)[
            [*GEOMETRY, "predicted_time_min", "naismith_time_min", "regression_time_min"]
        ].sum(min_count=1)


def predict_planned_route(gpx_path, hike_type, direction="one-way", *, artifact=None,
                          model_path=None, missing_elevation="error", break_artifact=None) -> PlannedPrediction:
    """Load a saved fit and predict GPX geometry; never read/write training tables.

    An absent default model uses a labeled baseline. An explicitly requested
    missing/corrupt model or incompatible schema raises a clear ValueError.
    Passing an artifact allows a web application to load its model just once.
    """
    if artifact is not None and model_path is not None:
        raise ValueError("Supply artifact or model_path, not both.")
    resolved_model = None
    if artifact is None:
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL
        artifact = load_model(path) if model_path is not None or path.exists() else ModelArtifact()
        if path.exists():
            resolved_model = path.resolve()
    if not isinstance(artifact, ModelArtifact):
        raise ValueError("Expected a supported ModelArtifact.")
    if artifact.estimator is not None:
        validate_estimator_schema(artifact.estimator, artifact.model_name)
    # A raw fitted estimator alone provides no evidence that it was evaluated.
    metrics = artifact.evaluation.get("segment_metrics", [])
    if artifact.estimator is not None and not any(
        row.get("model") == MODEL_LABELS.get(artifact.model_name)
        and row.get("observations", 0) > 0 and row.get("hikes", 0) >= 2 for row in metrics
    ):
        artifact = replace(artifact, estimator=None, model_name=None, selected=False,
                           reason="No evaluated regression is available in this artifact; using the Naismith baseline.")
    route = gpx_route(gpx_path, hike_type, artifact.config, direction=direction,
                      missing_elevation=missing_elevation)
    forecast = forecast_route(route.segments, artifact)
    # Checking only the combined out-and-back total could hide an unusual leg.
    if direction == "out-and-back":
        for leg, segments in route.segments.groupby("leg", sort=False):
            forecast.warnings.extend(f"{leg.title()}: {warning}" for warning in range_warnings(segments, artifact)
                                     if warning.startswith("Route total"))
    if break_artifact is None:
        break_artifact = load_break_model(project=Path(__file__).resolve().parents[2])
    totals = route.segments[GEOMETRY].sum()
    break_forecast = predict_break_time(totals.distance_miles, totals.elevation_gain_ft,
                                       totals.elevation_loss_ft, hike_type, break_artifact)
    return PlannedPrediction(route, forecast, artifact, Path(gpx_path).resolve(), resolved_model, break_forecast)
