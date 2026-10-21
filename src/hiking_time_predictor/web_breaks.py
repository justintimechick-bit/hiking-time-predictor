"""Historical break tables and charts for the existing Streamlit dashboard."""

from pathlib import Path

import pandas as pd
import plotly.express as px

from .web_charts import _style


def load_break_history(project):
    processed = Path(project) / "data/processed"
    events = pd.read_csv(processed / "break_events.csv", keep_default_na=False)
    for column in ("start_time", "end_time"):
        events[column] = pd.to_datetime(events[column], utc=True, format="mixed")
    hikes = pd.read_csv(processed / "break_hike_summary.csv")
    predictions = pd.read_csv(processed / "break_analysis/held_out_predictions.csv")
    return hikes, events, predictions


def break_history_figures(hikes, events, held_out, selected_model):
    breaks = events.loc[events.break_classification.eq("stationary_break")]
    figures = {
        "break_rate": px.bar(hikes, x="hike_id", y="break_minutes_per_mile", labels={"hike_id": "Recording", "break_minutes_per_mile": "Break minutes per mile"}),
        "break_durations": px.histogram(breaks, x="duration_minutes", nbins=20, labels={"duration_minutes": "Individual break duration (min)"}),
        "break_frequency": px.histogram(hikes, x="breaks_per_mile", labels={"breaks_per_mile": "Breaks per mile"}),
        "break_distance": px.scatter(hikes, x="distance_miles", y="total_break_minutes", hover_name="hike_id",
                                     labels={"distance_miles": "Measured moving distance (mi)", "total_break_minutes": "Ordinary break minutes"}),
    }
    if not held_out.empty:
        part = held_out.loc[held_out.model.eq(selected_model)]
        if not part.empty:
            figures["break_actual_predicted"] = px.scatter(part, x="actual_break_minutes", y="predicted_break_minutes", hover_name="hike_id",
                                                          labels={"actual_break_minutes": "Observed break minutes", "predicted_break_minutes": "Held-out predicted break minutes"})
    for fig in figures.values():
        _style(fig)
    return figures
