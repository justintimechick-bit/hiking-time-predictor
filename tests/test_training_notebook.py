"""Run notebook code against isolated, explicitly synthetic edge-case datasets.

Real notebook execution is separate. These tests protect the no-data/single-hike
guards and the optional C path without altering real data or notebook outputs.
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from hiking_time_predictor.modeling import TARGET


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "02_train_regression_models.ipynb"


@pytest.mark.parametrize("hikes", [0, 1, 4])
def test_notebook_code_handles_empty_single_and_known_type_cohorts(tmp_path, monkeypatch, hikes):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    rows = []
    for hike in range(hikes):
        for segment in range(3):
            distance = 0.5 if segment < 2 else 0.2
            gain = 10 * hike + 20 * segment
            loss = 5 * hike + 10 * (2 - segment)
            category = "day_hike" if hike < 2 else "backpacking"
            rows.append({"hike_id": f"synthetic-notebook-{hike}", "segment_index": segment,
                         "distance_miles": distance, "elevation_gain_ft": gain,
                         "elevation_loss_ft": loss, TARGET: 2 + distance*20 + gain*0.02,
                         "hike_type": category, "quality_flags": "", "unknown_time_min": 0})
    columns = ["hike_id", "segment_index", "distance_miles", "elevation_gain_ft",
               "elevation_loss_ft", TARGET, "hike_type", "quality_flags", "unknown_time_min"]
    segments = pd.DataFrame(rows, columns=columns)
    segments.to_csv(processed / "hiking_segments.csv", index=False)
    if hikes:
        summary = segments.groupby("hike_id").agg({TARGET: "sum", "distance_miles": "sum"}).reset_index()
        summary["quality_flags"] = ""
        summary["unknown_time_min"] = 0
        summary.to_csv(processed / "hiking_summary.csv", index=False)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    notebook = json.loads(NOTEBOOK.read_text())
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"training-notebook-cell-{number}", "exec"), namespace)
    run_summary = json.loads((processed / "regression" / "run_summary.json").read_text())
    assert run_summary["eligible_hikes_ab"] == hikes
    predictions = namespace["predictions"]
    if hikes < 2:
        assert predictions.empty
        assert namespace["segment_metrics"].empty
    else:
        assert namespace["c_evaluation"] is not None
        assert namespace["c_evaluation"].predictions.model.nunique() == 4
        assert namespace["c_hike_totals"].whole_hike_eligible.all()
