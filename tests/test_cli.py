"""Exercise terminal validation, default baselines, and the public module name."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from hiking_time_predictor.cli import format_duration, main
from hiking_time_predictor.gpx import ImportConfig
from hiking_time_predictor.predictor import load_model


MANUAL = ["predict", "--distance", "7.5", "--gain", "1800", "--loss", "1200", "--type", "backpacking"]


def test_public_module_command_without_model_works_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, "-m", "hiking_predictor", *MANUAL], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Predicted moving time: 3 h 24 min (baseline estimate: Naismith)" in result.stdout
    assert "Naismith's Rule estimated time: 3 h 24 min" in result.stdout
    assert "Difference (prediction - Naismith): +0.0 min" in result.stdout
    assert "Training range unavailable" in result.stdout
    assert not (tmp_path / "models").exists()  # Inference never trains or writes a model.


@pytest.mark.parametrize("args", [
    ["predict"],
    ["predict", "--distance", "1", "--type", "day_hike"],
    ["predict", "--distance", "1", "--gain", "0", "--loss", "0", "--type", "running"],
    ["predict", "--gpx", "route.gpx", "--distance", "1", "--type", "day_hike"],
    ["predict", "--gpx", "route.gpx", "--gain", "10", "--type", "day_hike"],
    ["predict", "--distance", "nan", "--gain", "0", "--loss", "0", "--type", "day_hike"],
    ["predict", "--distance", "1", "--gain", "-1", "--loss", "0", "--type", "day_hike"],
])
def test_invalid_cli_inputs_exit_with_usage_error(tmp_path, monkeypatch, capsys, args):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "Traceback" not in stderr


def test_explicit_missing_or_corrupt_model_does_not_silently_fall_back(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "model.joblib"
    for contents in (None, b"invalid"):
        if contents is not None:
            path.write_bytes(contents)
        with pytest.raises(SystemExit) as error:
            main([*MANUAL, "--model", str(path)])
        assert error.value.code == 2
        assert "Could not load the model" in capsys.readouterr().err


def test_planned_route_command_requires_no_timestamps(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "planned.gpx"
    path.write_text('<gpx version="1.1"><rte><rtept lat="0" lon="0"><ele>100</ele></rtept>'
                    '<rtept lat="0" lon="0.01"><ele>120</ele></rtept></rte></gpx>')
    main(["predict", "--gpx", str(path), "--type", "backpacking"])
    output = capsys.readouterr().out
    assert "planned route geometry" in output
    assert "baseline estimate" in output
    assert "actual hiking time is unavailable" in output
    assert "2 segment(s)" in output


def test_train_command_persists_evaluation_and_matching_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rows = []
    for hike in range(3):
        for segment in range(2):
            rows.append({"hike_id": f"synthetic-cli-{hike}", "segment_index": segment,
                         "distance_miles": 0.25, "elevation_gain_ft": 10 * hike,
                         "elevation_loss_ft": 5 * segment, "moving_time_min": 5 + hike,
                         "unknown_time_min": 0, "quality_flags": "", "hike_type": None})
    segments = pd.DataFrame(rows)
    segments.to_csv("segments.csv", index=False)
    settings = ImportConfig(segment_miles=0.25, elevation_window_m=50)
    Path("settings.json").write_text(json.dumps({"config": asdict(settings)}))
    main(["train", "--segments", "segments.csv", "--settings", "settings.json", "--model", "trained.joblib"])
    artifact = load_model("trained.joblib")
    assert artifact.config == settings
    assert not artifact.selected
    assert artifact.evaluation["eligible_hikes"] == 3
    assert artifact.provenance["segments_sha256"]
    assert artifact.provenance["summary_sha256"] is None
    assert "experimental" in capsys.readouterr().out
    main([*MANUAL, "--model", "trained.joblib"])
    output = capsys.readouterr().out
    assert "30 segment(s)" in output
    assert "Experimental" in output
    original = Path("segments.csv").read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["train", "--segments", "segments.csv", "--settings", "settings.json", "--model", "segments.csv"])
    assert error.value.code == 2
    assert "do not overwrite training inputs" in capsys.readouterr().err
    assert Path("segments.csv").read_bytes() == original


def test_duration_rounding_carries_minutes():
    assert format_duration(59.6) == "1 h 00 min"
    assert format_duration(120) == "2 h 00 min"
