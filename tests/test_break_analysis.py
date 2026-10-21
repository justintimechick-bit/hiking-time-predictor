"""Isolated end-to-end reprocessing uses synthetic GPX, never personal files."""

import pandas as pd
import pytest

from hiking_time_predictor.break_analysis import analyze_breaks
from hiking_time_predictor.break_model import load_break_model
from hiking_time_predictor.breaks import save_corrections
from hiking_time_predictor.cli import main


def create_project(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    samples = [(0, 0), (0.001, 60), (0.001, 120), (0.001, 180), (0.001, 240), (0.002, 300)]
    contents = '<gpx version="1.1"><trk><trkseg>' + ''.join(
        f'<trkpt lat="0" lon="{lon}"><ele>100</ele><time>2026-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}Z</time></trkpt>'
        for lon, seconds in samples) + '</trkseg></trk></gpx>'
    (raw / "synthetic.gpx").write_text(contents)
    (tmp_path / "data/hike_metadata.csv").write_text("source_file,hike_id,hike_type,pack_weight_lbs,notes\nsynthetic.gpx,synthetic,day_hike,,Synthetic test only\n")
    return tmp_path


def test_full_analysis_reuses_inputs_without_overwriting_moving_exports(tmp_path):
    project = create_project(tmp_path)
    processed = project / "data/processed"
    processed.mkdir()
    (processed / "hiking_summary.csv").write_text("moving export sentinel\n")
    (processed / "break_analysis").mkdir()
    (processed / "break_analysis/actual_predicted.png").write_text("stale generated chart sentinel")
    protected = [project / "data/raw/synthetic.gpx", project / "data/hike_metadata.csv", processed / "hiking_summary.csv"]
    before = {str(p): p.read_bytes() for p in protected}
    hikes, events, artifact = analyze_breaks(project)
    assert artifact.training_hikes == 1
    assert artifact.evaluation["metrics"] == []
    assert hikes.total_break_minutes.iloc[0] == 3
    assert hikes.break_count.iloc[0] == 1
    assert before == {str(p): p.read_bytes() for p in protected}
    assert (project / "BREAK_ANALYSIS.md").exists()
    assert len(list((processed / "break_analysis").glob("*.png"))) == 6
    assert not (processed / "break_analysis/actual_predicted.png").exists()
    assert pd.read_csv(processed / "break_events.csv").shape == events.shape
    loaded = load_break_model(project / "models/break_predictor.joblib", project=project)
    assert loaded.minutes_per_mile == artifact.minutes_per_mile


def test_saved_corrections_survive_complete_analysis_and_exclude_non_hiking_activity(tmp_path):
    project = create_project(tmp_path)
    _, events, _ = analyze_breaks(project)
    edited = events.loc[events.break_classification.eq("stationary_break")].copy()
    edited["break_classification"] = "non_hiking"
    edited["correction_notes"] = "synthetic trailhead activity"
    path = project / "data/break_corrections.csv"
    save_corrections(events, edited, path)
    before = path.read_bytes()
    hikes, reprocessed, artifact = analyze_breaks(project)
    assert path.read_bytes() == before
    assert reprocessed.break_classification.eq("non_hiking").sum() == 1
    assert hikes.total_break_minutes.iloc[0] == 0
    assert hikes.excluded_minutes.iloc[0] == 3
    assert not hikes.training_eligible.iloc[0]
    assert artifact.minutes_per_mile is None


def test_changed_review_or_recording_invalidates_saved_prediction(tmp_path):
    project = create_project(tmp_path)
    _, events, _ = analyze_breaks(project)
    model = project / "models/break_predictor.joblib"
    edited = events.loc[events.break_classification.eq("stationary_break")].copy()
    edited["correction_notes"] = "new review note"
    save_corrections(events, edited, project / "data/break_corrections.csv")
    assert load_break_model(model, project=project).minutes_per_mile is None
    analyze_breaks(project)
    (project / "data/raw/new-recording.gpx").write_bytes((project / "data/raw/synthetic.gpx").read_bytes())
    stale = load_break_model(model, project=project)
    assert stale.minutes_per_mile is None
    assert "GPX files changed" in stale.reason


def test_analyze_breaks_cli_and_no_recordings_error(tmp_path, capsys):
    project = create_project(tmp_path)
    main(["analyze-breaks", "--project", str(project)])
    output = capsys.readouterr().out
    assert "1 ordinary breaks totaling 3.00" in output
    assert "Too few hikes for MAE evaluation" in output
    (project / "data/raw/synthetic.gpx").unlink()
    with pytest.raises(ValueError, match="No recorded GPX"):
        analyze_breaks(project)
