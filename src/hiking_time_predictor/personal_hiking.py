"""Personal GPX/training workflows; reuse ML functions without cloud dependencies."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from uuid import uuid4
import zipfile

import pandas as pd

from .break_model import BreakArtifact, train_break_predictor
from .breaks import BreakConfig, CORRECTION_COLUMNS, NON_MOVING_TARGET, apply_break_corrections, extract_break_events, summarize_breaks
from .gpx import ImportConfig, process_hike
from .modeling import GEOMETRY, MODEL_FEATURES, prepare_segments
from .predictor import ModelArtifact, train_predictor
from .trusted_models import canonical
from .web_upload import _temporary_upload


def clean(value):
    """JSON-safe pandas/numpy output; missing measurements remain null."""
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient='records', date_format='iso', date_unit='us', double_precision=15))
    return json.loads(pd.Series(value).to_json(date_format='iso', date_unit='us', double_precision=15))


@dataclass
class RecordingPreview:
    id: str
    filename: str
    contents: bytes
    summary: dict
    segments: pd.DataFrame
    points: pd.DataFrame
    events: pd.DataFrame
    config: ImportConfig
    break_config: BreakConfig


def preview_recording(contents, filename, hike_type=None, *, config=None, break_config=None, confirmed_rests=None):
    if hike_type not in (None, 'day_hike', 'backpacking'):
        raise ValueError('Choose a known hike type or leave it unknown.')
    config, break_config = config or ImportConfig(), break_config or BreakConfig()
    hike_id = str(uuid4())
    with _temporary_upload(contents, filename) as path:
        summary, segments, points, intervals = process_hike(path, hike_id, {'hike_type': hike_type}, config, confirmed_rests)
        filename = path.name
    if (summary['trackpoint_count'] - summary['missing_timestamp_count'] < 2
            or pd.isna(summary['elapsed_time_min']) or summary['elapsed_time_min'] <= 0):
        raise ValueError('Completed historical activities need at least two timestamped trackpoints and positive elapsed time. To plan a route without timestamps, use Plan a hike.')
    events = extract_break_events(points, intervals, break_config)
    return RecordingPreview(hike_id, filename, contents, summary, segments, points, events, config, break_config)


def save_recording(repo, preview, *, name=None, training_enabled=False):
    digest = hashlib.sha256(preview.contents).hexdigest()
    duplicate = next((h for h in repo.list_hikes() if h['original_file_hash'] == digest), None)
    if duplicate:
        return duplicate['id'], False
    path = f'{repo.user_id}/{preview.id}.gpx'
    s = clean(preview.summary)
    row = {'id': preview.id, 'user_id': repo.user_id, 'name': (name or preview.filename)[:250],
           'activity_date': s['start_time_utc'], 'hike_type': s['hike_type'],
           **{k: s[k] for k in ('distance_miles', 'elevation_gain_ft', 'elevation_loss_ft', 'moving_time_min', 'elapsed_time_min')},
           'gps_quality_status': s['quality_flags'], 'original_gpx_storage_path': path,
           'original_file_hash': digest, 'training_enabled': training_enabled, 'summary': s,
           'import_config': asdict(preview.config), 'break_config': asdict(preview.break_config)}
    if path in repo.object_paths('hike-gpx'):
        if hashlib.sha256(repo.download('hike-gpx', path)).hexdigest() != digest:
            raise ValueError('An interrupted upload has different contents. Preview this recording again.')
    else:
        repo.upload('hike-gpx', path, preview.contents, 'application/gpx+xml')
    try:
        repo.import_hike(row, clean(preview.segments), clean(preview.events))
    except Exception:
        # Resolve an ambiguous network response before deleting any file.
        existing = repo.list_hikes()  # If this fails, leave private orphan for cleanup.
        committed = next((h for h in existing if h['id'] == preview.id), None)
        if committed:
            return committed['id'], True
        repo.remove_object('hike-gpx', path)
        duplicate = next((h for h in existing if h['original_file_hash'] == digest), None)
        if duplicate:
            return duplicate['id'], False
        raise
    return preview.id, True


def snapshot_fingerprint(snapshot):
    return hashlib.sha256(canonical(snapshot)).hexdigest()


def training_frames(snapshot, user_id):
    """Defend the adapter boundary and use only explicitly included recordings."""
    for table in ('hikes', 'segments', 'breaks'):
        if any(row['user_id'] != user_id for row in snapshot[table]):
            raise ValueError('Training snapshot contains a different user.')
    hikes = [h for h in snapshot['hikes'] if h['training_enabled'] and not h.get('deleting')]
    if not hikes:
        raise ValueError('Confirm at least one completed hike for training in My Hiking History.')
    ids = {h['id'] for h in hikes}
    config = ImportConfig(**hikes[0]['import_config'])
    break_config = BreakConfig(**hikes[0]['break_config'])
    if any(h['import_config'] != asdict(config) or h['break_config'] != asdict(break_config) for h in hikes):
        raise ValueError('Hikes use different preprocessing settings. Reimport them with consistent settings before training.')
    summaries = []
    for h in hikes:
        summaries.append({**h['summary'], 'hike_id': h['id'], 'hike_type': h['hike_type'],
                          **{k: h[k] for k in ('distance_miles', 'elevation_gain_ft', 'elevation_loss_ft', 'moving_time_min', 'elapsed_time_min')},
                          'quality_flags': h['gps_quality_status']})
    segments = []
    for row in snapshot['segments']:
        if row['hike_id'] in ids:
            segments.append({**row['data'], **{k: row[k] for k in ('hike_id', 'segment_index', 'distance_miles',
                           'elevation_gain_ft', 'elevation_loss_ft', 'moving_time_min', 'quality_flags')}})
    events = []
    for row in snapshot['breaks']:
        if row['hike_id'] in ids:
            events.append({**row['data'], 'hike_id': row['hike_id'], 'start_time': row['start_time'],
                           'end_time': row['end_time'], 'duration_minutes': row['duration_minutes'],
                           'distance_into_hike_miles': row['distance_into_hike_miles'],
                           'break_classification': row['classification'], 'quality_flag': row['quality_flags']})
    from .breaks import EVENT_COLUMNS
    summary = pd.DataFrame(summaries)
    for column in (*GEOMETRY, 'moving_time_min', 'unknown_time_min'):
        summary[column] = pd.to_numeric(summary[column], errors='coerce')
    segment_table = pd.DataFrame(segments)
    if segment_table.empty:
        from .gpx import SEGMENT_COLUMNS
        segment_table = pd.DataFrame(columns=SEGMENT_COLUMNS)
    event_table = pd.DataFrame(events, columns=EVENT_COLUMNS)
    for column in ('start_time', 'end_time'):
        event_table[column] = pd.to_datetime(event_table[column], utc=True)
    return summary, segment_table, event_table, config, break_config


def train_personal_models(repo, codec):
    snapshot = repo.snapshot()
    fingerprint = snapshot_fingerprint(snapshot)
    summary, segments, events, config, break_config = training_frames(snapshot, repo.user_id)
    moving = train_predictor(segments, summary, config)
    breaks, held_out = train_break_predictor(
        summarize_breaks(summary, events, break_config, target=NON_MOVING_TARGET), break_config)
    now = datetime.now(timezone.utc).isoformat()
    version = str(uuid4())
    stored = []
    for kind, artifact in (('moving', moving), ('break', breaks)):
        model_id = str(uuid4())
        contents, provenance = codec.encode(artifact, user_id=repo.user_id, model_id=model_id,
                                             kind=kind, dataset_fingerprint=fingerprint, version=version)
        path = f'{repo.user_id}/{model_id}.joblib'
        row = {'id': model_id, 'user_id': repo.user_id, 'model_type': kind,
               'algorithm': 'linear_regression_with_baseline', 'version': version, 'trained_at': now,
               'training_hike_count': moving.evaluation.get('eligible_hikes', 0) if kind == 'moving' else breaks.training_hikes,
               'training_segment_count': int(prepare_segments(segments).eligible_ab.sum()) if kind == 'moving' else 0,
               'training_features': MODEL_FEATURES.get(moving.model_name, []) if kind == 'moving' else ['distance_miles', 'elevation_gain_ft', 'elevation_loss_ft'],
               'evaluation_metrics': artifact.evaluation, 'storage_path': path, 'artifact_hash': provenance['sha256'],
               'provenance': provenance, 'dataset_fingerprint': fingerprint, 'is_active': False, 'status': 'ready'}
        repo.upload('user-models', path, contents, 'application/octet-stream')
        # Check round trip before activation, without trusting DB hashes alone.
        codec.decode(repo.download('user-models', path), row, repo.user_id)
        repo.insert_model(row)
        stored.append(row)
    if snapshot_fingerprint(repo.snapshot()) != fingerprint:
        raise ValueError('History changed while training. New versions were saved but not activated. Retrain with the updated history.')
    repo.activate_models(stored[0]['id'], stored[1]['id'], snapshot)
    return moving, breaks, stored


def load_personal_models(repo, codec):
    rows = [r for r in repo.models() if r['is_active']]
    if not rows:
        return ModelArtifact(), BreakArtifact(reason='No personal break model yet. Add recordings and train; moving time uses the Naismith baseline.'), []
    if len(rows) != 2 or {r['model_type'] for r in rows} != {'moving', 'break'} or len({r['version'] for r in rows}) != 1:
        raise ValueError('Active model pair is incomplete. Retrain your models.')
    artifacts = {}
    for row in rows:
        try:
            contents = repo.download('user-models', row['storage_path'])
        except Exception:
            raise ValueError('A saved model file is unavailable. Check Storage permissions or retrain from My Models.') from None
        artifacts[row['model_type']] = codec.decode(contents, row, repo.user_id)
    return artifacts['moving'], artifacts['break'], rows


def validate_break_edits(original, edited, config):
    changed = edited.break_classification.ne(original.break_classification)
    if (changed & (original.break_classification.eq('moving') | edited.break_classification.eq('moving'))).any():
        raise ValueError('Moving intervals cannot be relabelled in break review; use confirmed rest intervals and reimport.')
    if (edited.break_classification.eq('stationary_break') & edited.duration_minutes.lt(config.min_break_minutes)).any():
        raise ValueError('An ordinary break is shorter than the configured minimum duration.')
    corrections = edited.loc[edited.break_classification.ne(original.break_classification)
                              | edited.correction_notes.fillna('').ne(original.correction_notes.fillna('')),
                              ['hike_id', 'start_time', 'end_time', 'break_classification', 'correction_notes']].rename(columns={'correction_notes': 'notes'})
    return apply_break_corrections(original, corrections[CORRECTION_COLUMNS], config)


def export_personal_data(repo):
    """In-memory private export, including inactive versions and orphaned blobs."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for table in ('profiles', 'hikes', 'hike_segments', 'break_events', 'model_versions', 'predictions', 'legacy_imports'):
            archive.writestr(f'{table}.json', json.dumps(repo.rows(table), indent=2))
        for bucket in ('hike-gpx', 'user-models'):
            for path in repo.object_paths(bucket):
                archive.writestr(f'{bucket}/{path.split("/")[-1]}', repo.download(bucket, path))
    return output.getvalue()
