"""Explicit owner-only packaging and import; never unpickle legacy artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
from uuid import uuid4
import zipfile

import pandas as pd

from .breaks import BreakConfig, apply_break_corrections
from .gpx import ImportConfig
from .import_gpx import load_confirmed_rests, load_metadata
from .personal_hiking import preview_recording, save_recording

MAX_BUNDLE = 50 * 1024 * 1024
MAX_EXPANDED = 300 * 1024 * 1024


def allowed_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        return False
    return (bool(re.fullmatch(r'data/raw/[^/]+\.gpx(?:-\d+)?', name, re.I))
        or name in ('data/hike_metadata.csv','data/confirmed_rests.csv','data/break_corrections.csv',
                    'config/gpx_import.json','config/breaks.json')
        or (name.startswith('data/processed/') and p.suffix in ('.csv','.json'))
        or (len(p.parts) == 2 and p.parts[0] == 'models' and p.suffix in ('.joblib','.json')))


def package_project(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a new output filename; original files are never overwritten.')
    files = sorted(p for directory in ('data','models','config') for p in (source/directory).rglob('*')
                   if p.is_file() and not p.is_symlink() and allowed_path(p.relative_to(source).as_posix()))
    hashes = {}
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(source).as_posix()
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
        archive.writestr('migration_manifest.json', json.dumps({'schema':1,'files':hashes}, indent=2))
    return len(files)


def inspect_bundle(contents):
    if len(contents) > MAX_BUNDLE:
        raise ValueError('Migration bundles are limited to 50 MiB compressed to fit the private Storage bucket.')
    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
        names = archive.namelist()
        if (len(names) > 2000 or len(names) != len(set(names))
                or sum(i.file_size for i in archive.infolist()) > MAX_EXPANDED
                or any(not allowed_path(n) and n != 'migration_manifest.json' for n in names)):
            raise ValueError('Invalid or oversized migration archive.')
        data = {name: archive.read(name) for name in names}
        manifest = json.loads(data.pop('migration_manifest.json'))
        if manifest.get('schema') != 1 or manifest['files'] != {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}:
            raise ValueError('Migration bundle integrity check failed.')
        if not any(n.startswith('data/raw/') for n in data):
            raise ValueError('The bundle has no historical GPX recordings.')
        return data
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        raise ValueError('Use a ZIP created by the developer migration command.') from None


def import_bundle(repo, contents):
    data = inspect_bundle(contents)
    digest = hashlib.sha256(contents).hexdigest()
    runs = repo.client.table('legacy_imports').select('*').eq('user_id',repo.user_id).execute().data
    if runs and (runs[0]['status'] == 'complete' or runs[0]['bundle_hash'] != digest):
        raise ValueError('This account already has a migration. A pending migration must resume using the same bundle.')
    backup = runs[0]['backup_path'] if runs else f'{repo.user_id}/{uuid4()}.zip'
    if not runs:
        repo.client.table('legacy_imports').insert({'user_id':repo.user_id,'bundle_hash':digest,'backup_path':backup}).execute()
    if backup not in repo.object_paths('user-models'):
        repo.upload('user-models', backup, contents, 'application/octet-stream')
    if hashlib.sha256(repo.download('user-models',backup)).hexdigest() != digest:
        raise ValueError('Backup verification failed; migration remains pending.')
    config = ImportConfig(**json.loads(data.get('config/gpx_import.json', b'{}')))
    break_config = BreakConfig(**json.loads(data.get('config/breaks.json', b'{}')))
    imported = []
    # Private temporary directory, no extractall and no arbitrary paths.
    with TemporaryDirectory(prefix='hiking-migration-') as directory:
        temp = Path(directory)
        for name in ('hike_metadata.csv','confirmed_rests.csv'):
            if f'data/{name}' in data:
                (temp/name).write_bytes(data[f'data/{name}'])
        files = [Path(n).name for n in data if n.startswith('data/raw/')]
        metadata = load_metadata(temp/'hike_metadata.csv', [temp/n for n in files])
        rests = load_confirmed_rests(temp/'confirmed_rests.csv')
        corrections = pd.read_csv(io.BytesIO(data['data/break_corrections.csv']), keep_default_na=False) if 'data/break_corrections.csv' in data else pd.DataFrame()
        for name in files:
            row = metadata.loc[metadata.source_file.eq(name)].iloc[0]
            preview = preview_recording(data['data/raw/'+name], re.sub(r'\.gpx-\d+$','.gpx',name,flags=re.I), row.hike_type or None,
                config=config, break_config=break_config, confirmed_rests=rests.loc[rests.source_file.eq(name)])
            if not corrections.empty:
                part = corrections.loc[corrections.hike_id.eq(row.hike_id)].copy()
                part['hike_id'] = preview.id
                preview.events = apply_break_corrections(preview.events, part, break_config)
            hike_id, _ = save_recording(repo, preview, name=name, training_enabled=True)
            imported.append(hike_id)
    # Read back every imported hike before marking this one-time operation complete.
    if not set(imported).issubset({h['id'] for h in repo.list_hikes()}):
        raise ValueError('Migration verification incomplete. Retry the same bundle.')
    repo.client.table('legacy_imports').update({'status':'complete','imported_hike_ids':imported,
        'completed_at':datetime.now(timezone.utc).isoformat()}).eq('user_id',repo.user_id).execute()
    return len(set(imported))


def main():
    parser=argparse.ArgumentParser(description='Package the original local project for explicit import into your own account.')
    parser.add_argument('--source',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    count=package_project(args.source,args.output)
    print(f'Packaged {count} files. Originals preserved. Upload this private ZIP in Account Settings; retrain models afterward.')


if __name__=='__main__':
    main()
