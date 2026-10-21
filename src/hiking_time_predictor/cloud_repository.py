"""User-token-only Supabase adapter. PostgreSQL and Storage enforce ownership."""
from typing import Protocol
from uuid import UUID

from supabase import ClientOptions, create_client


class CloudError(ValueError):
    pass


def private_request_message(error):
    """Translate known database errors without exposing raw SDK responses."""
    from postgrest.exceptions import APIError
    if isinstance(error, APIError):
        if error.code in ('PGRST205', 'PGRST202', '42P01', '42883'):
            return ('Supabase database setup is incomplete: an application table or function is missing. '
                    'Apply the SQL migrations in this project with supabase db push, then refresh this page. '
                    'See docs/SUPABASE_SETUP.md, step 3.')
        if error.code in ('42501', '42P17'):
            return ('Supabase denied the private data request. Check that all project migrations, '
                    'table grants and row-level security policies are applied. '
                    'Keep using your user session; do not switch normal requests to an administrator key.')
        if error.code in ('PGRST301', 'PGRST302', 'PGRST303'):
            return 'Your Supabase session could not be authorized. Sign out and continue with Google again.'
    return ('The private data request could not be completed. Try again; if your session expired, '
            'sign in again. Check migrations and Storage policies if the problem persists.')


TABLES = ('profiles', 'hikes', 'hike_segments', 'break_events', 'model_versions', 'predictions', 'legacy_imports')


class HikingRepository(Protocol):
    user_id: str
    def list_hikes(self) -> list[dict]: ...
    def snapshot(self) -> dict: ...
    def upload(self, bucket: str, path: str, contents: bytes, content_type: str) -> None: ...
    def download(self, bucket: str, path: str) -> bytes: ...
    def models(self) -> list[dict]: ...


class SupabaseRepository:
    def __init__(self, config, session, *, client=None):
        self.user_id = str(UUID(session.user_id))
        # Construct a separate client for each user/request. Never cache this client.
        self.client = client or create_client(config.supabase_url, config.publishable_key,
            options=ClientOptions(headers={'Authorization': 'Bearer ' + session.access_token},
                                  auto_refresh_token=False, persist_session=False,
                                  postgrest_client_timeout=30, storage_client_timeout=30))

    def _path(self, path):
        if not path.startswith(self.user_id + '/') or '..' in path or len(path.split('/')) != 2:
            raise CloudError('Invalid private object path.')
        return path

    def rows(self, table):
        if table not in TABLES:
            raise CloudError('Unsupported table.')
        owner = 'id' if table == 'profiles' else 'user_id'
        result, offset = [], 0
        while True:
            page = self.client.table(table).select('*').eq(owner, self.user_id).order('user_id' if table == 'legacy_imports' else 'id').range(offset, offset + 499).execute().data
            result.extend(page)
            if len(page) < 500:
                break
            offset += 500
        if any(row.get(owner) != self.user_id for row in result):
            raise CloudError('Unexpected ownership in server response.')
        return result

    def profile(self):
        rows = self.rows('profiles')
        if len(rows) != 1:
            raise CloudError('Your profile is unavailable. Apply the database migrations before signing in.')
        return rows[0]

    def save_profile(self, display_name, preferred_hike_type):
        if preferred_hike_type not in ('day_hike', 'backpacking') or len(display_name) > 200:
            raise CloudError('Invalid profile preferences.')
        from datetime import datetime, timezone
        self.client.table('profiles').update({'display_name': display_name, 'preferred_hike_type': preferred_hike_type,
            'updated_at': datetime.now(timezone.utc).isoformat()}).eq('id', self.user_id).execute()

    def list_hikes(self):
        return sorted(self.rows('hikes'), key=lambda row: row['activity_date'], reverse=True)

    def snapshot(self):
        data = self.client.rpc('hiking_training_snapshot').execute().data
        for name in ('hikes', 'segments', 'breaks'):
            if any(row['user_id'] != self.user_id for row in data[name]):
                raise CloudError('Unexpected ownership in training snapshot.')
        return data

    def import_hike(self, hike, segments, events):
        if hike['user_id'] != self.user_id:
            raise CloudError('Ownership mismatch.')
        self.client.rpc('import_recorded_hike', {'p_hike': hike, 'p_segments': segments, 'p_breaks': events}).execute()

    def set_training(self, hike_id, enabled):
        self.client.table('hikes').update({'training_enabled': bool(enabled)}).eq('id', str(UUID(hike_id))).eq('user_id', self.user_id).execute()

    def correct_breaks(self, hike_id, edits):
        self.client.rpc('review_hiking_breaks', {'p_hike': str(UUID(hike_id)), 'p_edits': edits}).execute()

    def upload(self, bucket, path, contents, content_type):
        if bucket not in ('hike-gpx', 'user-models'):
            raise CloudError('Unsupported bucket.')
        self.client.storage.from_(bucket).upload(self._path(path), contents,
                                                {'content-type': content_type, 'upsert': 'false'})

    def download(self, bucket, path):
        return self.client.storage.from_(bucket).download(self._path(path))

    def remove_object(self, bucket, path):
        self.client.storage.from_(bucket).remove([self._path(path)])

    def object_paths(self, bucket):
        result, offset = [], 0
        while True:
            rows = self.client.storage.from_(bucket).list(self.user_id,
                {'limit': 100, 'offset': offset, 'sortBy': {'column': 'name', 'order': 'asc'}})
            result.extend(self._path(self.user_id + '/' + row['name']) for row in rows if row.get('id'))
            if len(rows) < 100:
                return result
            offset += 100

    def delete_hike(self, hike_id):
        rows = [h for h in self.list_hikes() if h['id'] == hike_id]
        if not rows:
            raise CloudError('Hike not found.')
        # Exclude from training before external Storage deletion. Retry is safe.
        self.client.table('hikes').update({'training_enabled': False, 'deleting': True}).eq('id', hike_id).eq('user_id', self.user_id).execute()
        self.remove_object('hike-gpx', rows[0]['original_gpx_storage_path'])
        self.client.table('hikes').delete().eq('id', hike_id).eq('user_id', self.user_id).execute()

    def models(self):
        return sorted(self.rows('model_versions'), key=lambda row: row['trained_at'], reverse=True)

    def insert_model(self, row):
        if row['user_id'] != self.user_id:
            raise CloudError('Ownership mismatch.')
        self.client.table('model_versions').insert(row).execute()

    def activate_models(self, moving_id, break_id, snapshot):
        self.client.rpc('activate_model_pair', {'p_moving': moving_id, 'p_break': break_id, 'p_snapshot': snapshot}).execute()

    def save_prediction(self, row):
        self.client.table('predictions').insert({**row, 'user_id': self.user_id}).execute()

    def predictions(self):
        return sorted(self.rows('predictions'), key=lambda row: row['created_at'], reverse=True)
