"""Privileged cleanup only. Normal application queries never use this client."""
import argparse
from datetime import datetime, timezone
from uuid import UUID

from supabase import ClientOptions, create_client


def admin_client(config):
    if not config.admin_key:
        raise ValueError('Account deletion is unavailable until the server has SUPABASE_SECRET_KEY.')
    return create_client(config.supabase_url, config.admin_key,
                         options=ClientOptions(auto_refresh_token=False, persist_session=False))


def delete_account_data(admin, user_id, *, delete_identity=True):
    user_id = str(UUID(user_id))
    # Persist the job first. A cron/CLI retry can resume after any interruption.
    admin.table('account_deletion_jobs').upsert({'user_id': user_id, 'status': 'pending',
        'delete_identity': delete_identity, 'completed_at': None, 'requested_at':datetime.now(timezone.utc).isoformat()}).execute()
    admin.table('profiles').update({'deletion_requested_at': datetime.now(timezone.utc).isoformat()}).eq('id', user_id).execute()
    # The RLS gate now denies new end-user writes. Include orphaned uploads.
    drain_objects(admin, user_id)
    for table in ('predictions', 'model_versions', 'hikes', 'legacy_imports'):
        admin.table(table).delete().eq('user_id', user_id).execute()
    for table in ('hikes','hike_segments','break_events','model_versions','predictions','legacy_imports'):
        if admin.table(table).select('user_id').eq('user_id',user_id).limit(1).execute().data:
            raise ValueError('Database cleanup could not be verified.')
    if delete_identity:
        from supabase_auth.errors import AuthApiError
        try:
            admin.auth.admin.delete_user(user_id)
        except AuthApiError as error:
            if error.status != 404:
                raise
        # Recheck after identity removal to catch uploads already in flight when locked.
        drain_objects(admin, user_id)
        if admin.table('profiles').select('id').eq('id',user_id).execute().data:
            raise ValueError('Identity cleanup could not be verified.')
        admin.table('account_deletion_jobs').update({'status':'complete', 'completed_at':datetime.now(timezone.utc).isoformat()}).eq('user_id',user_id).execute()
    else:
        admin.rpc('finish_hiking_data_reset', {'p_user':user_id}).execute()


def drain_objects(admin, user_id):
    for bucket in ('hike-gpx', 'user-models'):
        storage = admin.storage.from_(bucket)
        for _ in range(10000):
            objects = storage.list(user_id, {'limit': 100, 'offset': 0})
            paths = [user_id + '/' + item['name'] for item in objects if item.get('id')]
            if not objects:
                break
            if not paths or any('/' in item['name'] for item in objects):
                raise ValueError('Unexpected nested objects. Cleanup remains pending for operator review.')
            storage.remove(paths)
        else:
            raise ValueError('Cleanup did not converge; retry the pending job.')
        if storage.list(user_id, {'limit': 1}):
            raise ValueError('Storage cleanup could not be verified.')



def main():
    parser = argparse.ArgumentParser(description='Retry previously confirmed account cleanup jobs; no new deletions are inferred.')
    parser.add_argument('--retry-pending', action='store_true', required=True)
    parser.parse_args()
    from .cloud_config import CloudConfig
    from .auth_sessions import SessionVault
    config = CloudConfig.from_env(include_admin=True)
    admin = admin_client(config)
    jobs = admin.table('account_deletion_jobs').select('*').eq('status', 'pending').execute().data
    for job in jobs:
        delete_account_data(admin, job['user_id'], delete_identity=job['delete_identity'])
        if config.auth_mode == 'service':
            SessionVault(config.session_db, config.session_key).delete_user(job['user_id'])
    print(f'Completed {len(jobs)} pending cleanup jobs.')


if __name__ == '__main__':
    main()
