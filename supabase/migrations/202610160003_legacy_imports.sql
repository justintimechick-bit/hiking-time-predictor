-- A developer-created archive preserves local datasets and legacy artifacts as
-- opaque backup bytes. Backups are never passed to joblib.load.
create table public.legacy_imports (
 user_id uuid primary key references public.profiles(id) on delete cascade,
 bundle_hash text not null check(bundle_hash ~ '^[0-9a-f]{64}$'),
 backup_path text not null check(backup_path like user_id::text || '/%.zip'),
 status text not null default 'pending' check(status in ('pending','complete')),
 created_at timestamptz not null default now(), completed_at timestamptz,
 imported_hike_ids jsonb not null default '[]'
);
alter table public.legacy_imports enable row level security;
revoke all on public.legacy_imports from anon,authenticated;
grant select,insert,update,delete on public.legacy_imports to authenticated;
grant all on public.legacy_imports to service_role;
create policy own_legacy on public.legacy_imports for all to authenticated
using(user_id=(select auth.uid()) and public.hiking_account_live())
with check(user_id=(select auth.uid()) and public.hiking_account_live());
