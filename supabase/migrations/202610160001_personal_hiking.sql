-- Application data. Apply with `supabase db push`; never use dashboard edits.
create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text not null default '' check (length(display_name) <= 200),
  avatar_url text,
  preferred_hike_type text not null default 'day_hike' check (preferred_hike_type in ('day_hike','backpacking')),
  created_at timestamptz not null default now(), updated_at timestamptz not null default now(),
  deletion_requested_at timestamptz
);
create function public.initialize_hiking_profile() returns trigger language plpgsql security definer
set search_path = '' as $$
begin
  insert into public.profiles(id,display_name,avatar_url)
  values(new.id, left(coalesce(new.raw_user_meta_data->>'full_name',''),200), new.raw_user_meta_data->>'avatar_url');
  return new;
end $$;
create trigger hiking_profile_created after insert on auth.users for each row execute function public.initialize_hiking_profile();
-- Existing Auth accounts get profiles too. Provider metadata is display-only.
insert into public.profiles(id,display_name,avatar_url)
select id,left(coalesce(raw_user_meta_data->>'full_name',''),200),raw_user_meta_data->>'avatar_url' from auth.users on conflict do nothing;

create function public.hiking_account_live() returns boolean language sql stable security definer
set search_path = '' as $$
 select exists(select 1 from public.profiles where id=(select auth.uid()) and deletion_requested_at is null)
$$;
revoke all on function public.hiking_account_live() from public, anon;
grant execute on function public.hiking_account_live() to authenticated;

create table public.hikes (
 id uuid primary key, user_id uuid not null references public.profiles(id) on delete cascade,
 name text not null check(length(name) between 1 and 250), activity_date timestamptz not null,
 hike_type text check (hike_type in ('day_hike','backpacking')),
 distance_miles double precision not null check(distance_miles >= 0 and distance_miles < 'Infinity'::float8),
 elevation_gain_ft double precision check(elevation_gain_ft >= 0 and elevation_gain_ft < 'Infinity'::float8),
 elevation_loss_ft double precision check(elevation_loss_ft >= 0 and elevation_loss_ft < 'Infinity'::float8),
 moving_time_min double precision not null check(moving_time_min >= 0 and moving_time_min < 'Infinity'::float8),
 elapsed_time_min double precision not null check(elapsed_time_min > 0 and elapsed_time_min < 'Infinity'::float8),
 gps_quality_status text not null default '',
 original_gpx_storage_path text not null check(original_gpx_storage_path = user_id::text || '/' || id::text || '.gpx'),
 original_file_hash text not null check(original_file_hash ~ '^[0-9a-f]{64}$'),
 training_enabled boolean not null default false, summary jsonb not null,
 import_config jsonb not null, break_config jsonb not null,
 revision bigint not null default 1, created_at timestamptz not null default now(),
 unique(user_id, original_file_hash), unique(id,user_id)
);
create index hikes_user_date on public.hikes(user_id,activity_date desc);
create table public.hike_segments (
 id uuid primary key default gen_random_uuid(), user_id uuid not null,
 hike_id uuid not null, segment_index integer not null check(segment_index >= 0),
 distance_miles double precision not null check(distance_miles >= 0 and distance_miles < 'Infinity'::float8),
 elevation_gain_ft double precision, elevation_loss_ft double precision,
 moving_time_min double precision not null check(moving_time_min >= 0 and moving_time_min < 'Infinity'::float8),
 quality_flags text not null default '', data jsonb not null,
 foreign key(hike_id,user_id) references public.hikes(id,user_id) on delete cascade,
 unique(hike_id,segment_index)
);
create index hike_segments_user_hike on public.hike_segments(user_id,hike_id);
create table public.break_events (
 id uuid primary key default gen_random_uuid(), user_id uuid not null, hike_id uuid not null,
 start_time timestamptz not null, end_time timestamptz not null check(end_time > start_time),
 duration_minutes double precision not null check(duration_minutes > 0 and duration_minutes < 'Infinity'::float8),
 distance_into_hike_miles double precision not null check(distance_into_hike_miles >= 0 and distance_into_hike_miles < 'Infinity'::float8),
 classification text not null check(classification in ('moving','stationary_break','unknown','short_stop','camping','non_hiking')),
 quality_flags text not null default '', manually_verified boolean not null default false,
 data jsonb not null,
 foreign key(hike_id,user_id) references public.hikes(id,user_id) on delete cascade,
 unique(hike_id,start_time,end_time)
);
create index break_events_user_hike on public.break_events(user_id,hike_id);
create table public.model_versions (
 id uuid primary key, user_id uuid not null references public.profiles(id) on delete cascade,
 model_type text not null check(model_type in ('moving','break')), algorithm text not null,
 version uuid not null, trained_at timestamptz not null default now(),
 training_hike_count integer not null check(training_hike_count >= 0),
 training_segment_count integer not null check(training_segment_count >= 0),
 training_features jsonb not null, evaluation_metrics jsonb not null,
 storage_path text not null check(storage_path = user_id::text || '/' || id::text || '.joblib'),
 artifact_hash text not null check(artifact_hash ~ '^[0-9a-f]{64}$'),
 provenance jsonb not null, dataset_fingerprint text not null check(dataset_fingerprint ~ '^[0-9a-f]{64}$'),
 is_active boolean not null default false, status text not null check(status in ('ready','archived')),
 unique(id,user_id), unique(user_id,version,model_type)
);
create unique index one_active_model_per_type on public.model_versions(user_id,model_type) where is_active;
create index model_versions_user_trained on public.model_versions(user_id,trained_at desc);
create table public.predictions (
 id uuid primary key default gen_random_uuid(), user_id uuid not null references public.profiles(id) on delete cascade,
 moving_model_version_id uuid, break_model_version_id uuid,
 route_name text not null check(length(route_name) between 1 and 250),
 distance_miles double precision not null check(distance_miles > 0 and distance_miles < 'Infinity'::float8),
 elevation_gain_ft double precision not null, elevation_loss_ft double precision not null,
 hike_type text not null check(hike_type in ('day_hike','backpacking')),
 route_direction text not null check(route_direction in ('one-way','out-and-back')),
 predicted_moving_time_min double precision not null check(predicted_moving_time_min > 0 and predicted_moving_time_min < 'Infinity'::float8),
 predicted_break_time_min double precision check(predicted_break_time_min >= 0 and predicted_break_time_min < 'Infinity'::float8),
 predicted_total_time_min double precision not null check(predicted_total_time_min >= predicted_moving_time_min and predicted_total_time_min < 'Infinity'::float8),
 details jsonb not null default '{}', created_at timestamptz not null default now(),
 foreign key(moving_model_version_id,user_id) references public.model_versions(id,user_id) on delete cascade,
 foreign key(break_model_version_id,user_id) references public.model_versions(id,user_id) on delete cascade
);
create index predictions_user_created on public.predictions(user_id,created_at desc);

-- Durable cleanup jobs remain after Auth deletion so interrupted cleanup is auditable.
create table public.account_deletion_jobs (
 user_id uuid primary key, requested_at timestamptz not null default now(),
 completed_at timestamptz, status text not null default 'pending' check(status in ('pending','complete'))
);
alter table public.account_deletion_jobs enable row level security;
revoke all on public.account_deletion_jobs from anon,authenticated;
grant all on public.account_deletion_jobs to service_role;

alter table public.profiles enable row level security;
create policy profile_read on public.profiles for select to authenticated using(id=(select auth.uid()));
create policy profile_update on public.profiles for update to authenticated
using(id=(select auth.uid()) and public.hiking_account_live()) with check(id=(select auth.uid()) and public.hiking_account_live());
revoke all on public.profiles from anon,authenticated;
grant select on public.profiles to authenticated;
grant update(display_name,preferred_hike_type,updated_at) on public.profiles to authenticated;

-- USING protects existing rows; WITH CHECK protects new ownership after INSERT/UPDATE.
do $$ declare t text; begin
 foreach t in array array['hikes','hike_segments','break_events','model_versions','predictions'] loop
  execute format('alter table public.%I enable row level security',t);
  execute format('revoke all on public.%I from anon, authenticated',t);
  execute format('grant select,insert,update,delete on public.%I to authenticated',t);
  execute format('create policy own_select on public.%I for select to authenticated using(user_id=(select auth.uid()) and public.hiking_account_live())',t);
  execute format('create policy own_insert on public.%I for insert to authenticated with check(user_id=(select auth.uid()) and public.hiking_account_live())',t);
  execute format('create policy own_update on public.%I for update to authenticated using(user_id=(select auth.uid()) and public.hiking_account_live()) with check(user_id=(select auth.uid()) and public.hiking_account_live())',t);
  execute format('create policy own_delete on public.%I for delete to authenticated using(user_id=(select auth.uid()) and public.hiking_account_live())',t);
 end loop;
end $$;

-- A single import transaction creates the hike and both sets of children.
create function public.import_recorded_hike(p_hike jsonb, p_segments jsonb, p_breaks jsonb)
returns uuid language plpgsql security invoker set search_path = '' as $$
declare h public.hikes; s jsonb; e jsonb;
begin
 h := jsonb_populate_record(null::public.hikes,p_hike);
 if h.user_id is distinct from auth.uid() then raise exception 'Ownership mismatch'; end if;
 if coalesce((h.summary->>'trackpoint_count')::int,0) - coalesce((h.summary->>'missing_timestamp_count')::int,0) < 2 then
  raise exception 'Historical recordings require timestamps';
 end if;
 insert into public.hikes(id,user_id,name,activity_date,hike_type,distance_miles,elevation_gain_ft,elevation_loss_ft,
 moving_time_min,elapsed_time_min,gps_quality_status,original_gpx_storage_path,original_file_hash,training_enabled,summary,import_config,break_config)
 values(h.id,h.user_id,h.name,h.activity_date,h.hike_type,h.distance_miles,h.elevation_gain_ft,h.elevation_loss_ft,
 h.moving_time_min,h.elapsed_time_min,h.gps_quality_status,h.original_gpx_storage_path,h.original_file_hash,h.training_enabled,h.summary,h.import_config,h.break_config);
 for s in select value from jsonb_array_elements(p_segments) loop
  insert into public.hike_segments(user_id,hike_id,segment_index,distance_miles,elevation_gain_ft,elevation_loss_ft,moving_time_min,quality_flags,data)
  values(h.user_id,h.id,(s->>'segment_index')::int,(s->>'distance_miles')::float8,(s->>'elevation_gain_ft')::float8,
  (s->>'elevation_loss_ft')::float8,(s->>'moving_time_min')::float8,coalesce(s->>'quality_flags',''),s);
 end loop;
 for e in select value from jsonb_array_elements(p_breaks) loop
  insert into public.break_events(user_id,hike_id,start_time,end_time,duration_minutes,distance_into_hike_miles,classification,quality_flags,manually_verified,data)
  values(h.user_id,h.id,(e->>'start_time')::timestamptz,(e->>'end_time')::timestamptz,(e->>'duration_minutes')::float8,
  (e->>'distance_into_hike_miles')::float8,e->>'break_classification',coalesce(e->>'quality_flag',''),position('manual_review' in coalesce(e->>'quality_flag','')) > 0,e);
 end loop;
 return h.id;
end $$;
revoke all on function public.import_recorded_hike(jsonb,jsonb,jsonb) from public,anon;
grant execute on function public.import_recorded_hike(jsonb,jsonb,jsonb) to authenticated;

-- Query one consistent snapshot instead of independently reading changing tables.
create function public.hiking_training_snapshot() returns jsonb language sql stable security invoker
set search_path = '' as $$
 select jsonb_build_object(
 'hikes',coalesce((select jsonb_agg(to_jsonb(h) order by h.id) from public.hikes h where h.user_id=auth.uid()),'[]'::jsonb),
 'segments',coalesce((select jsonb_agg(to_jsonb(s) order by s.hike_id,s.segment_index) from public.hike_segments s where s.user_id=auth.uid()),'[]'::jsonb),
 'breaks',coalesce((select jsonb_agg(to_jsonb(e) order by e.hike_id,e.start_time) from public.break_events e where e.user_id=auth.uid()),'[]'::jsonb))
$$;
revoke all on function public.hiking_training_snapshot() from public,anon;
grant execute on function public.hiking_training_snapshot() to authenticated;

create function public.activate_model_pair(p_moving uuid,p_break uuid,p_snapshot jsonb) returns void
language plpgsql security invoker set search_path = '' as $$
declare v uuid;
begin
 perform 1 from public.profiles where id=auth.uid() for update;
 if public.hiking_training_snapshot() <> p_snapshot then raise exception 'History changed during training'; end if;
 select version into v from public.model_versions where id=p_moving and user_id=auth.uid() and model_type='moving' and status='ready';
 if v is null or not exists(select 1 from public.model_versions where id=p_break and user_id=auth.uid()
 and model_type='break' and status='ready' and version=v) then raise exception 'Incomplete model pair'; end if;
 update public.model_versions set is_active=false where user_id=auth.uid() and is_active;
 update public.model_versions set is_active=true where id in(p_moving,p_break) and user_id=auth.uid();
end $$;
revoke all on function public.activate_model_pair(uuid,uuid,jsonb) from public,anon;
grant execute on function public.activate_model_pair(uuid,uuid,jsonb) to authenticated;


alter table public.hikes add column deleting boolean not null default false;
create function public.bump_hiking_revision() returns trigger language plpgsql set search_path = '' as $$
begin new.revision := old.revision + 1; return new; end $$;
create trigger hiking_revision before update on public.hikes for each row execute function public.bump_hiking_revision();

create function public.review_hiking_breaks(p_hike uuid,p_edits jsonb) returns void
language plpgsql security invoker set search_path = '' as $$
declare item jsonb; old_event public.break_events; cfg jsonb;
begin
 select break_config into cfg from public.hikes where id=p_hike and user_id=auth.uid() for update;
 if not found then raise exception 'Hike unavailable'; end if;
 for item in select value from jsonb_array_elements(p_edits) loop
  select * into old_event from public.break_events where id=(item->>'id')::uuid and hike_id=p_hike and user_id=auth.uid();
  if not found then raise exception 'Break unavailable'; end if;
  if (old_event.classification='moving' or item->>'classification'='moving') and old_event.classification <> item->>'classification' then
   raise exception 'Moving intervals cannot be relabelled as breaks';
  end if;
  if item->>'classification'='stationary_break' and (old_event.duration_minutes < (cfg->>'min_break_minutes')::float8
   or old_event.duration_minutes > (cfg->>'max_break_minutes')::float8) then raise exception 'Invalid ordinary break duration'; end if;
  update public.break_events set classification=item->>'classification',manually_verified=true,
    data=data || jsonb_build_object('break_classification',item->>'classification','correction_notes',left(coalesce(item->>'notes',''),2000))
  where id=old_event.id and user_id=auth.uid();
 end loop;
 update public.hikes set training_enabled=training_enabled where id=p_hike and user_id=auth.uid();
end $$;
revoke all on function public.review_hiking_breaks(uuid,jsonb) from public,anon;
grant execute on function public.review_hiking_breaks(uuid,jsonb) to authenticated;
alter table public.account_deletion_jobs add column delete_identity boolean not null default true;

-- Serialize history writes with activation and deletion gates per account.
create function public.lock_hiking_account() returns trigger language plpgsql security invoker
set search_path = '' as $$
begin
 if tg_op='DELETE' then
  perform 1 from public.profiles where id=old.user_id for update;
  return old;
 end if;
 perform 1 from public.profiles where id=new.user_id for update;
 return new;
end $$;
create trigger lock_hike_account before insert or update or delete on public.hikes for each row execute function public.lock_hiking_account();
create trigger lock_segment_account before insert or update or delete on public.hike_segments for each row execute function public.lock_hiking_account();
create trigger lock_break_account before insert or update or delete on public.break_events for each row execute function public.lock_hiking_account();

grant all on public.profiles,public.hikes,public.hike_segments,public.break_events,public.model_versions,public.predictions to service_role;
