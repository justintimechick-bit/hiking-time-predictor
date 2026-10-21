-- Reset completion and unlocking happen atomically. Not callable by end users.
create function public.finish_hiking_data_reset(p_user uuid) returns void
language plpgsql security definer set search_path = '' as $$
begin
 if not exists(select 1 from public.account_deletion_jobs where user_id=p_user and status='pending' and not delete_identity) then
  raise exception 'No confirmed data reset pending';
 end if;
 if exists(select 1 from public.hikes where user_id=p_user)
 or exists(select 1 from public.model_versions where user_id=p_user)
 or exists(select 1 from public.predictions where user_id=p_user)
 or exists(select 1 from public.legacy_imports where user_id=p_user)
 or exists(select 1 from storage.objects where bucket_id in ('hike-gpx','user-models') and (storage.foldername(name))[1]=p_user::text) then
  raise exception 'Cleanup is incomplete';
 end if;
 update public.profiles set deletion_requested_at=null,display_name='',avatar_url=null,preferred_hike_type='day_hike' where id=p_user;
 update public.account_deletion_jobs set status='complete',completed_at=now() where user_id=p_user;
end $$;
revoke all on function public.finish_hiking_data_reset(uuid) from public,anon,authenticated;
grant execute on function public.finish_hiking_data_reset(uuid) to service_role;
