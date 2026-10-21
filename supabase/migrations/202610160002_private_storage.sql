-- Bucket limits are enforced by Storage as well as Python.
insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types) values
('hike-gpx','hike-gpx',false,20971520,array['application/gpx+xml','application/xml']),
('user-models','user-models',false,52428800,array['application/octet-stream'])
on conflict(id) do update set public=false,file_size_limit=excluded.file_size_limit,allowed_mime_types=excluded.allowed_mime_types;

create policy hiking_object_select on storage.objects for select to authenticated
using(bucket_id in ('hike-gpx','user-models') and (storage.foldername(name))[1]=(select auth.uid())::text and public.hiking_account_live());
create policy hiking_object_insert on storage.objects for insert to authenticated
with check(bucket_id in ('hike-gpx','user-models') and (storage.foldername(name))[1]=(select auth.uid())::text
 and array_length(storage.foldername(name),1)=1 and name !~ '\.\.' and public.hiking_account_live());
-- No UPDATE policy: original recordings and versioned model blobs are immutable.
create policy hiking_object_delete on storage.objects for delete to authenticated
using(bucket_id in ('hike-gpx','user-models') and (storage.foldername(name))[1]=(select auth.uid())::text and public.hiking_account_live());
