// Execute real PostgreSQL policies in an isolated WASM PostgreSQL engine.
// This verifies SQL behavior, not Supabase Auth or Storage HTTP/object-provider integration.
// PGLITE_MODULE_PATH=/tmp/hiking-postgres-validation/node_modules/@electric-sql/pglite/dist/index.js node tests/sql/validate_migrations.mjs
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const { PGlite } = await import(pathToFileURL(process.env.PGLITE_MODULE_PATH).href);
const db = new PGlite();
let checks = 0;
async function check(value) { assert(value); checks++; }
async function rejected(sql,params=[],code) {
  let error;
  try { await db.query(sql,params); } catch (e) { error=e; }
  assert(error, 'Expected rejected SQL: '+sql);
  if(code) assert.equal(error.code,code);
  checks++;
}
await db.exec(`
 create role anon; create role authenticated; create role service_role bypassrls;
 create schema auth; create schema storage;
 create table auth.users(id uuid primary key,raw_user_meta_data jsonb not null default '{}');
 create function auth.uid() returns uuid language sql stable as $$select nullif(current_setting('request.jwt.claim.sub',true),'')::uuid$$;
 create table storage.buckets(id text primary key,name text,public boolean,file_size_limit bigint,allowed_mime_types text[]);
 create table storage.objects(id uuid primary key default gen_random_uuid(),bucket_id text references storage.buckets(id),name text not null);
 alter table storage.objects enable row level security;
 create function storage.foldername(text) returns text[] language sql immutable as $$select (string_to_array($1,'/'))[1:array_length(string_to_array($1,'/'),1)-1]$$;
 grant usage on schema auth,storage to anon,authenticated,service_role;
 grant select,insert,update,delete on storage.objects to anon,authenticated,service_role;
`);
const directory = new URL('../../supabase/migrations/',import.meta.url);
for(const name of (await readdir(directory)).filter(n=>n.endsWith('.sql')).sort()) {
  await db.exec(await readFile(new URL(name,directory),'utf8'));
  console.log('Applied '+name);
}
const a=randomUUID(), b=randomUUID();
await db.query('insert into auth.users(id,raw_user_meta_data) values ($1,$3),($2,$3)',[a,b,JSON.stringify({full_name:'Synthetic hiker'})]);
const profileCount=await db.query('select count(*)::int as n from public.profiles');
await check(profileCount.rows[0].n===2);
async function asUser(uid) {
  await db.exec('reset role; set role authenticated;');
  await db.query("select set_config('request.jwt.claim.sub',$1,false)",[uid]);
}
function hike(uid,id,hash) {
 return {id,user_id:uid,name:'Synthetic recording',activity_date:'2026-01-01T00:00:00Z',hike_type:'day_hike',
 distance_miles:1,elevation_gain_ft:20,elevation_loss_ft:5,moving_time_min:20,elapsed_time_min:22,
 gps_quality_status:'',original_gpx_storage_path:uid+'/'+id+'.gpx',original_file_hash:hash,
 training_enabled:true,summary:{trackpoint_count:20,missing_timestamp_count:0},
 import_config:{segment_miles:0.5},break_config:{min_break_minutes:2,max_break_minutes:60}};
}
const ha=randomUUID(),hb=randomUUID();
const segments=[{segment_index:0,distance_miles:1,elevation_gain_ft:20,elevation_loss_ft:5,moving_time_min:20,quality_flags:''}];
const events=[{start_time:'2026-01-01T00:20:00Z',end_time:'2026-01-01T00:22:00Z',duration_minutes:2,distance_into_hike_miles:1,break_classification:'stationary_break'}];
for(const [uid,id] of [[a,ha],[b,hb]]) {
 await asUser(uid);
 await db.query('select public.import_recorded_hike($1,$2,$3)',[JSON.stringify(hike(uid,id,'a'.repeat(64))),JSON.stringify(segments),JSON.stringify(events)]);
 await db.query("insert into storage.objects(bucket_id,name) values ('hike-gpx',$1),('user-models',$2)",[uid+'/'+id+'.gpx',uid+'/'+randomUUID()+'.joblib']);
}
await asUser(a);
for(const table of ['hikes','hike_segments','break_events']) {
 await check((await db.query(`select * from public.${table} where user_id=$1`,[b])).rows.length===0);
 await check((await db.query(`update public.${table} set user_id=$1 where user_id=$2 returning *`,[a,b])).rows.length===0);
 await check((await db.query(`delete from public.${table} where user_id=$1 returning *`,[b])).rows.length===0);
}
await check((await db.query('select * from storage.objects')).rows.length===2);
await rejected("insert into storage.objects(bucket_id,name) values ('hike-gpx',$1)",[b+'/forged.gpx'],'42501');
await check((await db.query('delete from storage.objects where name like $1 returning *',[b+'/%'])).rows.length===0);
await rejected('update public.profiles set deletion_requested_at=now() where id=$1',[a],'42501');
await rejected('update public.hikes set user_id=$1 where id=$2',[b,ha],'42501');
await rejected(`insert into public.hike_segments(user_id,hike_id,segment_index,distance_miles,moving_time_min,data) values($1,$2,2,1,20,'{}')`,[a,hb],'23503');
await rejected(`insert into public.break_events(user_id,hike_id,start_time,end_time,duration_minutes,distance_into_hike_miles,classification,data) values($1,$2,'2026-01-02T00:00Z','2026-01-02T00:02Z',2,1,'stationary_break','{}')`,[a,hb],'23503');
const invalid=randomUUID();
await rejected('select public.import_recorded_hike($1,$2,$3)',[JSON.stringify(hike(a,invalid,'b'.repeat(64))),JSON.stringify([{...segments[0],moving_time_min:null}]),'[]'],'23502');
await check((await db.query('select * from public.hikes where id=$1',[invalid])).rows.length===0);
await rejected('select public.import_recorded_hike($1,$2,$3)',[JSON.stringify(hike(a,randomUUID(),'a'.repeat(64))),JSON.stringify(segments),'[]'],'23505');
async function models(uid) {
 const version=randomUUID(), ids=[randomUUID(),randomUUID()];
 for(let i=0;i<2;i++) await db.query(`insert into public.model_versions(id,user_id,model_type,algorithm,version,training_hike_count,training_segment_count,training_features,evaluation_metrics,storage_path,artifact_hash,provenance,dataset_fingerprint,status)
 values($1,$2,$3,'linear',$4,1,1,'[]','{}',$5,$6,'{}',$6,'ready')`,[ids[i],uid,i?'break':'moving',version,uid+'/'+ids[i]+'.joblib','c'.repeat(64)]);
 return ids;
}
const ids=await models(a);
let snapshot=(await db.query('select public.hiking_training_snapshot() as data')).rows[0].data;
await db.query('select public.activate_model_pair($1,$2,$3)',[...ids,JSON.stringify(snapshot)]);
await check((await db.query('select * from public.model_versions where is_active')).rows.length===2);
const nextIds=await models(a);
await db.query('update public.hikes set training_enabled=false where id=$1',[ha]);
await rejected('select public.activate_model_pair($1,$2,$3)',[...nextIds,JSON.stringify(snapshot)]);
await check((await db.query('select * from public.model_versions where is_active')).rows.every(r=>ids.includes(r.id)));
const event=(await db.query('select id from public.break_events')).rows[0];
await db.query('select public.review_hiking_breaks($1,$2)',[ha,JSON.stringify([{id:event.id,classification:'unknown',notes:'Synthetic review'}])]);
await check((await db.query('select manually_verified,classification from public.break_events')).rows[0].manually_verified);
await asUser(b);
await check((await db.query('select * from public.model_versions')).rows.length===0);
const bModels=await models(b);
await asUser(a);
await rejected(`insert into public.predictions(user_id,route_name,distance_miles,elevation_gain_ft,elevation_loss_ft,hike_type,route_direction,predicted_moving_time_min,predicted_total_time_min,moving_model_version_id)
 values($1,'Plan',1,1,1,'day_hike','one-way',20,20,$2)`,[a,bModels[0]],'23503');
await rejected('select public.finish_hiking_data_reset($1)',[a],'42501');
await db.exec('reset role;set role anon');
await db.query("select set_config('request.jwt.claim.sub','',false)");
await rejected('select * from public.hikes',[],'42501');
await check((await db.query('select * from storage.objects')).rows.length===0);
await db.exec('reset role');
await db.query('update public.profiles set deletion_requested_at=now() where id=$1',[a]);
await asUser(a);
await check((await db.query('select * from public.hikes')).rows.length===0);
await rejected("insert into storage.objects(bucket_id,name) values('hike-gpx',$1)",[a+'/after-lock.gpx'],'42501');
await db.exec('reset role;set role service_role');
await check((await db.query('select * from public.hikes')).rows.length===2);
await db.close();
console.log(`PASS: ${checks} PostgreSQL migration, transaction, RLS and ownership checks. Supabase HTTP integration remains separate.`);
