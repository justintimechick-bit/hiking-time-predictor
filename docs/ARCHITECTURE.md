# Implementation plan and architecture

## Streamlit Community Cloud mode (2026-10-21)

`AUTH_MODE=streamlit` selects the single-process hosted path documented in
[STREAMLIT_CLOUD.md](STREAMLIT_CLOUD.md). A bundled Google Identity Services
button receives only a public client ID and hashed random nonce. The raw nonce
stays in Streamlit's server session. Supabase verifies the ID token and nonce,
issues its own user session, and the application verifies `get_user` before
binding any private UI state. All repository calls still carry that user's JWT.
Refresh credentials remain only in that browser session's server memory, with a
lock for refresh rotation, and never enter a URL or model cache. Reconnects that
lose server state require another login. No SQLite or FastAPI service is used in
this mode. Destructive actions run inside Streamlit's XSRF/CORS-protected UI,
require typed confirmation and recent verified login, and invoke only the
existing dedicated cleanup path with a privileged key. The disk vault, cookie
and proxy discussion below applies to the retained `service` mode.

Review completed before the Supabase implementation, 2026-10-16. The original
suite passed **350 tests**. There was no Supabase SDK or Supabase implementation.
The abandoned Drive changes were configuration/document scaffolding only; they
have been removed. No Google Drive backend or native Streamlit identity was added.

## What already worked

`gpx.process_hike` reads recorded trackpoints, classifies moving/stationary/unknown
intervals, smooths elevations and builds distance segments. Its output includes
summary, segment, point and interval tables. `import_gpx.import_recordings` writes
these tables and a settings/provenance JSON. It accepts explicit paths, so the
underlying preprocessing can be reused without reading the owner's directories.

`modeling` and `predictor.train_predictor` perform hike-grouped validation of
linear regressions A/B/C against Naismith. At least ten verified complete held-out
hikes and better whole-hike MAE are required to promote regression. `breaks`
extracts ordinary rests separately from movement and ambiguous time;
`train_break_predictor` evaluates whole-hike regressions and a personal rate.
The existing break evaluator reports MAE; it does not report RMSE. No new score
is invented. Planned-route prediction already supports elevations, timestamps
optional, and separately predicted reverse geometry for out-and-back trips.

The old Streamlit entrypoint assumed everyone was the owner: project-relative
`data/` and `models/`, global model-resource caching, local file-mtime route
invalidation, local CSV break correction writes. That UI is preserved in
`local_streamlit_app.py` for offline coursework. Its original tests now explicitly
exercise that entrypoint. Public `streamlit_app.py` has no local-data fallback.

## Incremental implementation

1. Introduce a focused FastAPI authentication service using Supabase Google
   authorization code + PKCE, encrypted server-side sessions and an opaque cookie.
2. Add versioned PostgreSQL migrations, composite ownership foreign keys, RLS,
   private buckets, transactional import/activation and verified account cleanup.
3. Reuse the ML pipeline through a repository adapter and personal-training
   service. Sign model artifacts before private persistence; verify before load.
4. Connect authenticated dashboard/history/models/prediction/settings pages to
   the same design and charts. Add explicit owner migration, exports and deletion.
5. Run offline regressions and SDK/mock tests. Provide separate opt-in direct
   database/Storage tests; do not represent mocks as proof of deployed policies.

## Why a small authentication service?

A browser redirect can destroy a Streamlit WebSocket/session. Keeping a PKCE
verifier only in `st.session_state` is insufficient. FastAPI stores the verifier
and single-use state in a private encrypted SQLite vault before leaving the app.
The callback matches a random HttpOnly flow cookie, consumes the flow, exchanges
the code through Supabase Auth, then verifies the access token with `get_user`.
Supabase owns registration and identity. There is no second identity system.

The browser holds an opaque random session handle. Supabase JWTs and refresh
tokens remain encrypted on disk and are refreshed under a database transaction
so two tabs/processes cannot race token rotation. Session handles are hashed in
the vault. The cookie is HttpOnly, SameSite=Lax, Path=/, with Secure and `__Host-`
in production. Only localhost development may disable Secure. A seven-day
absolute application session limit requires a new Google login thereafter.

SQLite holds **only authentication sessions and OAuth flow state**. Profiles,
activities, training data, model metadata and predictions live in Supabase
PostgreSQL; GPX/model bytes live in private Supabase Storage. This simple session
vault requires both Python processes on one host with a shared persistent disk
and keys. Multi-instance deployment needs a shared encrypted session store and
cross-instance refresh lock; it must not copy the SQLite vault to separate hosts.

## Ownership and failures

Normal clients use the publishable key plus the current user's Supabase JWT.
RLS enforces `auth.uid()` on every table and object prefix; Python checks are an
additional guard. Child rows use composite `(hike_id,user_id)` foreign keys.
Profile ownership cannot be changed by end users. A transactional snapshot feeds
training. Activation locks the account, rechecks the snapshot and switches both
models together; old models stay active on failure or conflicting history edits.

Storage and PostgreSQL are separate systems, so there is no cross-service
transaction. A recording blob uploads first, followed by one DB transaction for
its summary/segments/breaks. Failed transactions remove the blob only after
checking whether the write actually committed. A process crash can leave a
private orphan; it is never used for training and is included in export/account
cleanup. Individual deletion first marks the hike as deleting and excludes it
from training, then removes Storage and DB data; an interrupted deletion can be
retried. Failed model runs can leave private inactive versions/orphans; they do
not replace active versions and are included in account cleanup and export.

An HMAC signature binds each model's bytes, owner UUID, model UUID, training
snapshot, version, model type and Python/scikit-learn/joblib runtime. Verification
precedes deserialization. A user can upload bytes through the Storage API within
their own prefix, but cannot create a valid application signature. Legacy pickle
files are preserved only as opaque backup bytes. They are never loaded or signed.
A compromised application signing key or server is outside this trust boundary.

## Deletion and migration

Destructive account actions require a recent Google session (ten minutes), an
Origin check, action-specific CSRF token and typed confirmation. Logout instead
uses Yes/No buttons with the same Origin and CSRF protections. Auth pages send
`Referrer-Policy: strict-origin` so browser form posts retain their Origin while
OAuth callback paths and codes never appear in the Referer header. Only the auth
service uses a privileged key for these actions. The CSP `form-action` permits
the auth origin and configured application origin so browsers can follow the
post-confirmation redirect to Streamlit, including across localhost ports.
The service persists a cleanup job,
locks the account against end-user writes, removes objects through Storage APIs,
checks cleanup, deletes application rows, then deletes the Supabase Auth identity.
Failed cleanup remains pending for an operator retry. The Google account itself
is never deleted. Deleting only hiking data retains the sign-in identity.

The developer packages selected local GPX/config/dataset/model files explicitly.
Uploading that ZIP is a separate confirmed one-time action into the current
account. The archive is verified after upload, recordings are reprocessed with
their original metadata/confirmations/corrections, and imported IDs are read back.
Original local files are never overwritten or removed. Retraining creates signed
cloud models; legacy models and processed datasets remain downloadable backups.
