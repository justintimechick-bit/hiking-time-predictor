# Run the multi-user application

For **Streamlit Community Cloud**, follow [STREAMLIT_CLOUD.md](STREAMLIT_CLOUD.md).
It uses `AUTH_MODE=streamlit`, a Google sign-in component, and server-memory
sessions without a FastAPI process. The two-process instructions below describe
the optional **`AUTH_MODE=service`** deployment. Set that mode explicitly when
copying `.env.example`, whose default is now the single-process Streamlit mode.

Python 3.11+ is required. The app has two small Python processes: Streamlit draws
the hiking pages, and FastAPI completes Google sign-in and manages secure
sessions. Supabase stores application data and private files. Neither process
uses native Streamlit OIDC or Google Drive.

## 1. Create a Supabase project

Create a project at [Supabase](https://supabase.com/dashboard). Save its project
URL and **publishable** API key from Project Settings → API. A legacy `anon` key
also works. Never use a secret/service-role key as the publishable key.

For account deletion, the auth service separately needs a server secret key (or
legacy `service_role` key). Supply this only in the trusted auth service's
server environment. The normal repository client always uses the publishable
key plus the signed-in user's access token.

## 2. Configure Google OAuth and the two different callbacks

In [Google Cloud Console](https://console.cloud.google.com/), configure the OAuth
consent screen/Google Auth Platform branding, audience and test users. Create an
OAuth client with type **Web application**. Request only the identity scopes
used by Supabase Google sign-in (openid, email and profile). No Drive API or
Drive permissions are required. Add your application homepage to authorized
JavaScript origins when requested by Google.

The Google OAuth client's **authorized redirect URI** is Supabase's callback:

```text
https://YOUR_PROJECT_REF.supabase.co/auth/v1/callback
```

Google sends its response to **Supabase**, not Streamlit or FastAPI. Copy the
exact callback shown by your Supabase Google provider settings if using a
custom domain or local Supabase.

In Supabase Authentication → Sign In / Providers → Google, enable Google and
enter that Google client ID and client secret. Do not enable other providers for
the production application. Keep nonce checks enabled. First-time Google login
automatically creates the Supabase Auth user and the database profile trigger
creates the app profile.

In Supabase Authentication → URL Configuration:

- Site URL, local: `http://localhost:8501`.
- Allowed application redirect, local: `http://localhost:8000/auth/callback**`.
- Allowed application redirect, deployed:
  `https://hiking.example.com/auth/callback**` (replace the hostname).

The suffix allows the random state query carried by each login. Keep the scheme,
host, port and callback path fixed. Do not allow arbitrary hosts or a whole-site
`/**` wildcard. The app itself uses only the configured callback and never accepts
an arbitrary `next`/return URL.

Supabase sends its **authorization code** to the application callback:

```text
http://localhost:8000/auth/callback?state=RANDOM&code=SINGLE_USE_CODE
```

FastAPI validates the browser's flow cookie and state, retrieves the encrypted
PKCE verifier and exchanges the code with Supabase. It then redirects to
Streamlit. These are two separate OAuth legs with separate callback URLs.

References: [Supabase Google provider](https://supabase.com/docs/guides/auth/social-login/auth-google),
[Python OAuth](https://supabase.com/docs/reference/python/auth-signinwithoauth),
[PKCE code exchange](https://supabase.com/docs/reference/python/auth-exchangecodeforsession),
[server-side session guidance](https://supabase.com/docs/guides/auth/server-side/advanced-guide).
The implementation was checked against installed `supabase-py` 2.33.0.

## 3. Apply the version-controlled migrations

Install the [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started).
From this repository:

```bash
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
```

Apply migrations to a fresh development project before production. They create
profiles, hikes, segments, breaks, versioned models, predictions, migration
tracking and cleanup jobs. They also create the private `hike-gpx` and
`user-models` buckets and their RLS policies. Do not manually recreate the tables
or make the buckets public. Storage limits are 20 MiB per GPX and 50 MiB per
model/backup object (migration archives over 50 MiB require splitting the legacy
source before packaging; the web ZIP validator additionally caps expansion).

The SQL files are:

```text
supabase/migrations/202610160001_personal_hiking.sql
supabase/migrations/202610160002_private_storage.sql
supabase/migrations/202610160003_legacy_imports.sql
supabase/migrations/202610160004_cleanup_finalization.sql
```

For a local Supabase stack, install Docker and the CLI, then use `supabase start`
and `supabase db reset`. Reset deletes local development data; never use reset
against a production database. `supabase status` prints local keys. For actual
local Google consent, enable the Google section of `supabase/config.toml`, set
`GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the CLI environment, and use the
local Supabase Auth callback (normally
`http://127.0.0.1:54321/auth/v1/callback`) in Google. A hosted development project
is easier for first-time Google setup.

## 4. Configure Python and local secrets

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Use any installed Python 3.11+ interpreter instead of `python3.13`. Put the first
new random value in `AUTH_SESSION_KEY` and the second in `MODEL_SIGNING_KEY`.
Fill in `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. Set local URLs exactly as
shown in `.env.example`, with `COOKIE_SECURE=false` only for localhost HTTP.
Set restrictive file permissions:

```bash
chmod 600 .env
```

Keep both keys stable across restarts. Losing the session key requires a new
Google login; losing the model key requires retraining the cloud models. The
session vault contains encrypted tokens and is ignored by Git. Protect and back
up its private volume and encryption key separately. Do not include `.env`,
`.auth/`, private history or migration archives in a public deployment image.

For local account deletion, supply `SUPABASE_SECRET_KEY` to the auth process.
For a deployed system, inject it into that process alone through the host's
secret manager; do not add it to browser code, repository files, URLs or logs.

## 5. Start the application

Terminal 1, with the environment activated:

```bash
python -m hiking_time_predictor.auth_service
```

Terminal 2, from the same repository and environment:

```bash
python -m streamlit run streamlit_app.py
```

Visit `http://localhost:8501`, always using the same hostname as the auth
service. Do not mix `localhost` and `127.0.0.1`. Select **Continue with Google**.
The auth service must remain running. Both processes must use the same session
DB path and encryption key. No tokens are placed in Streamlit query parameters.

Without `.env`/live credentials, the multi-user entrypoint displays a setup
landing page and does not expose local history. For the original offline owner
application and course workflows:

```bash
python -m streamlit run local_streamlit_app.py
python -m hiking_time_predictor.import_gpx --config config/gpx_import.json
python -m hiking_predictor train
python -m hiking_predictor analyze-breaks
```

Run the offline owner app only on your own machine; it intentionally reads your
original `data/` and `models/` and is not the multi-user deployment entrypoint.

## 6. Verify with two Google accounts

Use two browser profiles/private windows, one per Google account. Both should
initially show empty histories. Upload different **synthetic/non-sensitive**
recordings, preview and confirm them. Confirm training eligibility and explicitly
train both models. Check that each dashboard shows only its own hikes, models
and predictions; log out, sign back in, and check restoration.

Run direct authorization tests on a **disposable migrated project**, separately
from UI checks. Enable the Email provider only on that isolated test project so
the test suite can create two synthetic password users without automating Google
consent. Set `SUPABASE_TEST_URL`, `SUPABASE_TEST_PUBLISHABLE_KEY` and
`SUPABASE_TEST_SECRET_KEY`, then:

```bash
RUN_SUPABASE_INTEGRATION=1 python -m pytest tests/integration -q
```

These tests make actual PostgREST and Storage requests with user A's token,
user B's token and an anonymous client. They attempt reads, writes, forged
ownership/parent references, private-model downloads, public URLs and deletion.
Fixtures remove their test data afterward. Keep all testing keys server-side;
don't put them on a command line or in screenshots. Do not enable password
sign-in in the production app to run these tests there.

The offline suite uses synthetic data and mocked Auth HTTP responses through the
installed SDK:

```bash
python -m pytest -q
```

The integration tests are explicitly skipped unless opted in. A skip does not
verify deployed RLS or Storage policies. Live Google consent, callback cookies,
network/proxy configuration and cross-account isolation must be smoke-tested
with your configured project before inviting other hikers.

## 7. Import your existing project once

From your original local repository:

```bash
python -m hiking_time_predictor.migrate_local --source . --output /tmp/my-hiking.migration.zip
```

This command reads selected original GPX, metadata, confirmations, break
corrections, processed CSV/JSON datasets and model files. It does not overwrite
anything or load a legacy pickle. Treat the ZIP as sensitive location history.
Keep it below the private bucket's 50 MiB per-object limit.

Sign in to **your** Google account, check the account UUID/email in Account
Settings, upload that archive under **Import the original single-user project**,
and explicitly confirm it belongs to you. The app first verifies a private
backup in Storage, reprocesses recordings with the archived settings and labels,
and checks the imported records before completing the one-time marker. A failed
import can resume with the same archive. Exact duplicate recordings are kept
once per account. Afterward, train models from My Models and inspect the results.

Use **Download my data** to verify history and the archived original datasets
and model bytes. Existing trained models are preserved as backup bytes; they are
not trusted for execution in the web app. Newly trained cloud versions carry the
application signature required for loading. Do not delete original local files
until you have independently checked the import and downloaded backups. This
implementation never removes them.

## 8. Privacy and deletion

Individual hikes have a checkbox-confirmed delete action in My Hiking History.
This removes the GPX, segments and breaks. Existing model versions remain snapshots;
retrain to build versions without that hike. Full data/account deletion also
removes those historical model versions.

Logout opens an **Are you sure?** page with **Yes, log out** and **No, stay signed
in** buttons. No typing is needed. Yes revokes the Supabase refresh sessions,
removes local server sessions and clears the browser cookie; No returns to the
app and preserves the session. Restart the auth service after changing its code.
An older form or a session change opens a fresh Yes/No confirmation without
logging anyone out. If the session is already gone, logout clears the stale
cookie and returns to the app. The choice is carried in hidden form fields so
browser submissions do not depend on including the clicked button's value.

Data/account deletion links open typed confirmation pages. Deletion requires a Google
login in the last ten minutes. **Delete all hiking application data** retains the
Auth identity/profile. **Delete my application account** also removes the profile
and Supabase Auth identity. Neither deletes your Google account. If privileged
configuration is missing, no deletion is falsely reported as complete.

Storage and PostgreSQL cannot share a transaction. Cleanup jobs are durable;
failed jobs leave the account locked against normal writes until an operator
resumes them:

```bash
python -m hiking_time_predictor.account_cleanup --retry-pending
```

Schedule that command on the trusted auth host (for example every five minutes)
and monitor failures. It only retries deletions already explicitly requested;
it never guesses which accounts to remove. Storage objects are removed using
Storage APIs, not SQL deletes from `storage.objects`.

## Deployment considerations and troubleshooting

Deploy both Python processes on one trusted host with one private persistent
auth volume. Put them behind **one HTTPS hostname**: proxy `/auth/*`, `/account/*`
and `/health` to FastAPI port 8000, everything else (including WebSockets) to
Streamlit port 8501. Keep both internal ports private. Example Caddy routes:

```caddyfile
hiking.example.com {
    @auth path /auth/* /account/* /health
    reverse_proxy @auth 127.0.0.1:8000
    reverse_proxy 127.0.0.1:8501
}
```

Set `APP_URL` and `AUTH_PUBLIC_URL` to `https://hiking.example.com`, and
`COOKIE_SECURE=true`. Match Supabase's redirect allowlist. Keep Streamlit's XSRF
and CORS protections enabled. Do not log OAuth callback query strings; the
provided auth-service command disables access logs. Apply the same redaction to
your reverse proxy. Add rate limiting for login, callback and account endpoints
at the proxy. Production hosts should encrypt disks and back up keys securely.
A single Streamlit Community Cloud process alone cannot host this architecture.

- **Redirect mismatch:** Google redirects to Supabase; Supabase redirects to
  FastAPI. Check both exact URLs and the limited state-query allowlist pattern.
- **Login returns to landing:** check same hostname, cookie Secure mode, auth
  DB path/key, HTTPS proxy headers and Supabase provider setup. Never paste JWTs
  or raw session objects into logs to debug this.
- **Session expired:** sign in again. The server rotates refresh tokens under a
  lock and has a seven-day absolute application session lifetime.
- **Request origin rejected on logout:** restart the auth service and reload the
  confirmation page before selecting Yes. The service uses `strict-origin`
  referrer policy: it preserves the browser's Origin on form posts without sending
  paths or OAuth query strings in referrers. Keep `AUTH_PUBLIC_URL` consistent
  with the URL you open (localhost port 8000 in development). Origin and CSRF
  validation remain required; null or foreign origins are rejected.
- **Profile unavailable / permission denied:** apply all migrations, verify the
  publishable key/project, RLS and bucket policies. Do not work around it by using
  an administrator client for normal app queries.
- **Yes/No briefly reloads but stays on logout:** restart the auth service and
  reload the confirmation page to receive the updated Content Security Policy.
  Its `form-action` allows both the auth origin and the configured `APP_URL`;
  browsers can otherwise block the redirect from port 8000 to port 8501 after
  a successful form POST. Origin and CSRF validation still apply to the POST.
- **Database setup incomplete after successful Google login:** Auth is connected,
  but `PGRST205` means an application table is absent from the API schema cache;
  `PGRST202` means a required function is absent. From the repository root, run
  `npx --yes supabase login`, `npx --yes supabase link --project-ref YOUR_PROJECT_REF`,
  then `npx --yes supabase db push` if the CLI is not installed globally. Complete
  the interactive CLI login/password prompts in your terminal. The publishable
  and server API keys in `.env` do not replace the CLI/database credentials.
  Refresh Streamlit after all four migrations finish. The profile migration also
  initializes profiles for users who signed in before tables existed.
- **Model rejected:** signature, owner or runtime differs. Keep signing keys
  stable and retrain after Python/scikit-learn/joblib upgrades. Do not upload
  arbitrary joblib files or disable integrity checks.
- **Training has insufficient data:** the Naismith primary estimate is expected.
  Two hikes can support an experimental moving fit; promotion requires the
  existing stricter validation policy. Break estimates need eligible labeled
  history. Untimed routes never become training activities. Partially timestamped recordings
  remain flagged, and existing quality filters exclude uncertain observations.
- **Deleting hike / pending cleanup:** retry individual deletion, or have the
  operator retry already-confirmed account cleanup. Never report success until
  Storage and database cleanup are verified.

The session vault is intentionally a single-host design. Horizontal scaling
requires a shared encrypted session store with cross-process refresh locking,
plus normal backup/recovery and monitoring. No deployed authentication or live
Supabase policy verification is implied by the offline tests.
