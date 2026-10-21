# Verification and change inventory

## Community Cloud publication preparation — 2026-10-21

The current release adds single-process `AUTH_MODE=streamlit` Google ID-token
sign-in with Supabase nonce verification, server-memory session refresh, and
confirmed account cleanup. The older disk-backed auth service remains optional.
The public-source snapshot, tested without a local `.env`, private GPX or models,
passed **428 tests, 7 skipped**: six opt-in live integration tests and one
optional personal-GPX check. Hosted tests also cover recent-login/typed deletion
confirmation and clearing private UI state on account changes. Runtime dependency
and compile checks passed.
Live Google consent and direct authorization testing remain the operator's
manual checks; this preparation does not assert they passed.

Public-source preparation removes saved notebook outputs and excludes personal
analyses/reports, sessions, secrets and trained artifacts. Original README and
notebook outputs are preserved in ignored `private/publication-backup/`.
The repository includes environment/Cloud secrets examples, runtime version
constraints, publication scanning, CI, and a Community Cloud deployment guide.
The implementation notes below are the historical 2026-10-16 verification record.

Implementation date: 2026-10-16. No real Google credentials, personal files or
live Supabase project were used for automated verification. Original local
recordings, processed outputs and trained models were not migrated or removed.

## Executed checks

- Training selection follow-up (2026-10-17): My Hiking History shows each
  uploaded hike with a checkbox and **Use all uploaded data to train the models**.
  Selections save immediately; fitting still requires Train Models or Retrain
  Models in My Models. Streamlit automatic reload and polling are enabled so
  source updates are picked up by the local app. Offline UI tests exercise the
  actual authenticated entrypoint, navigating from selection to training both
  model types using only the selected hikes.
- Original suite before changes: `python -m pytest -q` — **350 passed**.
- Authenticated implementation and preserved offline suite:
  `python -m pytest -q` — **400 passed, 6 skipped**. The skips are the explicitly
  opt-in tests against a live, disposable Supabase Auth/PostgREST/Storage project.
  One upstream FastAPI/Starlette TestClient deprecation warning advises future
  migration from httpx to httpx2; it is not a test failure.
- Supabase SDK **2.33.0** was installed and exercised through mocked HTTP
  responses, including real SDK PKCE generation/code exchange, server session
  restoration, refresh rotation, cookie/CSRF handling and logout. No Google
  consent flow was exercised against a live provider.
- Streamlit AppTest exercised the no-credential landing, authenticated navigation,
  baseline predictions, explicit model training and switching between two
  synthetic accounts. It also ran the original offline upload/chart/navigation
  tests. No real private history was required.
- Logout follow-up: **28 auth/UI tests passed**, followed by the full suite above.
  Yes clears the browser cookie and all local sessions for the signed-in user;
  No preserves the session. Missing/null/foreign origins and invalid CSRF values
  cannot authorize logout. A same-origin stale/invalid confirmation redirects to
  a fresh GET form; null or foreign origins still return 403. Rendered hidden
  inputs carry Yes/No without depending on submitter-button fields. Tests also
  cover old typed forms, changed session cookies and already-expired sessions.
  Deletion retains typed confirmation and action-bound CSRF.
  The confirmation page uses `strict-origin`, replacing `no-referrer`, which can
  cause browser form posts to send `Origin: null`; see
  [MDN's explanation](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Referrer-Policy#effect_on_the_origin_header).
  The auth service was restarted on port 8000. Live localhost checks using
  synthetic requests without user cookies confirmed health 200, stale-form
  recovery 303 and foreign-origin rejection 403. These checks do not exercise
  the user's authenticated browser session; open a fresh logout page to use it.
  The localhost regression also checks that CSP `form-action` explicitly allows
  `http://localhost:8501`, since browsers may apply it to the logout POST's 303
  redirect from port 8000. HTTP test clients do not enforce browser CSP; a real
  Safari logout remains a manual verification step.
- Editable package 0.2.0 built and installed from the updated pyproject.
- `python -m pip check` — **No broken requirements found**.
- `python -m compileall -q streamlit_app.py local_streamlit_app.py src` — passed.
- All four SQL migrations parsed using PostgreSQL's parser via pglast.
- All four migrations **executed** in isolated PGlite PostgreSQL; **32 direct SQL
  checks passed**. The test establishes Supabase-shaped Auth/Storage schemas and
  switches real database roles to exercise the migrated policies, ownership
  constraints, import rollback, duplicate rejection, break review, private object
  metadata and activation conflicts. It is not a running Supabase service and
  does not verify Storage object bytes or its HTTP API.
- Streamlit started successfully on localhost port 8510, then was stopped. No
  connected browser was available, so visual browser QA was not performed.

The SQL execution check is reproducible without installing PostgreSQL globally:

```bash
npm install --prefix /tmp/hiking-postgres-validation --cache /tmp/hiking-npm-cache --ignore-scripts @electric-sql/pglite
PGLITE_MODULE_PATH=/tmp/hiking-postgres-validation/node_modules/@electric-sql/pglite/dist/index.js node tests/sql/validate_migrations.mjs
```

PGlite and pglast are optional validation tools, not application dependencies.
See [PGlite's official documentation](https://pglite.dev/docs/) for the isolated
PostgreSQL engine. For production authorization assurance, apply the migrations
to your disposable Supabase project and execute the six direct HTTP integration
tests as documented in [the setup guide](SUPABASE_SETUP.md), then test Google
consent and returning sessions in two separate browser profiles.

## Files created

- `local_streamlit_app.py`: preserved original three-page owner-only app.
- `src/hiking_time_predictor/cloud_config.py`: validated environment settings;
  separate public-user and privileged service credentials.
- `src/hiking_time_predictor/auth_sessions.py`: SDK PKCE, encrypted session vault,
  verified identity restoration, expiration/refresh and logout.
- `src/hiking_time_predictor/auth_service.py`: FastAPI callback/cookie service,
  protected account confirmation/deletion endpoints.
- `src/hiking_time_predictor/cloud_repository.py`: user-JWT Supabase adapter and
  replaceable repository interface.
- `src/hiking_time_predictor/personal_hiking.py`: historical preview/import,
  private snapshot training, active model loading and export.
- `src/hiking_time_predictor/trusted_models.py`: signed model provenance and
  integrity/runtime checks before joblib loading.
- `src/hiking_time_predictor/safe_xml.py`: bounded untrusted GPX XML validation.
- `src/hiking_time_predictor/cloud_ui.py`: authenticated dashboard, history,
  models, planned trails, settings and explicit migration controls.
- `src/hiking_time_predictor/web_components.py`: existing design, prediction
  charts and model evaluation components receiving supplied personal artifacts.
- `src/hiking_time_predictor/migrate_local.py`: explicit local packaging and
  verified one-time import, with legacy artifacts kept as opaque backups.
- `src/hiking_time_predictor/account_cleanup.py`: privileged, retryable and
  verified Storage/database/Auth cleanup.
- `.env.example`: environment template with placeholders only.
- `supabase/config.toml`: local development stack configuration.
- `supabase/migrations/202610160001_personal_hiking.sql`: tables, constraints,
  indexes, ownership policies, profile trigger, snapshot/import/activation RPCs.
- `supabase/migrations/202610160002_private_storage.sql`: private buckets and
  object RLS; immutable uploaded objects.
- `supabase/migrations/202610160003_legacy_imports.sql`: private one-time legacy
  import tracking.
- `supabase/migrations/202610160004_cleanup_finalization.sql`: privileged atomic
  data-reset finalization and unlocking.
- `tests/test_cloud_auth.py`, `tests/test_cloud_repository.py`,
  `tests/test_personal_hiking.py`, `tests/test_cloud_cleanup_migration.py`,
  `tests/test_cloud_ui.py`: offline synthetic/SDK/state-machine coverage.
- `tests/integration/test_supabase_authorization.py`: two-user direct
  PostgREST/Storage integration tests, explicitly opt-in.
- `tests/sql/validate_migrations.mjs`: executed PostgreSQL migration/RLS checks.
- `docs/ARCHITECTURE.md`, `docs/SUPABASE_SETUP.md`, `docs/VERIFICATION.md`:
  inspection/plan, deployment/setup and this verification report.

## Files modified

- `streamlit_app.py`: authenticated entrypoint; no owner-data fallback.
- `src/hiking_time_predictor/web_upload.py`: XML validation before existing GPX
  parsing; prediction algorithms otherwise unchanged.
- `tests/test_streamlit_app.py`: original owner-app tests explicitly target the
  preserved `local_streamlit_app.py`, including source-reload checks.
- `README.md`: new platform/run instructions above preserved offline ML reference.
- `pyproject.toml`, `requirements.txt`: cloud/auth/security dependencies and test
  marker; project version 0.2.0.
- `.gitignore`: secrets, credential vault and private migration archives ignored.
- `.streamlit/config.toml`: allow migration archive uploads; Python and Storage
  still enforce a separate 20 MiB limit on GPX recordings.

The cancelled Google Drive scaffolding/dependencies were removed. The workspace
has no Git repository metadata, so this inventory is supplied in place of a Git
diff. No changes were committed or deployed.

## Remaining setup and limitations

Configure the Google client/provider and both OAuth callbacks, apply all four
migrations, populate private environment keys, run both services, then execute
live integration and browser tests. Session persistence requires a single host
with a shared private volume and stable keys; horizontal scaling needs a shared
encrypted session store and refresh lock. A server secret is required only for
confirmed account deletion/cleanup. Schedule pending cleanup retries. Storage
and PostgreSQL are separate systems, so interruption can leave private orphans;
these are excluded from training and included in export/account cleanup. Legacy
models are never executed; retrain cloud models from the imported recordings.
