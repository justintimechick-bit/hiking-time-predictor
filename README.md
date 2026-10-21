# Personal Hiking Time Predictor

A Streamlit app for private hiking history, personal moving-time and break models,
and planned-route predictions. Sign in with Google through Supabase, import GPX
recordings, choose training hikes, and save predictions to your own account.

## Host on Streamlit Community Cloud

Follow [the deployment guide](docs/STREAMLIT_CLOUD.md). Deploy branch `main` with
entrypoint `streamlit_app.py`, Python 3.13, and the private settings shown in
[.streamlit/secrets.example.toml](.streamlit/secrets.example.toml).

The hosted app uses `AUTH_MODE=streamlit`: a bundled Google sign-in button,
Supabase user sessions, and no separate auth server or persistent app disk.
Configure the Google Web client origin and Supabase provider before signing in.
Browser refreshes and app restarts require signing in again; saved hiking data
and models persist in Supabase. Existing Google users in the same Supabase project
continue using their accounts.

## Local development

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -c constraints.txt -e '.[dev]'
cp .env.example .env
```

Fill in `.env`, including the public Google client ID and a random model signing
key. Generate that key with:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Configure Google to allow `http://localhost` and `http://localhost:8501`, enable
Google in Supabase, and apply the versioned SQL migrations:

```bash
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`. For the optional two-process deployment with a
persistent encrypted session vault, set `AUTH_MODE=service` and follow
[the service setup guide](docs/SUPABASE_SETUP.md). That mode is for a trusted
single host, not Community Cloud.

## Account workflow

- **My Dashboard:** personal activity totals, models and saved predictions.
- **My Hiking History:** preview and confirm recordings, review breaks, select
  individual hikes or use all uploaded history for training.
- **My Models:** explicitly train or retrain moving-time and break models.
- **Trail Predictor:** upload a planned route, choose hike type and one-way or
  out-and-back direction, and inspect the map, elevation and segment estimates.
- **Account Settings:** preferences, private exports, explicit legacy import,
  and confirmed data/account deletion.

Database row-level security and private Storage buckets enforce ownership using
the signed-in user's Supabase JWT. Normal queries never use an administrator
key. Model signatures are verified before deserialization. Never upload unsigned
pickle files or make the storage buckets public. See [SECURITY.md](SECURITY.md).

The account break allowance is trained on elapsed time minus moving time for
eligible completed hikes, including short pauses. Ordinary break-event counts
remain separate descriptive measures. Unknown GPS intervals can include
unmeasured movement, so these are planning estimates.

## Prediction and model evaluation

Recorded GPX files are processed into distance segments using configurable GPS
quality rules and elevation smoothing. Model evaluation holds out whole hikes;
neighboring points and segments are not independent training examples.

Moving-time models use distance, optionally ascent/descent and known hike type.
The app compares them with the stated Naismith baseline: 20 minutes per mile plus
0.03 minutes per ascent foot. A regression becomes the primary estimate only
when the conservative held-out whole-hike selection criteria pass. Otherwise
new users receive the labeled baseline. Break estimates use only personal
eligible history and are unavailable when that evidence is absent.

Planned routes need elevations but do not need timestamps. For out-and-back,
upload the outbound leg only; the return geometry is predicted separately.
Weather, footing, fatigue and pack weight are not modeled. Predictions are
estimates, not guaranteed arrival times.

## Tests and publication checks

```bash
python -m pytest -q
python -m pip check
# After staging the intended publication:
git add .
python scripts/check_publication.py
```

Offline tests use synthetic data and mocked services. Direct authorization tests
require a separate disposable migrated Supabase project:

```bash
RUN_SUPABASE_INTEGRATION=1 python -m pytest tests/integration -q
```

See [the setup guide](docs/SUPABASE_SETUP.md#6-verify-with-two-google-accounts)
for the required `SUPABASE_TEST_*` variables. Skipped integration tests do not
verify deployed authorization. Before inviting users, test Google sign-in and
cross-account isolation in two browser profiles against the deployed app.
GitHub Actions runs offline tests and publication checks on pushes and PRs.

## Offline analysis and original data

Install notebook tools with `python -m pip install -e '.[notebooks]'`. The
notebooks contain reusable analysis code without saved outputs. To use the
original owner-only interface and CLI locally:

```bash
python -m hiking_time_predictor.import_gpx --config config/gpx_import.json
python -m hiking_predictor train
python -m hiking_predictor analyze-breaks
python -m streamlit run local_streamlit_app.py
```

Personal recordings, trained models, generated reports and private analyses are
excluded from Git. Original notebook outputs and the prior local README are
backed up under the ignored `private/publication-backup/` directory. Empty CSV
templates remain in `data/`. Do not deploy the offline owner-only entrypoint.

To explicitly package existing local history for your own signed-in account:

```bash
python -m hiking_time_predictor.migrate_local --source . --output /tmp/my-hiking.migration.zip
```

Treat the archive as private location history. Import it from Account Settings,
then train signed cloud models. Legacy model bytes are kept as opaque backups
and are never executed. Export and verify your data before removing local copies.

[Architecture](docs/ARCHITECTURE.md) · [Community Cloud deployment](docs/STREAMLIT_CLOUD.md) · [Service setup](docs/SUPABASE_SETUP.md)
