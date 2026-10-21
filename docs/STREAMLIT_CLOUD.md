# Publish on Streamlit Community Cloud

Community Cloud runs `streamlit_app.py` as one Python process. Set
`AUTH_MODE=streamlit` to use the bundled Google sign-in component. **Do not run
the optional FastAPI auth service on Community Cloud.** The older `service` mode
is for a single trusted host with a persistent shared disk only.

## 1. Prepare Supabase

Create or select your production project and apply all four migrations:

```bash
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
```

Back up an existing database before applying migrations. Never run `db reset`
on production. Enable Google in **Authentication → Sign In / Providers** and
keep nonce validation enabled. The `hike-gpx` and `user-models` buckets must
remain private. Ordinary app requests use the publishable key and a verified
user token; the application does not use an administrator key for those queries.

The Google provider's Web client ID must match the `GOOGLE_CLIENT_ID` below.
Keep its Google client secret in Supabase's provider configuration. The Streamlit
app only needs the public Google client ID. Existing Supabase Google identities
remain in the same project; this is a different login UI for that provider.

## 2. Configure the Google Web client

Choose the Streamlit subdomain you intend to deploy, for example
`https://YOUR_APP.streamlit.app`. In Google Cloud's Web OAuth client settings,
add this exact origin to **Authorized JavaScript origins**. For local testing,
also add `http://localhost` and `http://localhost:8501`.

Keep the Supabase provider's callback in **Authorized redirect URIs**:
`https://YOUR_PROJECT_REF.supabase.co/auth/v1/callback`. The Community Cloud
mode uses Google's popup/button ID-token flow, so it does not use the local
FastAPI `/auth/callback` route or Streamlit's native OIDC `/oauth2callback` route.
Set Supabase's Site URL to your deployed app URL. If the Google consent screen
is in testing, add your intended testers before attempting sign-in.

Google's button runs in a bundled Streamlit component. Each login gets a random
server-side challenge. Only its SHA-256 hash enters the browser. Supabase
validates Google's token signature, audience, expiry and nonce before issuing a
user session, and the server verifies that session through `get_user`. Keep
Google nonce checks enabled; do not work around login errors by disabling them.

## 3. Deploy from GitHub

1. Sign into [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose **Create app**, select this GitHub repository and branch `main`.
3. Set the entrypoint to `streamlit_app.py` and choose your subdomain.
4. In **Advanced settings**, use Python **3.13** or the current default **3.14**.
   GitHub Actions checks both runtimes.
5. Copy `.streamlit/secrets.example.toml` into the Secrets editor and replace
   every placeholder. Generate `MODEL_SIGNING_KEY` locally with
   `python -c 'import secrets; print(secrets.token_urlsafe(48))'`.
6. Deploy. If the assigned URL differs from the planned URL, update `APP_URL`
   and Google's authorized JavaScript origin to match the actual URL.

Secrets belong in the private Secrets editor, never in GitHub. Keep the model
signing key stable between releases. `SUPABASE_SECRET_KEY` is a server-only
privileged key needed for confirmed account/data deletion. If omitted, regular
account features work, but cleanup cannot complete. Root-level Streamlit secrets
are used when equivalent environment variables are absent.

The project uses `requirements.txt`, runtime version constraints, and packaged
component HTML. Notebook and test tools are optional extras. Streamlit's CORS
and XSRF protections remain enabled. Do not deploy `local_streamlit_app.py`:
that entrypoint is intentionally an offline owner-only interface.

## 4. Test before sharing the URL

Use two Google accounts in separate browser profiles. Check that each starts
with its own history, can import synthetic recordings, train personal models,
save predictions, export data, and log out. Verify that neither can read the
other's database rows or storage objects. Run the direct integration suite on a
**disposable** migrated Supabase project, using the instructions in
[SUPABASE_SETUP.md](SUPABASE_SETUP.md#6-verify-with-two-google-accounts).

The new hosted login component needs a real authorized origin and browser Google
consent test. Offline tests cover server-side behavior; they do not establish
that the Google Console, Supabase provider, or live browser is configured correctly.

## Sessions and operation

Refresh credentials live only in the current server-side Streamlit browser
session, protected from logs and UI output. No tokens are written to the app's
filesystem or placed in URLs. Page refresh, a disconnected session that is
discarded, app restart, or Cloud hibernation requires another sign-in. Saved
history, models, and predictions persist in Supabase. Sessions have a seven-day
absolute application limit and refresh Supabase access tokens as needed.
Logout revokes Supabase refresh sessions and discards the current app session;
already issued access tokens can remain valid until their normal expiry.

Account/data deletion requires typed confirmation and a login within ten minutes.
It uses the existing durable cleanup job and locking flow. A failed cleanup
remains pending; an operator must retry it using private credentials:

```bash
python -m hiking_time_predictor.account_cleanup --retry-pending
```

Schedule retries on a trusted scheduler/host if you need unattended recovery;
Community Cloud's sleeping app is not a reliable background scheduler. Keep
keys backed up securely. A public beta should monitor resource/storage usage;
larger datasets and frequent concurrent model training may require a dedicated
host and a shared job queue.

## Publishing updates

Keep personal recordings, generated analyses, model files and notebook outputs
out of commits. The original local README and notebook outputs are backed up in
the ignored `private/publication-backup/` directory. To check the next commit:

```bash
git add .
python scripts/check_publication.py
python -m pytest -q
git commit -m "Describe the change"
git push
```

GitHub Actions repeats publication checks and the offline tests. Community Cloud
redeploys from the selected GitHub branch. No live authorization tests run in CI
without dedicated disposable-project credentials.

References: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
[Google Identity Services](https://developers.google.com/identity/gsi/web/reference/js-reference),
[Supabase ID-token sign-in](https://supabase.com/docs/reference/python/auth-signinwithidtoken).
