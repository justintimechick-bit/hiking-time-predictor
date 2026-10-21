# Security and private data

Do not attach personal GPX recordings, location history, populated environment
files, access tokens, session databases, or private model artifacts to public
issues or pull requests. Use synthetic recordings for bug reports.

Report suspected authorization or credential issues through the repository's
GitHub **Security → Report a vulnerability** option if available. If private
reporting is unavailable, contact the repository owner privately; do not post
exploit details or credentials publicly.

Normal application requests must use a publishable Supabase key and the signed-in
user's JWT. A privileged key is restricted to explicitly confirmed account
cleanup. Keep RLS enabled and the Storage buckets private. Never deserialize
unsigned model uploads or disable signature verification.

The Streamlit Community Cloud mode keeps refresh credentials only in the current
server-side browser session. App restarts and page refreshes require sign-in.
Keep Streamlit XSRF/CORS protections enabled. Use a disposable Supabase project
for direct integration tests. Passing offline tests does not verify deployed
policies, Google settings, or browser behavior.
