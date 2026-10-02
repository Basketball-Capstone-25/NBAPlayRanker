# Capstone hosting

The shared source repository is `Basketball-Capstone-25/NBAPlayRanker`, branch
`main`. Deploy from the current shared checkout, after its tests pass.

## Backend

- Google Cloud project: `nba-playranker`.
- Cloud Run service: `nba-playranker-api`, region `northamerica-northeast1`.
- API: `https://nba-playranker-api-937082897804.northamerica-northeast1.run.app`.
- Source build directory: `backend`, which contains the Dockerfile.
- `backend/.gcloudignore` excludes credentials and local dependencies while
  preserving the CSV and Parquet files required at runtime. Keep this separate
  from the root `.gitignore`, whose dataset exclusions must not remove deployment
  data. Preview uploads with `gcloud meta list-files-for-upload` from `backend`.
- Set the production Supabase URL and publishable key in Cloud Run's environment;
  local `.env` files are not uploaded. Never use a service-role key in the frontend.
- Set `FRONTEND_ORIGIN=https://nbaplayranker-seven.vercel.app`. CORS accepts only
  explicitly configured HTTP(S) origins; comma-separated values are supported.
  Without this setting, only the local Next.js origins on port 3000 are allowed.

Initial runtime recommendation: 1 vCPU, 2 GiB RAM, request-based CPU allocation,
minimum zero instances, maximum two instances, and concurrency two. Keep both
service and revision maximums at two. Set `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`LOKY_MAX_CPU_COUNT` to `1`, and `MPLBACKEND=Agg`. Some analysis endpoints train
models and some visualization endpoints copy datasets, so higher concurrency
requires measurement under those workloads.

## Frontend and authentication

Vercel project `nba_play_ranker` belongs to `pearson-manufacturing`; its production
URL is `https://nbaplayranker-seven.vercel.app`. The repository specifies Node 22
to match CI. Production builds require these environment variables:

- `NEXT_PUBLIC_API_BASE`: the Cloud Run API URL above.
- `NEXT_PUBLIC_SUPABASE_URL`: the capstone Supabase project URL.
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`: its publishable client key.
- `NEXT_PUBLIC_SITE_URL`: the production frontend URL above.

Supabase Authentication URL Configuration now uses the production Site URL and
allows the application's exact production callback URLs for
`/auth/callback?next=/login` and `/auth/callback?next=/reset-password`. The existing
local-development callback is preserved. Password sign-in alone does not verify
email confirmation or password reset callbacks.

Vercel's previous personal-repository Git connection has been removed. Connecting
the shared repository through Vercel's GitHub integration was unavailable, so
deploy the authorized shared `main` checkout through the CLI. GitHub commit/push
access works independently. Automatic Vercel deployments require authorizing
the Vercel GitHub integration for the shared repository.

## Spending controls

On October 2, 2026, the authorized allowance was CAD$10 per month. Billing was
restored and the following controls were configured:

- CAD$10 monthly project budget across services, with 50%, 80%, and 100% alerts
  (`b0939e04-1572-47ef-b27b-93e2a976619b`).
- CAD$7 monthly Cloud Run spend cap
  (`7102c048-fd97-4d5a-8123-2e151d585bcf`).
- Cloud Run service minimum zero and maximum two instances.

Budget alerts do not stop charges. Cloud Run's spend cap uses gross cost before
credits, enforcement can be delayed, and in-flight work can finish. It does not
cover builds, artifact storage, or other services. These controls reduce spending
risk but do not guarantee an absolute CAD$10 account-wide ceiling. Review actual
billing after testing and stop hosting when the semester ends.

## Deployment checks

After deployment, verify API health, rejection of unauthenticated API requests,
CORS from the production origin, and actual coach/analyst sign-in. Exercise
recommendations, visualizations, and exports using the hosted frontend. Check
Cloud Run logs for errors and memory pressure before increasing concurrency.
