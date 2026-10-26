# Local setup and recovery

Use the shared repository. Keep the checkout outside iCloud. The October 25 clean setup at historical commit `45973ed` used Node 22.16.0, npm 10.9.2 and Python 3.13.7. It occupied about 1.44 GB after installation and build; allow at least 5 GB for working space.

## Restore a local demo

1. Clone into a new folder and record the commit with `git rev-parse HEAD`.

   ```bash
   git clone git@github.com:Basketball-Capstone-25/NBAPlayRanker.git
   cd NBAPlayRanker
   npm ci
   python3 -m venv backend/.venv
   backend/.venv/bin/python -m pip install --prefer-binary -r backend/requirements-dev.txt
   backend/.venv/bin/python -m spacy download en_core_web_sm
   ```

2. Create ignored `.env.local` for Next.js with `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` and `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8017`. Create ignored `backend/.env` with `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `ALLOW_INSECURE_DEV_AUTH=false` and `FRONTEND_ORIGIN=http://127.0.0.1:3017`. Obtain values from the project settings; do not copy them into Jira or a handoff ZIP. Set each numeric thread limit listed in [access.md](access.md) to `1`.

3. The full Git clone contains the shot runtime file. The repaired October 25 file has 1,370,434 rows and SHA-256 `d2f647a048959ce14b798a3fe3f0b47e9eba860d7ecb35bec7b67d9968d657a4`. Its source and rebuild checks are in [shot repair evidence](../shot-repair/READ-ME.txt).

   A reduced handoff ZIP may omit the data. Use that ZIP's restore script and verify its stated commit and SHA-256. The historical file at `45973ed` is 11,634,107 bytes with SHA-256 `36f128eec443543c7b59dee537fcfc7162ae66b336eece5abf8d25c77cec0fce`. That older file was restored during the clean-setup exercise, but it retains the attribution defects. Do not use it to replace the repaired release.

4. Only if restoring that older snapshot's missing cache, run `backend/.venv/bin/python backend/data/etl/build_pbp_pipeline.py` without `--force`. This rebuilds from the retained clean file. It does not repair its contents. `--force` requires an authoritative raw source. Do not replace a repaired later release's data with this older snapshot.

5. Build and start the frontend:

   ```bash
   npm run build
   ./node_modules/.bin/next start -H 127.0.0.1 -p 3017
   ```

   In a second terminal, start the backend:

   ```bash
   cd backend
   .venv/bin/python -m uvicorn application.api_coordination.app:app --host 127.0.0.1 --port 8017
   ```

6. Open `http://127.0.0.1:3017`, sign in with an existing coach account and load TOR versus BOS, 2024–25. The local API should return five baseline recommendations. `/health` should return 200; the same ranking request without a session should return 401. This fallback uses the laptop for the API but still needs internet access and the shared Supabase service for sign-in.

The [clean-checkout result](evidence/clean-checkout-check.json) records the commands. [Local checks](evidence/local-recovery-checks.json) passed six checks, including real coach sign-in and five returned rankings. The temporary session was signed out. This was an HTTP/API check, not a complete browser walkthrough.

## Recover a database safely

The SQL migrations in `supabase/migrations` recreate the protected profile-role rules and coach note schema. They are not a backup of saved accounts or notes.

For the Free project, retain a private, encrypted export outside Git and the submission ZIP. Supabase recommends regular CLI exports for Free projects; downloadable daily backups are a paid-plan feature. Storage objects need a separate copy if storage is introduced. See [Supabase backups](https://supabase.com/docs/guides/platform/backups).

Use the current [Supabase backup/restore guide](https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore) and `supabase db dump --help` to create role, schema and data exports. A database administrator must use the private connection details. Do not reset a password just to obtain them, put a database URL in a report, or restore into production for a test.

Before a restore, confirm the backup timestamp, checksum, PostgreSQL compatibility and target identity. Restore into an empty, isolated database. Verify that the migrations exist, `profiles` and `gameplan_notes` have RLS, users cannot change their role, and one coach cannot read or overwrite another coach's notes. Check row counts and representative saved notes against the private backup. Only then plan a production recovery and its rollback.

The isolated restore drill completed on October 25. Role, schema, data and migration-history exports were restored into a disposable local PostgreSQL 17.11 Supabase database. Network listening was disabled; verification used its local Unix socket. All 36 COPY tables matched the backup values exactly, including accounts, saved notes and migration records.

The first Gameplan test caught extra permissions inherited from the new target's defaults. The schema export added the source grants without removing those defaults. Reapplying the permission block from `20261003211500_gameplan_state_schema.sql` fixed the isolated target. Both `profile_roles_verification.sql` and `gameplan_state_verification.sql` then passed, covering protected roles, owner-only notes, JSON constraints, revisions and anonymous access. Their fixture transactions rolled back and no test users remained. On a future restore, restore the custom Auth triggers from the existing migrations and explicitly reapply the app's restricted grants before running these tests.

The five SQL exports and private restore records were bundled with a file-hash manifest and encrypted with Fernet authenticated encryption. The key is stored separately in a private configuration directory. Decryption and every archived file hash were checked before the 20 plaintext scratch files were removed. Backup and key files have mode `0600`; their private directories have mode `0700`. Neither the archive nor its key belongs in Git or a submission ZIP.

The disposable database container and its volume were removed through the project-scoped Supabase stop command. The other five running containers were preserved. See [database restore results](evidence/database-restore-check.json). This was an isolated recovery drill; production data and deployment traffic were unchanged.

## Recover a deployment

First record the failing release, error and last working release. Inspect the current project and revision before changing traffic. Keep database migrations compatible with both versions; a code rollback does not reverse a database migration.

For Vercel, use the stable `nbaplayranker-seven.vercel.app` alias. Confirm `vercel project inspect --non-interactive` reports `pearson-manufacturing/nba_play_ranker`, then inspect the intended known-good deployment. The supported rollback command is `vercel rollback <verified-deployment-id>`. Do not use an old preview URL as the demo address.

For Cloud Run, the supported command is:

```bash
gcloud run services update-traffic nba-playranker-api \
  --project=nba-playranker --region=northamerica-northeast1 \
  --to-revisions=KNOWN_GOOD_REVISION=100
```

These are recovery instructions, not commands executed in this check. After any rollback, test sign-in, Data Explorer, a coach ranking and an export. If hosting is unavailable, use the verified local demo while investigating; do not disable authentication.
