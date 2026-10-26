# Access and private configuration

The October 25 check could read the shared GitHub repository, the correct Vercel project, the Google Cloud service and the Supabase project. Supabase was healthy. Successful access on this laptop does not prove that another team member can recover an account.

| Service | Project to recover | Working access checked | Team confirmation still needed |
| --- | --- | --- | --- |
| GitHub | Basketball-Capstone-25/NBAPlayRanker | Existing Gurk login and remote clone | Organization backup administrator and recovery method |
| Google Cloud | nba-playranker | Service and billing settings read with existing account | Backup project/billing administrator and recovery contact |
| Vercel | pearson-manufacturing/nba_play_ranker | Project, production deployment and environment names read | Backup team administrator and recovery contact |
| Supabase | qdodginqubodugfeyiwi | Project healthy; existing coach sign-in works | Backup organization administrator, private backup location and recovery contact |
| Visual Paradigm | Basketball Playtype Ranker, workspace project 4 | Published shared revision 60 and preserved teammate changes verified | Teammate account recovery method |

Keep recovery codes in each owner's password manager. Do not attach them to Jira, GitHub, VP or the assignment. Record confirmation that recovery works, not the codes themselves.

## Environment names

| Name | Used by | Meaning |
| --- | --- | --- |
| `NEXT_PUBLIC_SUPABASE_URL` | Browser and Next.js | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Browser and Next.js | Public project key; access still depends on the session and RLS |
| `NEXT_PUBLIC_API_BASE` | Frontend | Backend origin; local fallback uses the local API |
| `NEXT_PUBLIC_SITE_URL` | Frontend deployment | Stable public site URL |
| `SUPABASE_URL` | Backend | Project URL for JWT public keys and profile lookup |
| `SUPABASE_PUBLISHABLE_KEY` | Backend | Public key used with the caller's session to read the protected profile |
| `SUPABASE_JWT_SECRET` | Legacy backend configuration only | Only needed for legacy HS256; this project's ES256 verification uses public JWKS |
| `FRONTEND_ORIGIN` | Backend | Explicit allowed browser origins |
| `ALLOW_INSECURE_DEV_AUTH` | Backend | Must remain `false` for normal tests and hosted use |
| `MPLBACKEND` | Backend | Noninteractive plotting backend |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `LOKY_MAX_CPU_COUNT` | Backend | Set each to numeric `1` to preserve the small Cloud Run limits |

No service-role key or private signing key belongs in a `NEXT_PUBLIC_` variable. The runtime checks the protected `profiles.role`; a role submitted during signup is not an authorization source. Keep `.env`, `.env.local`, `backend/.env`, CLI tokens and test-account credentials outside Git.

## Checks and limits

The [private-value scan](evidence/private-value-scan.json) checked 636 tracked paths and 38 built browser files, including comparison against three known private values. It found no matches or tracked credential filenames. This is a scoped scan, not proof that every possible secret is absent.

The fresh environment passed 46 authentication/profile/baseline tests. Signed-out access returned 401, and an existing coach account could load five local rankings. The [final deployed checks](../verification/live-api-final/results.json) also passed sign-in, signed-out rejection, invalid-session rejection and role restrictions. [Seven local JWT checks](../verification/security/jwt-verification.json) cover signed expired tokens and invalid signatures separately.

The [dependency scan](evidence/dependency-scan-before.json) found six advisories in the starting commit, including a critical Next.js advisory. [The dependency update](https://github.com/Basketball-Capstone-25/NBAPlayRanker/commit/5fac9de) is part of this release. The updated source passed the [frontend tests](../verification/frontend-final/tests.txt) and [production build](../verification/frontend-final/build.txt); the original advisory report is retained.

Supabase still reports [leaked-password protection disabled](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection). This existing setting was not changed, and no paid upgrade was made. The [service check](evidence/service-access-check.json) records it.

The [curated evidence scan](../verification/security/iteration17-evidence-privacy.json) checks this iteration's saved files separately, including the native model's embedded-resource report. It does not scan every historical commit.

BSR-161 stays open until the team's recovery contacts are confirmed. No credentials were rotated and no access was granted or revoked during this check.
