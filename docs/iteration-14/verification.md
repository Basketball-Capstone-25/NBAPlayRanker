# Verification and interpretation

All dates below are 3 October 2026. UTC is used unless a raw tool artifact
includes its local offset. This is a dated execution record, not a promise that
future deployments or third-party services remain unchanged.

## Recorded results

| Check | Result | Evidence |
| --- | --- | --- |
| Integrated backend suite | **176 passed**, 0 failed/skipped, 57.37 seconds; one existing Starlette/httpx deprecation warning. | [JUnit](evidence/backend-tests.xml), [output](evidence/backend-tests.txt). |
| Initial frontend suite | **12 passed** in three auth/middleware test files. This run predates SCRUM-527 and does not claim calibration component unit coverage. | [JUnit](evidence/frontend-tests-initial.xml), [output](evidence/frontend-tests-initial.txt). |
| Initial TypeScript and production build | Both exit 0; build generated 21 static pages. | [Commands, timestamps and exits](evidence/frontend-validation-initial.json), [build output](evidence/frontend-build-initial.txt). |
| Export correction frontend validation | **33 passed** in five files, TypeScript and production build passed; includes server export authorization and response behavior. | [Validation](evidence/export-proxy-validation.json), initial proxy correction commit `c3044da`. |
| Final coach PDF compatibility validation | **34 passed**, 0 failed; TypeScript and production build passed. Final allowlist preserves the existing coach play-diagram PDF path. | [Final validation](evidence/export-proxy-validation-final.json), [JUnit](evidence/frontend-tests-final.xml), [output](evidence/frontend-tests-final.txt), commit `18baffc`. |
| GitHub CI | Initial release and final application commit `18baffc` both succeeded. | [Initial run 37154802950](https://github.com/Basketball-Capstone-25/NBAPlayRanker/actions/runs/37154802950); [final application run 37157802665](https://github.com/Basketball-Capstone-25/NBAPlayRanker/actions/runs/37157802665), [captured result](evidence/final-application-ci.json). |
| Gameplan database verification | Isolated local PostgreSQL and live Supabase rollback tests **PASS**; disposable fixtures rolled back. | [Local output](evidence/gameplan-local-test.txt), [live summary](evidence/gameplan-live-summary.json). |
| Backend production smoke | **8/8 passed** at 21:29:53 UTC: health, anonymous rejection and exact-origin/foreign-origin CORS. | [Smoke JSON](evidence/backend-production-smoke.json). |
| Production authenticated API/database checks | **49/49 passed**, 21:29:04–21:29:16 UTC, using real password sessions and protected profile roles. Includes API export parity and database ownership/revision checks. | [Execution JSON](evidence/live-verification.json). |
| Initial production browser walkthrough | Analyst login, 11-row Data Explorer and calibration display passed. Coach login, baseline, context, Gameplan build and coach/analyst page separation passed. **Analyst direct CSV export failed**; coach CSV download-event confirmation was **unverified**. | [Initial observations](evidence/browser-verification-initial.json), preserved as originally observed. |
| Corrected production export proxy | **14/14 passed**, 22:17:35–22:17:54 UTC: real session-cookie data/uplift/shot/coach CSV responses, analyst and coach PDF signatures and completion markers, and anonymous/coach/foreign-origin rejection. | [Live proxy checks](evidence/export-proxy-live-verification.json). This final run measures raw response bytes; the separately retained intermediate record used decoded UTF-8 lengths. |
| Corrected browser downloads | User confirmed Data Explorer CSV finishes and opens as CSV. Actual analyst and coach CSV/PDF files saved by browser UI actions were inspected, including coach CSV with five data rows and a readable one-page coach play PDF. | [Final browser record](evidence/browser-verification-final.json), [downloaded-file evidence](evidence/downloaded-file-verification.json). |
| Native model revision | Teamwork **revision 51**, author `asahang@sheridancollege.ca`, 21:59:49.692 UTC; no pending local history. | [Model package](../model/README.md), [revision evidence](../model/teamwork-revision-evidence.json). |
| Live-verifier fixture safety | **7 offline mock tests passed** for refusal to overwrite existing state, uncertain inserts, malformed responses, unexpected authorization success and cleanup failure reporting. | [Output](evidence/verifier-safety-tests.txt). This maintenance change did not rerun or change the historical 49-check production record. |
| Final Jira reconciliation | **32 issues** captured at 22:28:16.320 UTC: 13 current plan records Done, nine test cases plus suite Pass, seven risks Monitoring and one Resolved; future SCRUM-526 scheduled in Iteration 15. | [Project, risk and test exports](project-management/README.md). Seven recorded worklogs total 4,500 seconds (75 minutes), separately from estimates. |

Live API/database results are not a substitute for browser download verification.
The initial browser failure is retained even though authorized API CSV requests
passed. The later fix streams allowed exports through same-origin `/api/exports`
using the verified Supabase session; the server forwards authorization without
putting tokens in URLs. Errors are not served as attachments. The new browser
record and deployment identify the successful analyst retest. A successful
HTTP response alone is not described as a visually confirmed local file save.
The [downloaded-file inspection](evidence/downloaded-file-verification.json)
also records actual analyst files: an 11-row Data Explorer CSV, a 5,000-row Shot
Explorer CSV and a readable one-page, 75,158-byte Shot Plan PDF with the selected
teams and season. The final coach browser exports produced a five-row baseline
CSV and a readable one-page, 42,424-byte PRRollMan play diagram with correct
teams, season and play type. Both PDFs were rendered and visually inspected;
their headings, matchup, court graphic and table/caption were readable without
clipped content. Hashes and file timestamps are included. These
checks cover the protected backend exports; unrelated historical client-generated
metric downloads were not included in this regression's verification scope.

## SCRUM-475: Top-K uplift contract

An analyst can request `/metrics/topk-uplift`, `/metrics/topk-uplift.json` or
`/metrics/topk-uplift.csv` with the same season, team, opponent, K and offense
weight. Each uses the current protected profile role. Missing sessions are
rejected with 401; a coach requesting an analyst route receives 403.

The team-season reference is `sum(PTS) / sum(POSS)` over **all available
offensive play-type rows** for the team and season. It does not change when K
changes. The Top-K modeled PPP is `sum(PPP_PRED * POSS_OFF) / sum(POSS_OFF)` over
the selected recommendations. Absolute uplift is modeled PPP minus reference
PPP; relative uplift is null when the reference is zero. The exports retain
filters, source SHA-256, formulas, numerators, denominators and contributions.

The [recorded TOR/BOS, 2024-25, K=3 result](evidence/topk-uplift-live.json) has
reference PPP **0.969238** (8,255 points / 8,517 possessions), modeled Top-K PPP
**1.071696**, difference **0.102458 PPP**, and relative difference **10.570965%**.
These are properties of the historical model comparison. They are **not an
observed scoring improvement, a causal effect, or held-out evaluation**. The
selected play types retain historical possession weights, not equal usage.
The source covers available Synergy play-type possessions, not independently
verified whole-game possessions. SCRUM-474 retains its separate scope.

## SCRUM-476: continuous PPP calibration

Each expanding-window fold fits the scaler and a fixed Ridge model (`alpha=0.1`)
on strictly preceding seasons, then predicts the next held-out season.
Possession reliability normalizations use training-fold maxima. Held-out PPP
does not fit the scaler, select model hyperparameters, fit a correction, or set
bin boundaries. Bins have equal width over the predicted PPP range. Rows have
equal weight; they are team/play-type/offensive-season observations, not
individual games or possessions.

The [live report](evidence/calibration-live.json) contains 1,650 held-out rows
across 2020-21 through 2024-25, from 1,980 input rows:

| Metric | Recorded value in PPP |
| --- | ---: |
| Mean predicted | 0.998542085 |
| Mean realized | 0.998576566 |
| Signed bias, predicted minus realized | -0.000034482 |
| MAE | 0.008489534 |
| RMSE | 0.011340119 |
| Count-weighted absolute bin bias | 0.001926235 |

Positive bias means overprediction; negative bias means underprediction. PPP is
a continuous outcome, so these are not classification probability calibration
scores. Small aggregate bias can conceal errors within bins. One bin has only
19 rows and is flagged sparse; empty bins return null means. The UI displays
fold provenance, exact-value tables and sparse/retrospective warnings.

**Temporal separation of training rows does not establish pre-game forecast
accuracy.** Predictors still include same-season shooting and scoring rates,
which are contemporaneous outcome proxies. This evaluates retrospective PPP
reconstruction/generalization to a later season with its explanatory statistics
available. It is not evidence of forecasts made before those statistics exist.
Strictly lagged features and an untouched prospective-style time holdout are
the subject of SCRUM-526. The reported model is fixed Ridge; the numbers do not
automatically describe every existing Context/ML or recommendation model.

## SCRUM-481: database scope

The applied migration `20261003211500_gameplan_state_schema.sql` creates private
coach Gameplan state keyed by `(user_id, season, our_team, opp_team)`. JSON
shape/size constraints, immutable owner/matchup identity, server audit fields
and revision increments are enforced in PostgreSQL. RLS uses the current
protected `profiles.role`, not client-controlled signup metadata.

The [storage contract](../gameplan-storage-contract.md) defines exact bounds,
INSERT for new state and conditional PATCH of state fields with the last-read
revision. A stale revision must update zero rows. Generic upsert is not the
documented client protocol. Local/live SQL verification covers CRUD, role and
owner isolation, malformed state, optimistic revisions, audit fields and user
deletion cascade. Real-session checks separately verified insert, update,
stale-update rejection, analyst isolation, owner forgery rejection and fixture
cleanup. The live database advisor snapshot has no performance findings and
one pre-existing disabled leaked-password protection warning.

The current Gameplan browser still uses localStorage. Browser creation of a
Gameplan does not prove cloud state persistence; that integration is SCRUM-482.

## SCRUM-506: released runtime and controls

| Component | Recorded release |
| --- | --- |
| Source | Backend `e950869cb318c6b44220d3c06be4ba2aa16521da`; frontend `18baffc0cc6f99273f60e797a499804029b8b84f`. Feature commits `5092bf1`, `b9f145d`, `e950869`; export corrections `1068ef2`, `c3044da` and `18baffc`, all on canonical shared `main`. |
| Cloud Run | `nba-playranker-api-00005-hrn`, 100% traffic in `northamerica-northeast1`; build `7d519406-32d9-4196-be2b-6d640016974b`; deploy completed 21:28:03 UTC. |
| API | <https://nba-playranker-api-937082897804.northamerica-northeast1.run.app> |
| Final application Vercel release | `dpl_7JaSVKvgfxo7yQpia6ZhUmFXLh75`, READY, frontend `18baffc`. Initial `dpl_3VGStQCMoBAbXg97FedCrZZtBwez` evidence remains for the original failure. |
| Frontend | <https://nbaplayranker-seven.vercel.app> |
| Supabase | Project `qdodginqubodugfeyiwi`; Gameplan migration version `20261003211500`. |

Cloud Run retained its authentication/CORS environment and cost configuration:
minimum 0, maximum 2 instances, 1 CPU, 2 GiB RAM, concurrency 2, request-based CPU
and numeric-library thread limits of 1. Environment values are intentionally
absent from the evidence. The [backend release record](evidence/backend-deployment.json)
contains the image digest, source command and preceding rollback revision.

The approved allowance is CAD$10/month. A CAD$10 monthly all-services project
budget and CAD$7 Cloud Run spending cap were configured. Budget alerts do not
stop spending. The Cloud Run cap uses gross costs and delayed billing data;
in-flight usage and other products such as builds/storage can exceed it.
Neither is an absolute guarantee of a CAD$10 total. Supabase remains Free and
Vercel Hobby; no paid tier upgrade is part of this release.

The old personal GitHub-to-Vercel linkage was removed. Shared-organization Git
integration was not authorized, so the recorded frontend deployment was manual
from the canonical checkout. Do not describe automatic Vercel deployments as
connected. The native model's revision 51 is verified through the post-commit
project, pristine origin and synchronized desktop metadata; no claim is made of
fresh online-history access because that session expired. Its exact comment is:
`Gurkaranjit Asahan: elaborate analyst evidence APIs, coach Gameplan schema and deployed architecture (SCRUM-475/476/481/506)`.
School submission remains outside the integrated workflow.

## Reproduction

Run tests in the canonical checkout with the existing installed dependencies.
The outputs below use a chosen local evidence directory; it must not contain
credentials before being committed or shared.

```sh
cd /Users/gurkasahan/Developer/NBAPlayRanker/backend
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg \
  ALLOW_INSECURE_DEV_AUTH=false \
  .venv/bin/python -m pytest tests --junitxml=/tmp/backend-tests.xml

cd /Users/gurkasahan/Developer/NBAPlayRanker
node_modules/.bin/vitest run --maxWorkers=1 --minWorkers=1
node_modules/.bin/tsc --noEmit --incremental false
npm run build
```

Database rollback-test commands and prerequisites are in the
[storage contract](../gameplan-storage-contract.md). The reusable production
verification script is `scripts/verify_elaboration_live.py`; inspect its
documented account and environment requirements before running. Real-session
credentials belong only in ignored local inputs, never in the evidence pack.
Re-running tests does not itself redeploy a change or create a VPository revision.
