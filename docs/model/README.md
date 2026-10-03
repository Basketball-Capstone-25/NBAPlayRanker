# Capstone native model revision evidence

Native Visual Paradigm revisions for SCRUM-475, SCRUM-476, SCRUM-481 and
SCRUM-506 are committed as **teamwork revision 51**, under
`asahang@sheridancollege.ca`, at **2026-10-03 21:59:49.692 UTC**. The revision adds
four editable UML diagrams and updates deployment, authorization and
implementation documentation. The user completed the normal Desktop commit
after the vendor's commit CLI failed; the committed project, pristine origin
copy and synchronized repository metadata all confirm revision 51 with no
pending local history.

Native deliverable: [Basketball Playtype Ranker - Elaboration Revision 51.vpp](<Basketball Playtype Ranker - Elaboration Revision 51.vpp>).
Evidence: [teamwork-revision-evidence.json](teamwork-revision-evidence.json).

## Source and preservation

Use the **Basketball Playtype Ranker** teamwork project in workspace `tnfaiqmm`,
online project 4. Its current local source is the `Basketball Playtype Ranker.vpp`
inside VisualParadigm's `wss/teamwork_client/projects/Basketball Playtype Ranker`
directory. Do not replace it with the older `CapstoneVPP.vpp` Desktop attachment.

The user-saved, fully closed desktop baseline had project format `20260622`,
build `20260914`, 182 diagrams, 2,070 stored models and 1,633 diagram elements.
The completed local revision has **186 diagrams, 2,118 stored models and 1,682
diagram elements**. Every original ID survived. Each existing model's type,
parent, creator and creation timestamp was preserved. All 125 new object IDs,
including composite operations, attributes and message ends, survived official
XML re-export. Five vendor-rendered images were visually reviewed.

An earlier transient 183-diagram snapshot included **Elaboration Deployment
Model Revision** (`zV_L9wWFYDwCTaqs`). That object was absent after the user saved
and closed Desktop, before these final edits. Its snapshot remains separately
preserved at `/Users/gurkasahan/.cache/capstone-vp-import-l88kctbi/source-backup.vpp`;
it was not substituted for the user's saved model. The complete pre-import
teamwork directory is backed up under
`/Users/gurkasahan/.cache/capstone-vp-stable-zd2ua40h/pre-import-teamwork`.

The live source was modified only through the vendor's `ImportXML` command
after isolated preservation tests passed. No direct SQLite or history-table
writes were used. Prepared XML and verification reports are retained under
`/Users/gurkasahan/.cache/capstone-vp-stable-zd2ua40h`. Local source SHA-256 after
import is `c61a22736b39ee6b154ed05ec27fd42affed313a65328ac40a1e19d1feee818d`.

## Native diagrams and rendered evidence

| Diagram | Native ID | Render |
| --- | --- | --- |
| Elaboration - Analyst Evidence Design (475, 476) | `Elab56198592e4c5` | [Design](renders/analyst-evidence-design.jpg) |
| Elaboration - Coach Gameplan Storage (481) | `Elab8132fcbcc10d` | [Storage](renders/coach-gameplan-storage.jpg) |
| Detailed Sequence - Analyst Top-K Uplift (475) | `Elabcccf12e8a8ce` | [Uplift sequence](renders/topk-uplift-sequence.jpg) |
| Detailed Sequence - Inspect PPP Calibration (476) | `Elab86fdd4b13a52` | [Calibration sequence](renders/calibration-sequence.jpg) |
| Deployment Diagram, updated existing diagram | `8W4D.wWAUAAADc8a` | [Deployment](renders/deployment.jpg) |

Vendor evaluation watermarks are preserved. The diagrams contain native
editable classes, operations, attributes, dependencies, lifelines and messages;
they are not embedded pictures. [revision-manifest.json](revision-manifest.json)
maps created IDs and intended diagram changes to their source export.

The native project, official XML export and all 504 decompressed embedded
resources were checked for private keys, literal access tokens, service-role
secrets, nonempty credential assignments and current configured credential
values. No matches were found. Ordinary password-form labels and
`session.access_token` message names remain part of the authorized model. No
workspace authentication configuration, login logs, caches or backups are
included in this deliverable folder.

## Design changes

| Jira task | Existing diagram or element | Exact change |
| --- | --- | --- |
| SCRUM-475 | **API Coordination Overview** `UK3N.wWAUAAADbL0`; new Analyst Evidence Design | Added native `TopKUpliftEndpoints` and its router factory, documented JSON/CSV representations and the `analytics` guard. The new sequence shows protected-role authorization. |
| SCRUM-475 | **Statistical Analysis Overview** `VR3N.wWAUAAADbRG`; new Analyst Evidence Design | Added `TopKUplift.compute_topk_uplift` with the exact input contract in operation documentation. Uses baseline rankings and all team-season baseline rows. |
| SCRUM-475 | **Compute & Export Top-K Uplift (CSV/JSON)** `Ub9lX1mGAqAALhoz` | Updated UC29 with analyst preconditions, filters, formulas, export provenance and alternate failures. |
| SCRUM-476 | **API Coordination Overview** `UK3N.wWAUAAADbL0`; **Statistical Analysis Overview** `VR3N.wWAUAAADbRG`; new Analyst Evidence Design | Added `CalibrationEndpoints.create_calibration_router`, `Calibration.compute_calibration` and `summarize_calibration`; documented fixed Ridge, training-only scaling and later-season test folds. |
| SCRUM-476 | **Analytics UI Overview** `u03N.wWAUAAADbJ2`; existing `ModelMetricsPageClient` `QBAo4bmAUAAADR5m` | Added native `CalibrationPanel` and the new design's renders/Bearer-HTTP dependencies. Updated page/diagram documentation with chart, retry and service/infrastructure contracts. |
| SCRUM-481 | **Data Access Overview** `2l3N.wWAUAAADbTK`; new Coach Gameplan Storage diagram | Added `GameplanState`, `AuthUser`, `ProtectedProfile` and `GameplanStateGuards`, ten storage fields, ownership and validation dependencies. Browser persistence remains future SCRUM-482. |
| SCRUM-506 | **Deployment Diagram** `8W4D.wWAUAAADc8a` | Updated Node.js/runtime labels and deployment/authentication notes. Added Gameplan Notes Table and an RLS/validation dependency on PostgreSQL. Preserved all original nodes and connections. |

The existing **Time-Split Uplift Evaluation** requirement
`hIxUa1mGAqAALhen` is broader than SCRUM-475's endpoint. Do not mark that requirement
fully satisfied by an in-sample modeled uplift diagnostic. SCRUM-474 remains a
separate teammate-owned calculation/evaluation task.

## Uplift interaction

Added **Detailed Sequence - Analyst Top-K Uplift (475)** under the Interaction
Model, with existing classifier references and updated UC29 documentation.

1. The analyst requests `/metrics/topk-uplift`, `/metrics/topk-uplift.json`, or
   `/metrics/topk-uplift.csv`, supplying season, our team, opponent, K and offense
   weight with their Bearer token.
2. `AuthDependency` validates the JWT and reads the current protected
   `profiles.role`. The resource must be `analytics` and the role `analyst`.
3. The endpoint calls `rank_playtypes_baseline(...)` directly. This is an internal
   domain call, not an analyst request to the coach-only baseline HTTP route.
4. `TopKUplift` computes team seasonal PPP from **all available offensive
   play-type rows**: `sum(PTS) / sum(POSS)`. It retains this denominator when K
   changes.
5. The selected Top-K modeled PPP is
   `sum(PPP_PRED * POSS_OFF) / sum(POSS_OFF)` over selected rows. This retains
   historical usage weights within the selected set. Absolute uplift is the
   difference; relative uplift divides that difference by seasonal PPP.
6. Return the same evidence in the requested representation, including the
   source-file SHA-256, scope, numerators, denominators, actual K, formulas and
   per-play-type contributions.

The interaction documentation records these alternate outcomes:
missing/invalid token → 401; coach or pending role →
403; invalid filter → 400/422; empty or unusable data → 404. Seasonal PPP of zero
produces a null relative uplift with an explanation, while absolute uplift
remains defined. Missing points must fail explicitly rather than substituting
rounded PPP. Its scope note identifies this as an in-sample historical modeled comparison,
not observed future scoring improvement or a causal effect.

Implementation: `backend/application/api_coordination/topk_uplift_endpoints.py`
and `backend/domain/statistical_analysis/topk_uplift.py`.
Verification: `backend/tests/test_topk_uplift.py`.

## Calibration interaction

Added **Detailed Sequence - Inspect PPP Calibration (476)** under the Interaction
Model. `ModelMetricsPageClient` supplies the fold count to
`CalibrationPanel`; the panel obtains the session token and calls
`/metrics/calibration?n_splits=...&n_bins=10` through the service/infrastructure
functions. The backend performs the same analyst resource check.

The endpoint caches reports by fold/bin parameters. On a cache miss, the domain
function prepares offense rows, constructs expanding-window season splits,
fits the scaler and fixed Ridge model only on preceding training seasons,
predicts the later held-out season, and accumulates PPP residuals. Reliability
normalization uses training-fold maxima. The summary groups continuous PPP
predictions into equal-width bins and returns means, bias, MAE/RMSE, counts and
fold provenance. Empty bins have null means; sparse bins are flagged.

The UI displays a loading state, a chart and exact-value table, or an error with
a retry action. Aborted requests do not overwrite the current view. The scope
note explains that season-level same-season explanatory features make this retrospective
evaluation, not proof of pre-game forecasting performance. The report does not
fit a calibration correction to held-out outcomes.

Implementation: `backend/domain/statistical_analysis/calibration.py`,
`backend/application/api_coordination/calibration_endpoints.py`,
`app/ui/analytics/_components/CalibrationPanel.tsx`, and
`app/infrastructure/calibration.ts`.
Verification: the 16 tests in `backend/tests/test_calibration.py` and a successful
signed-in analyst browser render of the calibration chart and exact-value table.
The frontend unit suite does not specifically cover the calibration panel.

## Gameplan storage design

The native `GameplanState` class documents `public.gameplan_notes` attributes:

| Attribute | Constraint or meaning |
| --- | --- |
| `user_id: uuid` | Owner; defaults to `auth.uid()`; FK to `auth.users.id` with cascade deletion. |
| `season, our_team, opp_team: text` | Together with `user_id`, form the composite primary key. Season format and team abbreviations are checked; our team must differ from the opponent. The matchup is directed. |
| `notes: jsonb` | Object mapping play identifiers to text; at most 100 entries, keys 1–160 characters, values at most 4,000 characters. |
| `plan: jsonb` | Ordered array of at most 50 unique `{id, label}` objects; IDs 1–160 and labels 1–240 characters. |
| `roles: jsonb` | Exactly `ballHandler`, `screener`, `cornerSpacer`, `cutter`, `safety`, each a string up to 120 characters. These are basketball assignments, not authorization roles. |
| `revision: bigint` | Starts at 1; the server increments it on update. Future clients should compare the last-read revision to detect concurrent edits. |
| `created_at, updated_at: timestamptz` | Server-maintained audit fields. |

The combined JSON payload is limited to 65,536 bytes. The model represents
`auth.users` as the owner entity, `public.profiles` as the protected authorization record, and
`gameplan_notes` as private coach state. RLS requires both `user_id = auth.uid()`
and current `profiles.role = coach`. Anonymous, analyst and pending-role users
cannot access these rows. Clients cannot modify ownership, identity or server
audit columns. `GameplanStateGuards` documents `private.valid_gameplan_state` as
a pure validation function and `private.touch_gameplan_state` as the update trigger.

The browser currently stores Gameplan state locally. The model documentation
labels cloud persistence as **future integration** under SCRUM-482.
The schema alone does not satisfy that separate UI integration task.

Implementation: `supabase/migrations/20261003211500_gameplan_state_schema.sql`.
Verification: `supabase/tests/gameplan_state_verification.sql`.

## Authentication and deployment corrections

The existing **Behavioural Overview — Authenticate Analyst Access**
`WuPN.wWAUAAADbuO` had obsolete JWT-role labels. Message `9JPN.wWAUAAADbxP` now
names a verified JWKS signing key, and `YZPN.wWAUAAADbx2` names verified subject
claims with the role resolved from a protected profile. Its documentation points
to the new uplift sequence, which explicitly includes a Supabase Profiles
participant and a role response from that query. The new sequence documents
JWKS, issuer/audience/expiry validation and 401/403 alternatives. The optional
legacy HS256 secret is not the production role source. This revision does not
claim to rebuild every legacy authentication or alternate-flow diagram.

For SCRUM-506, the existing deployment model documents:

- Browser → Vercel: HTTPS pages and Next.js client assets. Production frontend is
  `https://nbaplayranker-seven.vercel.app`; Node.js 22 serves the application.
- Browser → Cloud Run: HTTPS REST/JSON with Bearer JWT and explicit frontend
  CORS. `nba-playranker-api` runs in `northamerica-northeast1`, inside the existing
  Python 3.11/Uvicorn container. Entrypoint is
  `application.api_coordination.app:app`.
- Cloud Run: one CPU, 2 GiB, minimum zero, maximum two instances and concurrency
  two. Numeric-library threads are limited to one. Cached reports are ephemeral
  instance memory; the source datasets are packaged with the application.
- Browser/server → Supabase Auth: authentication and exact production callbacks.
  Cloud Run → Supabase: JWKS verification plus authenticated protected profile
  reads. Do not put credentials or token values in the diagram.
- PostgreSQL: protected `profiles`, coach-owned `gameplan_notes`, RLS policies and
  validation/update triggers. The new state table is a storage contract; the
  separate future browser integration remains explicitly labeled.
- Deployment currently uses the canonical shared Git checkout and manual Vercel
  CLI deployment. Do not depict automatic Vercel Git deployment as working.

Reverify live revision identifiers when attaching final deployment evidence;
the diagram should record stable architecture, with dated deployment evidence
linked separately. Budget alerts and the Cloud Run spending cap do not provide
an absolute guarantee that total monthly charges cannot exceed CAD$10.

## Genuine teamwork revision

Before editing, an authenticated update confirmed local and server revision 50
with no available update. After the verified import, the official commit CLI
authenticated but raised a vendor `NullPointerException`. No revision was
created by that failed command. The user then performed the normal Desktop
**Team → Commit** action successfully.

Read-only verification after that commit found:

- Current VPP and pristine `.vpp.orig`: teamwork revision **51**, 186 diagrams,
  2,118 stored model rows, 1,682 diagram elements, and zero pending `TW_HISTORY`
  rows.
- Desktop repository metadata: local revision **51** and last server revision
  **51**.
- Recorded check-in account: `asahang@sheridancollege.ca` (workspace display name
  Gurk).
- Recorded commit time: **2026-10-03 21:59:49.692 UTC**.
- Exact recorded comment:
  `Gurkaranjit Asahan: elaborate analyst evidence APIs, coach Gameplan schema and deployed architecture (SCRUM-475/476/481/506)`.

The browser history session had expired, so this evidence is the user's commit
confirmation plus synchronized local repository/project records, not a new
browser-history screenshot. The native deliverable was copied only after a
stable-hash check. Its SHA-256 is
`53603cd3d1862aec40b3dbf18958c1c2c76a1e457020be435c7e4256487c5714`.

## Supported command path

The macOS `Info.plist` reported 17.3 while updater metadata reported 18.1. To
avoid assuming a version-specific classpath, working commands use the bundled
Java executable at
`/Applications/Visual Paradigm.app/Contents/Resources/jre.bundle/Contents/Home/bin/java`
and the exact `JavaVM.ClassPath` from the app's `Contents/Info.plist`, expanding
`$APP_PACKAGE` to the application path. Working directory is
`Contents/Resources/app/bin`. Official entry points are
`com.vp.cmd.ExportXML`, `ImportXML`, `ExportDiagramImage`,
`UpdateTeamworkProject` and `CommitTeamworkProject`.

XML import/export and image export worked in headless mode. Teamwork needs
normal Java graphics initialization and a fully quit Desktop application;
forcing headless mode caused initialization exceptions. No lock was removed
and the user's application was not force-killed. The normal update command
works; the user completed the commit through Desktop after the CLI exception.

[build_revision_xml.py](build_revision_xml.py) regenerates the additive import
from the preserved pre-import official XML export. It rejects exports already
containing this revision. It does not modify VPP files or commit projects.
Always apply imports to a backup first and repeat preservation checks.

Official workflow references:
[XML export and import](https://www.visual-paradigm.com/support/documents/vpuserguide/124/255/7349_exportingand.html),
[teamwork update](https://www.visual-paradigm.com/support/documents/vpuserguide/124/255/7357_updatingteam.html),
and [teamwork commit](https://www.visual-paradigm.com/support/documents/vpuserguide/124/255/84295_committingpr.html).
