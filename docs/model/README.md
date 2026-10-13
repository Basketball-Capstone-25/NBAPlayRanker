# Iteration 14 Visual Paradigm model

This folder holds the model completed for SCRUM-475, SCRUM-476, SCRUM-481 and
SCRUM-506. It was published as **Teamwork revision 51 on October 3, 2026**.
For the later sequence-diagram corrections, see the
[Assignment 1 revision](../assignment-1-revision/interaction/README.md).

Open [Basketball Playtype Ranker - Elaboration Revision 51.vpp](<Basketball Playtype Ranker - Elaboration Revision 51.vpp>)
to review this saved version. Continue new work in the shared **Basketball
Playtype Ranker** project, rather than replacing it with this older copy.

## What changed

| Task | Model changes |
| --- | --- |
| SCRUM-475 | Added the analyst Top-K uplift API, calculation contract and request sequence. The sequence shows the protected profile-role check and matching JSON/CSV results. |
| SCRUM-476 | Added calibration calculation and API classes, the analyst panel and its request sequence. The model describes training on earlier seasons and checking a later season. |
| SCRUM-481 | Added coach-owned Gameplan storage, state validation, ownership rules and revision checks. Connecting the browser to that storage remains SCRUM-482. |
| SCRUM-506 | Updated the existing deployment diagram for Vercel, Cloud Run, Supabase authentication and the Gameplan table. Original nodes and connections were retained. |

Four editable diagrams were added, bringing the model from 182 to **186 diagrams**.
The five images below include the updated deployment diagram.

| Diagram | Preview |
| --- | --- |
| Analyst Evidence Design | [View](renders/analyst-evidence-design.jpg) |
| Coach Gameplan Storage | [View](renders/coach-gameplan-storage.jpg) |
| Analyst Top-K Uplift sequence | [View](renders/topk-uplift-sequence.jpg) |
| PPP Calibration sequence | [View](renders/calibration-sequence.jpg) |
| Deployment Diagram | [View](renders/deployment.jpg) |

The VPP contains editable UML elements. The JPGs are review copies, with the
Visual Paradigm evaluation watermarks retained.

## Important limits

The uplift result compares modeled PPP with historical team-season PPP. It does
not show a measured future scoring improvement or complete teammate SCRUM-474.
The seasonal reference uses all available offensive play-type rows; changing K
does not change that reference.

Calibration uses earlier seasons for training, but its inputs still include
same-season rates. It checks historical results, not pre-game forecasting.
SCRUM-526 covers the separate work on lagged predictors and future-style testing.

Gameplan storage is private to its coach owner. The schema checks JSON shape and
size and records server revisions. The browser still used localStorage in this
release. The [storage contract](../gameplan-storage-contract.md) explains how
SCRUM-482 should load and save it.

The deployment diagram describes the released services and access rules.
Vercel deployment used the CLI; automatic shared-repository deployment was not
connected. Hosting alerts and caps do not guarantee a hard CAD$10 monthly total.

## Revision evidence

Revision 51 was verified in both the native project and its synchronized origin
copy, with no pending changes at that time. The original 182 diagram IDs and all
existing model identities, parents, creators and creation times were preserved.
The completed model contains 2,118 model rows and 1,682 diagram elements.

- [Teamwork revision evidence](teamwork-revision-evidence.json) records the published revision, author, comment and file hash.
- [Revision manifest](revision-manifest.json) lists changed and added model IDs.
- [Iteration verification](../iteration-14/verification.md) contains calculations, tests and deployment evidence.
- [XML builder](build_revision_xml.py) reproduces the additions from the preserved pre-import XML; test any import on a copy first.

The native file and embedded resources were checked for credentials before
publication. Workspace login files and private backups are not included here.
