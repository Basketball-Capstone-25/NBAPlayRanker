# Iteration 14 elaboration work

This folder records the October 3, 2026 results for Gurkaranjit Asahan's four
assigned tasks: SCRUM-475, SCRUM-476, SCRUM-481 and SCRUM-506.
For the later feedback fixes and newly found failures, read the
[Assignment 1 revision](../assignment-1-revision/READ-ME.txt).

| Task | Finished work | Remaining work |
| --- | --- | --- |
| [SCRUM-475](https://basketball-strategy.atlassian.net/browse/SCRUM-475) | Analyst Top-K PPP uplift API with matching JSON/CSV exports, input checks and role protection. | The result is a historical comparison. Teammate SCRUM-474 has its own calculation/evaluation scope. |
| [SCRUM-476](https://basketball-strategy.atlassian.net/browse/SCRUM-476) | Predicted-versus-realized PPP report with error measures, charts and a value table for analysts. | Same-season inputs limit this to historical analysis. SCRUM-526 covers future-game evaluation. |
| [SCRUM-481](https://basketball-strategy.atlassian.net/browse/SCRUM-481) | Coach-owned Gameplan database schema with data limits, access controls and revision checks. | SCRUM-482 must connect the browser, which still uses localStorage. |
| [SCRUM-506](https://basketball-strategy.atlassian.net/browse/SCRUM-506) | Deployed the backend/frontend and checked signed-in flows. Fixed the export download problem in SCRUM-527 and retested saved CSV/PDF files. | Later application issues are listed in the Assignment 1 revision. |

## Where to look

- [Task and test links](traceability.md) connect each feature to its development tasks, tests and source files.
- [Verification report](verification.md) explains the calculations, test results, deployments and limitations.
- [Jira records](project-management/README.md) contain the dated project, risk and test exports.
- [Visual Paradigm model](../model/README.md) contains the editable model published as Teamwork revision 51.
- [Evidence manifest](evidence/manifest.json) lists the supporting files and hashes.

The shared source is in the
[NBAPlayRanker repository](https://github.com/Basketball-Capstone-25/NBAPlayRanker).
The release recorded here used backend `e950869` and frontend `18baffc`.
Abdul's later SCRUM-502 NLP changes were included in the shared-source archive;
all 182 backend tests passed on that combined source. Those later changes were
not part of the recorded deployment. See the
[source integration record](evidence/source-integration/source-integration-evidence.json).

The October 3 Jira snapshot has 13 assigned work items Done, nine test cases and
their suite Pass, seven risks Monitoring and one Resolved. Seven worklogs total
75 minutes. These are historical records; later Jira status changes and test
failures are recorded separately.

The evidence keeps the initial export failure, its fix and the successful
retest. This folder does not record a school-portal submission.
