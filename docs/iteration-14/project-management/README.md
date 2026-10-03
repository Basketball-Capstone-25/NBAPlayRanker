# Final Jira project, risk and test records

This snapshot was captured from the live Basketball Strategy Jira site at
**2026-10-03 22:28:16.320 UTC**, after the final calibration-test clarification
and future-iteration allocation. It covers the four assigned Iteration 14
stories and their development, verification and risk records.

| Record | Final state | Export |
| --- | --- | --- |
| Current project plan | Four parent stories and nine development/verification subtasks: **13 Done**. Parent story-point estimates total 13. | [Current iteration plan](Jira-current-iteration-plan.csv) |
| Future work | SCRUM-526: **To Do**, Iteration 15 (ID 183), assigned to Gurkaranjit Asahan, 3 story points. | [Future iteration work](Jira-future-iteration-work.csv) |
| Test plan | BST-120 suite plus BST-121–129: **10 Pass records**, comprising nine test cases and their suite. | [Iteration tests](Jira-iteration-tests.csv) |
| Risk plan | BSR-43, 45, 96, 98, 108, 110 and 112: **Monitoring**; BSR-111: **Resolved**. Current monitoring comments are retained. | [Risk monitoring](Jira-risk-monitoring.csv) |

The [complete 32-issue snapshot](jira-final-snapshot.json) retains descriptions,
comments, worklogs and custom fields. All four 22-column CSV exports were
round-trip checked against that snapshot without truncating those fields.
[Export verification](jira-export-verification.json) records row counts, hashes
and the final validation. [Issue grouping and workflow metadata](jira-export-manifest.json)
preserve the `future_work` group and exact issue keys; the local
[copy manifest](manifest.json) verifies the files included here.

Seven recorded worklog entries total **4,500 seconds (75 minutes)**. These are
recorded active-work intervals, not story points or planning estimates. The
individual entries, authors, timestamps and comments remain in the snapshot
and exports; no additional duration is inferred from calendar gaps.

BST-123 describes the actual regression scope: training-only scaling and
reliability normalization, no later test-season PPP affecting those fitted
transforms or the separately tested earlier fold, and the limit imposed by
same-season explanatory features. It does not claim proof of future forecasting
accuracy. [SCRUM-526](https://basketball-strategy.atlassian.net/browse/SCRUM-526)
is scheduled in future Iteration 15 for strictly lagged predictors and untouched
time holdouts. SCRUM-482 remains the separate future Gameplan frontend migration;
teammate-owned SCRUM-474 remains outside this completed assignment scope.

The initial export failure and subsequent correction are preserved in Jira
comments and the [technical verification history](../verification.md). The final
Pass statuses describe the completed retest, not a rewritten initial result.

Live references: [project plan](https://basketball-strategy.atlassian.net/jira/software/projects/SCRUM/list/VWtX8Ho),
[risk board](https://basketball-strategy.atlassian.net/jira/software/projects/BSR/boards/3),
[test suite BST-120](https://basketball-strategy.atlassian.net/browse/BST-120).
The files are a dated export; later Jira changes will not modify this package.
