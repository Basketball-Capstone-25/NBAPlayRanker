# Iteration 14 elaboration evidence

This package records Gurkaranjit Asahan's assigned scope in Iteration 14
(29 September–5 October 2026): SCRUM-475, SCRUM-476, SCRUM-481 and SCRUM-506.
It contains the implementation, model, deployment, verification and project
management evidence captured on **3 October 2026** for that individual scope.

The reviewed implementation is on the shared
[NBAPlayRanker repository](https://github.com/Basketball-Capstone-25/NBAPlayRanker).
The deployed backend snapshot is commit `e950869cb318c6b44220d3c06be4ba2aa16521da`.
The frontend correction is committed as `18baffc` and deployed; the user
confirmed that the Data Explorer CSV finishes downloading and opens as CSV.

The final shared-source archive also preserves Abdul's later SCRUM-502 NLP
commits, integrated from `6886865`. All **182 backend tests passed** on that
combined source. The dated production deployment above remains the release
verified for the assigned work; the later NLP source changes are separately
attributed and recorded in the [source integration evidence](evidence/source-integration/source-integration-evidence.json).

| Assigned item | Implemented result | Evidence and remaining boundary |
| --- | --- | --- |
| [SCRUM-475](https://basketball-strategy.atlassian.net/browse/SCRUM-475) | Analyst-only Top-K PPP uplift JSON and CSV endpoints with matching results, validation and provenance. | Arithmetic/API tests and authenticated production exports passed. The result is a descriptive historical comparison, not causal or held-out uplift. Teammate SCRUM-474 remains separately owned. |
| [SCRUM-476](https://basketball-strategy.atlassian.net/browse/SCRUM-476) | Predicted-versus-realized PPP calibration on later-season holdouts, including bins, signed bias, MAE/RMSE and analyst presentation. | Backend tests, live API and analyst browser display passed. Contemporaneous rate features make this a retrospective diagnostic; prospective forecasting is SCRUM-526. |
| [SCRUM-481](https://basketball-strategy.atlassian.net/browse/SCRUM-481) | Applied coach-owned Gameplan storage schema with constraints, RLS, revision checks and a client contract. | Isolated and live rollback SQL tests passed; real-session ownership checks passed. Browser migration from localStorage is the separate future SCRUM-482. |
| [SCRUM-506](https://basketball-strategy.atlassian.net/browse/SCRUM-506) | Canonical backend and frontend deployed; live role, CORS and application checks run. | Backend checks passed. SCRUM-527 corrected the discovered export failure; 34 frontend tests and 14 live cookie-authenticated export checks passed. Saved analyst CSV/PDF files, coach baseline CSV and coach play-diagram PDF were inspected with valid content. |

Read the [traceability matrix](traceability.md) for development/test task links
and the [verification report](verification.md) for execution results, formulas,
deployment identifiers, reproduction commands and limitations. Machine-readable
results are in [evidence/](evidence/manifest.json); the manifest includes SHA-256
hashes and source artifact names. No account passwords, session tokens, API keys
or administrator environment files are included.

The [final project/risk/test records](project-management/README.md) contain the
fresh 32-issue Jira snapshot and four CSV exports: 13 current work items Done,
nine test cases plus their suite Pass, seven monitored risks and one resolved
risk, and the forecasting follow-up scheduled in Iteration 15. Seven recorded
worklogs total 75 minutes; planning estimates remain separate.

The [native model package](../model/README.md) contains the revised editable
Visual Paradigm project and **teamwork revision 51**, committed under
`asahang@sheridancollege.ca`. The user completed the Desktop commit; the project,
pristine origin copy and synchronized desktop metadata confirm server revision
51 with no pending local history. This is the basis of verification; the online
history session had expired. See the [revision evidence](../model/teamwork-revision-evidence.json).

The school submission portal was excluded from integration by the user. This
package does not assert that the assignment has been uploaded there. Planning
estimates are not actual effort; this report does not invent student work hours
or backdate revisions.

The external submission manifest records the final source archive's commit and
file hashes. Deployed application revisions remain identified separately in
this report. The initial export failure is retained alongside its correction
and retest; it has not been rewritten as an initial pass.
