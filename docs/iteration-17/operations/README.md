# Operations work

These notes cover the risk follow-ups moved from SCRUM-532–537. Each result links to its evidence. Source tickets remain part of the history; the destination cards track the remaining work.

| Work | Current result | Open item |
| --- | --- | --- |
| [BST-188](https://basketball-strategy.atlassian.net/browse/BST-188), from SCRUM-532 | A fresh clone returned five coach recommendations. The isolated database restore matched all 36 backed-up tables and passed both access-control suites; the backup is encrypted and the disposable target was removed. | None for the recovery drill. See [restore evidence](evidence/database-restore-check.json). |
| [BSR-161](https://basketball-strategy.atlassian.net/browse/BSR-161), from SCRUM-533 | Access, configuration and private-value scans passed. The dependency fixes passed the tests and build; final deployed sign-in and role checks passed. | Confirm recovery contacts. |
| [BSR-160](https://basketball-strategy.atlassian.net/browse/BSR-160), from SCRUM-534 | Budget controls and resource limits match the approved setup. A local fallback works. October 1–25 net cost is CAD $0.19; the month-end forecast is $0.19. Final API and coach browser checks passed. | Review billing again as new usage appears. |
| [BSR-158](https://basketball-strategy.atlassian.net/browse/BSR-158), from SCRUM-537 | The explanation contract, evaluation cases, fallback and spending rules are specified. | Choose a provider and approve its separate budget before implementation or paid evaluation. |

Use [local setup and recovery](recovery.md), [access and secrets](access.md), [hosting and fallback](hosting.md), and [explanation checks](explanations.md).

The clean-checkout run used shared commit `45973ed5bb0a8fdadbf8e314d0f34adad4ddfc9b`. It checks the published starting point for this work. The final release tests must cover the new repairs separately. Existing Assignment 2 packages are unchanged.

Current records are in [evidence](evidence). They contain no passwords, private tokens or database exports. No hours are inferred from installation time or automated test duration.
