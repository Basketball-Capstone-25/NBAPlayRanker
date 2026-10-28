# Remaining baseline work

The four future features reuse the regular-season calculations, controls, request handling and table styles from Alpha. The current estimate is **22 hours**, down from 40 after narrowing the views. These are planning estimates; no actual hours were added.

| Work | Scope | Estimate | Target |
| --- | --- | ---: | --- |
| [SCRUM-697](https://basketball-strategy.atlassian.net/browse/SCRUM-697) | Compare FG%, eFG%, turnover and free-throw possession rates with the league in a table. | 6 h | November 16 |
| [SCRUM-698](https://basketball-strategy.atlassian.net/browse/SCRUM-698) | Track one team's selected play type across seasons in a simple chart and table. | 6 h | November 20 |
| [SCRUM-699](https://basketball-strategy.atlassian.net/browse/SCRUM-699) | Compare two season-long team profiles side by side. | 4 h | November 30 |
| [SCRUM-700](https://basketball-strategy.atlassian.net/browse/SCRUM-700) | Show independent scoring, turnover and foul-related possession rates with clear definitions. | 6 h | December 7 |

FG% uses summed makes and attempts. eFG% uses attempt-weighted rounded source rates; turnover and free-throw rates use possessions. The league comparison adds these detailed rates beyond Alpha's usage/PPP view. Missing values stay unavailable. The outcome rates can overlap and will not be shown as a 100% distribution.

No extra charts or exports are planned beyond the simple season trend chart. Each feature still includes model work, tests, access checks and publication. The future design subtasks will update the planned VP use cases to match this refined scope before implementation.

| Week | Feature work | Existing review work | Total planned |
| --- | ---: | --- | ---: |
| November 3–9 | 4 h | 1 h: coaching scenarios and risks | 5 h |
| November 10–16 | 4.5 h | 1 h: stakeholder follow-up and Beta evidence | 5.5 h |
| November 17–23 | 4 h | 1.5 h: Beta report, artifacts and submission preparation | 5.5 h |
| November 24–30 | 4.5 h | 1 h: final model/report alignment | 5.5 h |
| December 1–7 | 5 h | 1 h: demo evidence and review | 6 h |
| December 8–14 | — | 4 h: rehearsal, report integration and artifact checks; up to 2 h for defects | 4–6 h |
| December 15–18 | — | Up to 4 h: final team review and handoff | Up to 4 h |

This plan needs availability near **5–6 hours per week** through December 7. At four hours per week, it is about **7.5 hours short** before that date. Review the estimate after the first Beta feature and whenever exams, feedback or defects change the workload. These allowances are not a promise that meetings or report changes will fit into exactly one hour.

Beta features are targeted for November 20 before the November 23 submission. Final features are targeted for December 7, leaving December 8–14 for rehearsal and corrections before the December 18 submission. SLATE lists a December 7 Data Analytics defence deadline, but the team's applicable specialization still needs confirmation. Do not treat that route-specific date as a confirmed team commitment.

Jira parent stories sit in their completion iteration. Child start/due dates show preparation in earlier weeks; they inherit the parent's sprint. Future tasks remain To Do and future tests remain Verify. Release dates, teammate assignments and sprint lifecycle were unchanged.

[BSR-140](https://basketball-strategy.atlassian.net/browse/BSR-140) remains open: Abdul's availability, backup roles and any exam constraints still need agreement. [BSR-55](https://basketball-strategy.atlassian.net/browse/BSR-55) tracks the reduced scope. The original estimates, revised values, rationale and preservation checks are in `capacity-review.json`.
