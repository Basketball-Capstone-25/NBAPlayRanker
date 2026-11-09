Season summary model — SCRUM-495 / SCRUM-777

Status: published as shared revision 88, with no pending changes.
Basketball Playtype Ranker.vpp is the exact committed native project.

UC30: On Demand KPI Block
Native use-case ID: 1fgVX1mGAqAALhpS
https://basketball-strategy.atlassian.net/browse/SCRUM-495
https://basketball-strategy.atlassian.net/browse/SCRUM-777

What changed
The use case now describes the season summary in coach Matchup and analyst
Data Explorer. It covers current filters, no data, access checks, timeout and
retry. The summary compares observed season PPP with the modeled Top-K average.
Usage is each selected play's share of all team-season offensive possessions.
These are historical comparisons, not a scoring guarantee or a forecast.

What to open
functional-overview.png — coach and analyst access to the existing UC30.
collaboration-overview.png — how the use case connects to the design.
structural-design.png — shared panel, client, API and calculation.
sequence-summary.png — request and response for the current season.
sequence-retry.png — access, invalid filters, unavailable data and retry.
validation-model.png — five tests linked to UC30 and BST-295 through BST-299.
wireflow.png — choose, load, result, hidden, no-data and retry screen states.

The native test cases include procedures, expected results and Jira links.
Their execution results stay in Jira; these pictures show the model design.

Preservation
The revision starts from shared revision 87 and preserves all 310 existing
diagrams and 2,877 existing diagram elements. It adds 13 diagrams: six UML/test
views, one wireflow and six screen designs. Total: 323 diagrams.

The 30 preparation checks passed before publication. The separate publication
check passed all 18 checks, including exact original diagram-element rows and
fresh exports of every new diagram. VP changed some commit metadata, new
connector bounds and preview caches; those differences are recorded in
publication-verification.json. Existing teammate work and deployment layouts
are preserved. The privacy check passed for the native file and its 1,053
embedded resources. No credentials are included.
