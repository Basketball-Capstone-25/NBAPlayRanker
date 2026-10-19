JIRA REVIEW GUIDE

Open the current records in Jira:
- SCRUM-480: data-quality results in Data Explorer
  https://basketball-strategy.atlassian.net/browse/SCRUM-480
- SCRUM-526: next-season PPP evaluation using previous-season inputs
  https://basketball-strategy.atlassian.net/browse/SCRUM-526
- SCRUM-486: CSV, JSON and PDF exports of the displayed results
  https://basketball-strategy.atlassian.net/browse/SCRUM-486
- SCRUM-542: Model Metrics loading repair
  https://basketball-strategy.atlassian.net/browse/SCRUM-542
- SCRUM-616: negative context values in report filenames
  https://basketball-strategy.atlassian.net/browse/SCRUM-616
- SCRUM-550: measured analyst uplift response times
  https://basketball-strategy.atlassian.net/browse/SCRUM-550

The new export suite is BST-170. Its eight Task-level cases cover the three
pages and formats, stale results, access control, validation, retry and file
contents. Case comments record the checks and evidence. BST-178 keeps the
initial filename failure and its linked repair. BST-145 keeps its earlier
loading failure and the passing retest.
https://basketball-strategy.atlassian.net/browse/BST-170
https://basketball-strategy.atlassian.net/jira/software/projects/BST/boards/4

The final snapshot has 141 cases: 122 Pass, 13 Fail and 6 Verify. These totals
include historical results; each case retains its own execution evidence.
The eight new export cases are on the board and passed this release's checks.

Every risk has formal links to project work. Current comments explain changes
that affect the risk; existing findings and owners remain in the history.
The full register is in risk-plan.csv.
https://basketball-strategy.atlassian.net/jira/software/projects/BSR/boards/3

MODEL PUBLICATION AND OPEN WORK

SCRUM-603 and SCRUM-606 are Done. Shared VP revision 56 contains the three
feature collaborations, with an overview, structural design and sequence for
each. Publication passed 23 checks. The original diagrams and model identities
are preserved.
https://basketball-strategy.atlassian.net/browse/SCRUM-606

SCRUM-609 keeps the response-time agreement, coach baseline acceptance and
startup check open. The small uplift timing sample does not test those items.
BST-111 remains Verify.
https://basketball-strategy.atlassian.net/browse/SCRUM-609

Other known failures retain linked repairs. Higher-priority repairs are planned
for Iteration 17 and medium-priority repairs for Iteration 18. These allocations
do not assert that the whole app is repaired or that team capacity is agreed.
The shared sprint lifecycle was not changed.

EFFORT

New development subtasks have numeric estimates. The 13 supplied Assignment 2
hours are recorded for October 16-18: 3 for data checks, 2 for forecasting,
2 for exports, 4 for VP, and 2 for testing/Jira. The previous Assignment 1
worklogs remain 1 hour for deployment, 4 hours for risks and 4 hours for testing.
No hours were counted twice.

FILES

jira-snapshot.json is the sanitized export behind the three CSV files. It lists
the exact project query and capture time. test-plan.csv and risk-plan.csv cover
their full projects; project-plan.csv covers this release and its linked
follow-up work. project-plan-all.csv gives the whole shared project in a
shorter format, including teammate work. Historical snapshots and planning
records are retained with their capture times. Use jira-snapshot.json for the
final captured statuses.

Run export-csv.py to rebuild the CSV files from that snapshot. Run
verify-tracking.py to check the course hierarchy, linked repairs, risk links,
current results, duplicate reconciliation and preservation of the known hours.
The output is tracking-verification.json.
