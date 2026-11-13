League comparison model (SCRUM-697 / SCRUM-714)

The coach chooses one team, season and side in the League comparison tab.
The table compares FG%, eFG%, turnover possession rate and free-throw possession
rate with the league. Each play type shows the team, league and difference,
with attempts and possessions beside the rates. The selected team is included
in the league values. Small-sample warnings do not change the calculations.

Open Basketball Playtype Ranker.vpp in Visual Paradigm. This is the published
Teamwork revision 89. It includes the team's existing diagrams.

Use Diagram Navigator to open:
- Collaboration Overview - Compare Play Types with League Average
- Structural Design - Compare Play Types with League Average
- Detailed Sequence - Compare Play Types with League Average
- Detailed Sequence - League Comparison Access and Retry
- League Comparison Test Plan
- Wireflow - League Comparison

The five Screen - League Comparison views show filters, loading, results,
missing data and retry. The PNG files are exports from this committed model.
They are design sketches, not screenshots of the running application.

The structural design explains the calculations and their source files.
FG is made shots divided by attempts. eFG uses attempts as weights for the
rounded source rates. Turnover and free-throw rates use possession weights.
Difference = 100 × (team fraction − league fraction), in percentage points.
Missing values and zero denominators stay unavailable.

The test plan links BST-244, BST-245 and BST-246 to the existing league use case.
Execution results are tracked in Jira. The new diagrams cover league comparison.
Earlier Season Summary layout edits are preserved in this shared snapshot.

Jira:
https://basketball-strategy.atlassian.net/browse/SCRUM-697
https://basketball-strategy.atlassian.net/browse/SCRUM-714
https://basketball-strategy.atlassian.net/browse/BST-244
https://basketball-strategy.atlassian.net/browse/BST-245
https://basketball-strategy.atlassian.net/browse/BST-246
