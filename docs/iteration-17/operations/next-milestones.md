# Next milestones and rehearsal

The current repair work and VP revision 60 are recorded in the [release guide](../README.md) and [Jira closeout](../tests/jira-closeout.json). The final milestones below still need their own review and rehearsal. Preparing these notes does not complete them.

| Work | What to do next | Start with |
| --- | --- | --- |
| [SCRUM-509: final model review](https://basketball-strategy.atlassian.net/browse/SCRUM-509) · [BSR-113](https://basketball-strategy.atlassian.net/browse/BSR-113) | Compare the final code with the shared model. Check the metadata and dictionary APIs, readiness checks, saved shot results, coaching conditions and overtime flow. Update only diagrams that need a real change, then publish and verify the shared revision. | [Published model and diagrams](../model/README.md) |
| [SCRUM-511: final report](https://basketball-strategy.atlassian.net/browse/SCRUM-511) · [BSR-114](https://basketball-strategy.atlassian.net/browse/BSR-114) | Combine both members' sections. Use the current release, source-data notes and test results. Explain the model limits and unfinished requirements. Confirm the exact course deadline and required files before packaging. | [Release guide](../README.md), [source notes](data-sources.md), [shot evaluation](../shot-repair/READ-ME.txt) |
| [SCRUM-515: correctness rehearsal](https://basketball-strategy.atlassian.net/browse/SCRUM-515) · [BSR-115](https://basketball-strategy.atlassian.net/browse/BSR-115) | Run the walkthrough below together. Show a past failure, its fix and the passing rerun. Also explain a requirement that is still open. Record what was actually rehearsed and any changes needed. | [Test board](https://basketball-strategy.atlassian.net/jira/software/projects/BST/boards/4), [browser checks](../verification/browser/browser-checks.json) |
| [SCRUM-516: individual contribution rehearsal](https://basketball-strategy.atlassian.net/browse/SCRUM-516) · [BSR-116](https://basketball-strategy.atlassian.net/browse/BSR-116), [BSR-139](https://basketball-strategy.atlassian.net/browse/BSR-139) | Each member explains their own code, diagrams and test work. Open the related Jira ticket and commit, explain the change, and answer questions about its limits. Keep each person's contribution clear. | [Shared repository](https://github.com/Basketball-Capstone-25/NBAPlayRanker), [meeting walkthrough](../README.md#meeting-walkthrough) |

## Short rehearsal

1. Open [NBA Play Ranker](https://nbaplayranker-seven.vercel.app). Use an analyst account for the data and model pages, and a coach account for Gameplan.
2. In Data Explorer, show the source rows, seasons and version. Explain that a version identifies the source file; it does not prove prediction accuracy. Open the Dictionary and explain one column.
3. Show a repaired shot CSV and its different team/opponent values. Explain that the replacement source was rebuilt from event team IDs. The historical raw file was not recovered.
4. Open Shot Model Metrics. Explain the earlier-season folds and separate 2025–26 comparison. The current model evaluates a supplied shot location and type. [BST-117](https://basketball-strategy.atlassian.net/browse/BST-117) and [SCRUM-641](https://basketball-strategy.atlassian.net/browse/SCRUM-641) still track the before-possession requirement.
5. In Gameplan, enter “OT, down 3, 28 seconds left, need a three” and Build once. Explain the displayed situation, supported conditions and first recommendation. Show how unsupported instructions are identified.
6. Open one revised sequence in VP. Explain the lifeline names, request parameters and returned values. Show revision 60 and the publication checks.
7. Open a Jira regression case. Show the original failure, linked fix and current result. Check warm, concurrent and first-request timing separately; one does not establish the others.

Keep the [verified screenshots and files](../verification/browser/) available if the network is unreliable. The [local fallback](recovery.md) still needs internet and Supabase for sign-in.

## Agreements and feedback still needed

- Run the [three coaching scenarios](stakeholder-review.md) with Navreet and record what he does and says. His general feedback led to simpler screens; it does not replace the scenario review.
- Agree Abdul's availability and the backup owners in the [weekly plan](weekly-plan.md). Have each member follow the setup/recovery instructions and record their own checks.
- Confirm the recovery contacts in [access.md](access.md) and the outstanding sharing permissions in [data-sources.md](data-sources.md).
- Review the final team package together, confirm the submission requirements and deadline, then have the designated member submit it. Keep the final review and rehearsal tickets open until those steps happen.
