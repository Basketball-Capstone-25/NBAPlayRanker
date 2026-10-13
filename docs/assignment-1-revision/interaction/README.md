# Assignment 1 interaction model revision

Updated all 17 sequence diagrams to address the lifeline and message feedback.
The model contains 95 participants and 241 messages across those sequences.

- Lifelines use `objectName : Classifier`, such as `calibrationApi : CalibrationEndpoints`.
- Calls include parameter names and types. Replies name the result and its type.
- The NLP sequence includes the missing parser, context-guard and API replies.
- The three Glossary sequences now show access, search/tag filters and denied access.

The diagrams describe the current code. Calibration is a retrospective check;
Gameplan frontend bugs and the planned data-quality/export UI remain separate work.

## Open and review

Open `Basketball Playtype Ranker - Assignment 1 Revision.vpp` to review the saved
model. Use Diagram Navigator to find the titles below. Glossary is at project
root; uplift and calibration are under Interaction Model; the other sequences
are in their existing use-case collaborations.

Check the lifeline headers and arrow labels. Open a message specification for
fields inside grouped request and response types. The JPGs below are review
copies; the VPP is the editable model.

## Shared publication

The changes are saved locally and in Git. The shared Teamwork commit is still
pending. Open the existing shared **Basketball Playtype Ranker** project and use
**Team > Commit** with:

> Clarify sequence lifelines, request parameters and return values (SCRUM-528)

The saved model is based on shared revision 54. See `teamwork-status.json` for
publication evidence. Opening the packaged copy does not publish the changes.

## Diagrams

| Diagram | Participants | Messages | Image |
| --- | ---: | ---: | --- |
| Behavioural Overview - Glossary Access (UC15) | 4 | 8 | [View](<renders/Behavioural Overview - Glossary Access (UC15).jpg>) |
| Detailed Sequence - Glossary Search and Tag Filters (UC15) | 2 | 8 | [View](<renders/Detailed Sequence - Glossary Search and Tag Filters (UC15).jpg>) |
| Detailed Sequence - Glossary Access Denied (UC15) | 3 | 7 | [View](<renders/Detailed Sequence - Glossary Access Denied (UC15).jpg>) |
| Behavioural Overview - Access Analyst Workspace | 8 | 29 | [View](<renders/Behavioural Overview - Access Analyst Workspace.jpg>) |
| Behavioural Overview - Ranked List via API (Baseline) | 7 | 17 | [View](<renders/Behavioural Overview - Ranked List via API (Baseline).jpg>) |
| Behavioural Overview - Authenticate Analyst Access | 9 | 31 | [View](<renders/Behavioural Overview - Authenticate Analyst Access.jpg>) |
| Detailed Sequence - Authenticate Analyst Access - Invalid or Missing Token | 9 | 17 | [View](<renders/Detailed Sequence - Authenticate Analyst Access - Invalid or Missing Token.jpg>) |
| Detailed Sequence - Authenticate Analyst Access - Role Lacks Permission | 9 | 16 | [View](<renders/Detailed Sequence - Authenticate Analyst Access - Role Lacks Permission.jpg>) |
| Behavioural Overview - Validate Coach Access | 9 | 24 | [View](<renders/Behavioural Overview - Validate Coach Access.jpg>) |
| Detailed Sequence - Validate Coach Access - No Valid Session | 5 | 14 | [View](<renders/Detailed Sequence - Validate Coach Access - No Valid Session.jpg>) |
| Detailed Sequence - Validate Coach Access - Role Lacks Permission | 3 | 8 | [View](<renders/Detailed Sequence - Validate Coach Access - Role Lacks Permission.jpg>) |
| Behavioural Overview - Rank Plays with Game Context | 7 | 29 | [View](<renders/Behavioural Overview - Rank Plays with Game Context.jpg>) |
| Detailed Sequence - Analyst Top-K Uplift (475) | 5 | 8 | [View](<renders/Detailed Sequence - Analyst Top-K Uplift (475).jpg>) |
| Detailed Sequence - Inspect PPP Calibration (476) | 5 | 9 | [View](<renders/Detailed Sequence - Inspect PPP Calibration (476).jpg>) |
| Detailed Sequence - NLP Context Fallback (502) | 4 | 8 | [View](<renders/Detailed Sequence - NLP Context Fallback (502).jpg>) |
| Detailed Sequence - Run Data Quality Checks (478) | 3 | 4 | [View](<renders/Detailed Sequence - Run Data Quality Checks (478).jpg>) |
| Detailed Sequence - Unified Export Contract (484) | 3 | 4 | [View](<renders/Detailed Sequence - Unified Export Contract (484).jpg>) |

## What was preserved

All 192 diagram IDs and the original model identities, parents, creators and
creation times were retained. The 175 diagrams outside the sequence revision
were unchanged. Shared revisions 52–54 were included before the edits.

The five existing complete use-case collaborations and their overview diagrams
were preserved. The export collaboration still lacks its overview pair, and
Glossary, uplift and calibration are not inside complete collaborations. This
revision addresses the assigned lifeline and message feedback.

The existing Validation Model has five suites and 64 cases. Those cases and
links were preserved; the Jira test evidence records current executions. Some
older model links still point to duplicate, unnumbered use cases.

## Evidence files

- `verification.json`: preservation and notation checks.
- `sequence-contracts.json`: participant and message definitions.
- `revision-manifest.json`: changed and added model IDs.
- `render-verification.json`: the 17 exported images.
- `teamwork-status.json`: publication status.
