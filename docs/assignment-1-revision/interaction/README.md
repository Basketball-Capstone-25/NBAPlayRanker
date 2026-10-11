# Assignment 1 interaction model revision

The sequence diagrams now use the same lifeline and message notation across the project. Open the native project to edit the diagrams, or use the image links below to review them.

- Lifelines use `objectName : Classifier`, such as `calibrationApi : CalibrationEndpoints`.
- Calls name their parameters and types. Replies name the returned value and type. The diagram and message specifications define the fields inside grouped request and response types.
- The NLP flow now includes the parser result, context guard result and HTTP response that were missing.
- Three empty sequence containers now show the implemented glossary page: access, search/tag filtering and denied access.
- Authentication uses the protected profile role. CSV requests describe the same-origin export proxy. Calibration remains a retrospective diagnostic.

All 17 sequence diagrams were updated: 95 participants and 241 messages. The project still contains 192 diagrams. The 175 diagrams outside the interaction model are unchanged.

The data-quality and unified-export sequences distinguish their tested backend libraries from their planned UI integration. The Gameplan NLP sequence records the current response contract; it does not claim the separate frontend context bugs are fixed.

## How to review

1. Open `Basketball Playtype Ranker - Assignment 1 Revision.vpp` in Visual Paradigm.
2. In the Diagram Navigator, find the sequence titles listed below. The three glossary diagrams are at project root. Uplift and calibration are directly under Interaction Model. The other sequences are inside their existing use-case collaborations.
3. Check the lifeline headers and the typed labels on the arrows.
4. Open a diagram or message specification to read the grouped parameter fields, response fields and source references.
5. Use the matching JPG when comparing the submitted model with the feedback. Vendor evaluation watermarks remain intact.

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

## Preservation and validation

The official Teamwork update first incorporated shared revisions 52–54. The edits were then tested on a copy and imported with Visual Paradigm's XML tools. Every pre-existing model, diagram, diagram element and composite ID was retained. Existing model types, parents, creators and creation times were preserved. Shared class, operation and use-case definitions were retained.

`verification.json` contains the identity and contract checks. `sequence-contracts.json` lists the final participants and message definitions. `revision-manifest.json` records the exact model IDs changed or added. `render-verification.json` lists all 17 rendered images. `teamwork-status.json` records the publication state of the native project.

The native file is editable. The JPGs are review copies, not replacements for the model.

## Course structure check

The five existing collaborations for Access Analyst Workspace (UC62), Ranked List via API (Baseline) (UC18), Rank Plays with Game Context (UC11), Authenticate Analyst Access (UC69) and Validate Coach Access (UC71) retain their use-case realization links and their collaboration, structural and behavioural overview diagrams. Their sequence participants refer to existing Design Model classifiers.

This correction addresses lifeline and message notation. It does not complete the broader collaboration structure: the Export Results collaboration still has only its sequence, uplift/calibration remain directly under Interaction Model, and the glossary sequences remain at project root.

The Validation Model already contains five native test suites with 64 cases: workspace (10), analyst authentication (14), baseline API (9), coach access (9) and context ranking (22). Each suite has a test-plan diagram. These existing model test cases and their trace links were preserved; fresh Jira execution results are recorded separately. Some older test-plan links refer to duplicate, unnumbered use-case models, so this revision does not claim the entire native test model was reconciled.
