# Model walkthrough

**Reviewed snapshot: shared revision 65.** The project contains 268 diagrams, including the intervening teammate Quick Call changes. All 207 diagrams from revision 63 and all prepared model and diagram IDs remain present. The checks cover 20 scoped use cases and four native wireflows with 24 editable screens. After publication, all 24 screens and 17 use-case links were verified again; all 28 wireflow/screen renders match the checked candidate. See the [publication checks](native-publication-check.json).

Published native-file SHA-256: `e0e8ef81d7af920d227c6f228e707a0e9673fd6f5131bd080d1bdbf75a2b0408`.

The [published native project](<Basketball Playtype Ranker.vpp>) contains the editable diagrams and screens. The [publication record](publication-checks.json) separates the preserved work from the incoming Quick Call view changes.

The shared project later advanced to [revision 66](server-head-observation.json) for Abdul's request-log work. This review and native file cover revision 65; the later revision was observed but was not reviewed here.

## What to open in Visual Paradigm

1. Open **Basketball Playtype Ranker** and find **Functional Overview - Current Analyst and Shot Evidence**. Use it to show the analyst, shared shot, offline maintenance and coach requirements together. The [overview image](<renders/Functional Overview - Current Analyst and Shot Evidence.png>) is available here too.
2. Choose a use case and open its **Flow of Events**. Show the normal steps, alternatives, preconditions and postconditions. The [flow text](use-case-flows.json) preserves the saved text for all 20 scoped use cases.
3. Open the matching native wireflow below. Follow an input through loading, success, changed inputs and failure. Its screen views are editable models of states and transitions, not exact screenshots of every app control.
4. Follow the use case’s realization to its collaboration overview, structural design and detailed sequence. The [model crosswalk](model-traceability.json) lists their exact names and IDs.
5. Open the corresponding row in the [code and test guide](../code-traceability.md). Show the implementation and the actual execution evidence beside the model.

## Screen flows

| Native wireflow | What it explains |
| --- | --- |
| [Data Explorer and Dictionary](<renders/Wireflow - Data Explorer and Dictionary.png>) | Source checks, loaded data versions, column definitions and clear unavailable/retry states. |
| [Shot Plan, Explorer and Heatmap](<renders/Wireflow - Shot Plan, Explorer and Heatmap.png>) | Current filters, page-specific results, plotted/excluded attempt counts and CSV/PDF/PNG downloads. |
| [Analyst Metrics and Reports](<renders/Wireflow - Analyst Metrics and Reports.png>) | Regular diagnostics, saved shot evidence, limitations and exports of the permitted results. |
| [Coach Context and Current Calls](<renders/Wireflow - Coach Context and Current Calls.png>) | First-Build inputs, supported conditions, overtime and hiding old results when inputs change. |

## Three useful design examples

- [Shot-plan sequence](<renders/Detailed Sequence - Review Shot-Plan Recommendations.png>): the page requests rankings and the image through separate API boundaries, then presents a result for the selected inputs. Rankings use season-wide points per attempt; image filters do not change those rankings.
- [Saved shot-evidence design](<renders/Structural Design - Review Saved Shot-Model Evidence.png>): the server reads validated saved results. Opening the page does not train a model. These are conditional shot diagnostics, not pre-possession forecasts.
- [Gameplan-storage sequence](<renders/Detailed Sequence - Coach-Owned Gameplan Storage.png>): only the owning coach can read or change the saved state. New rows use INSERT; updates include the last-read revision. A conflicting write requires reload and reconciliation. Gurk’s scope is the storage contract; teammate 482/483 owns browser sync.

The next-season evaluation and shot rebuild are offline processes, so they have input/output flows and collaborations rather than invented app screens. Shared CSV/JSON/PDF reports export captured displayed rows. Specialized shot CSV/PDF files regenerate from selected inputs and record the loaded versions; PNG and saved-metrics CSV also check the expected version.

The [preservation checks](preservation-check.json), [native reopen checks](native-wireflow-audit.json) and [source/model comparison](code-model-review.json) record what was verified. The [render index](render-index.json) contains all 67 selected diagram and screen images.
