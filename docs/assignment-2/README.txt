ASSIGNMENT 2 - NBA PLAY RANKER

Open the app: https://nbaplayranker-seven.vercel.app/
Shared source: https://github.com/Basketball-Capstone-25/NBAPlayRanker

WHAT TO REVIEW

1. Data Explorer - SCRUM-480
   The analyst page shows source-data checks, their result and a refresh button.
   These checks cover the play-type data. They do not validate shot attribution
   or prediction accuracy. See verification/browser/data-quality.png.

2. Next-season evaluation - SCRUM-526
   The offline evaluation uses previous-season inputs and keeps the final
   season out of model selection. It compares Ridge with prior-team and
   prior-league baselines. Start with forecast/READ-ME.txt, then report.json
   and predictions.csv. Regular Model Metrics is a separate retrospective
   diagnostic; its page now explains that distinction.

3. Shared report downloads - SCRUM-486
   Matchup, Context and Model Metrics export the displayed result as CSV, JSON
   or PDF. Changing a filter disables export until fresh results are shown.
   The downloaded rows were compared with the page. Files and screenshots are
   in verification/browser/. The retry fixture checks the real export control
   against a temporary failing service; it did not interrupt production.

4. Fixes found during testing
   SCRUM-542 fixes Model Metrics panels remaining in a loading state.
   SCRUM-616 preserves negative values in Context report filenames.
   BST-145 and BST-178 retain the original failures and the passing retests.
   SCRUM-575 keeps the demonstration on the stable production address.

5. Visual Paradigm - SCRUM-603
   Review the feature collaborations in model/: each connects its use case
   to an overview, structural design and detailed sequence. Existing diagrams
   are preserved. The nine additions are published in shared revision 56.
   model/teamwork-status.json and publication-verification.json record the
   commit. The included native project is the committed snapshot.

6. Jira
   project-management/README.txt links the stories, risk register and test
   board. The current snapshot and CSV exports preserve results and planned
   repairs. BST-170 contains the eight new export cases. Existing unresolved
   failures remain visible with linked future work.

HOW THE EVIDENCE FITS THE BRIEF

Model (20%): published native project, feature collaborations, diagram renders,
preservation checks and the shared revision 56 commit.

Implementation (60%): shared source, assigned feature results, real downloads,
automated checks, browser evidence and linked failures/retests.

Management (20%): stories, points, development subtasks, acceptance criteria,
concise updates, risk links and current test results. Estimates are separate
from the 13 supplied actual hours recorded for October 16-18: data checks 3,
forecasting 2, exports 2, VP 4, and testing/Jira 2.

VERIFIED RELEASE

Frontend source: 80d968be11b13ddefc6897bd0e9b63a27d6e90c1
Backend source: db7abfb763b7890d624fbe1969ffc48163948412
Backend revision: nba-playranker-api-00008-g5z
CI: https://github.com/Basketball-Capstone-25/NBAPlayRanker/actions/runs/39186958422

The final CI run passed 293 backend tests and 110 frontend tests, plus the
type check and production build. Four known shot-attribution tests remain
expected failures. The final live report campaign passed 47 checks. Earlier
test files retain their own source pins and capture times; use
verification/github-ci-final.json and verification/release.json for this
release. This does not mean every feature in the wider app has been repaired.

SCRUM-550 records a small response-time sample for analyst Top-K uplift.
SCRUM-609 keeps target agreement, coach-baseline acceptance and startup checks
open. The sample does not establish cold-start performance.

BEFORE SUBMISSION

The earlier 9 hours for Assignment 1 remain separate. Have the other team
member review the integrated release and include their contribution to each
graded part. Confirm the exact SLATE deadline and upload requirements. No
school submission has been made.
