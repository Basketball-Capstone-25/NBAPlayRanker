# Shot data and display recheck

This follow-up checks the shot dataset, recommendation calculations, heatmaps and downloads. It found additional defects after the earlier team-attribution repair. The earlier checks remain historical evidence.

The final frontend is `99a31e0`, deployment `dpl_DknqC3yLsJp2f7ytQNwjUAMYG4MA`. The backend is `nba-playranker-api-00012-nrd`, built from `0ba37bb` with all traffic assigned to it. [The release record](release/final-integrated-release.json) confirms both versions and teammate commit `1c283c5`. The backend passed the repeated 218-check shot campaign; the final frontend passed the five Quick Call refresh checks. The PNG browser download first passed on frontend `0ba37bb`; a [final-release browser check](browser/final-release-smoke.json) on `99a31e0`/`00012-nrd` also saved an identical PNG.

## What was corrected

- Corrected the three-point arc, corner transition and rectangular paint boundaries. This changed 68,118 zone labels.
- Recognized the provider's “Step Back Jump Shot” label. This changed 69,274 shot-type labels to Stepback.
- Excluded 14 games with inconsistent source scoring records (2,506 shots). The pinned raw files are preserved; the excluded games and reasons are listed in the source report.
- Kept ten implausible locations as unknown rather than moving them. Charts, spatial models and zone recommendations exclude these locations.
- Kept 110,206 unknown attempt values missing. Their zones use approximate geometry and carry that label.
- Tied shot results, captions and downloads to the same successful inputs. Changing filters clears old results until the new request completes.
- Corrected team-code handling for shot PDFs and made the PDF use the same canonical data, filters and sample cap as the displayed heatmap.
- Added a linear attempt-count legend. Matched attempts, plotted locations, half-court exclusions and sample-cap omissions are counted separately.

## Data used

The final dataset contains 1,367,928 shots from 7,725 games across 2020–21 through 2025–26, including regular season and playoffs. Every retained event was compared with its pinned ESPN/SportsDataverse source record. The original source files were not altered.

| File | SHA-256 |
| --- | --- |
| Clean shot events | 15078408bfaa84b6cc1a45cc7e10edd822e22b25166a3ab3f4931a0323f640cf |
| Canonical explorer/heatmap events | 80776b1dea23cbd5a088442bc60062653eb2424ebc4a14aef7b41d04dc516ae5 |
| Team aggregates | 0649554dd4f3efdbebc19b0f80f1d7d9ef03aca576964fc281c59a97dc6aff99 |
| League aggregates | b34603259d6465eceea6c59fcd05377898ec12d0a9c6e0f2019a078a72d649ab |

## Reading the screens

Shot Plan blends the selected team's season-long offense with the opponent's season-long defense. Its score is estimated **points per shot attempt**, not points per possession. Reliability weights reduce small-sample influence; they are not confidence probabilities.

The heatmap shows historical shot locations for that exact matchup and season. Its colors count attempts, including makes and misses. A darker area does not by itself mean better shooting or a defensive weakness. Shot-type and zone filters affect the heatmap, not the season-long rankings. Color limits vary by chart and are printed in its legend.

Shot locations are approximate provider coordinates. Source matching checks the pinned provider records; it does not independently prove each event or coordinate is correct in the real game. The 14 excluded games leave gaps in coverage. The ten flagged locations still contribute their known outcomes to shot-type totals, but not to charts, spatial models or zone recommendations. The 110,206 misses with unknown two/three-point attempt value remain missing; their geometry-based zones are approximate.

The saved model results evaluate a proposed shot with its location and type already supplied. They use a bounded 75,000-row sample: 62,020 earlier-season training rows, five game-grouped folds and 12,980 rows from 2025–26 for the later-season comparison. They do not establish before-possession forecast accuracy. The current API has no minimum-attempt filter, so the small-sample check verifies that a small count is reported correctly, not that a sample-size threshold is enforced.

## What each check covers

| Check | Verified scope | Evidence |
| --- | --- | --- |
| Provider records | Every retained event matched the pinned source fields checked by the audit; all 53 shot-type labels were reviewed. The excluded games and uncertain locations are listed separately. | [Source notes](data/README.md), [final audit](data/raw-event-audit-final.json), [exclusions](data/final-source-quarantine.json) |
| Aggregate arithmetic | 45 checks compare every saved aggregate and league table with independent event counts, makes, points, rates and reliability weights. | [Local calculations](live/local-parity.json) |
| Application calculations | 66 checks compare actual rank output with eight independently calculated contexts and compare six heatmaps with expected counts. | [Component comparison](live/local-component-parity.json) |
| Coordinates | 14 synthetic checks cover home/away rotation, basket placement, backcourt shots and invalid/outside-court locations. | [Coordinate checks](live/coordinate-transform.json) |
| Latest live API | 218 assertions across 30 requests on revision `00012-nrd` check source hashes, rankings, preview rows, heatmap counts, PDF/CSV content and access restrictions. All 37 saved-result comparisons with `00011-rp7` also pass. | [Integrated results](live/integrated/results.json), [release comparison](live/integrated/previous-release-comparison.json) |
| Capped PDF image | Seven checks verify a real 1,000-shot PDF cap. Its embedded court image is pixel-identical to the matching API heatmap PNG. | [PDF/PNG comparison](live/capped-pdf-image-parity.json) |
| PNG download route | 24 checks across six requests verify analyst/coach images, exact PNG bytes, private response headers, unsigned access, changed source version, invalid inputs and foreign origin. Both temporary sessions were closed locally. | [PNG route checks](live/png-proxy-results.json) |
| Browser behavior | Eight recorded checks cover visible results, changed-input guards, empty filters and recovery on Plan, Heatmap and Explorer. Actual PDF/CSV files were inspected. The later PNG browser retest saved the matching image. | [Page checks](browser/checks.json), [downloads](browser/downloads-verification.json), [PNG retest](browser/png-download-final.json) |

The rank and heatmap comparisons cover TOR/BOS in both directions for 2021–22 and 2025–26, LAL/GSW for 2024–25, and OKC/SAS for 2025–26. Rank checks include offense weights of 0, 0.7 and 1. Counts distinguish matched events, invalid locations, points outside the physical half court, displayed points and omissions caused by the display cap.

The live PDF cap check uses OKC/SAS 2025–26: **1,080 matched = 1,000 shown + 6 outside the half court + 74 omitted by the cap**. It verifies the image pixels as well as the PDF's labels and canonical/aggregate/league hashes. This was an authenticated API download; browser file-save checks are recorded separately.

Browser checks confirmed the visible GSW/BOS 2024–25 results and the 183-row CSV. All 140 cells in its ten-row preview match the downloaded file. Changing filters clears stale results and blocks exports on Plan, Heatmap and Explorer. Empty filters show a no-data message, and clearing them restores the results.

The direct PNG link did not produce a file in Downloads. The browser's download-manager status was not inspected, so this is not recorded as a “Stopped” result. [The original failure](browser/png-download-initial.json) is preserved. The replacement passed its [actual browser download retest](browser/png-download-final.json) on `0ba37bb`: the saved PNG is 134,676 bytes, measures 1,050 × 900 pixels and matches the earlier API image byte for byte. [The screenshot](browser/heatmap-png-verified.jpg) shows the result, and [the changed-input check](browser/heatmap-png-stale-guard.txt) records the download guard. The [repair record](display/png-download-repair.json) now records the successful retest and BST-205 Pass while preserving the original failure.

## Release and reproduction

The original shot release was [3a701c6](https://github.com/Basketball-Capstone-25/NBAPlayRanker/commit/3a701c6), with backend `00011-rp7`; its [deployment record](release/deployed-release.json), [first API campaign](live/results.json), browser checks and capped PDF comparison remain unchanged.

The integrated backend is [0ba37bb](https://github.com/Basketball-Capstone-25/NBAPlayRanker/commit/0ba37bb), revision `nba-playranker-api-00012-nrd`, build `7272e176-7217-402e-ac16-2f15ebccb296`. Its [separate API rerun](live/integrated/results.json) passed, and the [comparison](live/integrated/previous-release-comparison.json) confirms unchanged source hashes, ranking/preview values, PNG and CSV bytes, and PDF text. The [PNG browser record](browser/png-download-final.json) identifies frontend `0ba37bb` and its verified deployment. The final frontend [99a31e0](https://github.com/Basketball-Capstone-25/NBAPlayRanker/commit/99a31e0) adds the Quick Call refresh guard. Its [final deployment record](release/final-integrated-release.json) identifies deployment `dpl_DknqC3yLsJp2f7ytQNwjUAMYG4MA`.

- [Original shot-release backend tests](release/backend-tests-final.txt): 460 passed. [Integrated backend tests](release/backend-tests-integrated.txt): 478 passed.
- [Final integrated frontend tests](release/frontend-tests-integrated-final.txt): 257 passed; [TypeScript](release/typescript-integrated-final.txt) and [production build](release/frontend-build-integrated-final.txt) passed. The earlier 189-test shot-release log remains preserved separately.
- [Saved-model check](data/final-saved-model-check.json): both saved analysis files match the final clean-data hash. The selected gradient-boosted model has later-season MAE 1.09745 versus 1.15887 for the training-mean baseline; this is a conditional diagnostic on one season.
- [Reproduction guide](scripts/READ-ME.txt): local scripts recompute the aggregate, application and coordinate checks without network access or model fitting.
- [Original stale-result failure](browser/before-stale-plan.txt) and [Jira setup snapshot](display/jira-setup.json): preserve the initial failures and linked work. The setup snapshot is historical, not a current Jira status query.

## Integration guard and Jira closeout

The [Quick Call browser check](browser/quick-call-checks.json) passed five checks on frontend `99a31e0` and backend `00012-nrd`. Changing the score or overtime situation removes the old calls while the refreshed results load. [The screenshot](browser/quick-call-verified.jpg) shows the updated result. This verifies the integration guard; it does not claim ownership or completion of the teammate's whole Quick Call feature.

The three observed refreshes took 8.096 seconds, 365 ms and 740 ms. These are individual browser observations, not a percentile benchmark or a two-second response guarantee. Failed refreshes and late-response races are covered by focused tests, not a production network interruption.

[The Jira receipts](display/jira-closeout.json) and [readback](display/jira-closeout-readback.json) confirm SCRUM-642/643 are Done and BST-202–205 are Pass, with previous failures preserved. BST-114 remains Testing In Progress because BST-117 remains Fail; its separate before-possession work, SCRUM-641, is To Do. Quick Call tracking is recorded separately from this shot closeout.

The separate Quick-Call result-state fix is [SCRUM-644](https://basketball-strategy.atlassian.net/browse/SCRUM-644), now Done. [BST-206](https://basketball-strategy.atlassian.net/browse/BST-206) passed with the browser screenshot and regression checks. [The tracking receipt](display/quick-call-tracking.json) preserves the links to the teammate’s feature without changing its ownership or status.
