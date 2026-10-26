# Shot source and zone recheck

The recheck found three problems: the three-point zone started at 22 feet everywhere, the paint and corner shapes were too broad or too short, and the provider's `Step Back Jump Shot` label was treated as an ordinary jump shot. It also found source games with inconsistent scores and ten shot locations that do not plausibly match their declared attempt value.

The corrected dataset contains **1,367,928 shots from 7,725 games**, with all 30 teams in each of the six seasons. It changes 68,118 zone labels and 69,274 shot-type labels. The retained team, opponent, shot outcome, coordinates, distance, clock and margin values are unchanged.

| Check | Result |
|---|---:|
| Source games excluded for inconsistent scoring events | 14 |
| Shot rows in those games | 2,506 |
| Retained locations flagged as unreliable | 10 |
| Retained unknown attempted shot values | 110,206 |
| Retained known twos in three-point zones | 0 |
| Retained known threes in two-point zones | 0 |
| Raw participant, outcome, coordinate or score-transition disagreements | 0 |

No raw events were deleted or corrected. The excluded games remain in the six pinned provider files, and [the quarantine manifest](final-source-quarantine.json) lists their IDs and conflicting events. For example, game 401704805 contains two different made-dunk events at 1:55 and 1:54 with the same scoreboard. The whole game is excluded because selecting one event as the truth would be a guess.

Zone classification now uses a 16-foot-wide rectangular paint ending 13.75 feet in front of the hoop, a 23.75-foot arc, and 22-foot corner lines that meet that arc about 8.95 feet in front of the hoop. These dimensions come from the [NBA playing rules](https://cdn.nba.com/manage/2026/01/Official-2025-26-NBA-Playing-Rules.pdf) and [official court diagram](https://cms.nba.com/wp-content/uploads/sites/4/2025/10/2025-26-NBA-Officials-Guide.pdf).

An explicit two- or three-point attempt value takes precedence when approximate coordinates land close to the line. A discrepancy larger than the project's 1.5-foot screening tolerance is flagged rather than moved to an invented location. This tolerance is a data-quality rule, not a statement about the provider's measurement precision. Those ten records retain their points and shot-type counts, but have `ZONE=unknown` and `LOCATION_VALID=false`; charts, spatial models and zone recommendations exclude them.

Older missed attempts with no explicit point value remain null. Their zones use geometry and carry `ZONE_METHOD=geometry_only`, so approximate classification is visible. Coordinates retain the [provider's feet-based court frame](https://py.sportsdataverse.org/changelog-unreleased): raw hoop position (25, 0), transformed full-court origin at midcourt, with home plays rotated. The display must rotate by the attacking team, not by the sign of each individual point.

All 53 source shot-type labels were reviewed in [provider-type-audit.json](provider-type-audit.json). Existing single-category priorities are preserved: an alley-oop dunk is Dunk, and a bank stepback is Bank Shot. Generic or contradictory subtype labels remain Other.

The [final raw audit](raw-event-audit-final.json) compares every retained event with its pinned source, including both scoreboard changes, and records 100 fresh samples per season. The previous 600 samples still match their source; three belong to quarantined games and are excluded from the new dataset. [Final installation checks](final-installation.json) record preserved fields and all four dataset hashes. The pre-repair data and saved results were backed up outside the repository; earlier assignment packages are unchanged.

Saved models and analysis are regenerated locally against the new clean-file hash. Model evaluation keeps the same bounded 75,000-row sample limit, seed 42, five game-grouped training folds, and separate 2025–26 evaluation. Spatially invalid rows are removed before sampling. These results evaluate a proposed shot, not a future possession. The [build log](offline-model-build-final.txt) records completion; live-page verification is recorded separately by the release checks.
