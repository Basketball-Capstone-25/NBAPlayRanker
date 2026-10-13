# Iteration 15

This work covers SCRUM-480 and SCRUM-526.

## Data quality in Data Explorer

The Data Explorer now shows four checks on the loaded Synergy play-type source: required columns, missing values, duplicate rows, and team/season coverage. Each check has its own result. The panel also shows the source, row count, check time, and dataset version.

Refresh runs the checks again. If the service cannot return a valid report, the panel shows an error and a Retry button. It does not leave an old passing result on screen. The report requires an analyst account.

The report covers all source rows, independently of the preview filters. It does not check the separate shot dataset or measure prediction accuracy. The existing shot-data failures remain open.

To show it: sign in as an analyst, open [Data Explorer](https://nbaplayranker-seven.vercel.app/data-explorer), and click **Refresh data checks**. Change a table filter and point out that the report still covers the full source.

## Season-ahead PPP evaluation

The new evaluation predicts a team's play-type PPP using the previous season's statistics. Earlier seasons are used for training and tuning. The final model is then tested on 2024–25, using only 2023–24 inputs.

On 330 final-test rows, Ridge had an MAE of 0.05231 PPP. Repeating the previous team's PPP gave 0.06340; using the previous league PPP gave 0.05345. The improvement over the league baseline is small, and the model overpredicts by 0.02227 PPP on average.

The [forecast report](forecast/README.md) explains the inputs, time split, baselines, calibration, and limits. [Predictions](forecast/predictions.csv) and a [reproduction check](forecast/verification.json) are included. This is a season-ahead backtest, not a claim of pre-game accuracy.

To show it: open the forecast report, explain the time split, then compare the three error rows. The evaluation is separate from the existing Calibration page.

## Verification

- Frontend: 59 tests passed.
- Backend: 222 tests passed; four existing shot-data cases remain expected failures.
- TypeScript and the production build passed.
- GitHub [CI for the implementation](https://github.com/Basketball-Capstone-25/NBAPlayRanker/actions/runs/38345252833) passed both jobs.
- Repeating the forecast run produced identical reports and predictions.
- Live API checks passed 16/16, including analyst access, coach rejection, fresh reports, and the existing table preview.
- Browser checks passed for loading, unavailable service, Retry, Refresh, and changing preview filters.

The visual check found pale status text. [SCRUM-574](https://basketball-strategy.atlassian.net/browse/SCRUM-574) fixes the contrast. [BST-168](https://basketball-strategy.atlassian.net/browse/BST-168) keeps the failed result and the passing retest.

The verification folder contains the local results and release checks. Jira links the implementation tasks to their test cases.

## Jira walkthrough

| Work | Implementation | Tests |
|---|---|---|
| Data quality | [SCRUM-480](https://basketball-strategy.atlassian.net/browse/SCRUM-480), subtasks 568–570 | [BST-156](https://basketball-strategy.atlassian.net/browse/BST-156), cases 158–162 and 168 |
| PPP evaluation | [SCRUM-526](https://basketball-strategy.atlassian.net/browse/SCRUM-526), subtasks 571–573 | [BST-157](https://basketball-strategy.atlassian.net/browse/BST-157), cases 163–167 |

Open either story to see its acceptance criteria and completed subtasks. Follow a test link to see the steps, expected result, actual result, and evidence. The test cases are on the [test board](https://basketball-strategy.atlassian.net/jira/software/projects/BST/boards/4), grouped under their suites.

## Next steps

- Add source-data warnings to recommendation flows through SCRUM-479.
- Repair the separate shot-data attribution defect tracked in SCRUM-538.
- Validate forecasting on a separately collected later season or timestamped game data.
