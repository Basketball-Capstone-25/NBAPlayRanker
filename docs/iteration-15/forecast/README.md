# Season-ahead PPP evaluation

SCRUM-526 checks whether earlier seasons can predict a later season's points per possession (PPP). The existing Calibration page measures same-season relationships. This report answers a different question: before a season starts, how close can we get using only the previous regular season?

The dataset contains season totals, so this is a season-ahead backtest. It does not measure predictions made before individual games.

## What goes into a prediction

Each row is one team and one play type. The outcome is that team's points divided by possessions for the target regular season. Playoffs and defensive rows are excluded.

The eight inputs all come from the immediately previous regular season: team PPP, league PPP for the play type, log possessions, possession share, effective field-goal rate, turnover rate, free-throw rate, and scoring-possession rate. Rates are weighted by player possessions. The target season supplies the outcome only. Missing history is counted and excluded, never filled from another season.

## How the data is split

| Step | Target seasons | Rows | Purpose |
|---|---|---:|---|
| Training | 2020–21 to 2022–23 | 990 | Fit the scaler and Ridge candidates |
| Tuning | 2023–24 | 330 | Choose alpha from 0.1, 1, 10, and 100 using RMSE |
| Final fitting | 2020–21 to 2023–24 | 1,320 | Refit the selected model and scaler using earlier data |
| Final test | 2024–25 | 330 | Score the frozen model using 2023–24 inputs |

The model family and eight features are fixed. Tuning selected alpha **0.1**. Neither scaling nor model selection uses the final season. All three methods use the same 330 test rows: 30 teams × 11 play types.

## Results

All errors are in PPP. Lower MAE and RMSE are better; positive bias means the prediction is too high on average.

| Method | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Ridge using previous-season inputs | 0.05231 | 0.06712 | +0.02227 |
| Repeat previous team PPP | 0.06340 | 0.08135 | +0.01516 |
| Use previous league PPP for the play type | 0.05345 | 0.06817 | +0.01845 |

Ridge has slightly lower errors than the league baseline, but more average overprediction. That small difference on one season is not enough to claim a reliable advantage. Both baselines follow the same timing rule as the model.

[report.json](report.json) includes the tuning results, fitted scaler, per-play-type errors, and five calibration bins showing average predicted and actual PPP. Empty or small bins are marked sparse. No correction is fitted to the final outcomes. [predictions.csv](predictions.csv) contains every test row, its prior-season inputs, the outcome, and all three predictions.

## Run it again

From the repository root, with the backend dependencies installed:

```bash
backend/.venv/bin/python backend/data/etl/run_forecast_evaluation.py
backend/.venv/bin/python -m pytest backend/tests/test_forecast_evaluation.py backend/tests/test_calibration.py -q
```

On another setup, use the Python executable from your backend environment. `--output-dir /tmp/forecast-check` saves a comparison run elsewhere. The report records the dataset and source-file SHA-256 hashes, Git revision, package versions, and prediction-file hash. Match those versions to reproduce the saved numbers; other library versions may differ slightly.

Tests change the final season's points, possessions, and rates and confirm that the selected model, fitted scaler, earlier features, and predictions stay identical. They also check the split boundaries, excluded playoffs, missing history, shared test rows, and invalid input handling. One test repeats the final-season mutation using the actual bundled dataset.

## Limits and next step

The source contains 20,050 regular-season offensive player rows, with no rows below 10 possessions. The aggregates describe this available coverage. The CSV has no publication timestamps, so it cannot prove exactly what was available on a particular past date. Its latest season also appeared in earlier retrospective project reports; it is held out from selection here, rather than being a never-seen external dataset.

This covers one final season, and related team/play-type rows are not independent trials. It does not include roster changes, injuries, or opponents. The next step is a separately collected later season or timestamped game data before using this as a pre-game forecast in the app. The existing retrospective Calibration page remains separate.
