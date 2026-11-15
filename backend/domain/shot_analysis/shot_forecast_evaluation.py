"""Hold out the latest season while keeping the model and scaling frozen."""

import pandas as pd

from domain.shot_analysis.shot_forecast_features import FEATURE_COLUMNS
from domain.shot_analysis.shot_forecast_metrics import summarize_shot_errors
from domain.shot_analysis.shot_forecast_model import fit_pregame_model


def evaluate_pregame_forecasts(rows: pd.DataFrame, coverage: dict) -> tuple[dict, pd.DataFrame]:
    required = {"SEASON_STR", "GAME_DATE", "HISTORY_CUTOFF_DATE", *FEATURE_COLUMNS}
    if rows.empty or not required.issubset(rows.columns):
        raise ValueError("Forecast evaluation needs scored rows with prior-date history.")
    seasons = rows.SEASON_STR
    if not seasons.astype("string").str.fullmatch(r"\d{4}-\d{2}").fillna(False).all():
        raise ValueError("Forecast seasons must use YYYY-YY.")
    if not seasons.map(lambda value: int(value[-2:]) == (int(value[:4]) + 1) % 100).all():
        raise ValueError("Forecast season end years must match their start years.")
    if rows.HISTORY_CUTOFF_DATE.isna().any() or not (rows.HISTORY_CUTOFF_DATE < rows.GAME_DATE).all():
        raise ValueError("Every forecast must use history strictly before its game date.")
    final_season = sorted(seasons.unique())[-1]
    training = rows.loc[seasons < final_season]
    final = rows.loc[seasons.eq(final_season)].copy()
    if training.empty or training.GAME_DATE.max() >= final.GAME_DATE.min():
        raise ValueError("Training games must finish before the held-out season begins.")

    # The fixed four-feature model is chosen before reading held-out errors.
    model = fit_pregame_model(training)
    training_mean = float(training.POINTS.sum() / training.FGA.sum())
    final["RIDGE_POINTS"] = model.predict(final)
    final["TRAINING_MEAN_POINTS"] = training_mean
    final["PRIOR_LEAGUE_POINTS_PREDICTION"] = final.PRIOR_LEAGUE_PPS
    columns = {"ridge": "RIDGE_POINTS", "training_mean": "TRAINING_MEAN_POINTS",
        "prior_league": "PRIOR_LEAGUE_POINTS_PREDICTION"}
    metrics = {name: summarize_shot_errors(final, final[column]) for name, column in columns.items()}
    date_text = lambda value: pd.Timestamp(value).date().isoformat()
    report = {
        "task": "SCRUM-641",
        "evaluation": "Rolling pre-game regular-season shot forecast",
        "target": "Expected points per recorded field-goal attempt; outcomes are 0, 2 or 3.",
        "prediction_time": "Before the game, using only games on earlier Eastern calendar dates.",
        "input_contract": {
            "features": FEATURE_COLUMNS,
            "history": "Expanding regular-season totals across seasons; all games on the prediction date are excluded.",
            "excluded_current_shot_fields": ["X", "Y", "DIST", "ANGLE", "ZONE", "SHOT_TYPE", "SHOT_VALUE",
                "SHOOTER_ID", "PERIOD", "CLOCK_SEC", "MARGIN", "MADE", "POINTS"],
            "missing_history": "Use prior league points per attempt for a missing team/opponent history. Exclude dates with no league history.",
        },
        "fit": {
            "seasons": sorted(training.SEASON_STR.unique().tolist()),
            "first_game_date": date_text(training.GAME_DATE.min()),
            "last_game_date": date_text(training.GAME_DATE.max()),
            "game_team_rows": len(training), "shots": int(training.FGA.sum()),
            "parameters": model.parameters(),
            "weighting": "StandardScaler and Ridge both use field-goal-attempt weights, matching individual-shot squared error.",
            "selection": "Fixed Ridge alpha 1.0 and clipping to [0,3]; no tuning on held-out results.",
            "training_mean_points": training_mean,
        },
        "holdout": {
            "season": final_season, "first_game_date": date_text(final.GAME_DATE.min()),
            "last_game_date": date_text(final.GAME_DATE.max()),
            "games": int(final.GAME_ID.nunique()), "game_team_rows": len(final),
            "shots": int(final.FGA.sum()), "metrics": metrics,
            "history_updates": "Earlier held-out games update later input totals. The model, scaler and training-mean baseline stay frozen.",
            "primary_metric": "Shot-level RMSE; MAE and bias are also reported on the same shots for every model.",
        },
        "coverage": coverage,
        "limitations": [
            "This predicts points conditional on a field-goal attempt. It is not PPP, a whole-possession forecast or a prediction of whether a shot will occur.",
            "No current shot location/type, live clock/margin, player availability or lineup is supplied. Every recorded attempt for a team in a game receives the same prediction.",
            "The source is a corrected historical snapshot, not an archived feed as published on every game date. Quarantined or missing games can affect the histories.",
            "2025-26 appeared in earlier project diagnostics. It is held out from this evaluation's fitting and tuning, not a previously unseen external dataset.",
            "One held-out season and related shots do not establish future accuracy. MAE can favor zero when many attempts miss; RMSE is the primary mean-forecast measure.",
            "This offline evaluation does not replace the app's proposed-shot model or change its recommendations.",
        ],
    }
    return report, final.reset_index(drop=True)
