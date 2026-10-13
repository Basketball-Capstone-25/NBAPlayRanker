"""Season-ahead PPP evaluation from completed prior regular seasons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from domain.statistical_analysis.calibration import summarize_calibration

RATE_COLUMNS = ["EFG_PCT", "TOV_POSS_PCT", "FT_POSS_PCT", "SCORE_POSS_PCT"]
FEATURE_COLUMNS = [
    "PRIOR_PPP", "PRIOR_LEAGUE_PPP", "PRIOR_LOG_POSS", "PRIOR_POSS_SHARE",
    *[f"PRIOR_{name}" for name in RATE_COLUMNS],
]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
KEYS = ["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"]


def _season_label(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def aggregate_regular_seasons(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season offense, with no global normalization or imputation."""
    required = [*KEYS, "SEASON_ID", "PLAYER_ID", "TYPE_GROUPING", "POSS", "PTS", *RATE_COLUMNS]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Forecast data is missing columns: {', '.join(missing)}")
    # NBA season IDs distinguish regular season (2YYYY) from playoffs (4YYYY).
    frame = raw.loc[
        raw["SEASON_ID"].astype(str).str.fullmatch(r"2\d{4}")
        & raw["TYPE_GROUPING"].eq("Offensive"), required
    ].copy()
    if frame.empty or frame[KEYS].isna().any().any():
        raise ValueError("Forecast evaluation requires regular-season offense with known row keys.")
    years = frame["SEASON_ID"].astype(str).str[1:].astype(int)
    if not frame["SEASON"].eq(years.map(_season_label)).all():
        raise ValueError("SEASON and regular-season SEASON_ID must agree.")
    if frame.duplicated([*KEYS, "PLAYER_ID"]).any():
        raise ValueError("Duplicate player / team / play-type season rows would double-count possessions.")
    numeric = frame[["POSS", "PTS", *RATE_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Forecast input values must be finite.")
    if (frame["POSS"] <= 0).any() or (numeric[:, 1:] < 0).any():
        raise ValueError("Possessions must be positive; points and rates must not be negative.")

    for column in RATE_COLUMNS:
        frame[column] = frame[column] * frame["POSS"]
    totals = frame.groupby(KEYS, as_index=False, sort=True)[["POSS", "PTS", *RATE_COLUMNS]].sum()
    totals["PPP"] = totals["PTS"] / totals["POSS"]
    for column in RATE_COLUMNS:
        totals[column] = totals[column] / totals["POSS"]
    totals["LOG_POSS"] = np.log1p(totals["POSS"])
    totals["POSS_SHARE"] = totals["POSS"] / totals.groupby(["SEASON", "TEAM_ABBREVIATION"])["POSS"].transform("sum")
    league = totals.groupby(["SEASON", "PLAY_TYPE"])[["PTS", "POSS"]].transform("sum")
    totals["LEAGUE_PPP"] = league["PTS"] / league["POSS"]
    return totals


def build_forecast_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match each target to exactly the preceding season, never a later or skipped one."""
    totals = aggregate_regular_seasons(raw)
    seasons = sorted(totals["SEASON"].unique())
    prior = totals.copy().rename(columns={
        "SEASON": "FEATURE_SEASON",
        **{name: f"PRIOR_{name}" for name in ["PPP", "LEAGUE_PPP", "LOG_POSS", "POSS_SHARE", *RATE_COLUMNS]},
    })
    prior["SEASON"] = (prior["FEATURE_SEASON"].str[:4].astype(int) + 1).map(_season_label)
    prior = prior[[*KEYS, "FEATURE_SEASON", *FEATURE_COLUMNS]]
    targets = totals[[*KEYS, "PPP", "POSS"]].rename(columns={"PPP": "TARGET_PPP", "POSS": "TARGET_POSS"})
    joined = targets.merge(prior, on=KEYS, how="left", validate="one_to_one", indicator=True)
    coverage = []
    for season in seasons:
        selected = joined.loc[joined["SEASON"].eq(season)]
        matched = int(selected["_merge"].eq("both").sum())
        previous = prior.loc[prior["SEASON"].eq(season)]
        coverage.append({
            "target_season": season,
            "observed_target_rows": len(selected),
            "scored_rows_with_prior_season": matched,
            "target_rows_without_prior_season": len(selected) - matched,
            "prior_season_rows_without_target": len(previous) - matched,
        })
    rows = joined.loc[joined["_merge"].eq("both")].drop(columns="_merge")
    return rows.sort_values(KEYS).reset_index(drop=True), coverage


def _fit(rows: pd.DataFrame, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(rows[FEATURE_COLUMNS].to_numpy(), rows["TARGET_PPP"].to_numpy())
    return model


def evaluate_forecasts(raw: pd.DataFrame, n_bins: int = 5) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune on the penultimate season, freeze the choice, then score the latest."""
    if not 2 <= n_bins <= 20:
        raise ValueError("n_bins must be between 2 and 20.")
    rows, coverage = build_forecast_rows(raw)
    # Use regular-season coverage, rather than any unrelated playoff-only season.
    seasons = [entry["target_season"] for entry in coverage]
    if len(seasons) < 4:
        raise ValueError("Need at least four seasons: input history, training, tuning, and final test.")
    final_season, tuning_season = seasons[-1], seasons[-2]
    train = rows.loc[rows["SEASON"] < tuning_season]
    tuning = rows.loc[rows["SEASON"].eq(tuning_season)]
    final = rows.loc[rows["SEASON"].eq(final_season)].copy()
    if any(part.empty for part in (train, tuning, final)):
        raise ValueError("Training, tuning, and final test each need rows with the preceding season available.")
    candidates = []
    for alpha in ALPHAS:
        candidate = _fit(train, alpha)
        predicted = candidate.predict(tuning[FEATURE_COLUMNS].to_numpy())
        rmse = float(np.sqrt(np.mean((predicted - tuning["TARGET_PPP"].to_numpy()) ** 2)))
        candidates.append({"alpha": alpha, "tuning_rmse_ppp": rmse})
    chosen = min(candidates, key=lambda item: (item["tuning_rmse_ppp"], item["alpha"]))["alpha"]

    fitting = rows.loc[rows["SEASON"] < final_season]
    model = _fit(fitting, chosen)
    final["RIDGE_PPP"] = model.predict(final[FEATURE_COLUMNS].to_numpy())
    final["PRIOR_TEAM_BASELINE_PPP"] = final["PRIOR_PPP"]
    final["PRIOR_LEAGUE_BASELINE_PPP"] = final["PRIOR_LEAGUE_PPP"]
    outcomes = final["TARGET_PPP"].to_numpy()
    metrics = {
        name: summarize_calibration(final[column].to_numpy(), outcomes, n_bins)
        for name, column in {
            "ridge": "RIDGE_PPP", "prior_team": "PRIOR_TEAM_BASELINE_PPP", "prior_league": "PRIOR_LEAGUE_BASELINE_PPP",
        }.items()
    }
    by_play_type = []
    for play_type, group in final.groupby("PLAY_TYPE", sort=True):
        result = {"play_type": play_type, "count": len(group)}
        for name, column in {"ridge": "RIDGE_PPP", "prior_team": "PRIOR_TEAM_BASELINE_PPP", "prior_league": "PRIOR_LEAGUE_BASELINE_PPP"}.items():
            summary = summarize_calibration(group[column].to_numpy(), group["TARGET_PPP"].to_numpy(), n_bins)
            result[name] = {key: summary[key] for key in ["mae_ppp", "rmse_ppp", "bias_ppp"]}
        by_play_type.append(result)

    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    report = {
        "task": "SCRUM-526",
        "evaluation": "season-ahead regular-season PPP backtest",
        "prediction_time": "Before the target regular season, after the entire previous regular season has ended.",
        "horizon": "One full regular season; one row per team and play type.",
        "input_availability": "Every predictor and both baselines use only the immediately previous regular season. The target season supplies outcomes for scoring only.",
        "features": FEATURE_COLUMNS,
        "target": "Sum of player points divided by sum of player possessions for the target team, play type, and regular season.",
        "selection": {
            "training_target_seasons": sorted(train["SEASON"].unique()),
            "training_rows": len(train),
            "tuning_target_season": tuning_season,
            "tuning_rows": len(tuning),
            "candidates": candidates,
            "selected_alpha": chosen,
            "rule": "Fixed eight prior-season features and Ridge family; choose lowest tuning RMSE, break ties by smaller alpha.",
        },
        "final_fit": {
            "target_seasons": sorted(fitting["SEASON"].unique()),
            "rows": len(fitting),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "ridge_coefficients": ridge.coef_.tolist(),
            "ridge_intercept": float(ridge.intercept_),
            "preprocessing": "StandardScaler fitted again on all earlier training and tuning rows after choosing alpha. No final-test rows used to fit anything.",
        },
        "final_test": {
            "target_season": final_season,
            "feature_seasons": sorted(final["FEATURE_SEASON"].unique()),
            "rows": len(final),
            "teams": int(final["TEAM_ABBREVIATION"].nunique()),
            "play_types": int(final["PLAY_TYPE"].nunique()),
            "weighting": "Each team/play-type row has equal weight; all models score the same rows.",
            "metrics": metrics,
            "by_play_type": by_play_type,
        },
        "coverage": coverage,
        "limitations": [
            "This is a season-ahead evaluation, not a pre-game or matchup forecast. It does not use opponents, current rosters, injuries, or target-season statistics.",
            "The CSV is a historical snapshot, not an archived release with publication timestamps. Season lag enforces logical availability but cannot prove the exact information published on a past date.",
            "Player coverage and any source corrections limit the aggregate. Missing prior-season rows are excluded and counted; no future-season imputation is used.",
            "The latest season is excluded from model selection in this evaluation. It has appeared in earlier retrospective project reports, so it is not a previously unseen external dataset.",
            "Only one final season is scored. Team/play-type rows are related, so these counts do not imply independent experiments or prove generalization to future seasons.",
            "PPP is continuous, not a probability. Positive bias means overprediction. Calibration bins with fewer than 20 rows are marked sparse; no calibration correction is fitted to final outcomes.",
        ],
    }
    return report, final.reset_index(drop=True)
