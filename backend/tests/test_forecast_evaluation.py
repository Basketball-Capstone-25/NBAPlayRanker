"""Season-ahead forecasting must never learn from its final test season."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain.statistical_analysis import forecast_evaluation as forecasts


@pytest.fixture
def source():
    rng = np.random.default_rng(18)
    rows = []
    for year in range(2019, 2025):
        for team in range(8):
            for play in range(3):
                poss = float(rng.integers(30, 500))
                ppp = 0.75 + play * 0.2 + team * 0.01 + (year - 2019) * 0.01 + rng.normal(0, 0.03)
                rows.append({
                    "SEASON": f"{year}-{(year + 1) % 100:02d}", "SEASON_ID": 20000 + year,
                    "TEAM_ABBREVIATION": f"T{team}", "PLAY_TYPE": f"Play{play}",
                    "PLAYER_ID": team, "TYPE_GROUPING": "Offensive", "POSS": poss,
                    "PTS": poss * ppp, "FGA": poss * 0.8, "EFG_PCT": ppp / 2,
                    "TOV_POSS_PCT": 0.1 + rng.uniform(0, 0.05),
                    "FT_POSS_PCT": 0.1, "SCORE_POSS_PCT": ppp / 2,
                })
    return pd.DataFrame(rows)


def test_features_use_exactly_the_previous_regular_season(source):
    rows, coverage = forecasts.build_forecast_rows(source)
    assert len(rows) == 120
    assert coverage[0]["target_rows_without_prior_season"] == 24
    assert all(int(target[:4]) - int(feature[:4]) == 1 for target, feature in zip(rows.SEASON, rows.FEATURE_SEASON))
    row = rows.query("SEASON == '2024-25' and TEAM_ABBREVIATION == 'T0' and PLAY_TYPE == 'Play0'").iloc[0]
    previous = source.query("SEASON == '2023-24' and TEAM_ABBREVIATION == 'T0' and PLAY_TYPE == 'Play0'").iloc[0]
    assert row.PRIOR_PPP == pytest.approx(previous.PTS / previous.POSS)
    assert row.PRIOR_EFG_PCT == previous.EFG_PCT


def test_playoffs_and_defense_cannot_change_forecasts(source):
    ignored = source.copy()
    ignored["SEASON_ID"] += 20000
    ignored["PTS"] *= 50
    defense = source.assign(TYPE_GROUPING="Defensive", PTS=99999)
    original, _ = forecasts.evaluate_forecasts(source)
    changed, _ = forecasts.evaluate_forecasts(pd.concat([source, ignored, defense]))
    assert changed == original


def test_future_statistics_cannot_change_earlier_features(source):
    original, _ = forecasts.build_forecast_rows(source)
    changed = source.copy()
    future = changed.SEASON >= "2023-24"
    changed.loc[future, ["POSS", "PTS", *forecasts.RATE_COLUMNS]] *= 20
    perturbed, _ = forecasts.build_forecast_rows(changed)
    # 2023–24 predictions are made with 2022–23 inputs, before the changed seasons.
    columns = [*forecasts.KEYS, "FEATURE_SEASON", *forecasts.FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(original.loc[original.SEASON <= "2023-24", columns], perturbed.loc[perturbed.SEASON <= "2023-24", columns])


def test_final_test_mutation_cannot_change_selection_scaler_or_predictions(source):
    original, predictions = forecasts.evaluate_forecasts(source)
    changed = source.copy()
    last = changed.SEASON.eq("2024-25")
    changed.loc[last, "PTS"] *= 5
    changed.loc[last, "POSS"] *= 2
    changed.loc[last, forecasts.RATE_COLUMNS] *= 10
    perturbed, changed_predictions = forecasts.evaluate_forecasts(changed)
    assert original["selection"] == perturbed["selection"]
    assert original["final_fit"] == perturbed["final_fit"]
    columns = [*forecasts.FEATURE_COLUMNS, "RIDGE_PPP", "PRIOR_TEAM_BASELINE_PPP", "PRIOR_LEAGUE_BASELINE_PPP"]
    pd.testing.assert_frame_equal(predictions[columns], changed_predictions[columns])
    assert original["final_test"]["metrics"]["ridge"]["mae_ppp"] != perturbed["final_test"]["metrics"]["ridge"]["mae_ppp"]


def test_each_candidate_fits_only_training_rows_and_refit_stays_before_final(source, monkeypatch):
    fits = []
    original_fit = forecasts._fit

    def observe_fit(rows, alpha):
        fits.append((set(rows.SEASON), alpha))
        return original_fit(rows, alpha)

    monkeypatch.setattr(forecasts, "_fit", observe_fit)
    report, _ = forecasts.evaluate_forecasts(source)
    assert [seasons for seasons, _ in fits[:-1]] == [{"2020-21", "2021-22", "2022-23"}] * len(forecasts.ALPHAS)
    assert fits[-1][0] == {"2020-21", "2021-22", "2022-23", "2023-24"}
    rows, _ = forecasts.build_forecast_rows(source)
    fit_rows = rows.loc[rows.SEASON < "2024-25", forecasts.FEATURE_COLUMNS]
    np.testing.assert_allclose(report["final_fit"]["scaler_mean"], fit_rows.mean().to_numpy())
    expected_scale = fit_rows.std(ddof=0).to_numpy()
    expected_scale = np.where(expected_scale < 1e-12, 1, expected_scale)
    np.testing.assert_allclose(report["final_fit"]["scaler_scale"], expected_scale)


def test_no_missing_history_is_filled_from_another_season(source):
    source = source.loc[~((source.SEASON == "2023-24") & (source.TEAM_ABBREVIATION == "T0") & (source.PLAY_TYPE == "Play0"))]
    rows, coverage = forecasts.build_forecast_rows(source)
    assert rows.query("SEASON == '2024-25' and TEAM_ABBREVIATION == 'T0' and PLAY_TYPE == 'Play0'").empty
    assert coverage[-1]["target_rows_without_prior_season"] == 1
    assert coverage[-2]["prior_season_rows_without_target"] == 1


def test_models_share_test_rows_and_calibration_reports_all_samples(source):
    report, predictions = forecasts.evaluate_forecasts(source)
    assert report["final_test"]["target_season"] == "2024-25"
    assert report["final_test"]["feature_seasons"] == ["2023-24"]
    assert report["final_test"]["rows"] == len(predictions) == 24
    for name, column in {"ridge": "RIDGE_PPP", "prior_team": "PRIOR_TEAM_BASELINE_PPP", "prior_league": "PRIOR_LEAGUE_BASELINE_PPP"}.items():
        metrics = report["final_test"]["metrics"][name]
        assert metrics["count"] == len(predictions)
        assert sum(b["count"] for b in metrics["bins"]) == len(predictions)
        assert metrics["mae_ppp"] == pytest.approx(abs(predictions[column] - predictions.TARGET_PPP).mean())
    assert sum(item["count"] for item in report["final_test"]["by_play_type"]) == len(predictions)


@pytest.mark.parametrize("problem", ["duplicate", "nonfinite", "zero_possessions", "negative_points", "mismatched_season", "missing_schema"])
def test_invalid_inputs_fail_instead_of_silently_changing_coverage(source, problem):
    if problem == "duplicate":
        source = pd.concat([source, source.iloc[[0]]])
    elif problem == "nonfinite":
        source.loc[0, "EFG_PCT"] = np.nan
    elif problem == "zero_possessions":
        source.loc[0, "POSS"] = 0
    elif problem == "negative_points":
        source.loc[0, "PTS"] = -1
    elif problem == "mismatched_season":
        source.loc[0, "SEASON"] = "1999-00"
    elif problem == "missing_schema":
        source = source.drop(columns="POSS")
    with pytest.raises(ValueError):
        forecasts.evaluate_forecasts(source)


def test_insufficient_or_disconnected_seasons_are_rejected(source):
    with pytest.raises(ValueError, match="four seasons"):
        forecasts.evaluate_forecasts(source.loc[source.SEASON < "2022-23"])
    with pytest.raises(ValueError, match="preceding season"):
        forecasts.evaluate_forecasts(source.loc[source.SEASON != "2023-24"])


def test_row_order_cannot_change_the_result(source):
    original, predictions = forecasts.evaluate_forecasts(source)
    reordered, changed_predictions = forecasts.evaluate_forecasts(source.sample(frac=1, random_state=3))
    assert original == reordered
    pd.testing.assert_frame_equal(predictions, changed_predictions)


@pytest.mark.integration
def test_bundled_final_season_has_no_effect_on_model_or_preprocessing():
    source = pd.read_csv(Path(__file__).resolve().parents[1] / "data/synergy_playtypes_2019_2025_players.csv")
    report, predictions = forecasts.evaluate_forecasts(source)
    assert report["selection"]["training_rows"] == 990
    assert report["selection"]["tuning_rows"] == 330
    assert report["final_test"]["target_season"] == "2024-25"
    assert report["final_test"]["rows"] == 330
    final = source.SEASON.eq("2024-25")
    source.loc[final, "PTS"] *= 4
    source.loc[final, "POSS"] *= 2
    source.loc[final, forecasts.RATE_COLUMNS] *= 3
    changed, changed_predictions = forecasts.evaluate_forecasts(source)
    assert changed["selection"] == report["selection"]
    assert changed["final_fit"] == report["final_fit"]
    columns = ["RIDGE_PPP", "PRIOR_TEAM_BASELINE_PPP", "PRIOR_LEAGUE_BASELINE_PPP"]
    pd.testing.assert_frame_equal(predictions[columns], changed_predictions[columns])
