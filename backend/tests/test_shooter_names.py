"""Player names for the shot data come only from the play text (SCRUM-825)."""

import pandas as pd

from domain.shot_analysis.shooter_names import normalize_athlete_id, shooter_name, shooter_names


def test_names_come_from_make_and_miss_text():
    assert shooter_name("Jayson Tatum makes 26-foot three point jumper (Jrue Holiday assists)") == "Jayson Tatum"
    assert shooter_name("RJ Barrett misses driving layup") == "RJ Barrett"
    assert shooter_name("Shai Gilgeous-Alexander makes 12-foot pullup jump shot") == "Shai Gilgeous-Alexander"
    assert shooter_name("De'Aaron Fox makes free throw 1 of 2") == "De'Aaron Fox"
    assert shooter_name("Jaren Jackson Jr. makes two point shot") == "Jaren Jackson Jr."
    assert shooter_name("Kevin Love blocks Jalen Brunson 's 2-foot layup") is None
    assert shooter_name("End of the 1st Quarter") is None
    assert shooter_name(None) is None


def test_ids_match_whatever_type_they_were_stored_as():
    assert normalize_athlete_id(3136193) == "3136193"
    assert normalize_athlete_id(3136193.0) == "3136193"
    assert normalize_athlete_id("3136193.0") == "3136193"
    assert normalize_athlete_id(float("nan")) is None
    assert normalize_athlete_id("") is None


def test_each_id_gets_its_most_common_name_and_unnamed_ids_are_left_out():
    rows = pd.DataFrame({
        "athlete_id_1": [1.0, 1.0, 1.0, 2.0, 3.0, None],
        "text": [
            "Scottie Barnes makes layup", "Scottie Barnes misses jumper", "S. Barnes makes dunk",
            "Jakob Poeltl makes hook shot", "Timeout", "RJ Barrett makes layup",
        ],
    })
    table = shooter_names(rows, "athlete_id_1", "text")
    assert table.to_dict(orient="records") == [
        {"athlete_id": "1", "name": "Scottie Barnes", "plays": 2},
        {"athlete_id": "2", "name": "Jakob Poeltl", "plays": 1},
    ]
