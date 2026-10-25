"""Column definitions shared by the catalog API and Data Dictionary page."""

# Unknown columns stay visible instead of receiving a guessed definition.
DEFINITIONS = {
    "SEASON": ("NBA season covered by the row.", "season label"),
    "SEASON_ID": ("Season identifier supplied by NBA Stats.", "identifier"),
    "ENTITY_TYPE": ("Whether the source row describes a player or team.", "category"),
    "PLAYER_ID": ("NBA player identifier.", "identifier"),
    "PLAYER_NAME": ("Player name supplied by the source.", "text"),
    "TEAM_ID": ("NBA team identifier.", "identifier"),
    "TEAM_ABBREVIATION": ("Team code used for this row.", "team code"),
    "TEAM_NAME": ("Full team name supplied by the source.", "text"),
    "PLAY_TYPE": ("Possession category, such as isolation or pick-and-roll.", "category"),
    "TYPE_GROUPING": ("Whether the row measures offense or defense.", "category"),
    "SIDE": ("Normalized offense or defense label.", "category"),
    "PERCENTILE": ("Source ranking of efficiency against comparable rows.", "source percentile"),
    "GP": ("Games included in the source row.", "games"),
    "POSS": ("Possessions recorded for this play type.", "possessions"),
    "POSS_PCT": ("Share of possessions using this play type.", "fraction, 0–1"),
    "PPP": ("Points scored or allowed per possession.", "points per possession"),
    "FG_PCT": ("Made field goals divided by field-goal attempts.", "fraction, 0–1"),
    "EFG_PCT": ("Field-goal percentage adjusted for the extra value of three-pointers.", "fraction"),
    "SCORE_POSS_PCT": ("Share of possessions that produce points.", "fraction, 0–1"),
    "TOV_POSS_PCT": ("Share of possessions ending in a turnover.", "fraction, 0–1"),
    "SF_POSS_PCT": ("Share of possessions ending with a shooting foul.", "fraction, 0–1"),
    "FT_POSS_PCT": ("Share of possessions ending with free throws.", "fraction, 0–1"),
    "PLUSONE_POSS_PCT": ("Share of possessions with a made basket and an additional free throw.", "fraction, 0–1"),
    "PTS": ("Total points recorded in the source row.", "points"),
    "FGM": ("Made field goals.", "shots"),
    "FGA": ("Attempted field goals.", "shots"),
    "RELIABILITY_WEIGHT": ("Sample-size weight used when shrinking estimates toward league rates.", "weight, 0–1"),
    "SEASON_STR": ("NBA season label attached to the event or aggregate.", "season label"),
    "TEAM_ABBR": ("Shooting team code resolved from the event participant ID.", "team code"),
    "OPP_ABBR": ("Opposing team code. It must differ from the shooting team.", "team code"),
    "HOME_FLAG": ("Whether the shooting team is the home team.", "boolean"),
    "SHOT_TYPE": ("Description or category of the field-goal attempt.", "category"),
    "SHOT_VALUE": ("Point value of the attempted field goal when supplied or explicit in the event. Ambiguous older misses remain null; this column is excluded from model inputs.", "2 or 3 points"),
    "MADE": ("Whether the attempt was made.", "0 or 1"),
    "POINTS": ("Points scored by this shot.", "points"),
    "X": ("Shot-location coordinate along the length of the full court, with midcourt at zero.", "feet"),
    "Y": ("Shot-location coordinate across the court, with the centre line at zero.", "feet"),
    "DIST": ("Distance from the shooting hoop, calculated from the source hoop-relative coordinates.", "feet"),
    "ANGLE": ("Angle from the hoop toward the court; calculated from hoop-relative lateral and depth coordinates.", "degrees"),
    "ZONE": ("Court region assigned by the project shot pipeline.", "category"),
    "PERIOD": ("Game period; values above four represent overtime.", "period number"),
    "CLOCK_SEC": ("Time remaining in the event's period.", "seconds"),
    "MARGIN": ("Shooting-team score minus opponent score immediately before the shot, removing the current shot’s points from the event scores. Unknown margins remain missing.", "points"),
    "EVENT_ID": ("ESPN play-by-play event identifier retained for source comparisons.", "identifier"),
    "GAME_ID": ("Source game identifier used to keep games separate in evaluation.", "identifier"),
    "SHOOTER_ID": ("Source identifier for the player taking the shot.", "identifier"),
    "ROLE": ("Whether an aggregate describes offense or defense.", "category"),
    "ATTEMPTS": ("Number of shot attempts in the group.", "shots"),
    "MAKES": ("Number of made shots in the group.", "shots"),
    "EPA": ("Points scored divided by shot attempts in this group.", "points per attempt"),
    "LEVEL": ("Whether shots are grouped by type or court zone.", "category"),
    "LEAGUE_ATTEMPTS": ("League-wide shot attempts for the matching group.", "shots"),
    "LEAGUE_POINTS": ("League-wide points for the matching group.", "points"),
    "EPA_LEAGUE": ("League points divided by league shot attempts.", "points per attempt"),
    "LEAGUE_REL_WEIGHT": ("Sample-size reliability weight for a league shot group.", "weight, 0–1"),
    "PPP_PRED": ("Estimated points per possession from the saved model.", "points per possession"),
    "PPP_ML": ("Saved model estimate of offensive points per possession.", "points per possession"),
}
ALIASES = {"TEAM": "TEAM_ABBR", "OPP": "OPP_ABBR", "HOME": "HOME_FLAG", "IS_MAKE": "MADE"}


def dictionary_entries(catalog: list[dict]) -> list[dict]:
    entries = []
    for dataset in catalog:
        for column in dataset["columns"]:
            key = ALIASES.get(column.upper(), column.upper())
            definition = DEFINITIONS.get(key)
            entries.append({"dataset": dataset["id"], "dataset_name": dataset["name"],
                            "column": column, "meaning": definition[0] if definition else "Undocumented column.",
                            "unit": definition[1] if definition else "Unknown", "source": dataset["source"],
                            "documented": definition is not None})
    return entries
