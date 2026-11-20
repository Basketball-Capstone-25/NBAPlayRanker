"""Shared, read-only columns used by both shot heatmaps and PDF exports."""
from threading import RLock
from .pbp_constants import CLEAN_PARQUET
from .dataset_metadata import read_versioned_parquet

_FRAME = None
_LOCK = RLock()
# SCRUM-825: MADE, SHOOTER_ID and GAME_ID feed the player made-shot view.
COLUMNS = ["SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "SHOT_TYPE", "ZONE", "X", "Y", "HOME_FLAG", "LOCATION_VALID", "ZONE_METHOD",
           "MADE", "SHOOTER_ID", "GAME_ID"]

def get_shot_plot_df():
    global _FRAME
    # SCRUM-491: keep the loaded shots; a changed file is only read after an
    # analyst reload has checked it and cleared this cache.
    with _LOCK:
        if _FRAME is None:
            _FRAME = read_versioned_parquet(CLEAN_PARQUET, columns=COLUMNS)
        return _FRAME


def clear_shot_plot_cache() -> None:
    global _FRAME
    with _LOCK:
        _FRAME = None
