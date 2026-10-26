"""Shared, read-only columns used by both shot heatmaps and PDF exports."""
from threading import RLock
from .pbp_constants import CLEAN_PARQUET
from .dataset_metadata import read_versioned_parquet, source_version

_FRAME = None
_LOCK = RLock()
COLUMNS = ["SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "SHOT_TYPE", "ZONE", "X", "Y", "HOME_FLAG", "LOCATION_VALID", "ZONE_METHOD"]

def get_shot_plot_df():
    global _FRAME
    version = source_version(CLEAN_PARQUET)
    with _LOCK:
        if _FRAME is None or _FRAME.attrs.get("dataset_version") != version:
            _FRAME = read_versioned_parquet(CLEAN_PARQUET, columns=COLUMNS)
        return _FRAME
