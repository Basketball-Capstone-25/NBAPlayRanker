"""Runtime shot data must work from the distributed clean dataset alone."""

import pandas as pd
import pytest

from infrastructure.data_access import pbp_clean


@pytest.fixture
def clean_dataset(tmp_path, monkeypatch):
    for name, filename in {
        "CLEAN_PARQUET": "shots_clean.parquet",
        "SOURCE_PARQUET": "raw_missing.parquet",
        "ALT_SOURCE_PARQUET": "alternate_raw_missing.parquet",
        "CANONICAL_PARQUET": "cache/shots_canonical.parquet",
        "CANONICAL_META_JSON": "cache/shots_canonical_meta.json",
    }.items():
        monkeypatch.setattr(pbp_clean, name, tmp_path / filename)
    monkeypatch.setattr(pbp_clean, "CACHE_DIR", tmp_path / "cache")
    frame = pd.DataFrame([{
        "SEASON_STR": "2024-25", "TEAM_ABBR": "TOR", "OPP_ABBR": "BOS",
        "GAME_ID": "game-1", "X": 1.0, "Y": 4.0, "POINTS": 2,
        "MADE": 1, "SHOT_VALUE": 2, "SHOT_TYPE": "Jump Shot",
        "ZONE": "paint", "HOME_FLAG": True,
    }])
    frame.to_parquet(pbp_clean.CLEAN_PARQUET, index=False)
    return frame


def test_canonical_build_and_cache_work_without_raw_download(clean_dataset):
    path = pbp_clean.ensure_canonical_parquet()
    result = pd.read_parquet(path)
    assert result.loc[0, "team"] == "TOR"
    assert result.loc[0, "points"] == 2
    original_mtime = path.stat().st_mtime_ns

    assert pbp_clean.ensure_canonical_parquet() == path
    assert path.stat().st_mtime_ns == original_mtime
    assert not pbp_clean.SOURCE_PARQUET.exists()


def test_canonical_refreshes_when_clean_input_changes(clean_dataset):
    path = pbp_clean.ensure_canonical_parquet()
    updated = pd.concat([clean_dataset, clean_dataset.assign(GAME_ID="game-2")])
    updated.to_parquet(pbp_clean.CLEAN_PARQUET, index=False)

    assert pbp_clean.ensure_canonical_parquet() == path
    assert pd.read_parquet(path)["game_id"].tolist() == ["game-1", "game-2"]


def test_unknown_attempt_value_stays_missing_in_canonical(clean_dataset):
    clean_dataset.loc[0, "SHOT_VALUE"] = float("nan")
    result = pbp_clean.build_canonical_from_clean(clean_dataset)
    assert pd.isna(result.loc[0, "shot_value"])
    assert result.loc[0, "points"] == 2


def test_cache_survives_checkout_path_and_timestamp_changes(clean_dataset, tmp_path):
    from infrastructure.data_access.pbp_cache import fingerprint_file, build_meta, write_json_atomic, cache_valid
    import shutil
    original = pbp_clean.CLEAN_PARQUET
    copied = tmp_path / "copied.parquet"
    shutil.copyfile(original, copied)
    meta_path = tmp_path / "portable-cache.json"
    write_json_atomic(meta_path, build_meta(fingerprint=fingerprint_file(original, schema_version="v2")))
    assert cache_valid(meta_path, fingerprint_file(copied, schema_version="v2"))
    assert not cache_valid(meta_path, fingerprint_file(copied, schema_version="v3"))


def test_heatmap_filters_before_normalizing_full_source(monkeypatch):
    from infrastructure.data_access import pbp_viz
    import infrastructure.visualization_and_export.viz_shot_heatmap as plot
    source = pd.DataFrame([
        {"season":"2024-25","team":"TOR","opp":"BOS","shot_type":"Jump Shot","zone":"paint","x":1.0,"y":2.0},
        {"season":"2024-25","team":"BOS","opp":"TOR","shot_type":"Jump Shot","zone":"paint","x":1.0,"y":2.0},
        {"season":"2023-24","team":"TOR","opp":"BOS","shot_type":"Jump Shot","zone":"paint","x":1.0,"y":2.0},
    ])
    monkeypatch.setattr(pbp_viz, "_load_canonical", lambda **kwargs: source)
    original = pbp_viz._prepare_df
    sizes = []
    def prepare(frame):
        sizes.append(len(frame))
        return original(frame)
    monkeypatch.setattr(pbp_viz, "_prepare_df", prepare)
    monkeypatch.setattr(plot, "render_shot_heatmap_png", lambda **kwargs: b"png")
    assert pbp_viz.render_pbp_heatmap_png(season="2024-25",team="TOR",opp="BOS") == b"png"
    assert sizes == [1]
    assert list(source.columns) == ["season","team","opp","shot_type","zone","x","y"]


def test_home_flag_keeps_heaves_away_from_the_attacking_basket():
    import numpy as np
    from infrastructure.visualization_and_export.viz_shot_heatmap import _coords_to_sportypy_halfcourt
    class Axis:
        def get_xlim(self):
            return (0, 47)
    class Court:
        court_params = {"basket_center_to_baseline":5.25}
    frame = pd.DataFrame({"X":[-40.,40.,40.,-40.],"Y":[0.,0.,0.,0.],"HOME_FLAG":[True,False,True,False]})
    x, y = _coords_to_sportypy_halfcourt(df=frame, ax=Axis(), court=Court())
    np.testing.assert_array_equal(x, [-40.,-40.,40.,40.])
