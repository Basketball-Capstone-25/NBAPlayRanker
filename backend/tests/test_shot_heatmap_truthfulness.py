import base64
import io

import numpy as np
import pandas as pd
import pytest

from infrastructure.data_access import pbp_viz
from infrastructure.visualization_and_export.viz_shot_heatmap import heatmap_plot_data, render_shot_heatmap_png


def frame():
    return pd.DataFrame({
        'SEASON_STR': ['2024-25'] * 7, 'TEAM_ABBR': ['TOR'] * 7,
        'OPP_ABBR': ['BOS'] * 7, 'SHOT_TYPE': ['Jump'] * 7, 'ZONE': ['mid'] * 7,
        'X': [40., -40., 48., 30., 30., 30., np.nan],
        'Y': [0., 0., 0., 26., 0., 0., 0.],
        'HOME_FLAG': [True] * 7,
        'LOCATION_VALID': [True, True, True, True, False, True, True],
    })


def canonical(df):
    return df.rename(columns={'SEASON_STR':'season','TEAM_ABBR':'team','OPP_ABBR':'opp',
        'SHOT_TYPE':'shot_type','ZONE':'zone','X':'x','Y':'y','HOME_FLAG':'home','LOCATION_VALID':'location_valid'})


def test_physical_bounds_exclude_padding_heaves_and_invalid_locations():
    eligible, x, y, counts = heatmap_plot_data(frame())
    assert list(eligible.index) == [0, 5]
    assert x.tolist() == [40, 30]
    assert counts == {'n_shots_excluded_location': 2, 'n_shots_outside_halfcourt': 3}


def test_api_counts_reconcile_to_actual_render_and_never_include_exclusions(monkeypatch):
    monkeypatch.setattr(pbp_viz, '_load_canonical', lambda: canonical(frame()))
    payload = pbp_viz.render_pbp_heatmap_json(season='2024-25', team='TOR', opp='BOS', max_shots=1)
    assert payload['n_shots_total'] == 7
    assert payload['n_shots_rendered'] == 1
    assert payload['n_shots_excluded_location'] == 2
    assert payload['n_shots_outside_halfcourt'] == 3
    assert payload['n_shots_omitted_by_cap'] == 1
    assert base64.b64decode(payload['image_base64']).startswith(b'\x89PNG')
    assert 'Shown: 1' in payload['caption']
    a = pbp_viz._heatmap_data(season='2024-25', team='TOR', opp='BOS', max_shots=1)[1]
    b = pbp_viz._heatmap_data(season='2024-25', team='TOR', opp='BOS', max_shots=1)[1]
    pd.testing.assert_frame_equal(a, b)


def test_empty_filter_stays_no_data_instead_of_success(monkeypatch):
    monkeypatch.setattr(pbp_viz, '_load_canonical', lambda: canonical(frame()))
    with pytest.raises(Exception) as exc:
        pbp_viz.render_pbp_heatmap_json(season='2024-25', team='TOR', opp='BOS', zone='rim')
    assert exc.value.status_code == 404


def test_renderer_uses_labeled_linear_count_scale(monkeypatch):
    import matplotlib.figure
    seen = []
    original = matplotlib.figure.Figure.colorbar
    def colorbar(self, image, *args, **kwargs):
        result = original(self, image, *args, **kwargs)
        seen.append((image, result))
        return result
    monkeypatch.setattr(matplotlib.figure.Figure, 'colorbar', colorbar)
    png = render_shot_heatmap_png(shots_df=frame(), season='2024-25', our_team='TOR', opp_team='BOS')
    assert png.startswith(b'\x89PNG')
    image, legend = seen[0]
    from matplotlib.colors import Normalize, LogNorm
    assert isinstance(image.norm, Normalize) and not isinstance(image.norm, LogNorm)
    assert legend.ax.get_ylabel() == 'Attempts per hexagon'
    assert int(image.get_array().sum()) == 2


def test_pdf_uses_same_cap_filters_and_counted_heatmap(monkeypatch):
    from application.recommendation_services import export_shotplan_pdf as pdf
    from pypdf import PdfReader
    class Recommender:
        def rank(self, **kwargs):
            return {'top_shot_types': [], 'top_zones': []}
    monkeypatch.setattr(pdf, '_get_shot_rec', lambda: Recommender())
    monkeypatch.setattr(pdf, '_get_shots_df', frame)
    monkeypatch.setattr(pbp_viz, '_load_canonical', lambda: canonical(frame()))
    report, _ = pdf.build_shotplan_pdf(season='2024-25', our='TOR', opp='BOS', k=5, w_off=.7, max_shots=1)
    text = '\n'.join(p.extract_text() for p in PdfReader(report).pages)
    assert 'Matched 7 attempts; shown 1. Cap 1.' in text
    assert 'points per shot attempt, not points per possession' in text
