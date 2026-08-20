import numpy as np
import pandas as pd
import pytest

from ci_impact import (
    fraction_excluding_zero,
    pairwise_resolvability,
    rank_intervals,
    topk_stability,
    ci_impact_summary,
)
from dependence import add_shot_dependence, bootstrap_draw_matrix


def point_frame():
    """Three players: one clearly positive, one clearly negative, one straddling zero."""
    return pd.DataFrame({
        'player_id': [1, 2, 3],
        'dep_share': [0.30, -0.20, 0.02],
        'dep_share_ci_lower': [0.20, -0.30, -0.10],
        'dep_share_ci_upper': [0.40, -0.10, 0.14],
    })


def test_fraction_excluding_zero_counts_only_intervals_clear_of_zero():
    out = fraction_excluding_zero(point_frame(), 'dep_share')
    assert out['n_groups'] == 3
    assert out['n_excluding_zero'] == 2          # players 1 and 2
    assert np.isclose(out['fraction'], 2 / 3)


def test_fraction_excluding_zero_is_zero_when_every_interval_straddles():
    df = pd.DataFrame({
        'player_id': [1, 2],
        'dep_share': [0.05, -0.05],
        'dep_share_ci_lower': [-0.10, -0.20],
        'dep_share_ci_upper': [0.20, 0.10],
    })
    assert fraction_excluding_zero(df, 'dep_share')['fraction'] == 0.0


def test_pairwise_resolvability_counts_non_overlapping_pairs():
    """Players 1 and 2 are separated; player 3 overlaps 2 but not 1."""
    df = pd.DataFrame({
        'player_id': [1, 2, 3],
        'dep_share': [0.30, -0.20, -0.15],
        'dep_share_ci_lower': [0.20, -0.30, -0.25],
        'dep_share_ci_upper': [0.40, -0.10, -0.05],
    })
    out = pairwise_resolvability(df, 'dep_share')
    assert out['n_pairs'] == 3                    # (1,2), (1,3), (2,3)
    assert out['n_resolved'] == 2                 # (1,2) and (1,3); (2,3) overlap
    assert np.isclose(out['fraction'], 2 / 3)


def test_pairwise_resolvability_is_zero_when_all_intervals_overlap():
    df = pd.DataFrame({
        'player_id': [1, 2, 3],
        'dep_share': [0.10, 0.12, 0.14],
        'dep_share_ci_lower': [0.00, 0.02, 0.04],
        'dep_share_ci_upper': [0.20, 0.22, 0.24],
    })
    assert pairwise_resolvability(df, 'dep_share')['fraction'] == 0.0


def _draws_for(separation, n_boot=500, n_players=4, noise=0.02, seed=0):
    """Draw matrix where player j is centred at j*separation with the given noise."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame({'player_id': np.arange(n_players)})
    draws = np.column_stack([
        rng.normal(j * separation, noise, n_boot) for j in range(n_players)
    ])
    return keys, draws


def test_rank_intervals_are_tight_when_players_are_well_separated():
    keys, draws = _draws_for(separation=1.0, noise=0.01)
    out = rank_intervals(keys, draws)

    # Ranks are 1 = highest. Player 3 is centred highest and never loses its rank.
    assert list(out.sort_values('player_id')['rank_point']) == [4, 3, 2, 1]
    assert (out['rank_ci_lower'] == out['rank_ci_upper']).all(), 'separation should pin every rank'


def test_rank_intervals_widen_when_players_are_indistinguishable():
    keys, draws = _draws_for(separation=0.0, noise=0.5)
    out = rank_intervals(keys, draws)
    spans = out['rank_ci_upper'] - out['rank_ci_lower']
    assert (spans >= 2).all(), f'indistinguishable players need wide rank intervals, got {spans.tolist()}'


def test_rank_intervals_cover_the_point_rank():
    keys, draws = _draws_for(separation=0.05, noise=0.1)
    out = rank_intervals(keys, draws)
    assert (out['rank_ci_lower'] <= out['rank_point']).all()
    assert (out['rank_point'] <= out['rank_ci_upper']).all()


def test_topk_stability_is_one_when_the_top_group_is_unambiguous():
    keys, draws = _draws_for(separation=1.0, noise=0.01)
    assert np.isclose(topk_stability(draws, k=1)['mean_overlap'], 1.0)


def test_topk_stability_degrades_to_chance_when_all_groups_are_equal():
    """With 4 indistinguishable players, a top-2 list retains about half of itself."""
    keys, draws = _draws_for(separation=0.0, noise=0.5)
    out = topk_stability(draws, k=2)
    assert 0.3 < out['mean_overlap'] < 0.75, out['mean_overlap']


def test_topk_stability_rejects_k_larger_than_the_field():
    _, draws = _draws_for(separation=1.0)
    with pytest.raises(ValueError, match='k'):
        topk_stability(draws, k=99)


def test_summary_runs_end_to_end_on_real_shaped_input():
    """Smoke test over the actual pipeline path: shots -> draws -> impact summary."""
    rng = np.random.default_rng(4)
    n = 600
    three = rng.integers(0, 2, n)
    ep = np.where(three == 1, rng.normal(1.3, 0.2, n), rng.normal(1.05, 0.2, n))
    shots = add_shot_dependence(pd.DataFrame({
        'player_id': rng.integers(0, 12, n),
        '3pt': three,
        'expected_points': ep,
        'exp_pts_naive': np.where(three == 1, 1.05, ep),
    }))

    keys, draws = bootstrap_draw_matrix(shots, group=['player_id'], n_boot=200, random_state=0)
    point = pd.DataFrame({
        'player_id': keys['player_id'],
        'dep_share': draws.mean(axis=0),
        'dep_share_ci_lower': np.percentile(draws, 2.5, axis=0),
        'dep_share_ci_upper': np.percentile(draws, 97.5, axis=0),
    })

    summary = ci_impact_summary(point, keys, draws, stat='dep_share', k=5)
    assert set(summary) >= {'fraction_excluding_zero', 'pairwise_resolvability',
                            'topk_overlap', 'median_rank_span', 'n_groups'}
    assert 0 <= summary['pairwise_resolvability'] <= 1
    assert 0 <= summary['fraction_excluding_zero'] <= 1
    assert summary['n_groups'] == 12
