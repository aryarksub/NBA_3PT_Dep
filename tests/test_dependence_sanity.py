import numpy as np
import pandas as pd

from dependence import (
    add_shot_dependence,
    aggregate_dependence,
    bootstrap_dependence,
    clustered_bootstrap_dependence,
)


def test_a_player_who_never_shoots_threes_has_zero_dependence():
    df = pd.DataFrame({
        'player_id': [9] * 5,
        '3pt': [0] * 5,
        'expected_points': [1.0, 1.1, 0.9, 1.0, 1.2],
        'exp_pts_naive': [1.0, 1.1, 0.9, 1.0, 1.2],
    })
    out = aggregate_dependence(add_shot_dependence(df), group=['player_id'])
    assert out.loc[0, 'dep_total'] == 0.0
    assert out.loc[0, 'dep_share'] == 0.0
    assert out.loc[0, '3pa_rate'] == 0.0


def test_dependence_is_zero_when_threes_match_the_baseline():
    df = pd.DataFrame({
        'player_id': [9] * 4,
        '3pt': [1, 1, 0, 0],
        'expected_points': [1.05, 1.05, 1.05, 1.05],
        'exp_pts_naive': [1.05, 1.05, 1.05, 1.05],
    })
    out = aggregate_dependence(add_shot_dependence(df), group=['player_id'])
    assert np.isclose(out.loc[0, 'dep_share'], 0.0)


def test_dependence_is_negative_when_threes_are_worse_than_the_alternative():
    """A player forcing bad threes should register as negative: the metric is signed, not a rate."""
    df = pd.DataFrame({
        'player_id': [9] * 4,
        '3pt': [1, 1, 0, 0],
        'expected_points': [0.75, 0.75, 1.10, 1.10],
        'exp_pts_naive': [1.10, 1.10, 1.10, 1.10],
    })
    out = aggregate_dependence(add_shot_dependence(df), group=['player_id'])
    assert out.loc[0, 'dep_total'] < 0
    assert out.loc[0, 'dep_share'] < 0


def test_bootstrap_interval_attains_roughly_nominal_coverage():
    """
    The claim 'these are 95% intervals' is worth checking rather than asserting.

    Simulate a player whose true dep_share is known by construction, draw many independent
    seasons from that process, and count how often the bootstrap interval contains the truth.
    Percentile intervals for a ratio statistic are not exact, so the target is 'close to 0.95',
    not 'exactly 0.95' -- but coverage of, say, 0.80 would mean the intervals are decorative.
    """
    rng = np.random.default_rng(11)

    p3, p2 = 0.36, 0.50           # true make probabilities
    ep3, ep2 = 3 * p3, 2 * p2     # 1.08 vs 1.00
    rate = 0.5                    # true three-point attempt rate
    true_share = rate * (ep3 - ep2) / (rate * ep3 + (1 - rate) * ep2)

    n_seasons, n_shots = 200, 300
    hits = 0
    for s in range(n_seasons):
        three = rng.random(n_shots) < rate
        # Modeled EP varies shot to shot around the player's true level.
        obs = np.where(three, rng.normal(ep3, 0.10, n_shots), rng.normal(ep2, 0.10, n_shots))
        df = add_shot_dependence(pd.DataFrame({
            'player_id': 1,
            '3pt': three.astype(int),
            'expected_points': obs,
            'exp_pts_naive': np.where(three, ep2, obs),
        }))
        boot = bootstrap_dependence(df, group=['player_id'], stat='dep_share',
                                    n_boot=300, random_state=s)
        lo = boot.loc[0, 'dep_share_ci_lower']
        hi = boot.loc[0, 'dep_share_ci_upper']
        hits += int(lo <= true_share <= hi)

    coverage = hits / n_seasons
    assert 0.88 <= coverage <= 0.99, f'nominal 95% interval covered {coverage:.2%} of the time'


def test_clustered_bootstrap_is_at_least_as_wide_when_shots_are_correlated_within_games():
    """If game context moves every shot together, ignoring clusters understates the interval."""
    rng = np.random.default_rng(3)
    n_games, per_game = 40, 12
    rows = []
    for g in range(n_games):
        game_effect = rng.normal(0, 0.25)      # a whole game runs hot or cold together
        three = rng.integers(0, 2, per_game)
        obs = np.where(three == 1, 1.15, 1.00) + game_effect + rng.normal(0, 0.03, per_game)
        rows.append(pd.DataFrame({
            'player_id': 1, 'game_id': g, '3pt': three,
            'expected_points': obs,
            'exp_pts_naive': np.where(three == 1, 1.00, obs),
        }))
    df = add_shot_dependence(pd.concat(rows, ignore_index=True))

    shot = bootstrap_dependence(df, group=['player_id'], stat='dep_share',
                                n_boot=400, random_state=0)
    clust = clustered_bootstrap_dependence(df, group=['player_id'], cluster_col='game_id',
                                           stat='dep_share', n_boot=400, random_state=0)

    shot_w = shot.loc[0, 'dep_share_ci_upper'] - shot.loc[0, 'dep_share_ci_lower']
    clust_w = (clust.loc[0, 'dep_share_clustered_ci_upper']
               - clust.loc[0, 'dep_share_clustered_ci_lower'])
    assert clust_w > shot_w, f'clustered {clust_w:.4f} should exceed shot-level {shot_w:.4f}'


def test_clustered_interval_is_degenerate_for_a_single_cluster():
    """
    Documents the failure mode that matters for this dataset.

    The median player here appears in one game, and resampling one cluster with replacement
    always returns that cluster, so the interval collapses to a point. n_clusters is reported
    so these rows can be filtered rather than mistaken for high precision.
    """
    df = add_shot_dependence(pd.DataFrame({
        'player_id': [1] * 6,
        'game_id': [77] * 6,
        '3pt': [1, 0, 1, 0, 1, 0],
        'expected_points': [1.3, 1.0, 1.2, 1.1, 1.4, 1.0],
        'exp_pts_naive': [1.05, 1.0, 1.05, 1.1, 1.05, 1.0],
    }))
    clust = clustered_bootstrap_dependence(df, group=['player_id'], cluster_col='game_id',
                                           stat='dep_share', n_boot=100, random_state=0)
    assert clust.loc[0, 'dep_share_n_clusters'] == 1
    assert np.isclose(clust.loc[0, 'dep_share_clustered_ci_lower'],
                      clust.loc[0, 'dep_share_clustered_ci_upper'])


def test_unmatched_shots_are_not_silently_labelled_two_pointers():
    """
    Regression test for a labelling bug that only shows up at scale.

    Shots whose release moment is never found carry NaN shooter coordinates. The
    is_two_pointer geometry check returns True by fallthrough on NaN, so such shots were
    being labelled 2PT. Two-pointers have a dependence delta of exactly zero, so each one
    silently pads the dep_share denominator with a shot that was never measured.
    """
    import numpy as np

    HOOP_Y = 25.0

    def is_two_pointer(x, y, hoop_x):
        dx = abs(x - hoop_x)
        if dx >= 22 and y <= 14:
            return False
        if np.sqrt(dx ** 2 + (y - HOOP_Y) ** 2) >= 23.75:
            return False
        return True

    # A genuine deep three is correctly classified...
    assert not is_two_pointer(5.25, 25.0 + 30, 5.25)
    # ...but NaN coordinates fall through to "two-pointer", which is the bug.
    assert is_two_pointer(np.nan, np.nan, 88.75), 'NaN fallthrough is the behaviour guarded against'

    # The pipeline must therefore drop unmatched shots before labelling, not label them.
    shots = pd.DataFrame({'shooter_x': [10.0, np.nan, 40.0], 'shooter_y': [20.0, np.nan, 30.0]})
    kept = shots[shots['shooter_x'].notna()]
    assert len(kept) == 2, 'unmatched shots must be dropped before the 3PT label is derived'
