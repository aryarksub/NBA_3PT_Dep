import numpy as np
import pandas as pd

from dependence import (
    SHOT_DELTA_COL,
    add_shot_dependence,
    aggregate_dependence,
    bootstrap_dependence,
    signal_to_noise,
)


def toy_frame():
    """Two players. Player 1 takes two threes worth 1.5 EP against a 1.0 EP alternative."""
    return pd.DataFrame({
        'player_id': [1, 1, 1, 1, 2, 2, 2, 2],
        '3pt':       [1, 1, 0, 0, 1, 0, 0, 0],
        'expected_points':  [1.5, 1.5, 1.0, 1.0, 1.2, 1.1, 1.1, 1.1],
        'exp_pts_naive':    [1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1],
    })


def test_delta_is_zero_for_two_point_shots():
    out = add_shot_dependence(toy_frame())
    twos = out[out['3pt'] == 0]
    assert (twos[SHOT_DELTA_COL] == 0).all()


def test_delta_is_observed_minus_counterfactual():
    out = add_shot_dependence(toy_frame())
    assert out.loc[0, SHOT_DELTA_COL] == 0.5


def test_aggregate_computes_all_four_normalizations():
    out = aggregate_dependence(add_shot_dependence(toy_frame()), group=['player_id'])
    p1 = out[out['player_id'] == 1].iloc[0]

    assert p1['n_shots'] == 4
    assert p1['n_3pa'] == 2
    assert p1['dep_total'] == 1.0                     # 0.5 + 0.5
    assert p1['dep_per_shot'] == 0.25                 # 1.0 / 4
    assert p1['dep_per_3pa'] == 0.5                   # 1.0 / 2
    assert np.isclose(p1['dep_share'], 1.0 / 5.0)     # 1.0 / (1.5+1.5+1.0+1.0)
    assert p1['3pa_rate'] == 0.5


def test_dep_per_3pa_is_nan_when_player_took_no_threes():
    df = add_shot_dependence(toy_frame())
    df = df[df['3pt'] == 0]
    out = aggregate_dependence(df, group=['player_id'])
    assert out['dep_per_3pa'].isna().all()


def test_bootstrap_ci_brackets_the_point_estimate():
    df = add_shot_dependence(toy_frame())
    point = aggregate_dependence(df, group=['player_id'])
    boot = bootstrap_dependence(df, group=['player_id'], stat='dep_share',
                                n_boot=200, random_state=0)
    merged = point.merge(boot, on='player_id')

    assert (merged['dep_share_ci_lower'] <= merged['dep_share']).all()
    assert (merged['dep_share'] <= merged['dep_share_ci_upper']).all()
    assert (merged['dep_share_ci_lower'] <= merged['dep_share_ci_upper']).all()


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    df = add_shot_dependence(toy_frame())
    a = bootstrap_dependence(df, group=['player_id'], n_boot=100, random_state=7)
    b = bootstrap_dependence(df, group=['player_id'], n_boot=100, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_interval_narrows_as_the_shot_sample_grows():
    """The bootstrap must actually respond to sample size, not just return a plausible band."""
    rng = np.random.default_rng(0)

    def player(n, pid):
        three = rng.integers(0, 2, n)
        obs = np.where(three == 1, rng.normal(1.35, 0.25, n), rng.normal(1.05, 0.20, n))
        return pd.DataFrame({
            'player_id': pid,
            '3pt': three,
            'expected_points': obs,
            'exp_pts_naive': np.where(three == 1, 1.05, obs),
        })

    df = add_shot_dependence(pd.concat([player(40, 1), player(2000, 2)], ignore_index=True))
    boot = bootstrap_dependence(df, group=['player_id'], stat='dep_share',
                                n_boot=400, random_state=0)
    width = boot['dep_share_ci_upper'] - boot['dep_share_ci_lower']

    assert width.iloc[0] > 3 * width.iloc[1], (
        f'small-sample interval must be far wider, got {width.iloc[0]:.4f} vs {width.iloc[1]:.4f}'
    )


def test_a_player_with_no_dependence_gets_an_interval_covering_zero():
    """Negative control for the interval itself, not just for the point estimate."""
    n = 400
    rng = np.random.default_rng(1)
    ep = rng.normal(1.1, 0.15, n)
    df = add_shot_dependence(pd.DataFrame({
        'player_id': 1,
        '3pt': rng.integers(0, 2, n),
        'expected_points': ep,
        'exp_pts_naive': ep,
    }))
    boot = bootstrap_dependence(df, group=['player_id'], stat='dep_share',
                                n_boot=400, random_state=0)
    assert boot.loc[0, 'dep_share_ci_lower'] <= 0 <= boot.loc[0, 'dep_share_ci_upper']


def test_reliability_separates_a_real_spread_from_a_noise_spread():
    """
    Same observed spread, different sampling error: reliability must tell them apart.

    Population A's spread is far larger than its players' sampling noise, so it is real.
    Population B has the identical spread but sampling noise as large as the spread itself,
    so almost none of it is signal.
    """
    values = [0.05, 0.10, 0.15, 0.20, 0.25]

    real = pd.DataFrame({'dep_share': values, 'dep_share_se': [0.005] * 5})
    noise = pd.DataFrame({'dep_share': values, 'dep_share_se': [0.08] * 5})

    a = signal_to_noise(real, 'dep_share', 'dep_share_se')
    b = signal_to_noise(noise, 'dep_share', 'dep_share_se')

    assert a['reliability'] > 0.9, f'clean spread should be nearly all signal, got {a["reliability"]:.3f}'
    assert b['reliability'] < 0.2, f'noisy spread should be nearly all noise, got {b["reliability"]:.3f}'
    assert b['tau2'] >= 0, 'tau^2 is floored at zero and never negative'
    assert np.isclose(a['observed_var'], b['observed_var']), 'observed spread is identical by construction'
