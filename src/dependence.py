"""Behavioral (counterfactual) three-point dependence.

Behavioral dependence is the offensive value a player or team would forgo if each of their
three-point attempts were replaced by a plausible non-three alternative. At shot level it is
observed expected points minus counterfactual expected points; two-point attempts are their
own counterfactual and contribute exactly zero.

The expected-points column must be cross-fitted (see cross_fit.py). Feeding in-sample
predictions to these functions produces numbers that are not interpretable.
"""

import numpy as np
import pandas as pd

SHOT_DELTA_COL = 'dep_delta'

OBS_COL = 'expected_points'
CF_COL = 'exp_pts_naive'
THREE_COL = '3pt'

STATS = ['dep_total', 'dep_per_shot', 'dep_per_3pa', 'dep_share']


def add_shot_dependence(df, obs_col=OBS_COL, cf_col=CF_COL):
    """Shot-level dependence: observed EP minus counterfactual EP."""
    out = df.copy()
    out[SHOT_DELTA_COL] = out[obs_col] - out[cf_col]
    return out


def aggregate_dependence(df, group, delta_col=SHOT_DELTA_COL, obs_col=OBS_COL,
                         three_col=THREE_COL):
    """
    Aggregate shot-level dependence to the given grouping keys.

    `group` must be a list of column names, e.g. ['player_id'] or ['game_id', 'team_id'].

    Returns one row per group with four normalizations:
      dep_total     total expected points dependent on three-point selection
      dep_per_shot  dep_total spread over every shot attempt
      dep_per_3pa   average value added per three attempted
      dep_share     fraction of total observed expected points that is dependent  <- primary
    """
    if not isinstance(group, list):
        raise TypeError(f'group must be a list of column names, got {type(group).__name__}')

    out = (
        df.groupby(group)
        .agg(
            n_shots=(delta_col, 'size'),
            n_3pa=(three_col, 'sum'),
            ep_obs_total=(obs_col, 'sum'),
            dep_total=(delta_col, 'sum'),
        )
        .reset_index()
    )

    out['3pa_rate'] = out['n_3pa'] / out['n_shots']
    out['dep_per_shot'] = out['dep_total'] / out['n_shots']
    out['dep_per_3pa'] = np.where(out['n_3pa'] > 0, out['dep_total'] / out['n_3pa'], np.nan)
    out['dep_share'] = np.where(out['ep_obs_total'] > 0,
                                out['dep_total'] / out['ep_obs_total'], np.nan)
    return out


def _stat_from_arrays(delta, obs, three, stat):
    if stat == 'dep_total':
        return delta.sum()
    if stat == 'dep_per_shot':
        return delta.mean()
    if stat == 'dep_per_3pa':
        n3 = three.sum()
        return delta.sum() / n3 if n3 > 0 else np.nan
    if stat == 'dep_share':
        denom = obs.sum()
        return delta.sum() / denom if denom > 0 else np.nan
    raise ValueError(f'unknown stat {stat!r}, expected one of {STATS}')


def bootstrap_dependence(df, group, stat='dep_share', n_boot=1000, alpha=0.05,
                         random_state=0, delta_col=SHOT_DELTA_COL, obs_col=OBS_COL,
                         three_col=THREE_COL):
    """
    Percentile bootstrap CI for a dependence statistic, resampling shots within each group.

    Bootstrap rather than a closed-form SE because every statistic here is a ratio of two
    sums over the same random shot set, with a random-size numerator (only threes contribute)
    and correlated numerator and denominator. The delta-method variance for that is an
    approximation that degrades exactly where it is needed most: players near the 3PA
    qualification floor. The per-shot terms are also a zero-inflated mixture -- twos
    contribute exactly 0 -- so no parametric family describes them. The bootstrap assumes
    only exchangeability of shots within the group.

    Endpoints are percentiles of the bootstrap draws, not point +/- 1.96*SE, so the interval
    inherits the statistic's skew instead of being forced symmetric. The SE is returned for
    reporting, not used to build the interval.

    Resampling is within-group because the inferential target is the player's shot-selection
    policy: what dependence would look like if this player played the same season again.
    It is not an interval about the historical realization, which is known exactly.

    Not covered by this interval: uncertainty in the cross-fitted probabilities (treated as
    fixed), uncertainty in the merged ep_2pt baseline, and within-game correlation between
    shots (so the interval is somewhat narrow). See docs/behavioral_dependence.md.

    Returns one row per group with <stat>_boot_mean, _se, _ci_lower, _ci_upper.
    """
    if not isinstance(group, list):
        raise TypeError(f'group must be a list of column names, got {type(group).__name__}')

    rng = np.random.default_rng(random_state)
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []

    for key, sub in df.groupby(group):
        delta = sub[delta_col].to_numpy()
        obs = sub[obs_col].to_numpy()
        three = sub[three_col].to_numpy()
        n = len(sub)

        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            draws[b] = _stat_from_arrays(delta[idx], obs[idx], three[idx], stat)

        key_vals = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group, key_vals)),
            f'{stat}_boot_mean': np.nanmean(draws),
            f'{stat}_se': np.nanstd(draws, ddof=1),
            f'{stat}_ci_lower': np.nanpercentile(draws, lo_pct),
            f'{stat}_ci_upper': np.nanpercentile(draws, hi_pct),
        })

    return pd.DataFrame(rows)


def bootstrap_draw_matrix(df, group, stat='dep_share', n_boot=1000, random_state=0,
                          delta_col=SHOT_DELTA_COL, obs_col=OBS_COL, three_col=THREE_COL):
    """
    The raw bootstrap draws, as a (n_boot x n_groups) matrix, plus the group keys.

    bootstrap_dependence collapses each group's draws to four summary numbers immediately,
    which is all a per-player interval needs. Any question about how groups compare -- rank
    intervals, how often A really outranks B, whether a top-10 survives resampling -- needs
    every group's value *within the same replicate*, because those comparisons are made
    replicate by replicate. That is what this returns.

    Resampling is still independent across groups within a replicate: two players' shot
    samples are independent, so their draws should be too. What matters is that replicate b
    holds one simultaneous realization of the whole field, which can then be ranked.

    Returns (keys, draws) where keys is a DataFrame of the grouping columns in column order
    of draws, and draws[b, j] is the statistic for group j in replicate b.
    """
    if not isinstance(group, list):
        raise TypeError(f'group must be a list of column names, got {type(group).__name__}')

    rng = np.random.default_rng(random_state)
    keys, cols = [], []

    for key, sub in df.groupby(group):
        delta = sub[delta_col].to_numpy()
        obs = sub[obs_col].to_numpy()
        three = sub[three_col].to_numpy()
        n = len(sub)

        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            draws[b] = _stat_from_arrays(delta[idx], obs[idx], three[idx], stat)

        keys.append(key if isinstance(key, tuple) else (key,))
        cols.append(draws)

    keys_df = pd.DataFrame(keys, columns=group)
    return keys_df, np.column_stack(cols) if cols else np.empty((n_boot, 0))


def clustered_bootstrap_dependence(df, group, cluster_col='game_id', stat='dep_share',
                                   n_boot=1000, alpha=0.05, random_state=0,
                                   delta_col=SHOT_DELTA_COL, obs_col=OBS_COL,
                                   three_col=THREE_COL):
    """
    Bootstrap that resamples whole games instead of individual shots.

    Sensitivity check for the exchangeability assumption in bootstrap_dependence. Shots in one
    game share defense, pace, and game state, so treating them as independent understates the
    interval. This version resamples clusters, which respects that correlation -- at the cost
    of a small and therefore unstable cluster count for low-volume players. Report both; if
    they disagree materially, the clustered interval is the conservative one to quote.

    A group observed in a single cluster yields a zero-width interval, because resampling one
    cluster with replacement always returns that same cluster. The reported <stat>_n_clusters
    column exists so those degenerate rows can be identified and excluded rather than read as
    implausibly precise.
    """
    if not isinstance(group, list):
        raise TypeError(f'group must be a list of column names, got {type(group).__name__}')

    rng = np.random.default_rng(random_state)
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []

    for key, sub in df.groupby(group):
        blocks = [g for _, g in sub.groupby(cluster_col)]
        n_clusters = len(blocks)
        delta_b = [b[delta_col].to_numpy() for b in blocks]
        obs_b = [b[obs_col].to_numpy() for b in blocks]
        three_b = [b[three_col].to_numpy() for b in blocks]

        draws = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, n_clusters, n_clusters)
            draws[b] = _stat_from_arrays(
                np.concatenate([delta_b[i] for i in pick]),
                np.concatenate([obs_b[i] for i in pick]),
                np.concatenate([three_b[i] for i in pick]),
                stat,
            )

        key_vals = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group, key_vals)),
            f'{stat}_n_clusters': n_clusters,
            f'{stat}_clustered_ci_lower': np.nanpercentile(draws, lo_pct),
            f'{stat}_clustered_ci_upper': np.nanpercentile(draws, hi_pct),
        })

    return pd.DataFrame(rows)


def signal_to_noise(df, stat_col, se_col):
    """
    How much of the observed between-player spread is real rather than sampling noise.

    Decomposes the cross-player variance of a statistic into a true-heterogeneity component
    (tau^2) and the average within-player sampling variance already measured by the bootstrap:

        var(observed spread) = tau^2 + mean(se^2)

    Reliability = tau^2 / var(observed) is the fraction of the leaderboard's spread that is
    signal. Near 0 means the ranking is noise and the metric distinguishes nobody; near 1 means
    the spread is real heterogeneity in three-point dependence, which is the project's thesis.

    This is the diagnostic half of the empirical-Bayes machinery, deliberately kept while the
    shrinkage itself is not: it measures whether outliers are real without moving any estimate
    toward the mean. It is a population-level summary and must never be applied per player.
    """
    theta = df[stat_col].to_numpy(dtype=float)
    se = df[se_col].to_numpy(dtype=float)

    observed_var = float(np.nanvar(theta, ddof=1)) if len(theta) > 1 else np.nan
    sampling_var = float(np.nanmean(se ** 2)) if len(se) else np.nan
    tau2 = max(observed_var - sampling_var, 0.0) if np.isfinite(observed_var) else np.nan

    return {
        'n_groups': int(len(theta)),
        'observed_var': observed_var,
        'mean_sampling_var': sampling_var,
        'tau2': tau2,
        'tau': np.sqrt(tau2) if np.isfinite(tau2) else np.nan,
        'reliability': tau2 / observed_var if np.isfinite(tau2) and observed_var > 0 else np.nan,
    }


def calibration_table(y_true, p_hat, n_bins=10):
    """Reliability table: predicted probability vs realized make rate, by predicted-probability decile."""
    tbl = pd.DataFrame({'y': np.asarray(y_true, dtype=float), 'p': np.asarray(p_hat, dtype=float)})
    tbl['bin'] = pd.qcut(tbl['p'], n_bins, labels=False, duplicates='drop')
    return (
        tbl.groupby('bin')
        .agg(n=('y', 'size'), p_mean=('p', 'mean'), make_rate=('y', 'mean'))
        .reset_index()
    )


def brier_score(y_true, p_hat):
    """Mean squared error of the probability forecast. Lower is better."""
    return float(np.mean((np.asarray(p_hat, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))
