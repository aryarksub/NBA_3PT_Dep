"""What the confidence intervals actually change about the conclusions.

A dependence leaderboard invites the reader to treat the ordering as fact. These functions
quantify how much of that ordering survives the sampling uncertainty the bootstrap measures,
so the manuscript can say "this many players are distinguishable" instead of printing a rank
column and hoping.

Four questions, in the order a referee tends to ask them:

  fraction_excluding_zero  How many players have any dependence at all, signed and
                           distinguishable from none?
  pairwise_resolvability   Can the metric tell two given players apart?
  rank_intervals           What range of league positions is each player consistent with?
  topk_stability           Does the headline top-K list survive resampling?

The last two need the full draw matrix from dependence.bootstrap_draw_matrix, not the
collapsed per-player intervals, because they are comparisons *between* groups and must be
evaluated within a replicate.
"""

import numpy as np
import pandas as pd


def fraction_excluding_zero(point_df, stat='dep_share',
                            lo_col=None, hi_col=None):
    """
    Share of groups whose interval lies entirely on one side of zero.

    Zero is the meaningful null here: a player whose threes are worth exactly their
    two-point alternative has no behavioral dependence in either direction. An interval
    straddling zero means the sign itself is undetermined.
    """
    lo_col = lo_col or f'{stat}_ci_lower'
    hi_col = hi_col or f'{stat}_ci_upper'

    lo = point_df[lo_col].to_numpy(dtype=float)
    hi = point_df[hi_col].to_numpy(dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi)

    excludes = ok & ((lo > 0) | (hi < 0))
    n = int(ok.sum())
    return {
        'n_groups': n,
        'n_excluding_zero': int(excludes.sum()),
        'fraction': float(excludes.sum() / n) if n else np.nan,
    }


def pairwise_resolvability(point_df, stat='dep_share', lo_col=None, hi_col=None):
    """
    Share of group pairs whose intervals do not overlap.

    Non-overlapping intervals are a conservative test of "these two differ" -- more
    conservative than a direct test of the difference -- but they are exactly what a reader
    does when looking at a caterpillar plot, so this measures the plot as it will be read.

    A value near 0 means the ranking is decorative: almost no two players can be separated.
    """
    lo_col = lo_col or f'{stat}_ci_lower'
    hi_col = hi_col or f'{stat}_ci_upper'

    sub = point_df[[lo_col, hi_col]].dropna()
    lo = sub[lo_col].to_numpy(dtype=float)
    hi = sub[hi_col].to_numpy(dtype=float)
    n = len(lo)
    if n < 2:
        return {'n_pairs': 0, 'n_resolved': 0, 'fraction': np.nan}

    # pair (i, j) resolved when one interval lies entirely above the other
    disjoint = (lo[:, None] > hi[None, :]) | (hi[:, None] < lo[None, :])
    iu = np.triu_indices(n, k=1)
    resolved = int(disjoint[iu].sum())
    n_pairs = n * (n - 1) // 2

    return {
        'n_pairs': n_pairs,
        'n_resolved': resolved,
        'fraction': resolved / n_pairs,
    }


def rank_intervals(keys_df, draws, alpha=0.05):
    """
    The range of league ranks each group is consistent with.

    Ranks are recomputed inside every bootstrap replicate and then summarized, which is the
    only way to get an honest rank interval: ranking the point estimates and attaching each
    group's own interval to it would ignore that everyone else moves too.

    Rank 1 is the highest value of the statistic. Returns rank_point (rank of the median
    draw) alongside the interval endpoints.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError(f'draws must be 2-D (n_boot x n_groups), got shape {draws.shape}')

    n_boot, n_groups = draws.shape
    if n_groups != len(keys_df):
        raise ValueError(f'{n_groups} draw columns but {len(keys_df)} group keys')

    # rank within each replicate; descending so rank 1 is the largest
    order = (-draws).argsort(axis=1, kind='stable')
    ranks = np.empty_like(order)
    rows = np.arange(n_boot)[:, None]
    ranks[rows, order] = np.arange(1, n_groups + 1)[None, :]

    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    med = np.median(draws, axis=0)
    point_rank = (-med).argsort(kind='stable').argsort() + 1

    out = keys_df.copy().reset_index(drop=True)
    out['rank_point'] = point_rank
    out['rank_ci_lower'] = np.ceil(np.percentile(ranks, lo_pct, axis=0)).astype(int)
    out['rank_ci_upper'] = np.floor(np.percentile(ranks, hi_pct, axis=0)).astype(int)
    out['rank_span'] = out['rank_ci_upper'] - out['rank_ci_lower'] + 1
    return out


def topk_stability(draws, k=10):
    """
    How much of the point-estimate top-K survives resampling, on average.

    For each replicate, take that replicate's top K and measure its overlap with the top K
    of the median estimates. An overlap near 1 means the headline list is solid; an overlap
    near k/n_groups means it is no better than picking K groups at random.
    """
    draws = np.asarray(draws, dtype=float)
    n_boot, n_groups = draws.shape
    if not 1 <= k <= n_groups:
        raise ValueError(f'k must be between 1 and the number of groups ({n_groups}), got {k}')

    med = np.median(draws, axis=0)
    reference = set((-med).argsort(kind='stable')[:k].tolist())

    top_per_replicate = (-draws).argsort(axis=1, kind='stable')[:, :k]
    overlaps = np.array([len(reference & set(row.tolist())) / k for row in top_per_replicate])

    return {
        'k': k,
        'mean_overlap': float(overlaps.mean()),
        'chance_overlap': k / n_groups,
    }


def ci_impact_summary(point_df, keys_df, draws, stat='dep_share', k=10, alpha=0.05):
    """One dict of every headline number, for writing straight into a results table."""
    zero = fraction_excluding_zero(point_df, stat)
    pairs = pairwise_resolvability(point_df, stat)
    ranks = rank_intervals(keys_df, draws, alpha=alpha)
    k_eff = min(k, draws.shape[1])
    top = topk_stability(draws, k=k_eff)

    return {
        'n_groups': zero['n_groups'],
        'fraction_excluding_zero': zero['fraction'],
        'n_excluding_zero': zero['n_excluding_zero'],
        'pairwise_resolvability': pairs['fraction'],
        'n_pairs_resolved': pairs['n_resolved'],
        'n_pairs': pairs['n_pairs'],
        'median_rank_span': float(ranks['rank_span'].median()),
        'max_rank_span': int(ranks['rank_span'].max()),
        'topk_k': top['k'],
        'topk_overlap': top['mean_overlap'],
        'topk_chance_overlap': top['chance_overlap'],
    }
