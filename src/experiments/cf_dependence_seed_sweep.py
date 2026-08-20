"""How much does the dependence metric move when only the cross-fitting fold split changes?

The bootstrap in cf_dependence.py holds the cross-fitted probabilities fixed, so it cannot see
uncertainty coming from the expected-points model itself. Refitting under several fold seeds and
measuring the spread of dep_share gives a cheap lower bound on that component.

The per-player ep_2pt baseline is recomputed from each seed's own expected points. Reading it
from data/cf_dep_player.csv instead would pin every seed to whatever run last wrote that file,
which silently understates the spread this script exists to measure.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import (
    add_alt_exp_pts_cols,
    add_exp_pts_col,
    build_base_inputs,
    create_base_cf_dep_df,
)
from cross_fit import make_splits
from dependence import add_shot_dependence, aggregate_dependence


def distinct_split_patterns(df, seeds, out_feature='fgm', group_col='game_id', n_splits=5):
    """How many genuinely different fold partitions the requested seeds produce.

    StratifiedGroupKFold assigns whole groups greedily. When the number of groups is close to
    the number of folds -- 8 games into 5 folds, in the current sample -- the partition is
    fully determined and shuffling has no room to change it, so every seed yields the same
    split. This must be checked before a spread of zero is read as 'no model uncertainty'.
    """
    patterns = set()
    for seed in seeds:
        splits = make_splits(df, out_feature, group_col, n_splits, seed)
        patterns.add(tuple(
            tuple(sorted(set(df[group_col].iloc[test_idx]))) for _, test_idx in splits
        ))
    return len(patterns)

SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = 'results'
OUT_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_seed_sweep.csv')

MIN_SHOTS = 10
MIN_3PA = 3

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


if __name__ == '__main__':
    base_df, features, params = build_base_inputs()

    n_patterns = distinct_split_patterns(base_df, SEEDS)
    n_games = base_df['game_id'].nunique()
    print(f'{n_games} games, {len(SEEDS)} seeds -> {n_patterns} distinct fold partition(s)')
    if n_patterns == 1:
        print(
            'WARNING: every seed produces the identical fold split, so the spread below will\n'
            '         be exactly zero and measures NOTHING about model uncertainty. With only\n'
            f'        {n_games} games, StratifiedGroupKFold has no freedom to shuffle. This\n'
            '         diagnostic becomes informative once the sample covers more games.'
        )

    frames = []
    for seed in SEEDS:
        print(f'--- cross-fitting seed {seed} ---')
        ep = add_exp_pts_col(base_df, features, 'fgm', params, random_state=seed)

        # Baseline recomputed from this seed's expected points, not read from disk.
        baseline = create_base_cf_dep_df(ep, group=['player_id'])
        ep = add_alt_exp_pts_cols(ep, baseline_df=baseline)

        agg = aggregate_dependence(add_shot_dependence(ep), group=['player_id'])
        agg = agg[(agg['n_shots'] >= MIN_SHOTS) & (agg['n_3pa'] >= MIN_3PA)]
        frames.append(agg[['player_id', 'dep_share']].assign(seed=seed))

    wide = pd.concat(frames).pivot(index='player_id', columns='seed', values='dep_share')
    spread = pd.DataFrame({
        'n_seeds': wide.notna().sum(axis=1),
        'dep_share_seed_mean': wide.mean(axis=1),
        'dep_share_seed_sd': wide.std(axis=1, ddof=1),
        'dep_share_seed_range': wide.max(axis=1) - wide.min(axis=1),
    }).reset_index()
    spread.to_csv(OUT_FILE, index=False)

    print(f'\n{len(spread)} players qualified under at least one seed')
    print(f'Median across-seed SD of dep_share:    {spread["dep_share_seed_sd"].median():.5f}')
    print(f'Median across-seed range of dep_share: {spread["dep_share_seed_range"].median():.5f}')

    player_file = os.path.join(RESULTS_DIR, 'cf_dependence_player.csv')
    if os.path.exists(player_file):
        boot_se = pd.read_csv(player_file)
        boot_se = boot_se[boot_se['qualified']]['dep_share_se'].median()
        seed_sd = spread['dep_share_seed_sd'].median()
        print(f'Median bootstrap SE (shot sampling):   {boot_se:.5f}')
        print(f'Ratio seed SD / bootstrap SE:          {seed_sd / boot_se:.2f}')
        if n_patterns == 1:
            print(
                '\nThat ratio is zero only because the fold split cannot vary at this sample\n'
                'size (see the warning above). It is not evidence that the expected-points\n'
                'model contributes no uncertainty.'
            )
        else:
            print(
                '\nA ratio near or above 1 means fold choice moves the metric as much as the shot\n'
                'sample does, so the bootstrap interval alone understates total uncertainty.'
            )

    print(f'\nWritten to {OUT_FILE}')
