"""How much of the dependence leaderboard survives its own confidence intervals.

Reads the player results produced by cf_dependence.py, recomputes the bootstrap as a full
draw matrix (needed for between-player comparisons), and reports what the intervals change
about the conclusions: how many players have a detectable sign, how many pairs can be told
apart, how wide each player's plausible rank range is, and whether the top-K list is stable.

Run after src/experiments/cf_dependence.py.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE
from ci_impact import ci_impact_summary, rank_intervals
from dependence import add_shot_dependence, bootstrap_draw_matrix
import plots

RESULTS_DIR = 'results'
CF_DEP_PLOTS_DIR = os.path.join(plots.PLOTS_DIR, 'cf_dependence')
PLAYER_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_player.csv')

SUMMARY_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_ci_impact.csv')
RANKS_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_rank_intervals.csv')

N_BOOT = 1000
PRIMARY_STAT = 'dep_share'
TOP_K = 10

for directory in (RESULTS_DIR, CF_DEP_PLOTS_DIR):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_rank_intervals(ranks, fname, label_col='player_id', top_n=25):
    """Each player's plausible league position. Overlap here is the honest picture."""
    sub = ranks.sort_values('rank_point').head(top_n)
    y = np.arange(len(sub))
    lo = sub['rank_point'] - sub['rank_ci_lower']
    hi = sub['rank_ci_upper'] - sub['rank_point']

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(sub) + 2))
    ax.errorbar(sub['rank_point'], y, xerr=[lo, hi], fmt='o', color='#3b4a7a',
                ecolor='#9aa3bd', capsize=3, markersize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub[label_col].astype(str), fontsize=8)
    ax.set_xlabel('League rank by behavioral dependence (1 = highest)', fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    if not os.path.exists(PLAYER_FILE):
        raise FileNotFoundError(
            f'{PLAYER_FILE} not found. Run src/experiments/cf_dependence.py first.'
        )

    player = pd.read_csv(PLAYER_FILE)
    qualified = player[player['qualified']].copy()
    print(f'{len(qualified)} qualified players from {PLAYER_FILE}')

    shots = add_shot_dependence(pd.read_csv(ALT_SHOTS_EXP_PTS_FILE))
    shots = shots[shots['player_id'].isin(qualified['player_id'])]

    print(f'Recomputing bootstrap as a draw matrix ({N_BOOT} replicates)')
    keys, draws = bootstrap_draw_matrix(shots, group=['player_id'], stat=PRIMARY_STAT,
                                        n_boot=N_BOOT, random_state=0)

    # align the stored point estimates to the draw-matrix column order
    point = keys.merge(qualified, on='player_id', how='left')

    ranks = rank_intervals(keys, draws)
    ranks = ranks.merge(point[['player_id', PRIMARY_STAT]], on='player_id', how='left')
    ranks.sort_values('rank_point').to_csv(RANKS_FILE, index=False)

    summary = ci_impact_summary(point, keys, draws, stat=PRIMARY_STAT, k=TOP_K)
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    n = summary['n_groups']
    print('\n--- What the confidence intervals change ---')
    print(f'  players analysed                 {n}')
    print(f'  sign detectable (CI excludes 0)  {summary["n_excluding_zero"]}/{n} '
          f'= {summary["fraction_excluding_zero"]:.1%}')
    print(f'  player pairs separable           {summary["n_pairs_resolved"]}/{summary["n_pairs"]} '
          f'= {summary["pairwise_resolvability"]:.1%}')
    print(f'  median plausible rank span       {summary["median_rank_span"]:.0f} of {n} positions')
    print(f'  widest rank span                 {summary["max_rank_span"]} of {n} positions')
    print(f'  top-{summary["topk_k"]} list retained on resampling  '
          f'{summary["topk_overlap"]:.1%} (chance = {summary["topk_chance_overlap"]:.1%})')

    plot_rank_intervals(ranks, os.path.join(CF_DEP_PLOTS_DIR, 'player_rank_intervals.png'))

    print(f'\nWritten {SUMMARY_FILE}, {RANKS_FILE}')
    print(f'Plot written to {CF_DEP_PLOTS_DIR}/player_rank_intervals.png')
