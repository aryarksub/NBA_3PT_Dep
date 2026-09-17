import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE
from dependence import (
    add_shot_dependence,
    aggregate_dependence,
    bootstrap_dependence,
    clustered_bootstrap_dependence,
    signal_to_noise,
)

RESULTS_DIR = 'results'

RESULTS_FILES = {
    'player': os.path.join(RESULTS_DIR, 'cf_dependence_player.csv'),
    'team': os.path.join(RESULTS_DIR, 'cf_dependence_team.csv'),
    'game': os.path.join(RESULTS_DIR, 'cf_dependence_game.csv'),
}

# Reporting thresholds. A player needs enough total shots for the denominator of dep_share to
# be stable, and enough threes for the numerator to be more than one or two lucky attempts.
#
# These are the values the implementation plan specified. They were temporarily lowered to
# 10/3 while the sample was a single 8-game slice, in which 50/10 qualified nobody; the
# full 531-game sample qualifies 261 players at 50/10, with a median of 253 shots each, so
# the original thresholds are restored.
MIN_SHOTS = 50
MIN_3PA = 10

N_BOOT = 1000
PRIMARY_STAT = 'dep_share'

for directory in (RESULTS_DIR,):
    if not os.path.exists(directory):
        os.makedirs(directory)


def build_level(shots_df, group, n_boot=N_BOOT, stat=PRIMARY_STAT):
    """Point estimates plus bootstrap CIs for one grouping level."""
    print(f'Aggregating dependence over {group}')
    point = aggregate_dependence(shots_df, group=group)

    print(f'Bootstrapping {stat} over {len(point)} groups with {n_boot} resamples')
    boot = bootstrap_dependence(shots_df, group=group, stat=stat,
                                n_boot=n_boot, random_state=0)

    return point.merge(boot, on=group, how='left')


if __name__ == '__main__':
    print(f'Loading {ALT_SHOTS_EXP_PTS_FILE}')
    shots = pd.read_csv(ALT_SHOTS_EXP_PTS_FILE)

    missing = shots['exp_pts_naive'].isna().sum()
    if missing:
        raise ValueError(
            f'{missing} shots have a null counterfactual. Re-run src/cf_dep_processing.py '
            'with modify_exp_pts_df = True after applying the league-mean fallback.'
        )

    shots = add_shot_dependence(shots)
    print(f'{len(shots)} shots, {int(shots["3pt"].sum())} of them threes, '
          f'{shots["game_id"].nunique()} games, {shots["player_id"].nunique()} players')

    # --- player level -------------------------------------------------------
    player = build_level(shots, ['player_id'])
    player['qualified'] = (player['n_shots'] >= MIN_SHOTS) & (player['n_3pa'] >= MIN_3PA)
    print(f'{int(player["qualified"].sum())} of {len(player)} players qualify '
          f'(>= {MIN_SHOTS} shots, >= {MIN_3PA} 3PA)')

    qualified = player[player['qualified']].copy()
    if qualified.empty:
        raise ValueError(
            f'No player meets MIN_SHOTS={MIN_SHOTS} / MIN_3PA={MIN_3PA}. The sample is too '
            'small for player-level reporting; lower the thresholds or extend the data.'
        )

    # Clustered-by-game intervals for the qualified players only: a sensitivity check on the
    # within-player exchangeability assumption. Reported alongside the shot-level interval,
    # never as a replacement for it.
    clustered = clustered_bootstrap_dependence(
        shots[shots['player_id'].isin(qualified['player_id'])],
        group=['player_id'], cluster_col='game_id', stat=PRIMARY_STAT, n_boot=N_BOOT
    )
    qualified = qualified.merge(clustered, on='player_id', how='left')
    player = player.merge(clustered, on='player_id', how='left')
    player.to_csv(RESULTS_FILES['player'], index=False)

    n_single = int((qualified[f'{PRIMARY_STAT}_n_clusters'] <= 1).sum())
    multi = qualified[qualified[f'{PRIMARY_STAT}_n_clusters'] > 1]
    print(f'{n_single} of {len(qualified)} qualified players appear in a single game; their '
          'clustered interval is degenerate by construction and is excluded below')

    if not multi.empty:
        shot_width = multi[f'{PRIMARY_STAT}_ci_upper'] - multi[f'{PRIMARY_STAT}_ci_lower']
        clust_width = (multi[f'{PRIMARY_STAT}_clustered_ci_upper']
                       - multi[f'{PRIMARY_STAT}_clustered_ci_lower'])
        print(f'Median CI width ({len(multi)} multi-game players): '
              f'shot-level {shot_width.median():.4f}, '
              f'game-clustered {clust_width.median():.4f} '
              f'(ratio {clust_width.median() / shot_width.median():.2f})')
    else:
        print('No qualified player appears in more than one game; the clustered bootstrap '
              'cannot be evaluated on this sample.')

    # Does the leaderboard contain any real between-player signal, or is its whole spread
    # shot noise? This is the single most decisive number the bootstrap makes available.
    snr = signal_to_noise(qualified, PRIMARY_STAT, f'{PRIMARY_STAT}_se')
    pd.DataFrame([snr]).to_csv(
        os.path.join(RESULTS_DIR, 'cf_dependence_signal_to_noise.csv'), index=False)
    print(f'\nBetween-player spread of {PRIMARY_STAT}:')
    print(f'  observed variance      {snr["observed_var"]:.6f}')
    print(f'  mean sampling variance {snr["mean_sampling_var"]:.6f}')
    print(f'  true SD (tau)          {snr["tau"]:.5f}')
    print(f'  reliability            {snr["reliability"]:.3f}')

    # --- team and game levels ----------------------------------------------
    team = build_level(shots, ['team_id'])
    team.to_csv(RESULTS_FILES['team'], index=False)

    game = build_level(shots, ['game_id', 'team_id'], n_boot=200)
    game.to_csv(RESULTS_FILES['game'], index=False)

    # Figures are not drawn here. src/experiments/cf_dependence_figures.py owns the whole
    # figure set (see docs/figures.md) and reads the CSVs this script writes.
    rho = qualified[PRIMARY_STAT].corr(qualified['3pa_rate'], method='spearman')

    # --- normalization sensitivity, feeds section 6.7 -----------------------
    corr = qualified[['dep_total', 'dep_per_shot', 'dep_per_3pa', 'dep_share']].corr(method='spearman')
    corr.to_csv(os.path.join(RESULTS_DIR, 'cf_dependence_normalization_corr.csv'))
    print('\nRank correlation between normalizations:')
    print(corr.to_string())

    print(f'\nSpearman rho, dependence vs 3PA rate: {rho:.3f}')
    print(f'Results written to {RESULTS_DIR}/. '
          'Draw figures with src/experiments/cf_dependence_figures.py')
