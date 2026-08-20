import os
import sys

import matplotlib.pyplot as plt
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
import plots

RESULTS_DIR = 'results'
CF_DEP_PLOTS_DIR = os.path.join(plots.PLOTS_DIR, 'cf_dependence')

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

for directory in (RESULTS_DIR, CF_DEP_PLOTS_DIR):
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


def plot_dependence_distribution(df, fname, x_label='Behavioral dependence (share of EP)'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[PRIMARY_STAT].dropna(), bins=40, color='#3b4a7a', edgecolor='white')
    ax.axvline(0, color='#8f5560', linestyle='--', linewidth=1.5)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Players', fontsize=14)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_caterpillar(df, fname, label_col='player_id', top_n=20):
    """Highest and lowest dependence, with bootstrap intervals."""
    ranked = df.dropna(subset=[PRIMARY_STAT]).sort_values(PRIMARY_STAT)
    sub = pd.concat([ranked.head(top_n), ranked.tail(top_n)])
    sub = sub[~sub.index.duplicated(keep='first')]

    y = np.arange(len(sub))
    lo = sub[PRIMARY_STAT] - sub[f'{PRIMARY_STAT}_ci_lower']
    hi = sub[f'{PRIMARY_STAT}_ci_upper'] - sub[PRIMARY_STAT]

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(sub) + 2))
    ax.errorbar(sub[PRIMARY_STAT], y, xerr=[lo, hi], fmt='o', color='#3b4a7a',
                ecolor='#9aa3bd', capsize=3, markersize=4)
    ax.axvline(0, color='#8f5560', linestyle='--', linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(sub[label_col].astype(str), fontsize=8)
    ax.set_xlabel('Behavioral dependence (share of EP)', fontsize=14)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_vs_3pa_rate(df, fname):
    """Dependence against attempt rate. If these were the same measure the project has no thesis."""
    sub = df.dropna(subset=[PRIMARY_STAT, '3pa_rate'])
    rho = sub[PRIMARY_STAT].corr(sub['3pa_rate'], method='spearman')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub['3pa_rate'], sub[PRIMARY_STAT], s=18, alpha=0.6, color='#3b4a7a')
    ax.set_xlabel('Three-point attempt rate', fontsize=14)
    ax.set_ylabel('Behavioral dependence (share of EP)', fontsize=14)
    ax.set_title(f'Spearman rho = {rho:.3f}', fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    return rho


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

    # --- figures ------------------------------------------------------------
    plot_dependence_distribution(qualified, os.path.join(CF_DEP_PLOTS_DIR, 'player_dep_share_dist.png'))
    plot_caterpillar(qualified, os.path.join(CF_DEP_PLOTS_DIR, 'player_dep_share_caterpillar.png'))
    plot_caterpillar(team, os.path.join(CF_DEP_PLOTS_DIR, 'team_dep_share_caterpillar.png'),
                     label_col='team_id', top_n=15)
    rho = plot_vs_3pa_rate(qualified, os.path.join(CF_DEP_PLOTS_DIR, 'dep_share_vs_3pa_rate.png'))

    # --- normalization sensitivity, feeds section 6.7 -----------------------
    corr = qualified[['dep_total', 'dep_per_shot', 'dep_per_3pa', 'dep_share']].corr(method='spearman')
    corr.to_csv(os.path.join(RESULTS_DIR, 'cf_dependence_normalization_corr.csv'))
    print('\nRank correlation between normalizations:')
    print(corr.to_string())

    print(f'\nSpearman rho, dependence vs 3PA rate: {rho:.3f}')
    print(f'Results written to {RESULTS_DIR}/, plots to {CF_DEP_PLOTS_DIR}/')
