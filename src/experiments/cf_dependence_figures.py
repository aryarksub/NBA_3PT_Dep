"""The canonical figure set for behavioral three-point dependence.

One script, one figure list, documented in docs/figures.md. Everything here reads results
that are already on disk -- no bootstrap is recomputed -- so it is cheap to rerun.

    python src/experiments/cf_dependence_figures.py

Prerequisites: src/experiments/cf_dependence.py and src/experiments/cf_ci_impact.py have
been run, so results/cf_dependence_{player,team,game}.csv, results/ep_calibration.csv and
results/cf_dependence_rank_intervals.csv exist.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE
from dependence import add_shot_dependence
from pbp_shot_processing import GAMES_FILE

RESULTS_DIR = 'results'
OUT_DIR = os.path.join('plots', 'cf_dependence')
os.makedirs(OUT_DIR, exist_ok=True)

INK = '#3b4a7a'
ACCENT = '#8f5560'
MUTED = '#9aa3bd'
GREEN = '#4f7a5a'

STAT = 'dep_share'
LO, HI = f'{STAT}_ci_lower', f'{STAT}_ci_upper'

# Same thresholds the pipeline reports at.
MIN_SHOTS, MIN_3PA = 50, 10


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  wrote {path}')


# --------------------------------------------------------------------------------------
# 1. Expected points calibration
# --------------------------------------------------------------------------------------

def fig_calibration(calib):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lims = [min(calib.p_mean.min(), calib.make_rate.min()) - 0.03,
            max(calib.p_mean.max(), calib.make_rate.max()) + 0.03]
    ax.plot(lims, lims, '--', color=ACCENT, linewidth=1.5, label='perfect calibration')
    ax.plot(calib.p_mean, calib.make_rate, 'o-', color=INK, markersize=7,
            label='cross-fitted model')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Predicted make probability (decile mean)', fontsize=13)
    ax.set_ylabel('Realized make rate', fontsize=13)
    ax.set_title('Expected points model calibration, out of fold', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    save(fig, 'expected_points_calibration.png')


# --------------------------------------------------------------------------------------
# 2. Distribution of dependence
# --------------------------------------------------------------------------------------

def fig_distribution(qual):
    detectable = (qual[LO] > 0) | (qual[HI] < 0)
    bins = np.histogram_bin_edges(qual[STAT].dropna(), bins=40)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist([qual.loc[detectable, STAT], qual.loc[~detectable, STAT]], bins=bins,
            stacked=True, color=[INK, MUTED], edgecolor='white',
            label=[f'interval excludes zero ({int(detectable.sum())} players)',
                   f'interval includes zero ({int((~detectable).sum())} players)'])
    ax.axvline(0, color=ACCENT, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Share of expected scoring that depends on three-point selection', fontsize=13)
    ax.set_ylabel('Players', fontsize=13)
    ax.set_title(f'Dependence across {len(qual)} qualified players', fontsize=13)
    ax.legend(fontsize=10)
    save(fig, 'dependence_distribution.png')


# --------------------------------------------------------------------------------------
# 3 & 4. Player and team rankings
# --------------------------------------------------------------------------------------

def _caterpillar(df, label_col, title, name, top_n=None):
    ranked = df.dropna(subset=[STAT]).sort_values(STAT)
    sub = ranked if top_n is None else pd.concat([ranked.head(top_n), ranked.tail(top_n)])
    sub = sub[~sub.index.duplicated(keep='first')]

    y = np.arange(len(sub))
    lo = (sub[STAT] - sub[LO]).clip(lower=0)
    hi = (sub[HI] - sub[STAT]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(sub) + 2.2))
    ax.errorbar(sub[STAT], y, xerr=[lo, hi], fmt='o', color=INK, ecolor=MUTED,
                capsize=3, markersize=4)
    ax.axvline(0, color=ACCENT, linestyle='--', linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(sub[label_col].astype(str), fontsize=8)
    ax.set_xlabel('Share of expected scoring that depends on three-point selection', fontsize=13)
    ax.set_title(title, fontsize=13)
    save(fig, name)


# --------------------------------------------------------------------------------------
# 5. Rank uncertainty
# --------------------------------------------------------------------------------------

def fig_rank_uncertainty(ranks, top_n=25):
    sub = ranks.sort_values('rank_point').head(top_n)
    y = np.arange(len(sub))
    lo = (sub['rank_point'] - sub['rank_ci_lower']).clip(lower=0)
    hi = (sub['rank_ci_upper'] - sub['rank_point']).clip(lower=0)
    median_span = ranks['rank_span'].median()

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(sub) + 2.4))
    ax.errorbar(sub['rank_point'], y, xerr=[lo, hi], fmt='o', color=INK, ecolor=MUTED,
                capsize=3, markersize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['player_id'].astype(str), fontsize=8)
    ax.set_xlabel(f'Plausible league rank out of {len(ranks)} (1 = most dependent)', fontsize=13)
    ax.set_title('Rank uncertainty for the 25 most dependent players\n'
                 f'median plausible range across the field: {median_span:.0f} of '
                 f'{len(ranks)} positions', fontsize=13)
    ax.invert_yaxis()
    save(fig, 'player_rank_uncertainty.png')


# --------------------------------------------------------------------------------------
# 6. Volume versus edge
# --------------------------------------------------------------------------------------

def fig_volume_vs_edge(qual):
    """dep_share = 3pa_rate * dep_per_3pa / (expected points per shot).

    The contours hold the last term at the league median, so they are exact only for a
    player at that value; they are drawn as orientation, not as a lookup table.
    """
    sub = qual.dropna(subset=[STAT, 'dep_per_3pa', '3pa_rate']).copy()
    ep_per_shot = (sub['ep_obs_total'] / sub['n_shots']).median()

    x = np.linspace(sub['3pa_rate'].min() * 0.9, sub['3pa_rate'].max() * 1.05, 200)
    y = np.linspace(sub['dep_per_3pa'].min() * 1.05, sub['dep_per_3pa'].max() * 1.05, 200)
    XX, YY = np.meshgrid(x, y)
    ZZ = XX * YY / ep_per_shot

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cs = ax.contour(XX, YY, ZZ, levels=[-0.05, 0.0, 0.05, 0.10, 0.15],
                    colors=MUTED, linewidths=1.0, linestyles='--')
    ax.clabel(cs, fmt=lambda v: f'{v:.0%}', fontsize=9)
    ax.scatter(sub['3pa_rate'], sub['dep_per_3pa'], s=20, alpha=0.65, color=INK)
    ax.axhline(0, color=ACCENT, linewidth=1.0)
    ax.set_xlabel('Share of shots that are threes', fontsize=13)
    ax.set_ylabel('Expected points added per three attempted', fontsize=13)
    ax.set_title('Volume versus edge\n'
                 'dashed lines join players with equal dependence share\n'
                 f'(evaluated at the league median of {ep_per_shot:.2f} expected points per shot)',
                 fontsize=12)
    save(fig, 'volume_versus_edge.png')


# --------------------------------------------------------------------------------------
# 7. Dependence versus attempt rate, with residuals
# --------------------------------------------------------------------------------------

def fig_vs_attempt_rate(qual, n_label=5):
    sub = qual.dropna(subset=[STAT, '3pa_rate']).copy()
    rho = sub[STAT].corr(sub['3pa_rate'], method='spearman')

    slope, intercept = np.polyfit(sub['3pa_rate'], sub[STAT], 1)
    sub['resid'] = sub[STAT] - (slope * sub['3pa_rate'] + intercept)
    r2 = np.corrcoef(sub['3pa_rate'], sub[STAT])[0, 1] ** 2

    extremes = pd.concat([sub.nlargest(n_label, 'resid'), sub.nsmallest(n_label, 'resid')])

    xs = np.linspace(sub['3pa_rate'].min(), sub['3pa_rate'].max(), 50)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(sub['3pa_rate'], sub[STAT], s=20, alpha=0.55, color=INK)
    ax.plot(xs, slope * xs + intercept, color=ACCENT, linewidth=1.8, label='least-squares fit')
    ax.scatter(extremes['3pa_rate'], extremes[STAT], s=42, facecolors='none',
               edgecolors=ACCENT, linewidths=1.4)
    for _, r in extremes.iterrows():
        ax.annotate(str(int(r['player_id'])), (r['3pa_rate'], r[STAT]),
                    textcoords='offset points', xytext=(6, 3), fontsize=7.5, color=ACCENT)
    ax.axhline(0, color='#999999', linewidth=0.8)
    ax.set_xlabel('Share of shots that are threes', fontsize=13)
    ax.set_ylabel('Dependence share', fontsize=13)
    ax.set_title(f'Dependence versus attempt rate (Spearman rho = {rho:.3f}, '
                 f'linear R-squared = {r2:.2f})\n'
                 f'circled: {n_label} largest residuals each way -- more and less dependent '
                 'than volume implies', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    save(fig, 'dependence_versus_attempt_rate.png')

    return rho, r2, extremes[['player_id', '3pa_rate', STAT, 'resid']]


# --------------------------------------------------------------------------------------
# 8. Split-half reliability
# --------------------------------------------------------------------------------------

def fig_split_half(shots, min_shots_half=25, min_3pa_half=5):
    """Odd games versus even games, on the same player.

    A stability check that does not use the bootstrap, so it is an independent read on the
    reliability the bootstrap reports. Games are ordered by identifier -- which is
    chronological within a season -- and alternated, so each player's season is split
    without regard to opponent or outcome.
    """
    order = pd.DataFrame({'game_id': np.sort(shots['game_id'].unique())})
    order['half'] = np.arange(len(order)) % 2
    sh = shots.merge(order[['game_id', 'half']], on='game_id', how='left')
    # Estimate the naive alternative separately in each half; a shared baseline
    # otherwise creates artificial agreement. Probability models remain shared.
    baseline = sh[sh['3pt'] == 0].groupby(['player_id', 'half'])['expected_points'].mean()
    sh = sh.join(baseline.rename('half_baseline'), on=['player_id', 'half'])
    league = sh[sh['3pt'] == 0].groupby('half')['expected_points'].mean()
    sh['dep_delta'] = np.where(sh['3pt'] == 1,
                              sh['expected_points'] - sh['half_baseline'].fillna(sh['half'].map(league)), 0.)

    agg = (sh.groupby(['player_id', 'half'])
           .agg(n_shots=('dep_delta', 'size'), n_3pa=('3pt', 'sum'),
                dep_total=('dep_delta', 'sum'), ep_obs_total=('expected_points', 'sum'))
           .reset_index())
    agg[STAT] = np.where(agg['ep_obs_total'] > 0, agg['dep_total'] / agg['ep_obs_total'], np.nan)
    agg = agg[(agg['n_shots'] >= min_shots_half) & (agg['n_3pa'] >= min_3pa_half)]

    wide = agg.pivot(index='player_id', columns='half', values=STAT).dropna()
    wide.columns = ['odd', 'even']

    r = wide['odd'].corr(wide['even'], method='spearman')

    lims = [min(wide.min()) - 0.02, max(wide.max()) + 0.02]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(lims, lims, '--', color=MUTED, linewidth=1.2, label='perfect agreement')
    ax.scatter(wide['odd'], wide['even'], s=22, alpha=0.6, color=INK)
    ax.axhline(0, color='#bbbbbb', linewidth=0.8)
    ax.axvline(0, color='#bbbbbb', linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Dependence share, odd-numbered games', fontsize=13)
    ax.set_ylabel('Dependence share, even-numbered games', fontsize=13)
    ax.set_title(f'Split-half reliability ({len(wide)} players with at least '
                 f'{min_shots_half} shots and {min_3pa_half} threes per half)\n'
                 f'Spearman rho = {r:.3f}; separate half baselines, shared shot models', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    save(fig, 'split_half_reliability.png')

    return r, len(wide)


# --------------------------------------------------------------------------------------
# 9. Where dependence comes from
# --------------------------------------------------------------------------------------

def _binned_mean(df, col, value, n_bins=8):
    """Bin mean of `value`, positioned at the bin's MEDIAN x, not its interval midpoint.

    The top octile of defender distance runs to the far end of the court, so its midpoint
    sits around 35 feet while almost every shot in it is nearer 13. Plotting midpoints
    stretches the last point far to the right and makes the trend look like it continues
    smoothly out to distances at which barely any shots were taken.
    """
    b = pd.qcut(df[col], q=n_bins, duplicates='drop')
    g = df.groupby(b, observed=True).agg(
        mean=(value, 'mean'), sem=(value, 'sem'), size=(value, 'size'),
        centre=(col, 'median'),
    ).reset_index()
    return g


def fig_where_from(shots):
    threes = shots[shots['3pt'] == 1].dropna(subset=['close_def_dist', 'shot_dist'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, col, xlabel in (
        (axes[0], 'close_def_dist', 'Distance to closest defender (feet)'),
        (axes[1], 'shot_dist', 'Shot distance (feet)'),
    ):
        g = _binned_mean(threes, col, 'dep_delta')
        ax.errorbar(g['centre'], g['mean'], yerr=1.96 * g['sem'], fmt='o-', color=INK,
                    ecolor=MUTED, capsize=3, markersize=5)
        ax.axhline(0, color=ACCENT, linestyle='--', linewidth=1.3)
        ax.set_xlabel(xlabel, fontsize=13)
    axes[0].set_ylabel('Expected points added per three', fontsize=13)
    fig.suptitle('Where dependence comes from: value added per three, by octile\n'
                 f'({len(threes):,} three-point attempts; points sit at the bin median, '
                 'bars are 95% intervals on the bin mean)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'where_dependence_comes_from.png')


# --------------------------------------------------------------------------------------
# 10. Game-to-game versus between-team variation
# --------------------------------------------------------------------------------------

def fig_game_variation(game, team):
    order = team.dropna(subset=[STAT]).sort_values(STAT)['team_id'].tolist()
    data = [game.loc[game['team_id'] == t, STAT].dropna().to_numpy() for t in order]
    season = team.set_index('team_id').loc[order, STAT].to_numpy()

    within = np.median([np.std(d, ddof=1) for d in data if len(d) > 1])
    between = float(np.std(season, ddof=1))

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for box in bp['boxes']:
        box.set(facecolor='#dfe3ee', edgecolor=MUTED)
    for whisk in bp['whiskers'] + bp['caps']:
        whisk.set(color=MUTED)
    for med in bp['medians']:
        med.set(color=MUTED)
    ax.plot(np.arange(1, len(order) + 1), season, 'o', color=ACCENT, markersize=6,
            label='team season value')
    ax.axhline(0, color='#999999', linewidth=0.8)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([str(t) for t in order], rotation=90, fontsize=7)
    ax.set_xlabel('Team', fontsize=13)
    ax.set_ylabel('Dependence share', fontsize=13)
    ax.set_title('Game-to-game versus between-team variation\n'
                 f'median within-team spread across games {within:.3f}; '
                 f'spread between team season values {between:.3f} '
                 f'(ratio {within / between:.1f} to 1)', fontsize=12)
    ax.legend(fontsize=10)
    save(fig, 'game_to_game_variation.png')

    return within, between


# --------------------------------------------------------------------------------------
# 11. Threshold sensitivity
# --------------------------------------------------------------------------------------

def fig_threshold_sensitivity(player):
    grid = [(25, 5), (50, 10), (100, 20), (150, 30), (200, 40), (300, 60)]
    rows = []
    for ms, m3 in grid:
        sub = player[(player['n_shots'] >= ms) & (player['n_3pa'] >= m3)]
        s = sub[STAT].dropna()
        if s.empty:
            continue
        rows.append({'label': f'{ms} / {m3}', 'n': len(s), 'median': s.median(),
                     'q25': s.quantile(0.25), 'q75': s.quantile(0.75)})
    tbl = pd.DataFrame(rows)

    x = np.arange(len(tbl))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(x, tbl['q25'], tbl['q75'], color=MUTED, alpha=0.35,
                    label='middle half of players')
    ax.plot(x, tbl['median'], 'o-', color=INK, markersize=6, label='median player')
    ax.axhline(0, color=ACCENT, linestyle='--', linewidth=1.2)
    for xi, row in zip(x, tbl.itertuples()):
        ax.annotate(f'n={row.n}', (xi, row.q75), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=9, color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(tbl['label'], fontsize=10)
    ax.set_xlabel('Minimum shots / minimum threes required to qualify', fontsize=13)
    ax.set_ylabel('Dependence share', fontsize=13)
    ax.set_title('Qualification threshold sensitivity\n'
                 'the reported cutoff is 50 / 10; shaded band is the player interquartile range',
                 fontsize=12)
    ax.set_ylim(top=tbl['q75'].max() * 1.30)
    ax.legend(fontsize=10, loc='lower right')
    save(fig, 'qualification_threshold_sensitivity.png')

    return tbl


# --------------------------------------------------------------------------------------
# 12. External outcomes
# --------------------------------------------------------------------------------------

def fig_outcomes(team, game, shots):
    games = pd.read_csv(GAMES_FILE, low_memory=False)
    keep = set(game['game_id'].unique())
    games = games[games['game_id'].isin(keep)]

    home = games[['game_id', 'team_id_home', 'pts_home', 'fga_home', 'wl_home']].rename(
        columns={'team_id_home': 'team_id', 'pts_home': 'pts', 'fga_home': 'official_fga', 'wl_home': 'wl'})
    away = games[['game_id', 'team_id_away', 'pts_away', 'fga_away', 'wl_away']].rename(
        columns={'team_id_away': 'team_id', 'pts_away': 'pts', 'fga_away': 'official_fga', 'wl_away': 'wl'})
    long = pd.concat([home, away], ignore_index=True).dropna(subset=['pts'])
    long['win'] = (long['wl'] == 'W').astype(float)

    # Panel A: team season dependence versus win percentage.
    win_pct = long.groupby('team_id')['win'].mean().reset_index(name='win_pct')
    a = team.merge(win_pct, on='team_id', how='inner').dropna(subset=[STAT, 'win_pct'])
    rho_a = a[STAT].corr(a['win_pct'], method='spearman')

    # Panel B: team-game dependence versus points per field goal attempt.
    fga = (shots.groupby(['game_id', 'team_id']).size().reset_index(name='fga'))
    b = (game.merge(long[['game_id', 'team_id', 'pts', 'official_fga']], on=['game_id', 'team_id'], how='inner')
         .merge(fga, on=['game_id', 'team_id'], how='inner'))
    b['pts_per_fga'] = b['pts'] / b['official_fga']
    b = b.dropna(subset=[STAT, 'pts_per_fga'])
    rho_b = b[STAT].corr(b['pts_per_fga'], method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    axes[0].scatter(a[STAT], a['win_pct'], s=45, alpha=0.75, color=INK)
    axes[0].set_xlabel('Team dependence share (season)', fontsize=12)
    axes[0].set_ylabel('Win percentage in sampled games', fontsize=12)
    axes[0].set_title('A. Win percentage in sampled games\n'
                      f'{len(a)} teams, Spearman rho = {rho_a:.3f}',
                      fontsize=12)

    axes[1].scatter(b[STAT], b['pts_per_fga'], s=12, alpha=0.35, color=INK)
    axes[1].set_xlabel('Team dependence share (single game)', fontsize=12)
    axes[1].set_ylabel('Points per field goal attempt', fontsize=12)
    axes[1].set_title('B. Total points / official field-goal attempts\n'
                      f'{len(b)} team-games, Spearman rho = {rho_b:.3f}',
                      fontsize=12)

    fig.suptitle('Dependence and observed outcomes: descriptive associations', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, 'team_dependence_versus_outcomes.png')

    return rho_a, len(a), rho_b, len(b)


# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    player = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_dependence_player.csv'))
    team = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_dependence_team.csv'))
    game = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_dependence_game.csv'))
    ranks = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_dependence_rank_intervals.csv'))
    calib = pd.read_csv(os.path.join(RESULTS_DIR, 'ep_calibration.csv'))

    qual = player[player['qualified']].copy()
    print(f'{len(qual)} qualified players, {len(team)} teams, {len(game)} team-games')

    print('Loading shot-level data')
    shots = add_shot_dependence(pd.read_csv(ALT_SHOTS_EXP_PTS_FILE))
    print(f'{len(shots):,} shots')

    print('\nDrawing figures')
    fig_calibration(calib)
    fig_distribution(qual)
    _caterpillar(qual, 'player_id',
                 'Most and least dependent players (top and bottom 20)',
                 'player_dependence_ranking.png', top_n=20)
    _caterpillar(team, 'team_id', 'Team dependence, all teams',
                 'team_dependence_ranking.png')
    fig_rank_uncertainty(ranks)
    fig_volume_vs_edge(qual)
    rho, r2, extremes = fig_vs_attempt_rate(qual)
    r, n_half = fig_split_half(shots)
    fig_where_from(shots)
    within, between = fig_game_variation(game, team)
    thresh = fig_threshold_sensitivity(player)
    rho_a, n_a, rho_b, n_b = fig_outcomes(team, game, shots)

    print('\n--- numbers behind the new figures ---')
    print(f'dependence vs attempt rate: Spearman rho {rho:.3f}, linear R-squared {r2:.2f}')
    print(f'split-half ({n_half} players): rho {r:.3f}; half-specific baselines, shared shot models')
    print(f'within-team game spread {within:.4f} vs between-team spread {between:.4f} '
          f'(ratio {within / between:.1f})')
    print(f'team dependence vs win percentage: rho {rho_a:.3f} over {n_a} teams')
    print(f'team-game dependence vs points per attempt: rho {rho_b:.3f} over {n_b} team-games')
    print('\nthreshold sensitivity:')
    print(thresh.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
