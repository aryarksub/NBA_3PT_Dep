"""Figures for the counterfactual-strategy comparison.

Reads what src/experiments/cf_alternatives.py wrote. Documented in docs/figures.md.

    python src/experiments/cf_alternatives_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from counterfactual import build_support_grid, half_court_twos, in_support_vec
from cf_alternatives import MAX_RADIUS, CELL_SIZE

RESULTS_DIR = 'results'
OUT_DIR = os.path.join('plots', 'cf_alternatives')
os.makedirs(OUT_DIR, exist_ok=True)

INK = '#3b4a7a'
ACCENT = '#8f5560'
MUTED = '#9aa3bd'
GREEN = '#4f7a5a'

STAT = 'dep_share'

PRETTY = {'naive': 'naive (reference)'}


def label(s):
    if s in PRETTY:
        return PRETTY[s]
    if s.startswith('near_k'):
        return f'nearest, openness {int(s[6:]) / 100:g}x'
    q = int(s[6:9]) / 100
    lam = int(s[11:]) / 100
    return f'best, q{q:g}, closing {lam:g}'


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  wrote {path}')


# --------------------------------------------------------------------------------------

def fig_strategy_comparison(comp, player, strategies):
    """Where each specification puts the league, with the reference marked."""
    bad = comp.set_index('strategy').loc[strategies, 'degenerate'].to_numpy()
    y = np.arange(len(strategies))
    med = [player[s].median() for s in strategies]
    q25 = [player[s].quantile(0.25) for s in strategies]
    q75 = [player[s].quantile(0.75) for s in strategies]
    colours = [MUTED if d else (ACCENT if s == 'naive' else INK)
               for s, d in zip(strategies, bad)]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(strategies) + 3.0))
    for yi, lo, hi, c in zip(y, q25, q75, colours):
        ax.plot([lo, hi], [yi, yi], color=c, alpha=0.45, linewidth=2.6,
                solid_capstyle='round')
    ax.scatter(med, y, s=46, color=colours, zorder=3)
    ax.axvline(player['naive'].median(), color=ACCENT, linestyle='--', linewidth=1.2,
               label='naive median')
    ax.axvline(0, color='#999999', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(s) + ('  (high correlation)' if d else '')
                        for s, d in zip(strategies, bad)], fontsize=10)
    for tick, d in zip(ax.get_yticklabels(), bad):
        tick.set_color('#8a8a8a' if d else 'black')
    ax.invert_yaxis()
    ax.set_xlabel('Player dependence share (dot = median, bar = middle half)', fontsize=12)
    ax.set_title('Where each counterfactual specification puts the league\n'
                 'naive is retained as the reference, not replaced;\n'
                 'greyed rows have attempt-rate rank correlation >= 0.95', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    save(fig, 'strategy_comparison.png')


def fig_versus_naive(comp, player, strategies):
    """Player by player, each specification against the reference."""
    bad = dict(zip(comp['strategy'], comp['degenerate']))
    others = [s for s in strategies if s != 'naive']
    ncol = 4
    nrow = int(np.ceil(len(others) / ncol))
    lims = [min(player[strategies].min()) - 0.01, max(player[strategies].max()) + 0.01]

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), others):
        rho = player[s].corr(player['naive'], method='spearman')
        ax.plot(lims, lims, '--', color=MUTED, linewidth=1.0)
        ax.scatter(player['naive'], player[s], s=8, alpha=0.45,
                   color=MUTED if bad[s] else INK)
        ax.set_title(f'{label(s)}{"  (high correlation)" if bad[s] else ""}\nrho = {rho:.3f}',
                     fontsize=9, color='#8a8a8a' if bad[s] else 'black')
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(others):]:
        ax.axis('off')

    fig.supxlabel('Dependence share, naive reference', fontsize=12)
    fig.supylabel('Dependence share, this specification', fontsize=12)
    fig.suptitle('Every specification against the retained reference\n'
                 'dashed line is equality. Above it, this specification values the substitute '
                 'two lower,\nso the three looks more valuable. Greyed panels meet the '
                 'heuristic high-correlation screen', fontsize=11)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.93])
    save(fig, 'strategy_versus_naive.png')


def fig_degeneracy(comp, strategies):
    """A ranking-similarity diagnostic; not a test of metric validity."""
    c = comp.set_index('strategy').loc[strategies]
    rho = c['spearman_vs_3pa_rate']
    bad = c['degenerate'].to_numpy()
    colours = [MUTED if d else (ACCENT if s == 'naive' else INK)
               for s, d in zip(strategies, bad)]

    y = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(strategies) + 3.0))
    ax.barh(y, rho.to_numpy(), color=colours, height=0.62)
    ax.axvline(0.95, color=ACCENT, linestyle='--', linewidth=1.4,
               label='0.95: heuristic threshold for similar rankings')
    for yi, v, d in zip(y, rho.to_numpy(), bad):
        ax.text(v + 0.012, yi, f'{v:.2f}' + ('  high correlation' if d else ''),
                va='center', fontsize=8.5, color='#555555')
    ax.set_yticks(y)
    ax.set_yticklabels([label(s) for s in strategies], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Spearman correlation between dependence share and three-point attempt rate',
                  fontsize=11)
    ax.set_title('Similarity of dependence rankings to three-point attempt rate\n'
                 'passing this heuristic does not establish validity or useful information',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    save(fig, 'attempt_rate_degeneracy.png')


def fig_feasibility(diag):
    """What the openness requirement costs in coverage and in travel."""
    near = diag[diag['strategy'].str.startswith('near_k')].copy()
    near['kappa'] = near['strategy'].str[6:].astype(int) / 100

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].plot(near['kappa'], near['fallback_rate'], 'o-', color=INK, markersize=7)
    axes[0].set_xlabel('Openness required, as a multiple of the shot taken', fontsize=12)
    axes[0].set_ylabel('Share of threes with no feasible substitute', fontsize=12)
    axes[0].set_ylim(0, max(0.05, near['fallback_rate'].max() * 1.25))
    axes[0].set_title('A. Threes with no comparable two\n'
                      '(these fall back to the naive counterfactual)', fontsize=11)

    axes[1].plot(near['kappa'], near['median_displacement'], 'o-', color=INK, markersize=7,
                 label='median')
    axes[1].plot(near['kappa'], near['p90_displacement'], 'o--', color=MUTED, markersize=6,
                 label='90th percentile')
    axes[1].axhline(MAX_RADIUS, color=ACCENT, linestyle=':', linewidth=1.4,
                    label=f'{MAX_RADIUS:.0f} ft search limit')
    axes[1].set_xlabel('Openness required, as a multiple of the shot taken', fontsize=12)
    axes[1].set_ylabel('Distance from the real shot (feet)', fontsize=12)
    axes[1].set_title('B. How far the substitute sits from the three', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle('Cost of requiring a comparably open substitute', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'feasibility_by_openness.png')


def fig_spec_vs_sampling(player, n_all, n_live):
    """The number this whole exercise exists to produce, on both strategy sets.

    Reporting only the all-specification version would inflate the claim with specifications
    that failed the degeneracy check; reporting only the restricted version would hide the
    full range an analyst could reach. Both are shown.
    """
    both = player.replace([np.inf, -np.inf], np.nan)
    r_all = both['spec_over_sampling'].dropna()
    r_live = both['spec_over_sampling_non_degenerate'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    bins = np.histogram_bin_edges(pd.concat([r_all, r_live]), bins=45)
    axes[0].hist(r_live, bins=bins, color=INK, alpha=0.35,
                 label=f'{n_live} below threshold (median {r_live.median():.2f})')
    axes[0].hist(r_live, bins=bins, histtype='step', color=INK, linewidth=1.8)
    axes[0].hist(r_all, bins=bins, histtype='step', color=ACCENT, linewidth=1.8,
                 label=f'all {n_all} specifications (median {r_all.median():.2f})')
    axes[0].axvline(1.0, color='#555555', linestyle=':', linewidth=1.8,
                    label='equal to conditional CI width')
    axes[0].set_xlabel('Spread across specifications / bootstrap interval width', fontsize=12)
    axes[0].set_ylabel('Players', fontsize=12)
    axes[0].set_title(f'A. Specification range exceeds conditional CI width\n'
                      f'for {(r_all > 1).mean():.0%} of players on all '
                      f'specifications,\n{(r_live > 1).mean():.0%} on the below-threshold set',
                      fontsize=11)
    axes[0].legend(fontsize=9)

    sub = both.dropna(subset=['spec_range_non_degenerate', 'naive_ci_width'])
    axes[1].scatter(sub['naive_ci_width'], sub['spec_range_non_degenerate'], s=16,
                    alpha=0.55, color=INK)
    top = max(sub['naive_ci_width'].max(), sub['spec_range_non_degenerate'].max()) * 1.05
    axes[1].plot([0, top], [0, top], '--', color=ACCENT, linewidth=1.4,
                 label='equal magnitude')
    axes[1].set_xlim(0, top)
    axes[1].set_ylim(0, top)
    axes[1].set_xlabel('Bootstrap interval width, naive reference (sampling)', fontsize=12)
    axes[1].set_ylabel('Range across below-threshold specifications', fontsize=12)
    axes[1].set_title('B. Range and conditional interval width, restricted to the\n'
                      'specifications below the correlation threshold.\n'
                      'Above the line: range exceeds conditional CI width',
                      fontsize=11)
    axes[1].legend(fontsize=10)

    fig.suptitle('Specification range versus conditional bootstrap width', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, 'specification_versus_sampling.png')


def _mirror_to_right(x, y):
    """Fold both halves of the court onto the right half so densities are comparable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    left = x < 47.0
    x = np.where(left, 94.0 - x, x)
    y = np.where(left, 50.0 - y, y)
    return x, y


def fig_locations(comp, shots, locations, strategies):
    """Where the counterfactual sends the shooter, against where twos are actually taken."""
    twos = shots[shots['3pt'] == 0]
    ox, oy = _mirror_to_right(twos['shooter_x'], twos['shooter_y'])

    show = [s for s in ('near_k100', 'best_q090_l000', 'best_q090_l050') if f'{s}_x' in locations]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(4.6 * (len(show) + 1), 5.0),
                             sharex=True, sharey=True)

    bins = [np.linspace(47, 94, 48), np.linspace(0, 50, 51)]
    axes[0].hist2d(ox, oy, bins=bins, cmap='Blues')
    axes[0].set_title(f'Observed two-point attempts\n({len(twos):,} shots)', fontsize=11)

    bad = dict(zip(comp['strategy'], comp['degenerate']))
    for ax, s in zip(axes[1:], show):
        cx, cy = _mirror_to_right(locations[f'{s}_x'].dropna(), locations[f'{s}_y'].dropna())
        ax.hist2d(cx, cy, bins=bins, cmap='Blues')
        ax.set_title(f'{label(s)}{"  (high correlation)" if bad[s] else ""}\n'
                     f'({len(cx):,} substituted threes)', fontsize=11,
                     color='#8a8a8a' if bad[s] else 'black')

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('Court x (feet, both halves folded right)', fontsize=10)
    axes[0].set_ylabel('Court y (feet)', fontsize=11)

    fig.suptitle('Where each strategy puts the substitute shot\n'
                 'each panel is shaded against its own maximum, so compare shapes, not '
                 'intensities.\nNearest-feasible-two lands in a thin ring just inside the arc '
                 '-- consistent with its strong attempt-rate association', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, 'counterfactual_locations.png')


def fig_candidate_support(shots, support_cell=2.0, support_min=5):
    """Map candidate grid points that are inside the common-support cells.

    Shows observed two-point density and overlays candidate grid points colored by
    whether their 2x2-ft support cell is considered "in support" (>= min_count).
    """
    twos = shots[shots['3pt'] == 0]
    # build support grid from observed twos
    support = build_support_grid(twos, cell_size=support_cell, min_count=support_min)

    # full-court candidate grid (both halves)
    pts_left = half_court_twos(False, cell_size=1.0)
    pts_right = half_court_twos(True, cell_size=1.0)
    pts = np.vstack([pts_left, pts_right])

    in_sup = in_support_vec(support, pts)

    # mirror to right half for display
    def _mirror_to_right(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        left = x < 47.0
        x = np.where(left, 94.0 - x, x)
        y = np.where(left, 50.0 - y, y)
        return x, y

    ox, oy = _mirror_to_right(twos['shooter_x'], twos['shooter_y'])
    px, py = _mirror_to_right(pts[:, 0], pts[:, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [np.linspace(47, 94, 48), np.linspace(0, 50, 51)]
    ax.hist2d(ox, oy, bins=bins, cmap='Greys')
    ax.scatter(px[in_sup], py[in_sup], s=6, color='#4f7a5a', alpha=0.6, label='in support')
    ax.scatter(px[~in_sup], py[~in_sup], s=6, color='#8f5560', alpha=0.4, label='out of support')
    ax.set_aspect('equal')
    ax.set_xlabel('Court x (feet, folded right)')
    ax.set_ylabel('Court y (feet)')
    ax.set_title(f'Unused support diagnostic (filter disabled in the reported runs)\n'
                 f'>= {support_min} shots per {support_cell}ft cell; display grid = 1 ft')
    ax.legend(markerscale=2)
    save(fig, 'candidate_support_map.png')


# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    comp = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_alternatives_comparison.csv'))
    diag = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_alternatives_diagnostics.csv'))
    player = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_alternatives_player.csv'))
    shots = pd.read_csv(os.path.join('data', 'cf_alternatives_shots.csv'))
    locations = pd.read_csv(os.path.join('data', 'cf_alternatives_locations.csv'))

    strategies = comp['strategy'].tolist()
    print(f'{len(strategies)} strategies, {len(player)} qualified players')

    print('\nDrawing figures')
    fig_strategy_comparison(comp, player, strategies)
    fig_versus_naive(comp, player, strategies)
    fig_degeneracy(comp, strategies)
    fig_feasibility(diag)
    n_live = int((~comp['degenerate']).sum())
    fig_spec_vs_sampling(player, len(strategies), n_live)
    fig_locations(comp, shots, locations, strategies)
    fig_candidate_support(shots)
    print(f'\nWrote figures to {OUT_DIR}/')
