"""Diagnostic figures that were previously only reported as numbers.

1. Reliability curve for the cross-fitted shot probabilities -- the visual form of
   results/ep_calibration.csv, which is what justifies trusting expected points at all.
2. What the confidence intervals change, 8-game slice vs the full 531-game sample.
3. Distribution of plausible rank spans, i.e. how resolvable the ranking actually is.

Run after src/experiments/cf_ci_impact.py.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

import plots

RESULTS_DIR = 'results'
OUT_DIR = os.path.join(plots.PLOTS_DIR, 'cf_dependence')
os.makedirs(OUT_DIR, exist_ok=True)

INK = '#3b4a7a'
ACCENT = '#8f5560'
MUTED = '#9aa3bd'

SWEEP_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_subsample_sweep.csv')


def plot_calibration(fname):
    tbl = pd.read_csv(os.path.join(RESULTS_DIR, 'ep_calibration.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lims = [min(tbl.p_mean.min(), tbl.make_rate.min()) - 0.03,
            max(tbl.p_mean.max(), tbl.make_rate.max()) + 0.03]
    ax.plot(lims, lims, linestyle='--', color=ACCENT, linewidth=1.5, label='perfect calibration')
    ax.plot(tbl.p_mean, tbl.make_rate, 'o-', color=INK, markersize=7, label='cross-fitted model')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Predicted make probability (decile mean)', fontsize=13)
    ax.set_ylabel('Realized make rate', fontsize=13)
    ax.set_title('Shot probability model is well calibrated\nout of fold (slope 1.013, AUC 0.639)',
                 fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_ci_impact(fname):
    """50 games vs the full 531, same code and same 50/10 thresholds throughout.

    Top-K overlap is deliberately excluded: its chance baseline is 10/n_qualified, which is
    36% in a 28-player field and 3.8% in a 261-player one, so the raw numbers are not
    comparable across sample sizes. The three shown here are all scale-free.
    """
    sweep = pd.read_csv(SWEEP_FILE)
    small = sweep[sweep.label == '50 games'].mean(numeric_only=True)
    full = sweep[sweep.label.str.contains('full')].iloc[0]

    labels = ['Sign detectable\n(CI excludes 0)', 'Player pairs\nseparable',
              'Rank precision\n(100 - median span)']
    before = [small['fraction_excluding_zero'], small['pairwise_resolvability'],
              1 - small['median_rank_span_pct'] / 100]
    after = [full['fraction_excluding_zero'], full['pairwise_resolvability'],
             1 - full['median_rank_span_pct'] / 100]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(x - w/2, before, w, color=MUTED,
                label=f'50 games (~{small["n_qualified"]:.0f} players, mean of 3 draws)')
    b2 = ax.bar(x + w/2, after, w, color=INK,
                label=f'531 games ({full["n_qualified"]:.0f} players)')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.015,
                    f'{r.get_height():.0%}', ha='center', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Share (higher is better)', fontsize=13); ax.set_ylim(0, 1.0)
    ax.set_title('What the bootstrap intervals establish: 50 games vs 531\n'
                 '(identical code and thresholds; only sample size differs)', fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_convergence(fname):
    """How each headline diagnostic converges with sample size."""
    sweep = pd.read_csv(SWEEP_FILE)
    series = [
        ('spearman_vs_3pa_rate', 'Spearman rho vs 3PA rate', ACCENT),
        ('reliability', 'Reliability of dep_share', INK),
        ('fraction_excluding_zero', 'Players with detectable sign', '#4f7a5a'),
        ('pairwise_resolvability', 'Player pairs separable', MUTED),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for col, lab, colour in series:
        g = sweep.groupby('n_games')[col].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(g.n_games, g['mean'], 'o-', color=colour, label=lab, markersize=6)
        ax.fill_between(g.n_games, g['min'], g['max'], color=colour, alpha=0.15)

    ax.axhline(0, color='#999999', linewidth=0.8)
    ax.axhspan(0.5, 0.85, color=ACCENT, alpha=0.07)
    ax.text(60, 0.865, 'band the thesis predicts for rho', fontsize=9, color=ACCENT)
    ax.set_xlabel('Games in sample', fontsize=13)
    ax.set_ylabel('Value', fontsize=13)
    ax.set_title('Every diagnostic improves monotonically with sample size\n'
                 '(shaded = range over 3 independent draws)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_rank_span_distribution(fname):
    ranks = pd.read_csv(os.path.join(RESULTS_DIR, 'cf_dependence_rank_intervals.csv'))
    n = len(ranks)
    pct = 100 * ranks['rank_span'] / n
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pct, bins=40, color=INK, edgecolor='white')
    ax.axvline(pct.median(), color=ACCENT, linestyle='--', linewidth=1.8,
               label=f'median = {pct.median():.0f}% of the field')
    ax.set_xlabel(f'Width of 95% rank interval, as % of the {n}-player field', fontsize=13)
    ax.set_ylabel('Players', fontsize=13)
    ax.set_title('How precisely each player can be placed in the ranking', fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


if __name__ == '__main__':
    plot_calibration(os.path.join(OUT_DIR, 'ep_calibration_curve.png'))
    plot_ci_impact(os.path.join(OUT_DIR, 'ci_impact_50_vs_531.png'))
    plot_convergence(os.path.join(OUT_DIR, 'sample_size_convergence.png'))
    plot_rank_span_distribution(os.path.join(OUT_DIR, 'rank_span_distribution.png'))
    print(f'Wrote 4 figures to {OUT_DIR}/')
