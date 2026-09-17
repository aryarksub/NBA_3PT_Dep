"""What does sample size alone buy? Rerun the whole metric on a subsample of games.

The earlier 8-game figures are not a clean comparison against the full run: they were produced
before the unmatched-shot labelling fix and under lowered reporting thresholds, so they differ
from the 531-game results in three ways at once. This script changes exactly one thing --
the number of games -- holding code, thresholds, and model settings fixed.

Everything downstream of the raw feature table is recomputed per subsample, including the
cross-fitted shot probabilities and the per-player ep_2pt baseline. Reusing the full-sample
model would hand the subsample 531 games' worth of training and defeat the purpose.

    python src/experiments/cf_dependence_subsample.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import (
    MODEL_FEATURES, MODEL_PARAMS, add_alt_exp_pts_cols, add_exp_pts_col, create_base_cf_dep_df,
    create_base_df,
)
from ci_impact import ci_impact_summary, rank_intervals
from dependence import (
    add_shot_dependence, aggregate_dependence, bootstrap_dependence,
    bootstrap_draw_matrix, brier_score, signal_to_noise,
)
from pbp_shot_processing import FINAL_FILE

RESULTS_DIR = 'results'
OUT_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_subsample_sweep.csv')

# Held identical to src/experiments/cf_dependence.py so the comparison is like for like.
MIN_SHOTS = 50
MIN_3PA = 10
N_BOOT = 1000
PRIMARY_STAT = 'dep_share'
TOP_K = 10

GAME_COUNTS = [50, 150, 300]
N_REPS = 3          # independent draws per size, to show spread rather than one lucky sample

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(shots_raw, label, seed):
    """Full metric pipeline on one subsample. Returns a dict of headline diagnostics."""
    ep = add_exp_pts_col(shots_raw, list(MODEL_FEATURES), 'fgm', dict(MODEL_PARAMS),
                         n_splits=5, random_state=0)
    baseline = create_base_cf_dep_df(ep, group=['player_id'])
    ep = add_alt_exp_pts_cols(ep, baseline_df=baseline)
    shots = add_shot_dependence(ep)

    y, p = ep['fgm'].to_numpy(), ep['shot_prob'].to_numpy()
    base = y.mean()
    model_brier = brier_score(y, p)
    base_brier = float(np.mean((base - y) ** 2))

    agg = aggregate_dependence(shots, group=['player_id'])
    agg['qualified'] = (agg['n_shots'] >= MIN_SHOTS) & (agg['n_3pa'] >= MIN_3PA)
    qual = agg[agg['qualified']].copy()

    row = {
        'label': label, 'seed': seed,
        'n_games': shots['game_id'].nunique(), 'n_shots': len(shots),
        'n_players': agg['player_id'].nunique(), 'n_qualified': len(qual),
        'auc': roc_auc_score(y, p), 'brier': model_brier,
        'skill_score': 1 - model_brier / base_brier,
    }

    if len(qual) < 5:
        # Too few players to say anything about a ranking; record the counts and stop.
        row.update({k: np.nan for k in
                    ('reliability', 'tau', 'spearman_vs_3pa_rate', 'fraction_excluding_zero',
                     'pairwise_resolvability', 'median_rank_span_pct', 'topk_overlap')})
        return row

    sub = shots[shots['player_id'].isin(qual['player_id'])]
    boot = bootstrap_dependence(sub, group=['player_id'], stat=PRIMARY_STAT,
                                n_boot=N_BOOT, random_state=0)
    qual = qual.merge(boot, on='player_id', how='left')

    keys, draws = bootstrap_draw_matrix(sub, group=['player_id'], stat=PRIMARY_STAT,
                                        n_boot=N_BOOT, random_state=0)
    point = keys.merge(qual, on='player_id', how='left')
    summary = ci_impact_summary(point, keys, draws, stat=PRIMARY_STAT,
                                k=min(TOP_K, len(keys)))
    snr = signal_to_noise(qual, PRIMARY_STAT, f'{PRIMARY_STAT}_se')
    ranks = rank_intervals(keys, draws)

    row.update({
        'reliability': snr['reliability'], 'tau': snr['tau'],
        'spearman_vs_3pa_rate': qual[PRIMARY_STAT].corr(qual['3pa_rate'], method='spearman'),
        'fraction_excluding_zero': summary['fraction_excluding_zero'],
        'pairwise_resolvability': summary['pairwise_resolvability'],
        'median_rank_span_pct': 100 * ranks['rank_span'].median() / len(keys),
        'topk_overlap': summary['topk_overlap'],
    })
    return row


if __name__ == '__main__':
    # Same column selection and 3pt renaming the main pipeline uses, so the subsamples are
    # preprocessed identically to the full run.
    full = create_base_df()
    all_games = np.sort(full['game_id'].unique())
    print(f'{FINAL_FILE}: {len(full):,} shots over {len(all_games)} games\n')

    rows = []
    for n_games in GAME_COUNTS:
        if n_games > len(all_games):
            continue
        for rep in range(N_REPS):
            rng = np.random.default_rng(1000 * n_games + rep)
            pick = rng.choice(all_games, size=n_games, replace=False)
            sub = full[full['game_id'].isin(pick)].copy()
            print(f'--- {n_games} games, replicate {rep + 1}/{N_REPS} '
                  f'({len(sub):,} shots) ---', flush=True)
            rows.append(evaluate(sub, f'{n_games} games', rep))

    print(f'--- full sample ({len(all_games)} games) ---', flush=True)
    rows.append(evaluate(full.copy(), f'{len(all_games)} games (full)', 0))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)

    show = ['label', 'n_shots', 'n_qualified', 'auc', 'skill_score', 'reliability',
            'spearman_vs_3pa_rate', 'fraction_excluding_zero', 'pairwise_resolvability',
            'median_rank_span_pct', 'topk_overlap']
    print('\n' + '=' * 100)
    print(out[show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('=' * 100)

    print('\nMean over replicates by sample size:')
    agg = out.groupby('label', sort=False)[show[1:]].mean()
    print(agg.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'\nWritten to {OUT_FILE}')
