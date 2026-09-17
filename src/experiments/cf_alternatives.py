"""Location-aware counterfactual strategies, compared against the retained naive baseline.

Adds two alternatives from section 6.4.1 -- nearest feasible two and best feasible two -- as
additional columns beside `exp_pts_naive`, which is read and never written. Runs the existing
dependence stack over every strategy, including naive scored as one of them, and reports how
far the metric moves between specifications. That movement is the specification uncertainty,
which the bootstrap intervals do not capture.

    python src/experiments/cf_alternatives.py

Every counterfactual is scored by the fold model that produced its own shot's out-of-fold
probability. Anything else would leave the observed side of the difference out-of-fold and the
counterfactual side in-sample, which is the leak cross-fitting exists to remove.
"""

import os
import sys
import time
import json
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE, MODEL_FEATURES, MODEL_PARAMS
from counterfactual import (
    candidate_grid, defender_features_vec, defender_positions, feature_matrix,
)
from cross_fit import cross_fitted_probabilities
from dependence import add_shot_dependence, aggregate_dependence, bootstrap_dependence, signal_to_noise

RESULTS_DIR = 'results'
OUT_SHOTS = os.path.join('data', 'cf_alternatives_shots.csv')
COMPARISON_FILE = os.path.join(RESULTS_DIR, 'cf_alternatives_comparison.csv')
PLAYER_FILE = os.path.join(RESULTS_DIR, 'cf_alternatives_player.csv')
DIAGNOSTICS_FILE = os.path.join(RESULTS_DIR, 'cf_alternatives_diagnostics.csv')
PUBLISHED_PLAYER_FILE = os.path.join(RESULTS_DIR, 'cf_dependence_player.csv')

# Feasibility requires the substitute to be inside the arc and within MAX_RADIUS feet of where
# the three was actually taken (a shot 30 feet away happens in a different possession). The
# common-support filter remains available for sensitivity checks but is disabled for now.
CELL_SIZE = 0.25
MAX_RADIUS = 8.0
SUPPORT_CELL = 2.0
SUPPORT_MIN_COUNT = 5
USE_SUPPORT_FILTER = False

KAPPAS = [0.0, 0.5, 0.75, 1.0]
QUANTILES = [1.0, 0.95, 0.90]
LAMBDAS = [0.0, 0.25, 0.5]

CHUNK = 400
N_BOOT = 500
MIN_SHOTS, MIN_3PA = 50, 10
STAT = 'dep_share'
NAIVE_COL = 'exp_pts_naive'

# This is a descriptive ranking-similarity screen, not a validity test. A high correlation
# does not rule out additional information; see the residual repeatability analysis.
DEGENERACY_RHO = 0.95

os.makedirs(RESULTS_DIR, exist_ok=True)


def near_name(k):
    return f'near_k{int(round(k * 100)):03d}'


def best_name(q, lam):
    return f'best_q{int(round(q * 100)):03d}_l{int(round(lam * 100)):03d}'


STRATEGIES = ['naive'] + [near_name(k) for k in KAPPAS] + \
             [best_name(q, l) for l in LAMBDAS for q in QUANTILES]


def non_degenerate(comp, threshold=DEGENERACY_RHO):
    """Strategies below the descriptive correlation threshold."""
    return comp.loc[comp['spearman_vs_3pa_rate'].abs() < threshold, 'strategy'].tolist()


def predict_by_fold(models, fold_idx, X):
    """Score each row with the fold model that owns it."""
    out = np.empty(len(X), dtype=float)
    for f in np.unique(fold_idx):
        m = fold_idx == f
        out[m] = models[int(f)].predict_proba(X[m])[:, 1]
    return out


def build_counterfactuals(shots, models, fold_of_row):
    """One counterfactual expected-points column per strategy, plus per-shot diagnostics.

    Two-point attempts are their own counterfactual and are copied through unchanged. A three
    with no feasible substitute falls back to the naive counterfactual, and the fallback rate
    is reported per strategy -- a strategy that falls back often is telling you that no
    comparable two existed, and its column is a blend, not a pure alternative.
    """
    three_pos = np.flatnonzero(shots['3pt'].to_numpy() == 1)
    n3 = len(three_pos)
    print(f'{n3:,} three-point attempts to substitute')

    if USE_SUPPORT_FILTER:
        from counterfactual import build_support_grid, in_support_vec
        support = build_support_grid(shots[shots['3pt'] == 0], cell_size=SUPPORT_CELL,
                                     min_count=SUPPORT_MIN_COUNT)
        print(f'support grid: {len(support["cells"])} occupied cells of '
              f'{SUPPORT_CELL:.0f}x{SUPPORT_CELL:.0f} feet, built from two-point attempts only')
    else:
        print('support filter: disabled')

    near_cols = [near_name(k) for k in KAPPAS]
    best_cols = [best_name(q, l) for l in LAMBDAS for q in QUANTILES]

    p_hat = {c: np.full(n3, np.nan) for c in near_cols + best_cols}
    disp = {c: np.full(n3, np.nan) for c in near_cols + best_cols}
    loc_x = {c: np.full(n3, np.nan) for c in near_cols + best_cols}
    loc_y = {c: np.full(n3, np.nan) for c in near_cols + best_cols}

    n_no_candidates = 0
    n_cand_before, n_cand_after = 0, 0

    t0 = time.time()
    for start in range(0, n3, CHUNK):
        stop = min(start + CHUNK, n3)
        block = three_pos[start:stop]

        near_feats, near_fold, near_key = [], [], []
        best_feats = {l: [] for l in LAMBDAS}
        best_fold = {l: [] for l in LAMBDAS}
        best_span = {l: [] for l in LAMBDAS}   # (local_i, n_rows) per shot
        chunk_pts = {}                         # local_i -> the shot's candidate points

        for local_i, pos in enumerate(block, start=start):
            row = shots.iloc[pos]
            fold = int(fold_of_row[pos])

            pts = candidate_grid(row, CELL_SIZE, MAX_RADIUS)
            n_cand_before += len(pts)
            if USE_SUPPORT_FILTER and len(pts):
                pts = pts[in_support_vec(support, pts)]
            n_cand_after += len(pts)
            chunk_pts[local_i] = pts
            if not len(pts):
                n_no_candidates += 1
                for l in LAMBDAS:
                    best_span[l].append((local_i, 0))
                continue

            x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
            close_raw, avg = defender_features_vec(defender_positions(row), pts)
            travel = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)

            # --- nearest feasible two, one pick per kappa -------------------------------
            obs_close = float(row['close_def_dist'])
            for k in KAPPAS:
                ok = close_raw >= k * obs_close
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                j = idx[int(np.argmin(travel[idx]))]
                col = near_name(k)
                loc_x[col][local_i], loc_y[col][local_i] = pts[j, 0], pts[j, 1]
                disp[col][local_i] = travel[j]
                near_feats.append(feature_matrix(row, pts[j:j + 1],
                                                 close_raw[j:j + 1], avg[j:j + 1])[0])
                near_fold.append(fold)
                near_key.append((col, local_i))

            # --- best feasible two, one scoring pass per lambda -------------------------
            for l in LAMBDAS:
                close_pen = np.maximum(0.0, close_raw - l * travel)
                best_feats[l].append(feature_matrix(row, pts, close_pen, avg))
                best_fold[l].append(np.full(len(pts), fold))
                best_span[l].append((local_i, len(pts)))

        # --- score the chunk --------------------------------------------------------
        if near_feats:
            X = np.asarray(near_feats, dtype=float)
            p = predict_by_fold(models, np.asarray(near_fold), X)
            for (col, local_i), value in zip(near_key, p):
                p_hat[col][local_i] = value

        for l in LAMBDAS:
            if not best_feats[l]:
                continue
            X = np.concatenate(best_feats[l], axis=0)
            p = predict_by_fold(models, np.concatenate(best_fold[l]), X)

            off = 0
            for local_i, n_rows in best_span[l]:
                if n_rows == 0:
                    continue
                seg = p[off:off + n_rows]
                row = shots.iloc[three_pos[local_i]]
                pts = chunk_pts[local_i]
                x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
                for q in QUANTILES:
                    target = np.quantile(seg, q)
                    j = int(np.argmin(np.abs(seg - target)))
                    col = best_name(q, l)
                    p_hat[col][local_i] = seg[j]
                    loc_x[col][local_i], loc_y[col][local_i] = pts[j, 0], pts[j, 1]
                    disp[col][local_i] = float(np.hypot(pts[j, 0] - x0, pts[j, 1] - y0))
                off += n_rows

        done = stop
        rate = done / max(time.time() - t0, 1e-9)
        print(f'  {done:,}/{n3:,} threes  ({rate:.0f}/s, '
              f'{(n3 - done) / max(rate, 1e-9) / 60:.1f} min left)', flush=True)

    # --- assemble the columns ------------------------------------------------------
    naive = shots[NAIVE_COL].to_numpy(dtype=float)
    obs = shots['expected_points'].to_numpy(dtype=float)
    diagnostics = []

    for col in near_cols + best_cols:
        cf = obs.copy()                       # twos are their own counterfactual
        have = np.isfinite(p_hat[col])
        substituted = np.zeros(len(shots), dtype=bool)

        vals = np.where(have, 2.0 * p_hat[col], naive[three_pos])
        cf[three_pos] = vals
        substituted[three_pos] = have

        shots[f'exp_pts_{col}'] = cf
        diagnostics.append({
            'strategy': col,
            'n_3pa': n3,
            'n_substituted': int(have.sum()),
            'fallback_rate': float(1 - have.mean()),
            'median_displacement': float(np.nanmedian(disp[col])) if have.any() else np.nan,
            'p90_displacement': float(np.nanpercentile(disp[col], 90)) if have.any() else np.nan,
        })

    locations = pd.DataFrame({'pos': three_pos})
    for col in near_cols + best_cols:
        locations[f'{col}_x'] = loc_x[col]
        locations[f'{col}_y'] = loc_y[col]

    diag = pd.DataFrame(diagnostics)
    diag['support_rejection_rate'] = 1 - n_cand_after / max(n_cand_before, 1)
    diag['no_candidate_rate'] = n_no_candidates / n3
    diag['mean_candidates_per_shot'] = n_cand_after / n3
    return shots, diag, locations


def evaluate(shots, column, qualified_ids):
    """Point estimates, bootstrap interval, and reliability for one counterfactual column."""
    slim = shots[['player_id', 'game_id', 'team_id', '3pt', 'expected_points', column]].copy()
    slim = add_shot_dependence(slim, cf_col=column)

    agg = aggregate_dependence(slim, group=['player_id'])
    agg['qualified'] = agg['player_id'].isin(qualified_ids)

    sub = slim[slim['player_id'].isin(qualified_ids)]
    boot = bootstrap_dependence(sub, group=['player_id'], stat=STAT,
                                n_boot=N_BOOT, random_state=0)
    agg = agg.merge(boot, on='player_id', how='left')

    qual = agg[agg['qualified']].copy()
    snr = signal_to_noise(qual, STAT, f'{STAT}_se')
    league = float(slim['dep_delta'].sum() / slim['expected_points'].sum())

    return agg, qual, snr, league


def already_built():
    """The counterfactual columns from a previous run, or None.

    Building them costs a fold refit and many candidate model evaluations, so the
    analysis downstream of them is made rerunnable on its own. Delete OUT_SHOTS to force a
    rebuild after changing anything about how the counterfactuals are generated.
    """
    if not all(os.path.exists(p) for p in [OUT_SHOTS, DIAGNOSTICS_FILE, os.path.join('data', 'cf_alternatives_locations.csv')]):
        return None
    metadata_path = OUT_SHOTS + '.metadata.json'
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as handle:
        if json.load(handle) != cache_signature():
            return None
    df = pd.read_csv(OUT_SHOTS)
    needed = [NAIVE_COL] + [f'exp_pts_{s}' for s in STRATEGIES if s != 'naive']
    if any(c not in df.columns for c in needed):
        return None
    return df


def cache_signature():
    with open(ALT_SHOTS_EXP_PTS_FILE, 'rb') as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    return {'geometry': 'nba_parallel_corners_v2', 'grid_ft': CELL_SIZE,
            'radius_ft': MAX_RADIUS, 'support_filter': USE_SUPPORT_FILTER,
            'source_sha256': source_hash,
            'kappas': KAPPAS, 'quantiles': QUANTILES, 'lambdas': LAMBDAS,
            'support_cell': SUPPORT_CELL, 'support_min_count': SUPPORT_MIN_COUNT,
            'implementation_sha256': hashlib.sha256(b''.join(open(p, 'rb').read() for p in
                [__file__, os.path.join('src','counterfactual.py'), os.path.join('src','nba_geometry.py'), os.path.join('src','cross_fit.py'), os.path.join('src','cf_dep_processing.py')])).hexdigest()}


if __name__ == '__main__':
    cached = already_built()
    if cached is not None:
        print(f'Reusing counterfactual columns from {OUT_SHOTS} '
              f'({len(cached):,} shots); delete it to rebuild')
        shots = cached
        diag = pd.read_csv(DIAGNOSTICS_FILE)
    else:
        print(f'Loading {ALT_SHOTS_EXP_PTS_FILE}')
        shots = pd.read_csv(ALT_SHOTS_EXP_PTS_FILE)
        if NAIVE_COL not in shots.columns:
            raise ValueError(f'{NAIVE_COL} missing; the naive baseline is the reference '
                             'specification and must be present before alternatives are added')
        naive_before = shots[NAIVE_COL].to_numpy(dtype=float).copy()
        print(f'{len(shots):,} shots, {int(shots["3pt"].sum()):,} threes')

        # --- refit the folds and prove they are the ones behind the stored probabilities ---
        print('Refitting the cross-fitting folds to recover the models')
        oof, models, fold_of_row = cross_fitted_probabilities(
            shots, list(MODEL_FEATURES), 'fgm', dict(MODEL_PARAMS),
            group_col='game_id', n_splits=5, random_state=0, verbose=True, return_models=True,
        )
        stored = shots['shot_prob'].to_numpy(dtype=float)
        if not np.allclose(oof, stored, atol=1e-6):
            raise ValueError(
                'refitted out-of-fold probabilities do not match the stored shot_prob column. '
                'The seed, split settings or feature list has drifted, and every counterfactual '
                'scored by these models would be incomparable to the observed side.'
            )
        print(f'fold models recovered; max deviation from stored shot_prob '
              f'{np.abs(oof - stored).max():.2e}')

        # --- build every alternative column --------------------------------------------
        shots, diag, locations = build_counterfactuals(shots, models, fold_of_row)

        if not np.array_equal(naive_before, shots[NAIVE_COL].to_numpy(dtype=float)):
            raise ValueError('exp_pts_naive was modified; it is the reference specification '
                             'and must survive this run untouched')

        diag.to_csv(DIAGNOSTICS_FILE, index=False)
        keep = ['player_id', 'game_id', 'team_id', '3pt', 'shooter_x', 'shooter_y',
                'expected_points', NAIVE_COL] + \
               [f'exp_pts_{s}' for s in STRATEGIES if s != 'naive']
        shots[keep].to_csv(OUT_SHOTS, index=False)
        locations.to_csv(os.path.join('data', 'cf_alternatives_locations.csv'), index=False)
        with open(OUT_SHOTS + '.metadata.json', 'w') as handle:
            json.dump(cache_signature(), handle, indent=2)

    print('\n--- feasibility diagnostics ---')
    print(diag[['strategy', 'fallback_rate', 'median_displacement',
                'p90_displacement']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'support rejection rate {diag["support_rejection_rate"].iloc[0]:.3f}, '
          f'mean candidates per shot {diag["mean_candidates_per_shot"].iloc[0]:.0f}')

    # --- run the metric under every strategy, naive first --------------------------------
    published = pd.read_csv(PUBLISHED_PLAYER_FILE)
    qualified_ids = set(published.loc[published['qualified'], 'player_id'])
    print(f'\n{len(qualified_ids)} qualified players (>= {MIN_SHOTS} shots, >= {MIN_3PA} 3PA)')

    columns = {'naive': NAIVE_COL}
    columns.update({s: f'exp_pts_{s}' for s in STRATEGIES if s != 'naive'})

    rows, per_player = [], {}
    for name in STRATEGIES:
        print(f'evaluating {name}', flush=True)
        agg, qual, snr, league = evaluate(shots, columns[name], qualified_ids)
        per_player[name] = qual.set_index('player_id')

        rows.append({
            'strategy': name,
            'league_dep_share': league,
            'median_player_dep_share': float(qual[STAT].median()),
            'spearman_vs_3pa_rate': float(qual[STAT].corr(qual['3pa_rate'], method='spearman')),
            'reliability': snr['reliability'],
            'tau': snr['tau'],
            'median_ci_width': float((qual[f'{STAT}_ci_upper']
                                      - qual[f'{STAT}_ci_lower']).median()),
        })

    # naive must reproduce the published numbers, or the harness has drifted
    naive_q = per_player['naive']
    ref = published[published['qualified']].set_index('player_id').loc[naive_q.index]
    delta = np.abs(naive_q[STAT].to_numpy() - ref[STAT].to_numpy()).max()
    if delta > 1e-9:
        raise ValueError(f'naive dep_share recomputed by this harness differs from '
                         f'{PUBLISHED_PLAYER_FILE} by {delta:.2e}; the comparison is invalid')
    print(f'\nnaive reproduces the published player results exactly (max deviation {delta:.1e})')

    # --- rank agreement with the retained baseline -------------------------------------
    comparison = pd.DataFrame(rows)
    comparison['spearman_vs_naive'] = [
        float(per_player[s][STAT].corr(naive_q[STAT], method='spearman')) for s in STRATEGIES
    ]

    # --- specification uncertainty versus sampling uncertainty --------------------------
    wide = pd.DataFrame({s: per_player[s][STAT] for s in STRATEGIES})
    spec_range = wide.max(axis=1) - wide.min(axis=1)
    ci_width = naive_q[f'{STAT}_ci_upper'] - naive_q[f'{STAT}_ci_lower']
    ratio = spec_range / ci_width

    player_out = wide.copy()
    player_out['spec_range'] = spec_range
    player_out['naive_ci_width'] = ci_width
    player_out['spec_over_sampling'] = ratio
    player_out['3pa_rate'] = naive_q['3pa_rate']
    player_out['n_shots'] = naive_q['n_shots']

    # Report a below-threshold sensitivity subset alongside all strategies.
    # Neither the subset nor the range constitutes a validity or total-uncertainty test.
    live = non_degenerate(comparison)
    live_range = wide[live].max(axis=1) - wide[live].min(axis=1)
    live_ratio = live_range / ci_width
    player_out['spec_range_non_degenerate'] = live_range
    player_out['spec_over_sampling_non_degenerate'] = live_ratio
    player_out.to_csv(PLAYER_FILE)

    comparison['degenerate'] = ~comparison['strategy'].isin(live)
    comparison.to_csv(COMPARISON_FILE, index=False)

    print('\n--- strategy comparison (naive is the retained reference, first row) ---')
    print(comparison.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print('\n--- specification uncertainty vs sampling uncertainty ---')
    print(f'  median naive bootstrap CI width                      {ci_width.median():.4f}')
    print(f'  all {len(STRATEGIES)} specifications:')
    print(f'    median spread across specifications                {spec_range.median():.4f}')
    print(f'    median ratio to sampling noise                     {ratio.median():.2f}')
    print(f'    players above 1: {int((ratio > 1).sum())}/{len(ratio)} '
          f'({(ratio > 1).mean():.1%})')
    print(f'  {len(live)} below-threshold specifications '
          f'(|rho vs 3PA rate| < {DEGENERACY_RHO}):')
    print(f'    median spread across specifications                {live_range.median():.4f}')
    print(f'    median ratio to sampling noise                     {live_ratio.median():.2f}')
    print(f'    players above 1: {int((live_ratio > 1).sum())}/{len(live_ratio)} '
          f'({(live_ratio > 1).mean():.1%})')
    dropped = [s for s in STRATEGIES if s not in live]
    print(f'  above the correlation threshold: {", ".join(dropped)}')

    print(f'\nWritten {COMPARISON_FILE}, {PLAYER_FILE}, {DIAGNOSTICS_FILE}, {OUT_SHOTS}')
    print('Draw figures with src/experiments/cf_alternatives_figures.py')
