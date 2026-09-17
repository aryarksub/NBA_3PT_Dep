"""Prepare strictly forward-time shot predictions and 8-ft counterfactuals.

Models train only on the earliest 40% of observed dates. Earlier shots receive
game-grouped out-of-fold predictions; later shots use the early-only model.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb
from cf_dep_processing import MODEL_FEATURES, MODEL_PARAMS
from cross_fit import cross_fitted_probabilities
import candidate_search_sweep as sweep

OUT = Path('results/metric_validation/cache')


def chronological_blocks(dates):
    dates = pd.to_datetime(dates)
    unique = np.sort(dates.unique())
    a, b = int(len(unique) * .4), int(len(unique) * .7)
    if a < 2 or b <= a or b >= len(unique):
        raise ValueError('Insufficient distinct dates for three chronological blocks')
    return np.where(dates < unique[a], 0, np.where(dates < unique[b], 1, 2))


def load_inputs():
    shots = pd.read_csv('data/alt_exp_pts.csv')
    pbp = pd.read_csv('data/pbp_final.csv')
    for key in ['game_id', 'player_id', 'fgm', 'shooter_x', 'shooter_y']:
        if not np.allclose(shots[key], pbp[key], equal_nan=True):
            raise ValueError(f'Input row alignment failed: {key}')
    return shots, pbp, chronological_blocks(pbp.game_date)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shots, pbp, block = load_inputs()
    signature = hashlib.sha256(b''.join(Path(p).read_bytes() for p in
        ['data/alt_exp_pts.csv', 'data/pbp_final.csv', __file__, 'src/cross_fit.py'])).hexdigest()
    meta_path = OUT / 'models.json'
    if meta_path.exists():
        if json.loads(meta_path.read_text())['input_signature'] != signature:
            raise ValueError('Stale temporal models; archive cache before rebuilding')
        models = []
        for i in range(6):
            model = xgb.XGBClassifier(); model.load_model(OUT / f'fold_{i}.ubj')
            models.append(model)
        prob = np.load(OUT / 'prob.npy'); folds = np.load(OUT / 'folds.npy')
    else:
        early = block == 0
        params = dict(MODEL_PARAMS); params['n_jobs'] = 4
        p, models, f = cross_fitted_probabilities(shots.loc[early], MODEL_FEATURES,
            model_params=params, return_models=True, verbose=True)
        model = xgb.XGBClassifier(**params)
        model.fit(shots.loc[early, MODEL_FEATURES], shots.loc[early, 'fgm'])
        prob = np.empty(len(shots)); prob[early] = p
        prob[~early] = model.predict_proba(shots.loc[~early, MODEL_FEATURES])[:, 1]
        folds = np.full(len(shots), 5); folds[early] = f
        models.append(model)
        for i, model in enumerate(models): model.save_model(OUT / f'fold_{i}.ubj')
        np.save(OUT / 'prob.npy', prob); np.save(OUT / 'folds.npy', folds)
        meta_path.write_text(json.dumps({'input_signature': signature}, indent=2))
    shots['shot_prob'] = prob
    shots['expected_points'] = prob * (2 + shots['3pt'].to_numpy())
    # Candidate scoring does not use this column; all period baselines are rebuilt downstream.
    sweep.OUT = OUT
    sweep.run_grid(shots, models, folds, .25)
    periods = []
    for k in range(3):
        d = pbp.loc[block == k]
        periods.append({'block': k, 'first_date': str(d.game_date.min()),
            'last_date': str(d.game_date.max()), 'games': int(d.game_id.nunique()),
            'shots': len(d), 'threes': int(shots.loc[block == k, '3pt'].sum())})
    pd.DataFrame(periods).to_csv(OUT.parent / 'periods.csv', index=False)
    print('Temporal model and candidate preparation complete.', flush=True)


if __name__ == '__main__':
    main()
