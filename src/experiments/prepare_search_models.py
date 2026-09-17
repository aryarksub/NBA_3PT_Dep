"""Recover matched fold models for the optional radius/resolution sweep.

Run after cf_dep_processing.py. Refuses model drift instead of mixing predictions.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from cf_dep_processing import MODEL_FEATURES, MODEL_PARAMS
from cross_fit import cross_fitted_probabilities


def main():
    shots = pd.read_csv('data/alt_exp_pts.csv')
    params = dict(MODEL_PARAMS)
    params['n_jobs'] = 4
    oof, models, folds = cross_fitted_probabilities(
        shots, MODEL_FEATURES, model_params=params, return_models=True, verbose=True)
    if not np.allclose(oof, shots.shot_prob, atol=1e-6, rtol=0):
        raise ValueError('Model refit differs from stored shot_prob. Rebuild expected points '
                         'and dependence with this environment before running the sweep.')
    out = Path('results/search_sensitivity')
    out.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(out / f'fold_{i}.ubj')
    np.save(out / 'fold_of_row.npy', folds)
    print('Saved matched fold models. Existing sweep checkpoints validate their inputs on reuse.')


if __name__ == '__main__':
    main()
