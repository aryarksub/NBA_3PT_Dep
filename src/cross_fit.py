"""Out-of-fold (cross-fitted) shot probabilities.

The expected-points model must never score a shot it was trained on. Every downstream
dependence number is a difference of expected-points values, so an in-sample fit inflates
the observed side of that difference and biases the metric by an unknown amount.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(df, out_feature='fgm', group_col='game_id', n_splits=5, random_state=0):
    """
    Train/test index pairs that are stratified on the shot outcome and grouped by game.

    Grouping by game means no model is ever asked to score a shot from a game it has
    partially seen. Returns a list of (train_idx, test_idx) positional index arrays.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(df, df[out_feature].astype(int), df[group_col]))


def cross_fitted_probabilities(
    df, in_features, out_feature='fgm', model_params=None,
    group_col='game_id', n_splits=5, random_state=0, verbose=False,
    return_models=False
):
    """
    Out-of-fold predicted probability of a made shot, one value per row of df.

    Each row is scored by a model fitted on the folds that exclude that row's game.
    Returns a 1-D numpy array aligned to df's positional order.

    With return_models=True, also returns the fitted fold models and, for every row, the
    index of the fold model that scored it: (oof, models, fold_of_row).

    Those are needed to score anything *other* than the observed shot -- a counterfactual
    alternative action, say. Such a value must come from the same fold model that produced
    the shot's own out-of-fold probability. Refitting on all data and scoring the
    counterfactual with that would leave the observed side of the difference out-of-fold and
    the counterfactual side in-sample, which is the leak cross-fitting exists to remove.
    """
    model_params = {} if model_params is None else dict(model_params)
    X = df[in_features]
    y = df[out_feature].astype(int)

    oof = np.full(len(df), np.nan)
    models = []
    fold_of_row = np.full(len(df), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(
        make_splits(df, out_feature, group_col, n_splits, random_state)
    ):
        model = xgb.XGBClassifier(**model_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        models.append(model)
        fold_of_row[test_idx] = fold
        if verbose:
            print(f'  fold {fold + 1}/{n_splits}: trained on {len(train_idx)}, scored {len(test_idx)}')

    missing = int(np.isnan(oof).sum())
    if missing:
        raise ValueError(f'{missing} rows received no out-of-fold prediction')

    if not return_models:
        return oof

    unassigned = int((fold_of_row < 0).sum())
    if unassigned:
        raise ValueError(f'{unassigned} rows were not assigned to any fold model')

    return oof, models, fold_of_row
