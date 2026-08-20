import numpy as np
import pandas as pd
import pytest

from cross_fit import make_splits, cross_fitted_probabilities

FEATURES = ['f0', 'f1', 'f2', 'f3']
PARAMS = {'n_estimators': 60, 'max_depth': 5, 'learning_rate': 0.1}


def noise_frame(n_games=40, shots_per_game=15, seed=0):
    """Shots whose outcome is pure coin-flip noise, unrelated to any feature."""
    rng = np.random.default_rng(seed)
    n = n_games * shots_per_game
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df['game_id'] = np.repeat(np.arange(n_games), shots_per_game)
    df['fgm'] = rng.integers(0, 2, n)
    return df


def test_every_row_receives_a_prediction():
    df = noise_frame()
    p = cross_fitted_probabilities(df, FEATURES, 'fgm', PARAMS, n_splits=5)
    assert len(p) == len(df)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_no_game_appears_in_both_train_and_test():
    df = noise_frame()
    for train_idx, test_idx in make_splits(df, 'fgm', 'game_id', n_splits=5, random_state=0):
        train_games = set(df['game_id'].iloc[train_idx])
        test_games = set(df['game_id'].iloc[test_idx])
        assert train_games.isdisjoint(test_games)


def test_every_row_is_tested_exactly_once():
    df = noise_frame()
    counts = np.zeros(len(df), dtype=int)
    for _, test_idx in make_splits(df, 'fgm', 'game_id', n_splits=5, random_state=0):
        counts[test_idx] += 1
    assert (counts == 1).all()


def test_cross_fitting_does_not_reproduce_the_in_sample_leak():
    """On pure noise, an in-sample fit looks skillful and an out-of-fold fit does not."""
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    df = noise_frame()
    y = df['fgm'].to_numpy()

    in_sample_model = xgb.XGBClassifier(**PARAMS)
    in_sample_model.fit(df[FEATURES], y)
    in_sample_auc = roc_auc_score(y, in_sample_model.predict_proba(df[FEATURES])[:, 1])

    oof_auc = roc_auc_score(y, cross_fitted_probabilities(df, FEATURES, 'fgm', PARAMS, n_splits=5))

    assert in_sample_auc > 0.90, f'expected the leak to show up, got {in_sample_auc:.3f}'
    assert oof_auc < 0.60, f'cross-fitted AUC should be near chance, got {oof_auc:.3f}'


def test_raises_when_a_row_would_go_unpredicted(monkeypatch):
    df = noise_frame(n_games=6, shots_per_game=5)
    import cross_fit

    original = cross_fit.make_splits   # capture before patching, or truncated_splits recurses

    def truncated_splits(*args, **kwargs):
        return original(*args, **kwargs)[:1]

    monkeypatch.setattr(cross_fit, 'make_splits', truncated_splits)
    with pytest.raises(ValueError, match='out-of-fold'):
        cross_fit.cross_fitted_probabilities(df, FEATURES, 'fgm', PARAMS, n_splits=3)
