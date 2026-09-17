"""Reusable, outcome-blind matching and forward prediction helpers."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def match_controls(rate, efficiency, rate_limit=.02, efficiency_limit=.05):
    """Greedy disjoint closest pairs, selected using controls only.

    Tie ordering follows input positions. Limits are absolute fractions and EP/shot.
    """
    edges = []
    for i in range(len(rate)):
        for j in range(i + 1, len(rate)):
            dr, de = abs(rate[i] - rate[j]), abs(efficiency[i] - efficiency[j])
            if dr <= rate_limit and de <= efficiency_limit:
                edges.append(((dr/rate_limit)**2 + (de/efficiency_limit)**2, i, j))
    used, pairs = set(), []
    for distance, i, j in sorted(edges):
        if i not in used and j not in used:
            pairs.append((i, j, distance)); used.update([i, j])
    return pairs


def controls(panel):
    r = panel['rate']
    return np.column_stack([r, r*r, r*r*r, panel['mean_ep'],
                            panel['real_ppfga'], np.log1p(panel['n'])])


def fit_predict(x_train, y_train, x_test):
    """Fixed ridge penalty; all scaling is fitted on the training rows only."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
    model.fit(x_train, y_train)
    return model.predict(x_test)


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y)-prediction)**2)))


def shuffled_values(values, groups, rng):
    """Permute only specified strata; unlisted entries (including fallback) stay fixed."""
    result = values.copy()
    for indices in groups:
        result[indices] = values[rng.permutation(indices)]
    return result


def holm_adjust(p):
    """Holm family-wise adjustment in the original hypothesis order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result = np.empty_like(p); result[order] = adjusted
    return result
