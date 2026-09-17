"""Alternative-action counterfactuals: what a three-point attempt could have been instead.

Every function here answers one question -- if this shot had been taken from somewhere else,
with the defense frozen where it actually was, what would the model have predicted? The
frozen-defense assumption is the central limitation and is stated wherever it bites.

Feature definitions must match pbp_shot_processing exactly. The model was trained on those
definitions; a counterfactual computed on a different scale is not comparable to the observed
side of the difference.

This module imports nothing from the pipeline, so it stays cheap to unit-test.
"""

from functools import lru_cache

import numpy as np
from nba_geometry import is_three, is_three_vec

# Matches compute_shot_dist in pbp_shot_processing: hoop centres at (5, 25) and (89, 25),
# with the shorter of the two distances taken so the attacking end need not be known.
# Note this differs by a quarter foot from the 5.25 / 88.75 used by the 3PT geometry check;
# both are reproduced faithfully rather than reconciled, because changing either would mean
# retraining the model.
HOOPS = ((5.0, 25.0), (89.0, 25.0))

# 3PT geometry is shared with pbp_shot_processing through nba_geometry.
THREE_HOOPS = (5.25, 88.75)
ARC_RADIUS = 23.75
HOOP_Y = 25.0

COURT_X, COURT_Y = 94.0, 50.0
HALF_X = 47.0

DEF_IDX = range(1, 6)
CARRIED = ('seconds_rem', 'streak', 'shot_clock', 'home', 'def_hull_area')
MODEL_FEATURES = ['seconds_rem', 'streak', 'shot_dist', 'shot_clock',
                  'close_def_dist', 'avg_def_dist', 'def_hull_area', 'home']


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def shot_distance(x, y):
    """Distance to the nearer hoop, as pbp_shot_processing.compute_shot_dist defines it."""
    return min(float(np.hypot(x - hx, y - hy)) for hx, hy in HOOPS)


def shot_distance_vec(pts):
    """shot_distance for an (n, 2) array of points."""
    pts = np.asarray(pts, dtype=float)
    d = np.stack([np.hypot(pts[:, 0] - hx, pts[:, 1] - hy) for hx, hy in HOOPS], axis=1)
    return d.min(axis=1)


def defender_positions(row):
    """The five defender (x, y) pairs for one shot."""
    return np.array([[row[f'def{i}_x'], row[f'def{i}_y']] for i in DEF_IDX], dtype=float)


def defender_features(row, x, y):
    """(closest, average) defender distance from a candidate location."""
    d = np.hypot(*(defender_positions(row) - np.array([x, y])).T)
    return float(d.min()), float(d.mean())


def defender_features_vec(defs, pts):
    """(closest, average) defender distance for every point in an (n, 2) array."""
    pts = np.asarray(pts, dtype=float)
    d = np.hypot(pts[:, 0, None] - defs[None, :, 0], pts[:, 1, None] - defs[None, :, 1])
    return d.min(axis=1), d.mean(axis=1)


def recompute_features(row, x, y):
    """The full model feature vector for this shot's context, taken from (x, y) instead."""
    close, avg = defender_features(row, x, y)
    out = {c: row[c] for c in CARRIED}
    out['shot_dist'] = shot_distance(x, y)
    out['close_def_dist'] = close
    out['avg_def_dist'] = avg
    return out


def feature_matrix(row, pts, close=None, avg=None):
    """Model-ready (n, 8) feature matrix for candidate locations, in MODEL_FEATURES order.

    The vectorized twin of recompute_features. `close` and `avg` may be supplied when they
    have already been computed (and possibly penalised) by the caller.
    """
    pts = np.asarray(pts, dtype=float)
    if close is None or avg is None:
        close, avg = defender_features_vec(defender_positions(row), pts)

    n = len(pts)
    cols = {
        'seconds_rem': np.full(n, float(row['seconds_rem'])),
        'streak': np.full(n, float(row['streak'])),
        'shot_dist': shot_distance_vec(pts),
        'shot_clock': np.full(n, float(row['shot_clock'])),
        'close_def_dist': np.asarray(close, dtype=float),
        'avg_def_dist': np.asarray(avg, dtype=float),
        'def_hull_area': np.full(n, float(row['def_hull_area'])),
        'home': np.full(n, float(row['home'])),
    }
    return np.stack([cols[c] for c in MODEL_FEATURES], axis=1)


# --------------------------------------------------------------------------------------
# Common support
# --------------------------------------------------------------------------------------

def build_support_grid(shots_df, cell_size=2.0, min_count=5,
                       x_col='shooter_x', y_col='shooter_y'):
    """Occupancy grid of observed shot locations, in feet.

    Returns a dict with the cell size and the set of occupied cells. A counterfactual location
    outside this set is one no player in the sample shot from, so the model has no evidence
    there and its prediction is extrapolation.
    """
    xs = np.floor(shots_df[x_col].to_numpy(float) / cell_size).astype(int)
    ys = np.floor(shots_df[y_col].to_numpy(float) / cell_size).astype(int)
    cells, counts = np.unique(np.stack([xs, ys], axis=1), axis=0, return_counts=True)
    keep = {tuple(c) for c, n in zip(cells.tolist(), counts.tolist()) if n >= min_count}
    return {'cell_size': cell_size, 'cells': keep}


def in_support(grid, x, y):
    c = grid['cell_size']
    return (int(np.floor(x / c)), int(np.floor(y / c))) in grid['cells']


def in_support_vec(grid, pts):
    """in_support for an (n, 2) array of points."""
    pts = np.asarray(pts, dtype=float)
    c = grid['cell_size']
    xs = np.floor(pts[:, 0] / c).astype(int)
    ys = np.floor(pts[:, 1] / c).astype(int)
    cells = grid['cells']
    return np.fromiter(((int(a), int(b)) in cells for a, b in zip(xs, ys)),
                       dtype=bool, count=len(pts))


# --------------------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def half_court_twos(attacking_right, cell_size=1.0):
    """Every two-point grid location on one half of the court.

    Cached: the driver calls it once per shot and the answer depends only on which half.
    """
    x_lo, x_hi = (HALF_X, COURT_X) if attacking_right else (0.0, HALF_X)
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")
    xs = np.arange(int(np.ceil(x_lo / cell_size)), int(np.floor(x_hi / cell_size)) + 1) * cell_size
    ys = np.arange(int(np.floor(COURT_Y / cell_size)) + 1) * cell_size
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return pts[~is_three_vec(pts)]


def candidate_grid(row, cell_size=1.0, max_radius=None):
    """Candidate two-point locations on the shooter's own half, on a regular grid.

    Restricted to the attacking half so a counterfactual never lands at the far basket.
    """
    x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
    pts = half_court_twos(bool(x0 >= HALF_X), cell_size)

    if max_radius is not None:
        pts = pts[np.hypot(pts[:, 0] - x0, pts[:, 1] - y0) <= max_radius]
    return pts


def feasible_candidates(row, kappa=None, support=None, cell_size=1.0, max_radius=None,
                        lam=0.0):
    """Candidate points that survive the support check, with their defender features.

    Returns (points, close_def_dist, avg_def_dist). `close_def_dist` already carries the
    closing penalty when lam > 0. When kappa is given, only points retaining at least kappa
    of the shot's openness are kept.
    """
    pts = candidate_grid(row, cell_size, max_radius)
    if len(pts) and support is not None:
        pts = pts[in_support_vec(support, pts)]
    if not len(pts):
        return pts, np.empty(0), np.empty(0)

    x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
    close, avg = defender_features_vec(defender_positions(row), pts)

    if lam:
        travel = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)
        close = np.maximum(0.0, close - lam * travel)

    if kappa is not None:
        ok = close >= kappa * float(row['close_def_dist'])
        pts, close, avg = pts[ok], close[ok], avg[ok]

    return pts, close, avg


# --------------------------------------------------------------------------------------
# Strategy: nearest feasible two
# --------------------------------------------------------------------------------------

def nearest_feasible_two(row, kappa=1.0, support=None, cell_size=1.0, max_radius=None):
    """Closest two-point location that retains at least kappa of the shot's openness.

    kappa = 1.0 demands a substitute at least as open as the three that was taken; kappa = 0
    imposes no openness requirement and degenerates to the nearest point inside the arc, which
    is retained only as a reference. Returns None when no candidate qualifies, which is a
    meaningful outcome: no comparable two was available.
    """
    x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
    pts, close, avg = feasible_candidates(row, kappa=kappa, support=support,
                                          cell_size=cell_size, max_radius=max_radius)
    if not len(pts):
        return None

    disp = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)
    j = int(np.argmin(disp))
    return {'x': float(pts[j, 0]), 'y': float(pts[j, 1]),
            'displacement': float(disp[j]), 'close_def_dist': float(close[j]),
            'avg_def_dist': float(avg[j])}


# --------------------------------------------------------------------------------------
# Strategy: best feasible two
# --------------------------------------------------------------------------------------

def apply_closing_penalty(close_def_dist, travel, lam):
    """Shrink openness to reflect defenders closing while the shooter relocates.

    The frozen-defence assumption is most damaging for the best-feasible strategy, which
    actively seeks out spots where the defence happens not to be. lam is the feet of closure
    per foot travelled; lam = 0 is the frozen-defence case, retained for comparison.
    """
    return max(0.0, float(close_def_dist) - lam * float(travel))


def pick_at_quantile(p, quantile):
    """Index of the candidate sitting at a high quantile of predicted value.

    Not the argmax. The argmax over many noisy predictions is selected partly on its own
    error and so overestimates; a high quantile removes most of that winner's curse. The
    guard is a shrinkage -- the chosen value can never exceed the maximum.
    """
    p = np.asarray(p, dtype=float)
    target = np.quantile(p, quantile)
    return int(np.argmin(np.abs(p - target)))


def best_feasible_two(row, scorer, quantile=0.90, support=None, lam=0.0,
                      cell_size=1.0, max_radius=None):
    """Highest-value two-point location, guarded against both optimism biases.

    `scorer` maps a list of feature dicts to predicted make probabilities, and must be backed
    by the fold model that produced this shot's own out-of-fold probability.

    quantile < 1 replaces the raw maximum with a high quantile of candidate values, which
    removes most of the winner's curse. lam > 0 penalises openness by travel distance.
    """
    x0, y0 = float(row['shooter_x']), float(row['shooter_y'])
    pts, close, avg = feasible_candidates(row, kappa=None, support=support,
                                          cell_size=cell_size, max_radius=max_radius,
                                          lam=lam)
    if not len(pts):
        return None

    feats = []
    for (px, py), c, a in zip(pts, close, avg):
        f = {col: row[col] for col in CARRIED}
        f['shot_dist'] = shot_distance(px, py)
        f['close_def_dist'] = float(c)
        f['avg_def_dist'] = float(a)
        feats.append(f)

    p = np.asarray(scorer(feats), dtype=float)
    j = pick_at_quantile(p, quantile)
    return {'x': float(pts[j, 0]), 'y': float(pts[j, 1]), 'p_hat': float(p[j]),
            'displacement': float(np.hypot(pts[j, 0] - x0, pts[j, 1] - y0)),
            'close_def_dist': float(close[j]), 'avg_def_dist': float(avg[j])}
