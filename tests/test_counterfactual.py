import os

import numpy as np
import pandas as pd
import pytest

from counterfactual import (
    HOOPS, apply_closing_penalty, best_feasible_two, build_support_grid, candidate_grid,
    defender_features, feature_matrix, in_support, is_three, nearest_feasible_two,
    recompute_features, shot_distance, MODEL_FEATURES,
)

DEF_COLS = [f'def{i}_{a}' for i in range(1, 6) for a in ('x', 'y')]


def one_shot():
    """A shot from the right wing with five defenders scattered."""
    return pd.Series({
        'shooter_x': 70.0, 'shooter_y': 40.0,
        'def1_x': 72.0, 'def1_y': 38.0, 'def2_x': 80.0, 'def2_y': 25.0,
        'def3_x': 85.0, 'def3_y': 30.0, 'def4_x': 78.0, 'def4_y': 45.0,
        'def5_x': 88.0, 'def5_y': 20.0,
        'seconds_rem': 400.0, 'streak': 1.0, 'shot_clock': 12.0,
        'home': 1.0, 'def_hull_area': 150.0,
        'close_def_dist': np.hypot(70.0 - 72.0, 40.0 - 38.0),
    })


# --------------------------------------------------------------------------------------
# Task 2: geometry and feature recomputation
# --------------------------------------------------------------------------------------

def test_shot_distance_uses_the_nearer_hoop():
    # (5, 25) and (89, 25); a point at x=70 is nearer the right hoop
    assert np.isclose(shot_distance(70.0, 25.0), 19.0)
    assert np.isclose(shot_distance(20.0, 25.0), 15.0)


def test_shot_distance_matches_the_pipeline_definition():
    """Must reproduce compute_shot_dist in pbp_shot_processing, or the model is fed
    features on a different scale than it was trained on."""
    for x, y in [(5.0, 25.0), (89.0, 25.0), (47.0, 25.0), (70.0, 40.0)]:
        expected = min(np.hypot(x - 5, y - 25), np.hypot(x - 89, y - 25))
        assert np.isclose(shot_distance(x, y), expected)


def test_defender_features_are_distances_from_the_candidate_location():
    row = one_shot()
    close, avg = defender_features(row, 70.0, 40.0)
    dists = [np.hypot(70.0 - row[f'def{i}_x'], 40.0 - row[f'def{i}_y']) for i in range(1, 6)]
    assert np.isclose(close, min(dists))
    assert np.isclose(avg, float(np.mean(dists)))


def test_hull_area_is_invariant_to_the_shooter_location():
    """def_hull_area is the defenders' own convex hull, so relocating the shooter cannot move it."""
    row = one_shot()
    a = recompute_features(row, 70.0, 40.0)
    b = recompute_features(row, 30.0, 10.0)
    assert a['def_hull_area'] == b['def_hull_area'] == row['def_hull_area']


def test_unchanged_features_are_carried_through():
    row = one_shot()
    f = recompute_features(row, 30.0, 10.0)
    for col in ('seconds_rem', 'streak', 'shot_clock', 'home'):
        assert f[col] == row[col]


def test_recomputing_a_shot_at_its_own_location_is_the_identity():
    """Round trip: the recomputed features at the observed location must equal the stored ones."""
    row = one_shot()
    row['shot_dist'] = shot_distance(row.shooter_x, row.shooter_y)
    close, avg = defender_features(row, row.shooter_x, row.shooter_y)
    row['close_def_dist'], row['avg_def_dist'] = close, avg

    f = recompute_features(row, row.shooter_x, row.shooter_y)
    for col in ('shot_dist', 'close_def_dist', 'avg_def_dist'):
        assert np.isclose(f[col], row[col]), col


def test_is_three_agrees_with_the_pipeline_classifier():
    assert is_three(5.25, 25.0 + 30.0)      # deep arc three
    assert is_three(70.0, 3.0)              # right corner three
    assert not is_three(20.0, 25.0)         # mid-range two
    assert not is_three(89.0, 25.0)         # at the rim


def test_feature_matrix_agrees_with_the_scalar_path():
    """The vectorized path the driver uses must equal the scalar one the tests validate."""
    row = one_shot()
    pts = np.array([[70.0, 40.0], [30.0, 10.0], [80.0, 22.0]])
    m = feature_matrix(row, pts)
    assert m.shape == (3, len(MODEL_FEATURES))
    for i, (px, py) in enumerate(pts):
        f = recompute_features(row, px, py)
        for j, col in enumerate(MODEL_FEATURES):
            assert np.isclose(m[i, j], float(f[col])), (i, col)


# --------------------------------------------------------------------------------------
# Task 3: common support
# --------------------------------------------------------------------------------------

def test_support_mask_accepts_dense_regions_and_rejects_empty_ones():
    from counterfactual import build_support_grid, in_support
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'shooter_x': rng.normal(70, 4, 4000),
                        'shooter_y': rng.normal(25, 4, 4000)})
    grid = build_support_grid(obs, cell_size=2.0, min_count=5)
    assert in_support(grid, 70.0, 25.0)        # dense centre
    assert not in_support(grid, 10.0, 2.0)     # nowhere near any observed shot


def test_support_grid_respects_the_minimum_count():
    obs = pd.DataFrame({'shooter_x': [70.0] * 4, 'shooter_y': [25.0] * 4})
    assert not in_support(build_support_grid(obs, 2.0, min_count=5), 70.0, 25.0)
    assert in_support(build_support_grid(obs, 2.0, min_count=4), 70.0, 25.0)


# --------------------------------------------------------------------------------------
# Task 4: nearest feasible two
# --------------------------------------------------------------------------------------

def test_candidate_grid_stays_inside_the_arc_and_on_the_shooters_half():
    row = one_shot()
    pts = candidate_grid(row, cell_size=1.0, max_radius=None)
    assert len(pts) > 0
    assert not any(is_three(px, py) for px, py in pts)
    assert (pts[:, 0] >= 47.0).all()       # shooter is on the right half


def test_nearest_two_is_inside_the_arc():
    row = one_shot()
    grid = None  # support check disabled for the unit test
    res = nearest_feasible_two(row, kappa=0.0, support=grid)
    assert res is not None
    assert not is_three(res['x'], res['y'])


def test_kappa_zero_picks_a_closer_point_than_kappa_one():
    """Requiring equal openness must push the substitute further away, or the constraint
    is doing nothing and the counterfactual is degenerate."""
    row = one_shot()
    lax = nearest_feasible_two(row, kappa=0.0, support=None)
    strict = nearest_feasible_two(row, kappa=1.0, support=None)
    if strict is not None:
        assert strict['displacement'] >= lax['displacement']


def test_returns_none_when_no_candidate_meets_the_openness_requirement():
    row = one_shot()
    assert nearest_feasible_two(row, kappa=99.0, support=None) is None


def test_nearest_two_retains_the_required_openness():
    row = one_shot()
    res = nearest_feasible_two(row, kappa=1.0, support=None)
    if res is not None:
        assert res['close_def_dist'] >= float(row['close_def_dist']) - 1e-9


# --------------------------------------------------------------------------------------
# Task 5: best feasible two
# --------------------------------------------------------------------------------------

def test_best_two_returns_a_high_value_candidate():
    row = one_shot()
    scorer = lambda feats: np.array([1.0 - f['shot_dist'] / 100.0 for f in feats])
    res = best_feasible_two(row, scorer, quantile=1.0, support=None)
    assert res is not None and not is_three(res['x'], res['y'])


def test_quantile_is_never_above_the_maximum():
    """The winner's-curse guard must be a shrinkage, not an inflation."""
    row = one_shot()
    scorer = lambda feats: np.array([1.0 - f['shot_dist'] / 100.0 for f in feats])
    top = best_feasible_two(row, scorer, quantile=1.0, support=None)
    q90 = best_feasible_two(row, scorer, quantile=0.90, support=None)
    assert q90['p_hat'] <= top['p_hat'] + 1e-12


def test_closing_penalty_reduces_openness_with_travel_distance():
    assert apply_closing_penalty(10.0, travel=0.0, lam=0.5) == 10.0
    assert apply_closing_penalty(10.0, travel=8.0, lam=0.5) == 6.0
    assert apply_closing_penalty(1.0, travel=20.0, lam=0.5) == 0.0   # floored, never negative


def test_closing_penalty_moves_the_chosen_spot_closer_to_the_shooter():
    """With travel penalised, a far-away open spot should stop looking so attractive."""
    row = one_shot()
    scorer = lambda feats: np.array([f['close_def_dist'] for f in feats])
    frozen = best_feasible_two(row, scorer, quantile=1.0, support=None, lam=0.0)
    penalised = best_feasible_two(row, scorer, quantile=1.0, support=None, lam=1.0)
    assert penalised['displacement'] <= frozen['displacement']


# --------------------------------------------------------------------------------------
# Task 2 Step 5: the round-trip validation against real data
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_round_trip_reproduces_stored_features_on_real_data():
    """If feature recomputation is correct, a real shot scored at its own location through
    the counterfactual path must return exactly its stored feature values."""
    from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE
    if not os.path.exists(ALT_SHOTS_EXP_PTS_FILE):
        pytest.skip('run the pipeline first')

    df = pd.read_csv(ALT_SHOTS_EXP_PTS_FILE).sample(500, random_state=0)
    for _, row in df.iterrows():
        f = recompute_features(row, row['shooter_x'], row['shooter_y'])
        assert np.isclose(f['shot_dist'], row['shot_dist'], atol=1e-6)
        assert np.isclose(f['close_def_dist'], row['close_def_dist'], atol=1e-6)
        assert np.isclose(f['avg_def_dist'], row['avg_def_dist'], atol=1e-6)


@pytest.mark.slow
def test_stored_three_point_flag_agrees_with_the_geometry_helper():
    """is_three must reproduce the pipeline's stored 3pt label on real coordinates."""
    from cf_dep_processing import ALT_SHOTS_EXP_PTS_FILE
    if not os.path.exists(ALT_SHOTS_EXP_PTS_FILE):
        pytest.skip('run the pipeline first')

    df = pd.read_csv(ALT_SHOTS_EXP_PTS_FILE).sample(2000, random_state=0)
    got = np.array([is_three(r.shooter_x, r.shooter_y) for r in df.itertuples()])
    assert (got.astype(int) == df['3pt'].to_numpy()).all()
