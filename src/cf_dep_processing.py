import pandas as pd
import numpy as np
import os

from cross_fit import cross_fitted_probabilities
from dependence import brier_score, calibration_table
from pbp_shot_processing import FINAL_FILE

SHOTS_EXP_PTS_FILE = os.path.join('data', 'exp_pts.csv')
ALT_SHOTS_EXP_PTS_FILE = os.path.join('data', 'alt_exp_pts.csv')
CF_DEP_PLAYER_FILE = os.path.join('data', 'cf_dep_player.csv')
CF_DEP_TEAM_FILE = os.path.join('data', 'cf_dep_team.csv')


def create_base_df():
    full_df = pd.read_csv(FINAL_FILE)
    keep_features = [
        "game_id", "player_id", "team_id", "period", "seconds_rem", "new_3pt", "fgm", 
        "streak", "shot_dist", "shot_clock", "close_def_id", "close_def_team_id", 
        "close_def_dist", "avg_def_dist", "def_hull_area", "shooter_x", "shooter_y",
        "tmate1_x", "tmate1_y", "tmate2_x", "tmate2_y", "tmate3_x", "tmate3_y", 
        "tmate4_x", "tmate4_y", "def1_x", "def1_y", "def2_x", "def2_y", "def3_x", 
        "def3_y", "def4_x", "def4_y", "def5_x", "def5_y", "team_id_home",
        "team_id_away", "home"
    ]
    df_sub = full_df[keep_features].copy()
    df_sub = df_sub.rename(columns={"new_3pt": "3pt"})
    return df_sub

def add_exp_pts_col(df, model_in, model_out, model_params, n_splits=5, random_state=0):
    """
    Add cross-fitted shot probability and expected points columns.

    Each shot's probability comes from a model trained on the folds that exclude that shot's
    game, so no shot informs its own expected-points value.
    """
    df_mod = df.copy()
    df_mod['shot_prob'] = cross_fitted_probabilities(
        df_mod, model_in, model_out, model_params,
        group_col='game_id', n_splits=n_splits, random_state=random_state, verbose=True
    )

    shot_values = np.where(df_mod["3pt"] == 1, 3, 2)
    # Expected points
    df_mod["expected_points"] = df_mod["shot_prob"] * shot_values

    return df_mod

def create_base_cf_dep_df(df, group=['player_id', 'team_id']):
    # all-shot average EP for each group
    ep_all = (
        df.groupby(group)['expected_points']
        .mean()
        .reset_index(name='ep_all')
    )

    # 2PT-only average EP for each group
    ep_2pt = (
        df[df['3pt'] == 0]
        .groupby(group)['expected_points']
        .mean()
        .reset_index(name='ep_2pt')
    )

    # merge together
    final_ep = ep_all.merge(
        ep_2pt,
        on=group,
        how='left'
    )

    # expected points difference
    final_ep['delta_ep'] = (
        final_ep['ep_all'] - final_ep['ep_2pt']
    )

    return final_ep

def add_alt_exp_pts_cols(orig_df, baseline_df=None):
    """
    Add the naive counterfactual column: each 3PA valued at the shooter's mean 2PT EP.

    `baseline_df` supplies the per-player ep_2pt lookup. It defaults to reading
    CF_DEP_PLAYER_FILE, which is correct for the standard pipeline run. Callers that
    recompute expected points (e.g. the cross-fitting seed sweep) must pass the baseline
    derived from those same expected points, or the counterfactual silently stays pinned
    to whatever run last wrote the CSV.
    """
    df = orig_df.copy()
    if baseline_df is None:
        baseline_df = pd.read_csv(CF_DEP_PLAYER_FILE)

    # Add naive expected points calculation: avg number of points from all 2PT shots for that player
    print('Adding naive expected points column')
    lookup = baseline_df[['player_id', 'ep_2pt']].copy()
    df = df.merge(lookup, on='player_id', how='left')

    # A player with no 2PT attempts in the sample has no personal baseline; fall back to the
    # league-wide mean 2PT expected points so the shot still contributes to the aggregates.
    league_ep_2pt = df.loc[df['3pt'] == 0, 'expected_points'].mean()
    n_fallback = int(df['ep_2pt'].isna().sum())
    if n_fallback:
        print(f'{n_fallback} shots fall back to the league mean 2PT EP ({league_ep_2pt:.4f})')

    df['exp_pts_naive'] = np.where(
        df['3pt'] == 1,
        df['ep_2pt'].fillna(league_ep_2pt),
        df['expected_points']
    )

    df = df.drop(columns=['ep_2pt'])

    return df


# Best features/parameters as found by running shot_prob_model.py.
MODEL_FEATURES = ['seconds_rem', 'streak', 'shot_dist', 'shot_clock', 'close_def_dist', 'avg_def_dist', 'def_hull_area', 'home']
MODEL_PARAMS = {'learning_rate': np.float64(0.10504923529404636), 'max_depth': 5, 'max_leaves': 7, 'n_estimators': 149, 'reg_alpha': np.float64(0.5989426731764285), 'reg_lambda': np.float64(1.686054542997947), 'subsample': 0.9}


def build_base_inputs():
    """Base shot frame plus the feature list and hyperparameters used to model make probability.

    Extracted so callers that need to re-run cross-fitting under different settings (the seed
    sweep) use exactly the same inputs as the main pipeline.
    """
    return create_base_df(), list(MODEL_FEATURES), dict(MODEL_PARAMS)


if __name__=='__main__':
    create_exp_pts_df = True

    if create_exp_pts_df or not os.path.exists(SHOTS_EXP_PTS_FILE):
        print('Creating base dataset from shot/play-by-play dataset')
        base_df, features, params = build_base_inputs()

        print('Using best features/parameters as found by running shot_prob_model.py')
        print(f"Best features: {features}\nBest params: {params}")
        print('Adding cross-fitted shot probability/expected points columns to dataset')
        df_exp_pts = add_exp_pts_col(base_df, features, 'fgm', params)

        df_exp_pts.to_csv(SHOTS_EXP_PTS_FILE, index=False)

        calib = calibration_table(df_exp_pts['fgm'], df_exp_pts['shot_prob'], n_bins=10)
        calib.to_csv(os.path.join('results', 'ep_calibration.csv'), index=False)
        print(calib.to_string(index=False))
        print(f"Brier score: {brier_score(df_exp_pts['fgm'], df_exp_pts['shot_prob']):.5f}")
    else:
        print('Loading existing expected points dataset')
        df_exp_pts = pd.read_csv(SHOTS_EXP_PTS_FILE)

    create_ep_df = True

    if create_ep_df or not os.path.exists(CF_DEP_PLAYER_FILE):
        print('Creating base player counterfactual dependence dataset')
        base_cf_def_player_df = create_base_cf_dep_df(df_exp_pts, group=['player_id'])

        print('Creating base team counterfactual dependence dataset')
        base_cf_def_team_df = create_base_cf_dep_df(df_exp_pts, group=['team_id'])

        base_cf_def_player_df.to_csv(CF_DEP_PLAYER_FILE, index=False)
        base_cf_def_team_df.to_csv(CF_DEP_TEAM_FILE, index=False)
    else:
        print('Loading player counterfactual dependence dataset')
        cf_dep_player_df = pd.read_csv(CF_DEP_PLAYER_FILE)

        print('Loading team counterfactual dependence dataset')
        cf_dep_team_df = pd.read_csv(CF_DEP_TEAM_FILE)

    modify_exp_pts_df = True

    if modify_exp_pts_df or not os.path.exists(ALT_SHOTS_EXP_PTS_FILE):
        print('Adding additional expected points columns to dataset')
        alt_df_exp_pts = add_alt_exp_pts_cols(df_exp_pts)

        alt_df_exp_pts.to_csv(ALT_SHOTS_EXP_PTS_FILE, index=False)
    else:
        print('Loading expected points dataset (with alternative shot choices)')
        alt_df_exp_pts = pd.read_csv(ALT_SHOTS_EXP_PTS_FILE)
        
