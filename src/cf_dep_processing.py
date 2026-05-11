import pandas as pd
import numpy as np
import os
from sklearn.utils.validation import check_is_fitted

from pbp_shot_processing import FINAL_FILE
from shot_prob_model import xgb_classifier, train_xgb_clf

SHOTS_EXP_PTS_FILE = os.path.join('data', 'exp_pts.csv')


def create_base_df():
    full_df = pd.read_csv(FINAL_FILE)
    keep_features = [
        "game_id", "player_id", "team_id", "period", "seconds_rem", "3pt", "fgm", 
        "streak", "shot_dist", "shot_clock", "close_def_id", "close_def_team_id", 
        "close_def_dist", "avg_def_dist", "def_hull_area", "shooter_x", "shooter_y",
        "tmate1_x", "tmate1_y", "tmate2_x", "tmate2_y", "tmate3_x", "tmate3_y", 
        "tmate4_x", "tmate4_y", "def1_x", "def1_y", "def2_x", "def2_y", "def3_x", 
        "def3_y", "def4_x", "def4_y", "def5_x", "def5_y", "team_id_home",
        "team_id_away", "home"
    ]
    df_sub = full_df[keep_features]
    return df_sub

def add_exp_pts_col(df, model_in, model_out, model_params):
    xgb_clf = xgb_classifier(**model_params)
    xgb_clf_trained, _ = train_xgb_clf(df, model_in, model_out, xgb_clf, verbose=True)
    # Will raise error if model is not fitted
    check_is_fitted(xgb_clf_trained)

    X = df[model_in]
    # Shot probability (class 1)
    shot_probs = xgb_clf_trained.predict_proba(X)[:, 1]
    df_mod = df.copy()
    df_mod['shot_prob'] = shot_probs

    shot_values = np.where(df_mod["3pt"] == 1, 3, 2)
    # Expected points
    df_mod["expected_points"] = df_mod["shot_prob"] * shot_values

    return df_mod


if __name__=='__main__':
    create_exp_pts_df = False

    if create_exp_pts_df or not os.path.exists(SHOTS_EXP_PTS_FILE):
        print('Creating base dataset from shot/play-by-play dataset')
        base_df = create_base_df()

        print('Using best features/parameters as found by running shot_prob_model.py')
        features = ['seconds_rem', 'streak', 'shot_dist', 'shot_clock', 'close_def_dist', 'avg_def_dist', 'def_hull_area', 'home']
        params = {'learning_rate': np.float64(0.10504923529404636), 'max_depth': 5, 'max_leaves': 7, 'n_estimators': 149, 'reg_alpha': np.float64(0.5989426731764285), 'reg_lambda': np.float64(1.686054542997947), 'subsample': 0.9}
        print(f"Best features: {features}\nBest params: {params}")
        print('Adding shot probability/expected points columns to dataset')
        df_exp_pts = add_exp_pts_col(base_df, features, ['fgm'], params)
        
        df_exp_pts.to_csv(SHOTS_EXP_PTS_FILE, index=False)
    else:
        print('Loading existing expected points dataset')
        df_exp_pts = pd.read_csv(SHOTS_EXP_PTS_FILE)
