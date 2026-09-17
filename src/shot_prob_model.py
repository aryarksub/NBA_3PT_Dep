import itertools
import sys
import os

from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
import xgboost as xgb
import pandas as pd

from pbp_shot_processing import FINAL_FILE

def xgb_classifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, max_leaves=8, 
        subsample=1, reg_alpha=0, reg_lambda=1, **kwargs
):
    return xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        max_leaves=max_leaves, subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
        **kwargs
    )

def train_xgb_clf(df, in_features, out_features, xgb_clf=None, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(df[in_features], df[out_features], train_size=0.75)
    model = xgb_classifier() if xgb_clf is None else xgb_clf
    model.fit(X_train, y_train.values.ravel())
    score = model.score(X_test, y_test)
    if verbose:
        print(f"Accuracy: {score}")
    return model, score

def get_feature_importances(model, df, in_features, out_features):
    X = df[in_features]
    y = df[out_features]
    model.fit(X,y)
    importance = model.feature_importances_
    return sorted(
        [(in_features[i], importance[i]) for i in range(len(in_features))],
        key = (lambda x : x[1]),
        reverse=True
    )

def get_best_feature_subset(model, df, all_in_features, out_features, verbose=False):
    X = df[all_in_features]
    y = df[out_features]

    best_score = -1
    best_features = None

    for r in range(1, len(all_in_features) + 1):
        print(f'Running evaluations for feature subsets of size {r}')

        for subset in itertools.combinations(all_in_features, r):
            X_sub = X[list(subset)]
            scores = cross_val_score(
                model, X_sub, y, cv=5, scoring='roc_auc'
            )
            mean_score = scores.mean()
            
            if verbose:
                print(f'Score for subset {subset}: {mean_score}')

            if mean_score > best_score:
                if verbose:
                    print(f'Subset {subset} is the new best')
                best_score = mean_score
                best_features = subset
    
    return list(best_features), best_score

def best_params_for_model(model, param_spaces, df, in_features, out_features):
    # See https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf for random search
    opt_model = RandomizedSearchCV(model, param_spaces, n_iter=50, cv=5, verbose=3)
    fit_model = opt_model.fit(df[in_features], df[out_features])
    return fit_model.best_params_, fit_model.best_score_

if __name__ == '__main__':
    df = pd.read_csv(FINAL_FILE)

    res_file = os.path.join('results', 'shot_prob_xgb.txt') # sys.__stdout__

    features = [
        'period', 'seconds_rem', 'streak', 'shot_dist', 'shot_clock', 
        'close_def_dist', 'avg_def_dist', 'def_hull_area', 'home'
    ]
    target = ['fgm']

    with open(res_file, 'w') as f:
        basic_xgb_model  = xgb_classifier()
        feature_importances = get_feature_importances(basic_xgb_model, df, features, target)
        print('Feature importances:', feature_importances, file=f)

        # Set to True to go through process of finding best features with exhaustive subset search
        find_best_features = True

        if find_best_features:
            best_features, best_score = get_best_feature_subset(
                basic_xgb_model, df, features, target, verbose=False
            )
            print('Best features:', best_features, file=f)
            print('Model score using best features:', best_score, file=f)
        else:
            # Best features is everything except "period"
            print('Using best features found based on previous runs:', features[1:], file=f)
            best_features = features[1:]

        param_space_dict = {
            "n_estimators" : range(100, 151),
            "learning_rate" : stats.uniform(loc=0.05, scale=0.1),
            "max_depth" : [3,4,5],
            "max_leaves" : range(4,11),
            "subsample" : [0.6 + 0.1*x for x in range(5)],
            "reg_alpha" : stats.uniform(loc=0.5, scale=0.25),
            "reg_lambda" : stats.uniform(loc=1.5, scale=0.5)
        }
        
        best_params, best_score = best_params_for_model(basic_xgb_model, param_space_dict, df, best_features, target)
        print(f"Best params for XGBoost: {best_params}\nCorresponding score: {best_score}", file=f)