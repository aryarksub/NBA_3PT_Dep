import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import os
import sys
from patsy import dmatrix
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from pbp_shot_processing import SHOT_HISTORY_DEF_FILE
import plots

RESULTS_DIR = 'results'
STREAK_REG_RESULTS_FILE = os.path.join(RESULTS_DIR, 'streak_reg_all.csv')
STREAK_REG_PLOTS_DIR = os.path.join(plots.PLOTS_DIR, 'streak_reg')

def streak_regression_ols(df, streak_term='streak', metric='close_def_dist'):
    """
    Ordinary Least Squares (OLS) regression model. Dependent variable is given by the metric 
    argument and independent variables are a streak term (for number of previously consecutive
    made shots), shot distance, shot clock, period, seconds remaining, and player_id. 
    Players are fixed effects.

    Args:
        df (pd.DataFrame): Shot dataset
        streak_term (str, optional): Streak-related expressions to use as independent variable. Defaults to 'streak'.
        metric (str, optional): Defense metric to use in regression (dependent variable). Defaults to 'close_def_dist'.

    Returns:
        model: Fit OLS model to data
    """
    formula = f'{metric} ~ {streak_term} + shot_dist + shot_clock + seconds_rem + C(period) + C(player_id)'

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["player_id"]}
    )

    return model

def get_streak_term_for_model(m_type):
    """
    Generate streak-related term for usage in regression model based on the given model type.
    Supported model types are of the form 'poly_{d}' (polynomial of degree d), 'exp' (exponential),
    and 'spline_{i}_{j}' (spline with degree i and j degrees of freedom).

    Args:
        m_type (str): Model type

    Raises:
        ValueError: Occurs when a given model type is not supported.

    Returns:
        str: Streak term
    """
    if 'poly' in m_type:
        # Terms for each degree up to and including poly_deg
        poly_deg = int(m_type.split('_')[1])
        new_streak_cols = [f'I(streak**{d})' for d in range(2, poly_deg+1)]
    elif 'exp' in m_type:
        # Exponential expression
        new_streak_cols = ['I(np.exp(streak))']
    elif 'spline' in m_type:
        # Terms for each coefficient in spline up to and including spline degree
        tokens = m_type.split('_')
        poly_deg, deg_free = int(tokens[1]), int(tokens[2])
        return ' + '.join([f"streak_spline_{i}" for i in range(poly_deg + 1)])
    else:
        raise ValueError(f'Model type {m_type} not supported')

    all_streak_cols = ['streak'] + new_streak_cols
    return ' + '.join(all_streak_cols)

def add_spline_col(df, m_type='spline_2_3'):
    """
    Add spline-transformed streak columns to given DataFrame using B-spline basis functions.

    Args:
        df (pd.DataFrame): Shot dataset
        m_type (str, optional): Spline model type. Defaults to 'spline_2_3'.

    Returns:
        tuple: Tuple of updated DataFrame and streak term for the model
    """
    tokens = m_type.split('_')
    poly_deg, deg_free = int(tokens[1]), int(tokens[2])
    spline_df = dmatrix(
        f"bs(streak, df={deg_free}, degree={poly_deg}, include_intercept=False) - 1",
        data=df,
        return_type='dataframe'
    )
    spline_df.columns = [f"streak_spline_{i}" for i in range(spline_df.shape[1])]
    df = pd.concat([df, spline_df], axis=1)
    streak_term = ' + '.join(spline_df.columns)
    return df, streak_term

def create_mean_prediction_df(df, streak_col='streak', extra_streak_cols=[]):
    """
    Create metric prediction DataFrame for confidence interval usage.

    Args:
        df (pd.DataFrame): Shot dataset
        streak_col (str, optional): Streak column in shot dataset. Defaults to 'streak'.
        extra_streak_cols (list, optional): Extra columns to add to prediction DataFrame. Defaults to [].

    Returns:
        pd.DataFrame: Mean metric prediction DataFrame
    """
    streak_grid = np.arange(0, df[streak_col].max() + 1)
    pred_dict = {
        'streak': streak_grid,
        'shot_dist': df['shot_dist'].mean(),
        'shot_clock': df['shot_clock'].mean(),
        'seconds_rem': df['seconds_rem'].mean(),
        'period': df['period'].mode()[0],
        'player_id': df['player_id'].iloc[0]
    }
    for col in extra_streak_cols:
        pred_dict[col] = df[col].mean()
    return pd.DataFrame(pred_dict)

def f_test(model, term_like='streak'):
    """
    Run a joint significance ordered difference F-test on the given fit model where the model used the given shot window.
    The hypothesis are of the form B_0 = B_1 = B_2 = ... = 0 (i.e. coefficient equality to 0).

    Args:
        model (model): Fit OLS model
        term_like (str, optional): Term for which model parameters should be matched. Defaults to 'streak'.

    Returns:
        float: F-test p-value
    """
    terms = model.params.filter(like=term_like).index
    hypothesis = " = ".join(terms) + " = 0"
    f_test_results = model.f_test(hypothesis)
    return f_test_results.pvalue

def finite_diff_trend_test(model, shots_df, mean_col='mean', extra_streak_cols=[]):
    """
    Test for systematic trend in model predictions across streak values using finite differences.

    Args:
        model (model): Fit OLS model
        shots_df (pd.DataFrame): Shots dataset
        mean_col (str, optional): Column for mean in prediction DataFrame. Defaults to 'mean'.
        extra_streak_cols (list, optional): Extra columns to use when creating prediction DataFrame. Defaults to [].

    Returns:
        tuple: Tuple of trend test p-value and first differences array
    """
    # Make predictions on streak grid
    pred_df = create_mean_prediction_df(shots_df, extra_streak_cols=extra_streak_cols)
    pred = model.get_prediction(pred_df).summary_frame()

    # Compute first differences
    delta = np.diff(pred[mean_col])

    # Test whether average difference is not 0
    t_stat, p_val = stats.ttest_1samp(delta, 0)

    return p_val, delta

def directionally_monotone_inference(delta, bootstrap_n=1000):
    """
    Inferential trend check for directional monotonicity in a sequence of finite differences.

    Args:
        delta (list): List of finite differences
        bootstrap_n (int, optional): Number of bootstrap resamples for confidence interval
        estimation. Defaults to 1000.

    Returns:
        tuple: Tuple of persistence metric (percentage of same difference steps), number of
        times direction changes, confidence interval, boolean of if zero is in the CI
    """
    # Sign consistency
    direction = np.sign(delta)
    persistence = np.mean(direction[1:] == direction[:-1])
    num_sign_changes = np.sum(direction[1:] != direction[:-1])
    
    # Bootstrap confidence interval (check if 95% CI excludes 0)
    boot_means = []
    for _ in range(bootstrap_n):
        sample = np.random.choice(delta, size=len(delta), replace=True)
        boot_means.append(sample.mean())

    ci = np.percentile(boot_means, [2.5, 97.5])
    return persistence, num_sign_changes, ci, ci[0] < 0 < ci[1]

def model_metrics_driver(
    shots_df, 
    metrics=['close_def_dist', 'avg_def_dist', 'def_hull_area'],
    model_types=['poly_1'],
    save=True,
    verbose=False
):
    """
    Driver function for computing and storing statistical results (AIC/Log-Likelihood) for an
    ensemble of specified models and metrics.

    Args:
        shots_df (pd.DataFrame): Shots dataset
        metrics (list, optional): List of defensive metrics to use as the dependent variable in 
        regression models. Defaults to ['close_def_dist', 'avg_def_dist', 'def_hull_area'].
        model_types (list, optional): List of model types to use for regression. Defaults to ['poly_1'].
        save (bool, optional): True if results should be saved to CSV; False otherwise. Defaults to True.
        verbose (bool, optional): True if descriptive messages should be printed; False otherwise. Defaults to False.

    Returns:
        pd.DataFrame: Dataset of results (model type, metric, AIC, log-likelihood, additional test results).
    """
    results = []
    
    for metric in metrics:        
        for m_type in model_types:
            if verbose:
                print(f'Running {m_type} model for {metric}')
                
            # Ensure that the shots dataset contains non-null data for all the necessary columns
            cols_needed = [
                metric, 'streak', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
            ]
            shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()
            if 'spline' in m_type:
                shots_df_no_nan, streak_term = add_spline_col(shots_df_no_nan, m_type=m_type)
                extra_streak_cols = streak_term.split(' + ')
            else:
                streak_term = get_streak_term_for_model(m_type)
                extra_streak_cols = []

            fit_model = streak_regression_ols(shots_df_no_nan, streak_term=streak_term, metric=metric)

            aic, bic, loglik = fit_model.aic, fit_model.bic, fit_model.llf
            f_test_p_val = f_test(fit_model, term_like='streak')
            trend_test_p_val, delta = finite_diff_trend_test(fit_model, shots_df_no_nan, mean_col='mean', extra_streak_cols=extra_streak_cols)
            direction_persistence, num_dir_changes, sign_ci, zero_in_sign_ci = directionally_monotone_inference(delta)
            results.append((
                m_type, metric, aic, bic, loglik, f_test_p_val, trend_test_p_val,
                direction_persistence, num_dir_changes, sign_ci[0], sign_ci[1], zero_in_sign_ci
            ))
    
    results_df = pd.DataFrame(
        results,
        columns=[
            'model', 'metric', 'aic', 'bic', 'loglik', 'f_test_p_val', 'trend_test_p_val',
            'dir_persistence', 'dir_changes', 'sign_ci_low', 'sign_ci_high', 'zero_in_ci'
        ]
    )

    if save:
        results_df.to_csv(STREAK_REG_RESULTS_FILE, index=False)
    
    return results_df

def prepare_and_plot_ols_model(df, model, metric, fname):
    """
    Create confidence interval plots for a given OLS model.

    Args:
        df (pd.DataFrame): Shot dataset
        model (model): Fit OLS model
        metric (str): Defensive metric used as dependent variable in regression model
        fname (str): File name to save plot
    """
    pred_df = create_mean_prediction_df(df)
    pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
    pred['streak'] = range(df['streak'].max() + 1)

    metric_with_unit = f'{metric} (feet)' if 'dist' in metric else f'{metric} (square feet)'

    plots.plot_cis(
        dfs=[pred], 
        x_col='streak', mean_col='mean', low_col='mean_ci_lower', high_col='mean_ci_upper',
        x_label=f'Number of previous consecutive makes',
        y_labels=[metric_with_unit],
        save_file=fname
    )

def main_driver(plot=True, redo=False):
    """
    Main driver for running the experiment.

    Args:
        plot (bool, optional): True if plots should be generated; False otherwise. Defaults to False.
        redo (bool, optional): True if models should be re-fit even if their results dataset exists; False
        otherwise. Defaults to False.
    """
    shots_df = pd.read_csv(SHOT_HISTORY_DEF_FILE)
    model_types = [f'poly_{d}' for d in range(1, 5)] + ['exp'] + [
        f'spline_{i}_{j}' for i in range(2, 5) for j in range(i+1, i+3)
    ]

    if redo or not os.path.exists(STREAK_REG_RESULTS_FILE):
        model_metrics_driver(
            shots_df, metrics=['close_def_dist', 'avg_def_dist', 'def_hull_area'], model_types=model_types, save=True, verbose=True
        )
    
    results = pd.read_csv(STREAK_REG_RESULTS_FILE)
    # Include model name as tiebreaker to give preference to simpler poly models over spline models
    sorted_model_results = results.sort_values(by=['aic', 'loglik', 'model'], ascending=[True, False, True])
    best_rows = sorted_model_results.groupby('metric').first()

    for metric, row in best_rows.iterrows():
        best_model_results = row.to_dict()
        best_model_type = best_model_results['model']

        cols_needed = [
            metric, 'streak', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
        ]
        shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

        best_model = streak_regression_ols(
            shots_df_no_nan, streak_term=get_streak_term_for_model(best_model_type), metric=metric
        )
        
        if plot:
            if not os.path.exists(STREAK_REG_PLOTS_DIR):
                os.makedirs(STREAK_REG_PLOTS_DIR)
            prepare_and_plot_ols_model(
                shots_df_no_nan, best_model, metric,
                fname=os.path.join(STREAK_REG_PLOTS_DIR, f'{metric}_{best_model_type}_BEST_ci.png')
            )

def additional_runs():
    """
    Scratch function for alternate runs of models other than the (best) one chosen in the main
    driver pipeline.
    """
    shots_df = pd.read_csv(SHOT_HISTORY_DEF_FILE)
    m_types = ['poly_2', 'poly_2']
    metrics = ['avg_def_dist', 'def_hull_area']
    for m_type, metric in zip(m_types, metrics):
        cols_needed = [
            metric, 'streak', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
        ]
        shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

        model = streak_regression_ols(
            shots_df_no_nan, streak_term=get_streak_term_for_model(m_type), metric=metric
        )
        
        prepare_and_plot_ols_model(
            shots_df_no_nan, model, metric,
            fname=os.path.join(STREAK_REG_PLOTS_DIR, f'{metric}_{m_type}_ci.png')
        )
    
    plots.merge_three_png_plots(
        [
            os.path.join(STREAK_REG_PLOTS_DIR, 'avg_def_dist_poly_2_ci.png'),
            os.path.join(STREAK_REG_PLOTS_DIR, 'close_def_dist_poly_2_BEST_ci.png'),
            os.path.join(STREAK_REG_PLOTS_DIR, 'def_hull_area_poly_2_ci.png')
        ],
        os.path.join(STREAK_REG_PLOTS_DIR, 'streak_poly_2_plots.png'),
        layout='stack21'
    )


if __name__=='__main__':
    main_driver(plot=True, redo=False)
    additional_runs()