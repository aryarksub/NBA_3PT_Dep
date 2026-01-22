import statsmodels.formula.api as smf
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from pbp_shot_processing import SHOT_HISTORY_DEF_FILE
import plots

RESULTS_DIR = 'results'
DEF_METRIC_HEAT_REG_PLOTS_DIR = os.path.join(plots.PLOTS_DIR, 'def_metric_heat_reg')
DEF_METRIC_HEAT_REG_RESULTS_FILES = {
    'ols': os.path.join(RESULTS_DIR, 'def_metric_heat_reg_ols.csv'),
    'gam': os.path.join(RESULTS_DIR, 'def_metric_heat_reg_gam.csv'),
    'all': os.path.join(RESULTS_DIR, 'def_metric_heat_reg_all.csv')
}

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Experiment: Defensive reaction curves to recent "heat"
# Aim: After how many recent makes in a fixed shot memory do defenders notice and react?

def defense_metric_heat_regression_ols(df, model_type='ols_cat', window=3, metric='close_def_dist'):
    """
    Ordinary Least Squares (OLS) regression model. Dependent variable is given by the metric 
    argument and independent variables are shot distance, shot clock, period, seconds remaining,
    and player_id. Games and players are fixed effects.

    Args:
        df (pd.DataFrame): Shot dataset
        model_type (str, optional): OLS model to use: ols_cat treats the heat metric (total makes in
        previous shots) as categorical; ols treats it as numerical. Defaults to 'ols_cat'.
        window (int, optional): Number of previous shots to consider for heat metric. Defaults to 3.
        metric (str, optional): Defense metric to use in regression (dependent variable). Defaults to 'close_def_dist'.

    Returns:
        model: Fit OLS model to data
    """
    # Treat heat metric (tot{window} feature) as categorical
    if 'cat' in model_type:
        heat_col = f'C(tot{window}) - 1'
    else:
        heat_col = f'tot{window}'

    model = smf.ols(
        f'{metric} ~ {heat_col} + shot_dist + shot_clock + C(period) + seconds_rem + C(player_id)',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['game_id']})
    
    return model

def defense_metric_heat_regression_gam(df, model_type='gam_3_4', window=3, metric='close_def_dist'):
    """
    Generalized Additive Model (GAM) regression model. Dependent variable is given by the metric 
    argument and independent variables are shot distance, shot clock, period, seconds remaining,
    and player_id. Games and players are fixed effects.

    Args:
        df (pd.DataFrame): Shot dataset
        model_type (str, optional): GAM model to use in the form 'gam_{i}_{j}', where i is the 
        degree of the polynomial to use and j is the degrees of freedom. Defaults to 'gam_3_4'.
        window (int, optional): Number of previous shots to consider for heat metric. Defaults to 3.
        metric (str, optional): Defense metric to use in regression (dependent variable). Defaults to 'close_def_dist'.

    Returns:
        model: Fit GAM model to data
    """
    tokens = model_type.split('_')
    poly_deg, deg_free = int(tokens[1]), int(tokens[2])
    # Basic-spline smoother for categorical total makes variable
    x_spline = BSplines(df[[f'tot{window}']], df=[deg_free], degree=[poly_deg])
    # Avoiding the dummy variable trap by dropping one category
    player_dummies = pd.get_dummies(df['player_id'], drop_first=True)
    exog_vars = pd.concat([df[['shot_dist', 'shot_clock', 'seconds_rem']], player_dummies], axis=1).astype(float)

    model = GLMGam(
        df[metric],
        exog=exog_vars,
        smoother=x_spline
    ).fit()
    
    return model

def f_test(model, window):
    """
    Run a joint significance ordered difference F-test on the given fit model where the model used the given shot window.
    The hypotheses are of the form B_1 - B_0 = 0, B_2 - B_1 = 0, ... (i.e. pairwise coefficient differences).

    Args:
        model (model): Fit OLS model
        window (int): Shot history window

    Returns:
        float: F-test p-value
    """
    hypotheses = []
    for k in range(1, window + 1):
        hypotheses.append(
            f"C(tot{window})[{k*1.0}] - C(tot{window})[{(k-1)*1.0}] = 0"
        )

    f_test = model.f_test(hypotheses)
    return f_test.pvalue


def trend_test(model, window):
    """
    Linear trend test for heat metric coefficients in the given model using the given shot window.

    Args:
        model (model): Fit OLS model
        window (int): Shot history window

    Returns:
        float: Trend test p-value
    """
    coef_names = model.params.index
    L = np.zeros(len(model.params))

    for i, term in enumerate(coef_names):
        if f"C(tot{window})" in term:
            k = float(term.split("[")[-1].rstrip("]"))
            L[i] = k

    trend_test = model.t_test(L)
    return trend_test.pvalue

def model_metrics_driver(
        shots_df, 
        metrics=['close_def_dist', 'avg_def_dist'], 
        model_types=['ols', 'gam'], 
        windows=range(1, 11),
        save=True, verbose=False
    ):
    """
    Driver function for computing and storing statistical results (AIC/Log-Likelihood) for an
    ensemble of specified models, metrics, and shot history windows.

    Args:
        shots_df (pd.DataFrame): Shot dataset
        metrics (list, optional): List of defensive metrics to use as the dependent variable in 
        regression models. Defaults to ['close_def_dist', 'avg_def_dist'].
        model_types (list, optional): List of model types to use for regression. Defaults to ['ols', 'gam'].
        windows (list, optional): List of shot history windows to use. Defaults to range(1, 11).
        save (bool, optional): True if results should be saved to CSV; False otherwise. Defaults to True.
        verbose (bool, optional): True if descriptive messages should be printed; False otherwise. Defaults to False.

    Returns:
        pd.DataFrame: Dataset of results (model type, metric, window, AIC, log-likelihood).
    """
    results = []
    models = []
    # For OLS, run regression using both the categorical and non-categorical version
    if 'ols' in model_types:
        models += ['ols_cat', 'ols']
    # For GAM, use poly degree 2-4 and degrees of freedom = poly degree + {1, 2}
    if 'gam' in model_types:
        models += [f'gam_{i}_{j}' for i in range(2,5) for j in range(i+1, i+3)]
    
    for metric in metrics:        
        for model in models:
            for window in windows:
                if verbose:
                    print(f'Running {model} model for {metric} with window {window}')
                    
                # Ensure that the shots dataset contains non-null data for all the necessary columns
                cols_needed = [
                    metric, f'tot{window}', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
                ]
                shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

                if 'ols' in model:
                    fit_model = defense_metric_heat_regression_ols(shots_df_no_nan, model_type=model, window=window, metric=metric)
                elif 'gam' in model:
                    fit_model = defense_metric_heat_regression_gam(shots_df_no_nan, model_type=model, window=window, metric=metric)

                aic, loglik = fit_model.aic, fit_model.llf
                # F-test and trend test is only supported for categorical OLS model
                f_test_p_val = f_test(fit_model, window) if model == 'ols_cat' else None
                trend_test_p_val = trend_test(fit_model, window) if model == 'ols_cat' else None
                results.append((model, metric, window, aic, loglik, f_test_p_val, trend_test_p_val))
    
    results_df = pd.DataFrame(
        results,
        columns=['model', 'metric', 'window', 'aic', 'loglik', 'f_test_p_val', 'trend_test_p_val']
    )

    if save:
        ols_results = results_df[results_df["model"].str.contains("ols", case=False, na=False)]
        ols_results.to_csv(DEF_METRIC_HEAT_REG_RESULTS_FILES['ols'], index=False)
        gam_results = results_df[results_df["model"].str.contains("gam", case=False, na=False)]
        gam_results.to_csv(DEF_METRIC_HEAT_REG_RESULTS_FILES['gam'], index=False)
        results_df.to_csv(DEF_METRIC_HEAT_REG_RESULTS_FILES['all'], index=False)
    
    return results_df

def prepare_and_plot_ols_model(df, model, metric, window, fname):
    """
    Create plots for a given OLS model. Two plots will be created: one with the model
    coefficients for each number of makes within the specified window, and another
    with the actual metric values.

    Args:
        df (pd.DataFrame): Shot dataset
        model (model): Fit OLS model
        metric (str): Defensive metric used as dependent variable in regression model
        window (int): Shot history window
        fname (str): File name to save plot
    """
    def create_coefficient_df():
        """
        Create DataFrame with model coefficients (mean, lower/upper bounds of confidence interval).

        Returns:
            pd.DataFrame: Model coefficient DataFrame
        """
        rows = []
        for k in range(window + 1):
            term = f"C(tot{window})[{float(k)}]"
            coef = model.params[term]
            se = model.bse[term]
            
            # 1.96 factor corresponds to 95% confidence interval
            rows.append({
                "k": k,
                "mean": coef,
                "mean_ci_lower": coef - 1.96 * se,
                "mean_ci_upper": coef + 1.96 * se
            })
        return pd.DataFrame(rows)
    
    def create_prediction_df():
        """
        Create metric prediction DataFrame for confidence interval usage.

        Returns:
            pd.DataFrame: Metric prediction DataFrame
        """
        rows = []
        base = df.iloc[0].copy()

        # Mean values for each independent variable
        base["shot_dist"] = df["shot_dist"].mean()
        base["shot_clock"] = df["shot_clock"].mean()
        base["seconds_rem"] = df["seconds_rem"].mean()
        base["period"] = df["period"].mode()[0]
        base["player_id"] = df["player_id"].iloc[0]  # arbitrary

        for k in range(window + 1):
            row = base.copy()
            row[f"tot{window}"] = k
            rows.append(row)

        return pd.DataFrame(rows)
    
    coef_df = create_coefficient_df()
    pred_df = create_prediction_df()
    pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
    pred['k'] = range(window + 1)

    metric_with_unit = f'{metric} (feet)' if 'dist' in metric else f'{metric} (square feet)'

    plots.plot_cis(
        dfs=[coef_df, pred], 
        x_col='k', mean_col='mean', low_col='mean_ci_lower', high_col='mean_ci_upper',
        x_label=f'Number of makes in previous {window} shots',
        y_labels=[f'{metric} coefficient', metric_with_unit],
        save_file=fname
    )

def main_driver(plot=False, redo=False):
    """
    Main driver for running the experiment.

    Args:
        plot (bool, optional): True if plots should be generated; False otherwise. Defaults to False.
        redo (bool, optional): True if models should be re-fit even if their results dataset exists; False
        otherwise. Defaults to False
    """
    shots_df = pd.read_csv(SHOT_HISTORY_DEF_FILE)

    # Run OLS/GAM regression models (if results files don't exist)
    for model_type, file in sorted(DEF_METRIC_HEAT_REG_RESULTS_FILES.items()):
        if redo or not os.path.exists(file):
            if model_type == 'all':
                model_metrics_driver(shots_df, save=True, verbose=True)
                break
            elif model_type == 'ols':
                model_metrics_driver(shots_df, model_types=['ols'], save=True, verbose=True)
            elif model_type == 'gam':
                model_metrics_driver(shots_df, model_types=['gam'], save=True, verbose=True)
    
    # Find and re-fit model that works the best (according to loglik/AIC) for each metric
    all_model_results = pd.read_csv(DEF_METRIC_HEAT_REG_RESULTS_FILES['all'])
    sorted_model_results = all_model_results.sort_values(by=['loglik', 'aic'], ascending=[False, True])
    best_rows = sorted_model_results.groupby('metric').first()

    for metric, row in best_rows.iterrows():
        best_model_results = row.to_dict()
        best_model_type, best_model_window = best_model_results['model'], best_model_results['window']

        cols_needed = [
            metric, f'tot{best_model_window}', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
        ]
        shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

        if 'ols' in best_model_type:
            best_model = defense_metric_heat_regression_ols(
                shots_df_no_nan, model_type=best_model_type, window=best_model_window, metric=metric
            )
        elif 'gam' in best_model_type:
            best_model = defense_metric_heat_regression_gam(
                shots_df_no_nan, model_type=best_model_type, window=best_model_window, metric=metric
            )

        # Plotting CIs for GAM doesn't work properly so just do it if the best model is OLS
        if plot and 'ols' in best_model_type:
            if not os.path.exists(DEF_METRIC_HEAT_REG_PLOTS_DIR):
                os.makedirs(DEF_METRIC_HEAT_REG_PLOTS_DIR)
            prepare_and_plot_ols_model(
                shots_df_no_nan, best_model, metric, best_model_window,
                fname=os.path.join(DEF_METRIC_HEAT_REG_PLOTS_DIR, f'{metric}_{best_model_type}_{best_model_window}_BEST_ci.png')
            )

def alternate_runs():
    """
    Scratch function for alternate runs of models other than the (best) one chosen in the main
    driver pipeline.
    """
    shots_df = pd.read_csv(SHOT_HISTORY_DEF_FILE)

    for window in [3,4,5,6]:
        for metric in ['close_def_dist', 'avg_def_dist']:
            cols_needed = [
                metric, f'tot{window}', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
            ]
            shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

            model_type = 'ols_cat'
            model = defense_metric_heat_regression_ols(
                shots_df_no_nan, model_type=model_type, window=window, metric=metric
            )
            prepare_and_plot_ols_model(
                shots_df_no_nan, model, metric, window,
                fname=os.path.join(DEF_METRIC_HEAT_REG_PLOTS_DIR, f'{metric}_{model_type}_{window}_ci.png')
            )


if __name__=='__main__':
    main_driver(plot=True, redo=False)
    alternate_runs()