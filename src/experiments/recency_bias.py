import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from pbp_shot_processing import SHOT_HISTORY_DEF_FILE
import plots


RESULTS_DIR = 'results'
REC_BIAS_RESULTS_FILE = os.path.join(RESULTS_DIR, 'recency_bias.csv')
REC_BIAS_PLOTS_DIR = os.path.join(plots.PLOTS_DIR, 'recency_bias')

def add_miss_make_last_col(df):
    """
    Add two columns (make_last, miss_last) to given DataFrame that represent whether the
    previous shot was a make or miss.

    Args:
        df (pd.DataFrame): Shot DataFrame

    Returns:
        pd.DataFrame: Updated DataFrame with new columns
    """
    df["make_last"] = df["fgm1"] == 1
    df["miss_last"] = df["fgm1"] == 0

    df["make_last"] = df["make_last"].astype(int)
    df["miss_last"] = df["miss_last"].astype(int)

    return df

def miss_make_ols(df, metric='close_def_dist', make_col='make_last', miss_col='miss_last'):
    """
    Ordinary Least Squares (OLS) regression model. Dependent variable is given by the metric 
    argument and independent variables are make/miss_last columns, shot distance, shot clock, 
    period, seconds remaining, and player_id. Players are fixed effects.

    Args:
        df (pd.DataFrame): Shot dataset
        metric (str, optional): Defense metric to use in regression (dependent variable). Defaults to 'close_def_dist'.
        make_col (str, optional): Column that stores previous shot made status. Defaults to 'make_last'.
        miss_col (str, optional): Column that stores previous shot miss status. Defaults to 'miss_last'.

    Returns:
        model: Fit OLS model to data
    """
    formula = f'{metric} ~ {make_col} + {miss_col} + shot_dist + shot_clock + seconds_rem + C(period) + C(player_id)'

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["player_id"]}
    )
    return model

def f_test(model, make_col='make_last', miss_col='miss_last'):
    """
    Run a joint significance ordered difference F-test on the given fit model where the model used the given
    make/miss columns. The null hypotheses are of the form B_0 = 0, B_1 = 0, B_0 = B_1, where B_0 and B_1 are
    the coefficients for make/miss last shots, respectively.

    Args:
        model (model): Fit OLS model
        make_col (str, optional): Column that stores previous shot made status. Defaults to 'make_last'.
        miss_col (str, optional): Column that stores previous shot miss status. Defaults to 'miss_last'.

    Returns:
        tuple: F-test p-values
    """
    make_hyp = f'{make_col} = 0'
    make_test = model.f_test(make_hyp)

    miss_hyp = f'{miss_col} = 0'
    miss_test = model.f_test(miss_hyp)

    asym_hyp = f"{make_col} = {miss_col}"
    asym_test = model.f_test(asym_hyp)
    return make_test.pvalue, miss_test.pvalue, asym_test.pvalue

def model_metrics_driver(
    shots_df, 
    metrics=['close_def_dist', 'avg_def_dist', 'def_hull_area'],
    save=True,
    verbose=False
):
    """
    Driver function for computing and storing statistical results (AIC/Log-Likelihood) for an OLS model across metrics.

    Args:
        shots_df (pd.DataFrame): Shots dataset
        metrics (list, optional): List of defensive metrics to use as the dependent variable in 
        regression models. Defaults to ['close_def_dist', 'avg_def_dist', 'def_hull_area'].
        save (bool, optional): True if results should be saved to CSV; False otherwise. Defaults to True.
        verbose (bool, optional): True if descriptive messages should be printed; False otherwise. Defaults to False.

    Returns:
        pd.DataFrame: Dataset of results (metric, AIC, log-likelihood, additional test results).
    """
    results = []
    
    for metric in metrics:        
        if verbose:
            print(f'Running OLS model for {metric}')
            
        # Ensure that the shots dataset contains non-null data for all the necessary columns
        cols_needed = [
            metric, 'make_col', 'miss_col', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
        ]
        shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()

        fit_model = miss_make_ols(shots_df_no_nan, metric=metric)

        aic, loglik = fit_model.aic, fit_model.llf
        f_test_p_vals = f_test(fit_model)
        results.append((
            metric, aic, loglik, *f_test_p_vals
        ))
    
    results_df = pd.DataFrame(
        results,
        columns=[
            'metric', 'aic', 'loglik', 'f_test_p_val_make', 'f_test_p_val_miss', 'f_test_p_val_asym'
        ]
    )

    if save:
        results_df.to_csv(REC_BIAS_RESULTS_FILE, index=False)
    
    return results_df

def make_rec_bias_plot(model, metric, fname):
    """
    Make recency bias plot.

    Args:
        model (model): Fit OLS model.
        metric (str): Defense metric used as regression dependent variable.
        fname (str): File name to save plot.
    """
    coefs = model.params[["make_last", "miss_last"]]
    ses = model.bse[["make_last", "miss_last"]]

    labels = ["Make (previous shot)", "Miss (previous shot)"]
    x = np.arange(len(labels))

    plt.figure(figsize=(5, 4))
    plt.bar(x, coefs, yerr=1.96 * ses, capsize=6)
    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(x, labels, rotation=10)
    plt.ylabel(f"Change in {metric}")
    plt.title("Defensive Response to Shot Outcomes")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def make_pred_plot(model, metric, fname):
    """
    Make mean prediction plot with intercept as baseline.

    Args:
        model (model): Fit OLS model.
        metric (str): Defense metric used as regression dependent variable.
        fname (str): File name to save plot.
    """
    baseline = model.params["Intercept"]

    pred_make = baseline + model.params["make_last"]
    pred_miss = baseline + model.params["miss_last"]

    plt.figure(figsize=(5, 4))
    plt.bar(["Make", "Miss"], [pred_make, pred_miss])
    plt.ylabel(f"Predicted {metric}")
    plt.title("Defensive Adjustment by Last Shot Outcome")
    plt.savefig(fname, dpi=300)


def main_driver(plot=True, redo=False):
    """
    Main driver for running the experiment.

    Args:
        plot (bool, optional): True if plots should be generated; False otherwise. Defaults to False.
        redo (bool, optional): True if models should be re-fit even if their results dataset exists; False
        otherwise. Defaults to False.
    """
    shots_df = pd.read_csv(SHOT_HISTORY_DEF_FILE)

    if redo or not os.path.exists(REC_BIAS_RESULTS_FILE):
        model_metrics_driver(
            shots_df_no_nan, metrics=['close_def_dist', 'avg_def_dist', 'def_hull_area'], save=True, verbose=True
        )
    
    results = pd.read_csv(REC_BIAS_RESULTS_FILE)
    sorted_model_results = results.sort_values(by=['aic', 'loglik'], ascending=[True, False])
    best_rows = sorted_model_results.groupby('metric').first()

    rec_bias_plots = []
    pred_plots = []

    for metric, row in best_rows.iterrows():
        cols_needed = [
            metric, 'fgm1', 'shot_dist', 'shot_clock', 'period', 'seconds_rem', 'game_id', 'player_id'
        ]
        shots_df_no_nan = shots_df.dropna(subset=cols_needed).copy()
        shots_df_updated = add_miss_make_last_col(shots_df_no_nan)

        best_model = miss_make_ols(
            shots_df_updated, metric=metric
        )
        print(f'{metric} Coefficients for make_last, miss_last:', best_model.params['make_last'], best_model.params['miss_last'])
        
        if plot:
            if not os.path.exists(REC_BIAS_PLOTS_DIR):
                os.makedirs(REC_BIAS_PLOTS_DIR)
            rec_bias_plot_name = os.path.join(REC_BIAS_PLOTS_DIR, f'{metric}_rec_bias.png')
            pred_plot_name = os.path.join(REC_BIAS_PLOTS_DIR, f'{metric}_pred.png')
            make_rec_bias_plot(best_model, metric=metric, fname=rec_bias_plot_name)
            make_pred_plot(best_model, metric=metric, fname=pred_plot_name)
            rec_bias_plots.append(rec_bias_plot_name)
            pred_plots.append(pred_plot_name)
    
    if plot:
        plots.merge_three_png_plots(rec_bias_plots, os.path.join(REC_BIAS_PLOTS_DIR, f'rec_bias_plots.png'))
        plots.merge_three_png_plots(pred_plots, os.path.join(REC_BIAS_PLOTS_DIR, f'pred_plots.png'))

if __name__=='__main__':
    main_driver()