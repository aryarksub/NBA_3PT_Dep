import matplotlib.pyplot as plt
import os

PLOTS_DIR = 'plots'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

plt.style.use("grayscale")

def plot_cis(dfs, x_col, mean_col='mean', low_col='mean_ci_lower', high_col='mean_ci_upper', x_label="", y_labels=[], save_file=None):
    """
    Plot multiple confidence intervals in a vertical layout.

    Args:
        dfs (list): List of DataFrames to plot confidence intervals for
        x_col (str): Name of column to use for x-axis
        mean_col (str, optional): Column name for mean values. Defaults to 'mean'.
        low_col (str, optional): Column name for CI lower bound values. Defaults to 'mean_ci_lower'.
        high_col (str, optional): Column name for CI upper bound values. Defaults to 'mean_ci_upper'.
        x_label (str, optional): Label for x-axis. Defaults to "".
        y_labels (list, optional): List of labels for y-axes. Defaults to [].
        save_file (str, optional): File name for saving plots. Defaults to None.
    """
    # One subplot per df confidence interval
    fig, axes = plt.subplots(
        nrows=len(dfs), ncols=1,
        figsize=(7, 10),
        sharex=True
    )

    if len(dfs) == 1:
        axes = [axes]

    for ax, df, y_label in zip(axes, dfs, y_labels):
        ax.plot(df[x_col], df[mean_col], marker='o')
        ax.fill_between(df[x_col], df[low_col], df[high_col], alpha=0.25)
        ax.set_ylabel(y_label)
        # Mean line for first data point (baseline)
        ax.axhline(df[mean_col].iloc[0], linestyle="--", alpha=0.6)
    
    axes[-1].set_xlabel(x_label)
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300)
    else:
        plt.show()