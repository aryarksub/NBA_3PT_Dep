# Final model package

This flat, reproducible package compares three final approaches on exactly the same fixed split.

Run `python3 run_hgb.py`, `python3 run_nn.py`, and `python3 run_cluster.py` from this directory. Each script reads only `train.csv` and `test.csv`. Running scripts in any order updates the shared `model_metrics.csv`, `train_predictions.csv`, `test_predictions.csv`, and comparison figures without deleting another model's columns.

- HGB: 18 location/pressure buckets, no prior-shot-history inputs, plus retrospective full-season player 2P% and 3P%.
- NN: unified 2PT+3PT player-embedding MLP with the 30 base numeric inputs and 14 leakage-controlled prior-history inputs; 100 epochs and threshold 0.5.
- Cluster: 12 train-fitted location clusters and an exponentially distance-weighted make rate among up to 500 same-cluster Train shots. Standardized shot distance and closest-defender distance receive squared-distance weights 0.40 and 0.60.

Shared outputs contain Train and Test metrics/predictions. The three-model comparison figures are `confusion_matrix_comparison.png`, `roc_curve_comparison.png`, `calibration_curve_comparison.png`, and `probability_distribution_comparison.png`. Model-specific figures are `hgb_feature_importance.png`, `nn_loss_curve.png`, and `cluster_visualization.png`.

## Counterfactual dependence

Open `dependence_report.html` for the dependence report. The dependence cohort includes players with at least 20 season 3PT attempts and uses all retained season shots (both 2PT and 3PT) from those players. The model classification metrics remain Test-only. Analysis tables are in `results/model_based_dependence/`, and figures are in `plots/model_based_dependence/`.

The dependence script reads the parent project’s `pbp_final.csv` to recover all five absolute defender coordinates by `source_row_index`. Reproducing the report requires that source file and the matching expected-point Train/Test input and prediction CSVs.
