# Final figure catalog

The original nearest/best analysis contains 27 figures: 12 naive-reference figures, seven
alternative comparisons, seven sensitivity comparisons, and one historical geometry comparison.
Sensitivity and geometry figures also have SVG exports. All paths below link to the final PNGs.
The historical geometry plots are deliberate controlled comparisons; they are not current
candidate output. Contact sheets, temporary audits, and raw snapshots are excluded from Git.
Other existing plot directories concern the repository's separate defensive-response research.

## cf_dependence

| Figure | What it measures |
|---|---|
| [expected_points_calibration](../plots/cf_dependence/expected_points_calibration.png) | Observed make frequencies against cross-fitted probabilities. |
| [dependence_distribution](../plots/cf_dependence/dependence_distribution.png) | Distribution of naive player dependence shares. |
| [player_dependence_ranking](../plots/cf_dependence/player_dependence_ranking.png) | High/low naive player estimates and conditional bootstrap intervals. |
| [team_dependence_ranking](../plots/cf_dependence/team_dependence_ranking.png) | Naive team estimates and conditional intervals. |
| [player_rank_uncertainty](../plots/cf_dependence/player_rank_uncertainty.png) | Bootstrap uncertainty in player rank. |
| [volume_versus_edge](../plots/cf_dependence/volume_versus_edge.png) | Attempt frequency versus expected advantage per three. |
| [dependence_versus_attempt_rate](../plots/cf_dependence/dependence_versus_attempt_rate.png) | Naive dependence share versus three-point attempt rate. |
| [split_half_reliability](../plots/cf_dependence/split_half_reliability.png) | Naive repeatability with half-specific baselines; shot models remain shared. |
| [where_dependence_comes_from](../plots/cf_dependence/where_dependence_comes_from.png) | Mean naive advantage per three by octiles of defender distance and shot distance, with intervals on bin means. |
| [game_to_game_variation](../plots/cf_dependence/game_to_game_variation.png) | Within-team game variation compared with between-team variation. |
| [qualification_threshold_sensitivity](../plots/cf_dependence/qualification_threshold_sensitivity.png) | Effects of player sample qualification thresholds. |
| [team_dependence_versus_outcomes](../plots/cf_dependence/team_dependence_versus_outcomes.png) | Descriptive outcome associations using official field-goal-attempt denominators. |

## cf_alternatives

| Figure | What it measures |
|---|---|
| [strategy_comparison](../plots/cf_alternatives/strategy_comparison.png) | Median and interquartile range of player dependence by strategy. |
| [strategy_versus_naive](../plots/cf_alternatives/strategy_versus_naive.png) | Player estimates against the retained naive reference. |
| [attempt_rate_degeneracy](../plots/cf_alternatives/attempt_rate_degeneracy.png) | Attempt-rate rank correlation; 0.95 is a heuristic screen, not a validity test. |
| [feasibility_by_openness](../plots/cf_alternatives/feasibility_by_openness.png) | Fallback fraction and displacement across nearest openness constraints. |
| [specification_versus_sampling](../plots/cf_alternatives/specification_versus_sampling.png) | Specification range compared with conditional bootstrap width; not total uncertainty. |
| [counterfactual_locations](../plots/cf_alternatives/counterfactual_locations.png) | Selected substitute locations under the final 8 ft / 0.25 ft search. |
| [candidate_support_map](../plots/cf_alternatives/candidate_support_map.png) | Diagnostic illustration of the unused empirical support filter; display grid is 1 ft. |

## search_sensitivity

| Figure | What it measures |
|---|---|
| [geometry_controlled_comparison](../plots/search_sensitivity/geometry_controlled_comparison.png) | Saved and matched-model boundary comparisons, holding radius/grid at 15 ft / 1 ft. |
| [radius_resolution_tradeoffs](../plots/search_sensitivity/radius_resolution_tradeoffs.png) | Correlation and fallback across radii and grid spacings. |
| [resolution_convergence](../plots/search_sensitivity/resolution_convergence.png) | Player estimate differences from the 0.1 ft numerical reference. |
| [dependence_and_volume](../plots/search_sensitivity/dependence_and_volume.png) | Dependence versus attempt rate, nonlinear trend, and fallback. |
| [beyond_volume_repeatability](../plots/search_sensitivity/beyond_volume_repeatability.png) | Half-specific residual repeatability beyond a cubic attempt-rate trend. |
| [radius_tradeoffs](../plots/search_sensitivity/radius_tradeoffs.png) | Travel, fallback, and correlation as radius changes. |
| [all_strategy_diagnostics](../plots/search_sensitivity/all_strategy_diagnostics.png) | Association, residual repeatability, and fallback for all implemented variants. |

## geometry_audit

| Figure | What it measures |
|---|---|
| [three_point_geometry_verification](../plots/geometry_audit/three_point_geometry_verification.png) | Archived erroneous boundary, affected locations, and invalid historical substitutes; explicitly a before-fix comparison. |

## Regeneration

Install `requirements.txt`; run from the repository root. Canonical figures require the raw-data
pipeline to have generated `data/alt_exp_pts.csv` and `data/game.csv`:

```text
python src/experiments/cf_dependence.py
python src/experiments/cf_ci_impact.py
python src/experiments/cf_dependence_figures.py
python src/experiments/cf_alternatives.py
python src/experiments/cf_alternatives_figures.py
```

The optional full sensitivity sweep requires the same processed shots. Each shot is scored
by its held-out-game fold model. Checkpoint reuse validates data, model, code, and settings:

```text
python src/experiments/prepare_search_models.py
python src/experiments/candidate_search_sweep.py --steps 1 0.5 0.25 0.1
python src/experiments/evaluate_search_sensitivity.py
python src/experiments/search_sensitivity_figures.py
```

`publish_search_reference.py` optionally copies the selected 8 ft / 0.25 ft cache to the canonical
alternative outputs and rebuilds their summaries and figures. The regular `cf_alternatives.py`
entry point can generate those outputs without sensitivity caches.

The seven sensitivity figures can be regenerated directly from the versioned `summary.csv`,
`players.csv`, `convergence.csv`, and `geometry_comparison.csv` tables under
`results/search_sensitivity/`, without raw shots or model caches. The geometry illustration can
also be regenerated entirely from versioned compact inputs:

```text
python src/experiments/search_sensitivity_figures.py
python src/experiments/three_point_geometry_audit.py
```

Historical geometry tables are fixed research results; recomputing them with the evaluator's
`--geometry` option requires the original local pre-fix snapshots, which are not distributed.
Seed and sample-size result tables come from `cf_dependence_seed_sweep.py` and
`cf_dependence_subsample.py`. They are supplemental diagnostics, not extra final figures.

See [the evaluation](dependence_search_evaluation.md) for interpretation and limitations,
and [the methods](behavioral_dependence.md) for formulas and replacement rules.


## Additional information-beyond-volume validation

Six further figures accompany [the validation report](dependence_validation.md), bringing the
published set to 33 figures. These use current full-sample matching or newly fitted forward-time
models, as identified in their captions. The temporal outputs do not overwrite the original analysis.

| Figure | Measurement |
|---|---|
| [Attempt-rate bands](../plots/metric_validation/attempt_rate_bands.png) | Dependence variation within 5-pp attempt-rate bands |
| [Matched pairs](../plots/metric_validation/matched_player_pairs.png) | Differences and conditional game-bootstrap intervals for control-selected pairs |
| [Matched strategy summary](../plots/metric_validation/matched_strategy_summary.png) | All 125 disjoint pairs across every strategy |
| [Forward prediction](../plots/metric_validation/forward_prediction.png) | Held-out scoring prediction improvement from adding dependence or advantage per three |
| [Forward residual repeatability](../plots/metric_validation/forward_residual_repeatability.png) | Persistence after applying an early-fitted control trend to later periods |
| [Alternative placebo](../plots/metric_validation/alternative_placebo.png) | Actual statistics compared with 500 context-stratified shuffles per strategy |

Each also has an SVG export. Regenerate from versioned tables with
`python src/experiments/dependence_validation_figures.py`; see the report for model and analysis commands.
