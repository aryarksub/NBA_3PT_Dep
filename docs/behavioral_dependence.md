# Behavioral three-point dependence

See also [the additional matched-player, forward-time, and placebo validation](dependence_validation.md). These tests assess incremental usefulness beyond the earlier correlation and split-half diagnostics.


This repository compares three implemented replacement rules: the naive two-point baseline,
nearest feasible two, and best feasible two. The current search uses an **8 ft radius and
0.25 ft grid**, with the corrected NBA arc and parallel corner boundaries.

## Definition

For each three-point attempt, subtract replacement expected points from observed expected
points. Two-point attempts contribute zero. Aggregate these differences for a player or team:

- `dep_total`: total expected-point difference.
- `dep_per_shot`: total difference divided by all attempts.
- `dep_per_3pa`: total difference divided by three-point attempts.
- `dep_share`: total difference divided by total observed expected points.

Negative values are allowed: the modeled alternative can be more valuable than the three.
The identity `dep_share = 3pa_rate * dep_per_3pa / mean_observed_EP` makes attempt volume an
explicit component. These quantities describe a counterfactual assumption, not observed causal effects.

## Expected points and geometry

Observed EP is make probability times the geometric shot value. Probabilities are cross-fitted
in five folds grouped by game. Each candidate uses the same fold model as its original shot;
the model must not have trained on that game. Recovered models are checked against stored
probabilities before building alternatives.

Geometric threes use the 23.75 ft arc and lines at y=3 and y=47 on a 94 by 50 ft court.
Boundary points count as twos. The program-derived labels remain distinct from source
play-by-play scoring labels. Tracked player positions do not identify shooting-foot placement.
The shot-distance feature retains its existing basket coordinates (5/89 ft), while the boundary
uses the official 5.25/88.75 ft coordinates; the sensitivity experiment holds model features fixed.

## Replacement rules

**Naive:** replace each three with the player's mean modeled two-point EP, falling back to the
league mean when that player has no twos. This is retained as `exp_pts_naive`.

**Nearest feasible:** search two-point grid locations on the same court half within 8 ft.
Require nearest-defender distance at least k times the original distance, then choose the
shortest displacement. Tested k values are 0, 0.5, 0.75, and 1. Output names are `exp_pts_near_k*`.

**Best feasible:** score the same geographic candidates. Adjust nearest-defender distance to
`max(0, candidate_distance - lambda * displacement)`, with lambda 0, 0.25, or 0.5. Candidate
average defender distance and shot distance are recomputed. Other original shot context is
retained. Select the candidate closest to the q quantile of model make probabilities, with
q 1, 0.95, or 0.90. Only q=1 is a strict maximum. The quantile and closing adjustments are
sensitivity assumptions; neither guarantees removal of selection bias or realistic defense.
Outputs are `exp_pts_best_q*_l*` and candidate EP equals twice the selected probability.

All methods retain two-point shots unchanged. Nearest/best fall back to the naive replacement
when no candidate qualifies. Report fallback alongside dependence. The empirical shot-location
support filter is available but disabled in the published analysis. The current best rule only
searches shot locations for the original shooter; passing and policy-based alternatives remain
unimplemented. Passing requires receiver context from the tracking data.

## Published findings and uncertainty

The corrected sample contains 88,694 shots, 24,306 geometric threes, and 531 games. Player
qualification is at least 50 shots and 10 threes (271 players). At the selected settings,
Spearman correlations with attempt rate are 0.703 for naive, 0.948 for equal-openness nearest,
and 0.972 for best q=.95 / lambda=.25. Local alternatives are strongly volume-driven.
Residual split-half correlations suggest some repeatable information beyond frequency,
but shared shot models and persistent shot mix prevent interpreting this as independent validation.

The legacy CSV field `degenerate` means only absolute attempt-rate correlation >=0.95.
It is a heuristic screen, not proof of invalidity. Specification ranges divided by bootstrap
widths compare two quantities; they do not estimate total uncertainty. Alternative summaries
use 500 shot-bootstrap resamples, holding probabilities and baselines fixed. These intervals
omit model, baseline-estimation, label, defensive-response, and replacement-rule uncertainty.

See [the full evaluation](dependence_search_evaluation.md) for controlled boundary comparisons,
radius and resolution sweeps, numerical convergence, and limitations. See [the figure catalog](figures.md)
for exact output files and regeneration commands. Baseline and alternative results remain separate.
