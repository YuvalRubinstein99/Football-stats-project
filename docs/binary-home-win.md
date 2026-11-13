# Direct home-win classification

This experiment trains on the binary target itself:

```text
1 = home score > away score
0 = home score <= away score (draw or away win)
```

A regularized logistic regression and a 500-tree random-forest classifier learn
home-win probabilities directly. Neither fits goal counts or converts Poisson
score probabilities. The forest uses log-loss classification, depth 5,
`max_features=0.3`, and `max_samples=0.5`. Both models use the observed class
frequencies, without class reweighting or oversampling.

## Inputs and chronological comparison

The eight feature sets from the four-improvement experiment are retained:
original baseline, old positional features, rematched positions, compact gaps,
separately weighted unit strengths, lineup deviations, FIFA priors/experience,
and all features including monthly learned player corrections.

The sample is unchanged: 8,521 training matches and 1,065 test matches from
Spain, Germany and Italy. Median imputation, missing indicators, category
encoding and logistic numeric standardization are fitted only on the training
portion of each fold. Unknown team categories are handled explicitly.

Forward seasons 2019–20 and 2020–21 select logistic `C` from 0.01, 0.1, 1 and 10
for each candidate, then the candidate within each model family, using minimum
mean binary log loss. Random-forest settings stay fixed. Choices are saved
before evaluating 2021–22 and 2022–23. All candidates are reported without
choosing a new winner from test results.

The decision rule is fixed at `P(home win) >= 0.5`, including ties. The models
are compared with the same feature sets' earlier goal-model home-win
probabilities and the normalized market reference. Binary log loss, binary
Brier score, AUC, correct counts and accuracy are reported. Paired differences
use 2,000 calendar-week bootstrap samples; these intervals do not capture all
team dependence or the uncertainty from repeated experiments.

Earlier results can update later months' dynamic player features; prediction
model weights remain fixed during each evaluated season. The audit/test data
have been inspected previously, and FIFA publication dates remain unverified.
The player effects are starter-based approximations without substitution
minutes. These are exploratory sequential pre-match results.

## Saved models and inference

Outputs are isolated under `results/lineup_improvements/binary_retrained/`.
There is one model per family/candidate plus `Logistic_selected.joblib` and
`RF_binary_selected.joblib`, fitted on all training matches through 2021–22.

`src/home_win_model.py` exposes `fit`, `target`, and `predict`. Inference needs
the prepared pre-match feature columns, not final scores. `predict` returns
two columns in the saved, checked order:

```text
[p_no_home_win, p_home_win]
```

`p_no_home_win` combines draw and away win. It is not the probability of a
straight away-win bet. The binary classifier cannot separately price draws
and away wins without another model. This experiment does not apply odds
weighting, skip filters or optimize a betting threshold.

Run from the repository root after preparing the lineup-improvement features:

```powershell
python scripts/test_binary_home_win.py
python scripts/train_binary_home_win.py
python scripts/verify_binary_home_win.py
```

`report.html` compares the selected and all-candidate results. `predictions.csv`
contains per-match binary probabilities and decisions. `validation.csv`,
`selection.json`, `plan.json`, `paired_differences.csv`, and `feature_weights.csv`
record selection, source hashes and diagnostics. Logistic numeric coefficients
are per training standard deviation; forest feature weights are impurity-based
importances. Saved-model reload predictions and exported metrics are verified.

## Initial results

Earlier-season validation selected `improved_positions` with C=0.01 for
logistic regression, and `fifa_prior` for the binary forest. On the 1,065-match
2022–23 test sample, with the fixed 50% threshold:

| Model | Correct | Accuracy | Binary log loss |
|---|---:|---:|---:|
| Selected direct logistic | 676 | 63.47% | 0.63669 |
| Earlier selected Poisson, converted to binary | 677 | 63.57% | 0.63437 |
| Selected direct binary forest | 679 | 63.76% | 0.63142 |
| Earlier selected goal forest, converted to binary | 681 | 63.94% | 0.62967 |

The direct logistic model with original baseline features reached 683 correct
(64.13%), but was not the validation-selected logistic variant. All candidates
remain available; test performance does not override the saved choices.
Paired test log-loss intervals cross zero. Direct binary training did not
establish a reliable advantage over the existing goal-model conversion.

Four unit tests passed, including saved-model round trips. All 34,080 direct
binary forecast rows reconcile to the reported metrics; selected models
reproduce their full test predictions after reloading.

## Betting without draw bets

Run `python scripts/backtest_binary_strategies.py`; check the settlement rules
with `python scripts/test_binary_strategies.py`. Results are saved under
`results/lineup_improvements/binary_retrained/betting_no_draw/`.

Only straight home and away wagers are placed. Drawn matches are retained and
lose either wager. Mixed H/A rules use `P(no home win)` as the away selection
score; it is a heuristic proxy and overstates the actual away-win probability
by the unknown draw probability. Thus their away-side edge and tier scores
cannot be interpreted as genuine expected return or Kelly fractions. Home-only
expected-return rules use the model's correct paying-event probability.

The adaptations evaluate probability, return-score, log-odds, square-root odds,
square-root profit, legacy market-power and legacy variance rules. Home-only
rules test a 50% probability cutoff, positive expected return, an expected
return above 2%, and a square-root-profit comparison against the away proxy.
The fixed legacy variance filter retains odds 4–1000 and score edge above 2%.
No thresholds are tuned on the resulting returns. Synthetic double-chance
rules are omitted; draw-to-home variants duplicate other rules here.

Flat staking places one unit on every rule selection. The existing 1–5-unit
tier rule further skips non-positive score edges and raises stakes by one tier
per percentage point of `edge/(odds-1)`, capped at five units. Both total stakes
and flat profit on the exact same selected bets are reported to separate
selection from sizing. There is no compounding or bankroll constraint.

For the validation-selected binary forest on the 1,065-match test sample:

| Rule | Bets | Flat net units | ROI |
|---|---:|---:|---:|
| Most likely side | 1,065 | -78.59 | -7.38% |
| Return-score filter | 1,065 | -134.52 | -12.63% |
| Log-odds weighting | 1,065 | -147.10 | -13.81% |
| Square-root odds weighting | 1,065 | -82.53 | -7.75% |
| Square-root profit weighting | 1,065 | -123.60 | -11.61% |
| Legacy market-power weighting | 1,065 | -70.70 | -6.64% |
| Legacy variance weighting | 1,065 | -102.23 | -9.60% |
| Filtered legacy variance | 457 | -71.16 | -15.57% |
| Home only, probability >=50% | 382 | -13.72 | -3.59% |
| Home only, positive expected return | 248 | +28.62 | +11.54% |
| Home only, expected return >2% | 169 | +40.57 | +24.01% |

The home-only square-root comparison produced +24.25 units from just five
bets, an extremely small sample; its earlier audit sample was one losing bet.
The home-only >2% rule had 78 winning and 91 losing test bets, including 36
draw losses. Tier sizing on the same 169 bets staked 648 units and returned
+160.91 units (+24.83% ROI), compared with +40.57 at flat one-unit stakes.
Its earlier audit result was -0.76 flat units over 184 bets (-0.41% ROI), or
+9.74 with tiers on 694 units staked (+1.40% ROI).

For the selected logistic model, home-only positive expected return returned
+24.92 flat units from 281 bets, and >2% returned +26.09 from 231 bets. Both
rules lost flat units in the earlier audit season. These historical comparisons
do not establish a dependable profitable strategy, and they do not exclude
losing matches after observing results.

Four strategy tests passed. The 102,240 selected-model fixture/strategy/staking
rows and all 96 summaries were reconciled independently to recorded odds,
scores and stakes. Complete tables for every model candidate, selected-model
ledgers and weekly returns are saved alongside the report.

## Training-period forward predictions

Run `python scripts/backtest_binary_training_oos.py`, followed by
`python scripts/plot_binary_training_oos.py` and
`python scripts/report_binary_training_oos.py`. Validate with
`python scripts/test_binary_training_oos.py`. Outputs are under
`results/lineup_improvements/binary_retrained/training_oos/`.

The fixed-specification experiment refits the same selected `fifa_prior`
binary forest on all earlier seasons, separately forecasting 2015-16 through
2021-22. The first 1,066 matches in 2014-15 are training warm-up; they cannot
receive earlier-season predictions from this dataset. Every imputer, encoder
and forest is independently fitted using earlier seasons only. However, the
feature specification was selected retrospectively using 2019 and 2020, so
this is a forecast check of the chosen specification, not an independent
assessment of its selection.

The nested experiment additionally chooses among the eight feature sets
using only the preceding two forward validation seasons' mean binary log
loss. It forecasts 2017-18 through 2021-22, after 2014-17 initializes fitting
and validation. Each choice is recorded before evaluating the forecast
season. Forest settings remain fixed at 500 trees and the original depth,
feature subsampling and seed. The two betting thresholds and stake rules
remain unchanged; returns are never used to select features.

Lineup-history and monthly player-effect features continue to update from
already completed matches, as in the original sequential evaluation.
Unverified FIFA snapshot publication dates and previous researcher exposure
to these seasons remain limitations. Nested selection does not erase earlier
research decisions about the candidate grid or forest settings.

Both plots settle all matches on a date together and include drawn matches
as losing home bets. Their endpoints reconcile to the saved per-match ledger.
The report includes season, league and odds-band diagnostics, an odds-only
probability benchmark, and descriptive four-week-block bootstrap ROI intervals.
Those intervals do not incorporate fitting or research-selection uncertainty.

The same-specification run covers 7,455 forecast matches. At flat stakes,
positive EV returns -21.47 units on 2,500 bets (-0.86% ROI), and EV >2%
returns -6.44 on 2,022 bets (-0.32%). Tiers return +21.89 on 9,656 units
staked (+0.23%) and +56.10 on 8,716 staked (+0.64%), respectively. Daily
peak-to-trough losses reach 148.15/110.46 flat units and 479.04/417.32 tier
units. These totals do not replicate the strong 2022-23 performance.

Nested selection covers 5,324 forecast matches. Positive EV returns -68.30
flat units (-3.89%) and -310.30 tier units (-4.77%); EV >2% returns -71.65
flat units (-5.18%) and -313.93 tier units (-5.44%). The shorter period also
matters: on the identical 2017-21 seasons, the fixed specification loses
-89.29/-69.31 flat units and -300.14/-258.53 tier units. Thus the difference
between the full fixed and nested totals cannot be attributed solely to
feature-selection hindsight.

The nested >2% flat portfolio predicted +175.22 units but realized -71.65.
Among its 354 bets at odds >=5, the mean home-win forecast was 18.94%, while
14.41% won. Normalized bookmaker probabilities also beat the forest on
aggregate log loss (0.60150 versus 0.60439) and Brier loss (0.20714 versus
0.20833). These diagnostics motivate testing past-only probability calibration
and a market-probability blend before increasing stakes. They do not establish
that either change will improve returns. Compare calibration and blending on
later forward folds; choose thresholds within earlier folds and reserve fresh
seasons for the final assessment. Audit rating and odds availability timestamps
before interpreting any result as executable historical profit.

## In-sample training-fit curves

Run `python scripts/plot_binary_training_in_sample.py` to apply the saved
`RF_binary_selected.joblib` model directly to all 8,521 rows it was fitted
on (2014-15 through 2021-22). This uses ordinary fitted-model probabilities,
not out-of-bag or cross-validated predictions, and does not refit the model.
The script checks the saved model's input hashes, row count and training
cutoff, independently reconciles all 34,084 betting settlements, and verifies
the four plotted endpoints against the summaries.

Outputs are under `results/lineup_improvements/binary_retrained/training_in_sample/`:
the PNG/SVG graph, predictions, daily curves, settlement ledger, summary, and
model provenance. The graph compares home-only EV >0% and >2% with flat and
existing 1-5-unit stakes. Draws remain losses. These are explicitly labeled
training-fit results and should not be interpreted as unseen-match performance.

## Regularization, past-only calibration and a frozen 2025-26 holdout

Run these scripts in order:

1. `python scripts/train_regularized_home_win.py`
2. `python scripts/prepare_regularized_holdout.py`
3. `python scripts/evaluate_regularized_holdout.py`
4. `python scripts/test_regularized_home_win.py`
5. `python scripts/verify_regularized_home_win.py`
6. `python scripts/report_regularized_home_win.py`

Outputs are in `results/lineup_improvements/binary_retrained/regularized/`.
The experiment leaves the original saved model unchanged. A separate
`frozen_model.joblib` and hashed selection/plan record define the candidate.

The declared grid contains the bookmaker probability baseline and nine fitted
models: odds-only logistic regression; compact logistic models with C=0.01
or 0.1; and 500-tree forests with depth/leaf-size settings (2,100) and (3,50).
Compact features comprise normalized market log odds, overround, overall
rating difference, last-five goal-difference and points gaps, and matchweek.
The positional variant adds six matchup/coverage features, giving 12 inputs.
There are no team-identity dummies or redundant transformations of ratings.

For each forward season, raw predictions and sigmoid-calibrated predictions
are evaluated. Calibration uses logits and a C=1 logistic map fitted on the
previous two seasons of raw out-of-sample predictions (one season initializes
2016). The classifier, preprocessing and calibration never use outcomes from
the season being predicted. The annual model/calibration choice minimizes
mean log loss over the two previous seasons; outer evaluation covers 2018-22.

Betting cutoff selection considers 0%, 2%, 5% and 10%, requiring at least
100 prior bets combined and positive flat profit in both previous seasons.
Among eligible cutoffs it maximizes prior total flat net; otherwise it
selects no betting. This abstained in every historical outer season. The
fixed 0% and 2% filter charts are separate diagnostic comparisons; their
profits must not be attributed to the adaptive cutoff-selection policy.

On identical 5,326 historical replay fixtures, log loss is 0.60939 for the
new selection process, 0.61012 for the original forest, and 0.60892 for
normalized bookmaker probabilities. Thus the improvement over the original
forest is small and does not surpass the market baseline. These historical
seasons were previously examined and are not untouched tests.

Before opening 2025-26 results, the final choice was frozen using 2021 and
2022 validation: the 12-feature depth-2 forest with minimum leaf size 100,
raw probabilities, and the >0% home-only cutoff. Calibration was tested but
not selected. It was fitted on 9,586 matches ending June 4, 2023. The final
in-sample flat ROIs are 4.65% (>0%) and 6.36% (>2%); this training period
includes 2022-23 and therefore differs from the earlier 8,521-match fit plot.

The reserved 2025-26 sample includes Spain, Italy and Germany. Of 1,066
matches, 286 have complete unique lineups, at least ten dated starter ratings
per side and a unique fixture join. Exclusions: 634 insufficient dated ratings
and 146 missing/ambiguous fixture joins. All models use the same eligible
rows in the paired comparison; the all-fixture market metric is labeled
separately. Ratings are dated September 26, 2022 or September 22, 2023 and
therefore stale. Actual historical lineups are used; odds capture time relative
to lineup announcements is not verified. The eligible subset is not a
representative full-season test, and global lack of prior researcher exposure
to the raw data cannot be established from local experiment records.

| Frozen holdout filter | Bets | Flat net | Flat ROI | Tier net | Tier ROI |
|---|---:|---:|---:|---:|---:|
| EV >0%, selected | 95 | -11.21 | -11.80% | -42.85 | -13.06% |
| EV >2%, fixed comparison | 67 | -12.73 | -19.00% | -50.17 | -17.48% |

Holdout log loss is 0.62660 for the model versus 0.62372 for the market on
the same 286 matches. The >0% flat ROI's descriptive four-week block-bootstrap
95% interval is approximately -40.1% to +14.9%; uncertainty is large. These
results do not demonstrate a betting edge. The holdout was not used to revise
the model, calibration or cutoff, and is now an inspected evaluation sample.

## Old versus new on identical training and test rows

Run `python scripts/compare_binary_same_split.py` and then
`python scripts/plot_binary_same_split.py`. Outputs are saved separately in
`results/lineup_improvements/binary_retrained/matched_comparison/`.

Both fixed specifications are refitted on the exact same 8,521 training rows
(2014-15 through 2021-22), and tested on the same 1,065 rows (2022-23). The old
model retains its full `fifa_prior` feature set and depth-5 forest; the new
model retains its 12 features, depth-2 trees and minimum leaf size 100. Both
use 500 trees, the same seed and the same home-only betting rules. Preprocessing
is fitted independently using only those common training rows. This changes
both feature representation and forest regularization, so their separate
causal effects are not isolated.

The 2025-26 table lacks the old specification's complete rating/history inputs.
The original common complete split is used rather than fabricating those
inputs. This is a retrospective diagnostic: 2022-23 influenced the new model
selection, so it is not an untouched test of that selection procedure.

| Model | Training log loss | Test log loss | Test minus training |
|---|---:|---:|---:|
| Old | 0.579214 | 0.631419 | 0.052205 |
| New | 0.603450 | 0.630652 | 0.027201 |
| Normalized market | 0.602245 | 0.632411 | 0.030167 |

The log-loss gap is about 48% smaller, and the Brier-score gap decreases from
0.022348 to 0.012373. Training fit becomes worse while test log loss changes
only slightly. A paired four-week block-bootstrap interval for new minus old
test log loss is -0.00441 to +0.00337, with point difference -0.00077. This
does not establish an improvement in test prediction quality and excludes
uncertainty from prior model-selection exposure.

| Model and filter | Training flat ROI | Test flat ROI |
|---|---:|---:|
| Old, EV >0% | 43.63% | 11.54% |
| New, EV >0% | 3.79% | 16.42% |
| Old, EV >2% | 53.45% | 24.01% |
| New, EV >2% | 6.91% | 8.23% |

Thus the new model's training betting returns are much less optimistic, but
test betting performance is mixed across the two filters. The paired plots
compare models directly with identical dates in each panel; the other figure
uses the existing 1-5-unit stake rule. Every panel has its own vertical scale.

Verification checks equality of all row identities, scores and odds between
models, excludes outcomes from inference, confirms saved-model reloads, and
independently reconciles all 76,688 settlements and 16 graph endpoints. The old
refit reproduces its original saved model; the new refit's test predictions
reproduce its earlier raw 2022 forward fit. Original saved artifacts remain
unchanged.
