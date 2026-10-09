# Methodology

## Models

1. **Separate RFs:** two 200-tree Poisson-criterion regressors predict home and away goal means. Three-fold training-only cross-validation selects the feature fraction using goal MSE.
2. **Calibrated independent control:** mean-scale corrections are fitted to out-of-fold training predictions.
3. **Bivariate Poisson:** a five-coefficient joint-likelihood layer takes the separate RF predictions as offsets and estimates three rates per match. Home and away scores share an additive Poisson component. This allows nonnegative covariance; it is not end-to-end joint training of the forests.
4. **Vector RF:** one native multi-output forest predicts `[home_goals, away_goals]` using shared splits. It uses 2,000 trees and squared-error loss. Depth 5, feature fraction 0.3, and sample fraction 0.5 are retained from an earlier training-CV selection. Training out-of-bag MSE is recorded. The two predictions still feed an independent-Poisson outcome conversion; a vector of means alone is not a joint probability distribution.
5. **Market baseline:** normalized inverse recorded decimal odds.

Exact bivariate outcome probabilities follow from the unshared Poisson rates: the common additive component cancels in the goal difference. A separate fixed-seed simulation draws 50,000 score pairs per match. Small differences between simulated and exact strategy returns can arise from Monte Carlo noise changing decisions near thresholds.

## Evaluation

- The existing `data/train.csv` and `data/test.csv` split is retained. Seven incomplete training rows are dropped; all 1,447 test matches remain.
- Goal metrics: MSE, RMSE, and MAE. Outcome metrics: accuracy, log loss, multiclass Brier score, and one-vs-rest AUC.
- The model uses recorded betting odds as features, matching the original workflow. It is therefore not an odds-free forecast.
- Cross-validation is not chronological or nested. Dates are absent from the prepared CSVs; chronological separation and upstream feature leakage have not been verified.
- Repeated inspection of the same test set influenced subsequent experiments. Treat these results as exploratory, not as a fresh model-selection holdout.
- In the RF comparison, the vector model has 2,000 trees and separate models have 200 each; this is not a controlled comparison of tree structure alone.

## Betting accounting

One unit is the total stake per selected match. A correct ordinary bet returns `decimal_odds - 1` in net profit; an incorrect bet returns `-1`. ROI is total net profit divided by total units staked. The most-likely rule chooses the highest probability; expected-return betting chooses the highest `p × odds` and bets only when that value exceeds one.

The other original rules are retained for comparison. Labels beginning with **Legacy** indicate heuristic scores. Moving draw probability into the home column does not turn a home-win wager into a double-chance bet; actual draws still lose that legacy home-win wager. The original variance-based score is not a Sharpe ratio.

### Binary decisions

- Home loses: `P(away win)` versus `P(home win) + P(draw)`.
- Home wins: `P(home win)` versus `P(away win) + P(draw)`.

No quoted double-chance odds are available. A combined outcome is synthesized by splitting a one-unit stake across its two mutually exclusive outcomes. For decimal odds `o1` and `o2`, effective odds are `D = 1 / (1/o1 + 1/o2)`, with component stakes `D/o1` and `D/o2`. Either covered outcome returns `D` units. This assumes both prices are available simultaneously with fractional stakes, excluding fees and rounding.

Square-root-profit weighting uses `p × sqrt(odds − 1)`. Synthetic payouts at or below one receive zero weight because they cannot generate a positive profit even when correct. The home-not-to-lose hedge has this property in 13 matches.

### Home win or skip

Keep only the home-win selections from the binary square-root-profit rule. Compare `p_home × sqrt(home_odds − 1)` with `(p_away + p_draw) × sqrt(max(D − 1, 0))`, where `D = 1 / (1/away_odds + 1/draw_odds)`. If the home score is at least as high, stake one unit on home; otherwise stake zero. Ties favor home, matching the binary rule. A draw or away win loses a placed home bet. Skips have zero profit and do not enter the ROI denominator. The synthetic odds are used only for comparison; no away/draw hedge is placed. No extra loss penalty or positive-expected-return filter is applied.

## Diagnostics

Individual matchweek rankings are saved in `matchweek_errors.csv` for every model. They report wrong-outcome counts and rates, mean log loss, and goal MAE/MSE, with separate descending error ranks (ties share the minimum rank). The notebook highlights the ten highest outcome-error weeks and plots all observed weeks for the vector RF. Weeks with fewer than 30 matches are flagged. These groups pool test matches by matchweek number across available competitions and seasons; differences may reflect their composition rather than a matchweek effect. Market probabilities have no goal-mean predictions, so market goal metrics remain missing.

The notebook includes actual-outcome recall, goal-total slices, player-rating gaps, matchweek, market-favorite strength, prediction confidence, teams, and the largest wrong-prediction log losses. Actual-outcome and goal-total slices are retrospective. Team groups overlap. Groups below 30 matches are flagged, and team rankings require at least 30 matches. These are descriptions of where errors occur, not evidence of their causes.

The binned scatter uses fixed probability-gap bins of width 0.10, where gap means `max(P away, P draw, P home) - min(...)`. Each point shows mean gap versus realized mean net profit on most-likely-outcome bets; it is not an estimated profitable betting threshold.

## Reproducibility artifacts

`results/run_details.json` records the seed, package versions, input SHA-256 hashes, model parameters, optimizer diagnostics, and feature list. Model selection and training have not changed as part of the repository reorganization. Read `results/methodology.txt` for the generated run notes.
