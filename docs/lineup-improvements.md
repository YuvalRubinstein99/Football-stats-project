# Four improvements to the lineup model

This experiment implements identity recovery, changes from a team's recent
lineups, separately learned unit weights, and dynamic player strengths with
FIFA priors. It preserves the earlier experiment's 8,521 training and 1,065 test
matches and saves results separately under `results/lineup_improvements/`.

## 1. Identity recovery

The new matcher adds explicit club aliases and uses unique names from earlier
FIFA editions when they refer to the same stable EA player ID. Club-constrained
full-name tokens, explicit initials and substantial given-name prefixes can
identify a player when an exact name is unavailable. It does not apply fuzzy
edit-distance matching or use outcomes to select identities. Ambiguities remain
unmatched. Conflicting duplicate IDs raise an error, and a repeated starter ID
invalidates that fixture's positional features.

The old pipeline required a recovered player's overall rating to equal the
legacy lineup's overall value. The revised pipeline records that agreement but
does not use it to reject an independently matched identity. Both values,
matching method, club agreement and source row remain in `identity_audit.csv`.
Exactly identical FIFA snapshot rows are deduplicated; FIFA 23 contained 119
such duplicate rows. Stable IDs come from `sofifa_id`, or the FIFA 23 image URL.

| Test-set coverage | Old matching | Improved matching |
|---|---:|---:|
| Matched starter appearances | 19,946 | 21,890 |
| All positional comparisons available | 567 | 793 |
| All 22 starters matched with recognized positions | 75 | 299 |

Positions are still first-listed FIFA preferred positions. Minimum coverage
rules remain unchanged: at least eight accepted starters per side; at least
one attacker/goalkeeper or two midfielders/defenders for unit means. Missing
values are imputed within downstream training folds, with missing indicators.

## 2. Lineup changes

For each club, subtract its mean unit strength over up to five previous lineups
from the current unit strength. History includes home and away appearances,
uses a 365-day maximum age and requires three valid historical observations.
The eight strengths cover attacking shooting/dribbling/pace, midfield
passing/defending, defensive defending/pace, and goalkeeper overall.

All matches on the same date read the history before that date is added. A new
season can use recent prior-season lineups; it cannot use future lineups. Each
feature has a historical observation count. These are changes relative to the
club's recent starting selections, not a claim about its strongest available XI.

## 3. Independent matchup weights

The compact model receives home and away unit strengths separately, plus unit
counts. Separate home-goal and away-goal Poisson regressions can therefore learn
different coefficients for an attack's shooting and the opposing defense's
defending. No equal-and-opposite coefficient constraint is imposed.

Controls retain fixed differences over the same strength attributes. Numeric
inputs are standardized using training data only. Poisson L2 alpha is selected
from 0.01, 0.1 and 1.0 on the validation seasons; the random forest retains the
existing 500-tree, depth-five configuration. `coefficients.csv` exports Poisson
coefficients: numeric coefficients are per training standard deviation, while
team indicators are unscaled. These are conditional associations, not causal
estimates of a player's influence.

## 4. Dynamic FIFA-prior player strengths

This is a starter-based approximation inspired by
[FIFA-prior adjusted plus-minus](https://arxiv.org/abs/1810.08032) and
[offensive/defensive player ratings](https://www.mdpi.com/2076-3417/10/20/7345).
It is not a reproduction of either paper's full method. The source lineups do
not provide reliable substitution exposure or red-card timing.

Each player's initial log-strength is `0.02 * (overall - 70)`, for both attack
and defense. A penalized Poisson model learns separate attacking and defensive
corrections for stable player IDs, accounting for the other matched starters
and opponents. A side's expected goal rate is:

```text
log(expected goals) = league/venue intercept
                   + mean own FIFA prior - mean opposing FIFA prior
                   + mean own attacking correction
                   - mean opposing defensive correction
```

All matched starters contribute equally to the relevant averages. Goalkeepers
are included in the starter effects. Both sides must have at least eight
matched starters for a match to update the model. Player corrections have a
zero-centered Gaussian prior (L2 precision 1.0); league/venue intercepts receive
a small 0.001 penalty. Historical observations have a 365-day half-life. These
parameters are fixed before evaluation. Unknown players retain zero correction.

The player model is refitted before every calendar month using only earlier
months. Every fit checks convergence and its gradient. Corrections and prior
appearance counts become features in the downstream outcome model. A control
includes the same FIFA priors and appearance counts without learned corrections.

## Evaluation and outputs

Eight candidates isolate the changes: `baseline`, `old_positions`,
`improved_positions`, `compact_gaps`, `learned_weights`, `lineup_changes`,
`fifa_prior`, and `full_dynamic`. The last four progressively add independent
unit strengths, lineup deviations, matched FIFA priors/experience and learned
player corrections. All retain the original model's team names, form, odds,
matchweek and overall-rating inputs.

The 2019–20 and 2020–21 forward folds select Poisson penalties per candidate and
then a candidate per model family. Choices are persisted before reporting the
2021–22 audit and 2022–23 test. Outcome-model parameters stay fixed within each
evaluation season. Features update sequentially: earlier audit/test results can
affect later months' player estimates, and earlier lineups can affect later
lineup deviations. This is a sequential pre-match evaluation, not a simulation
of forecasting the entire season before it begins.

The audit/test seasons have previously been inspected. FIFA publication dates
remain unverified. Partial lineups and uncertain actual roles limit attribution.
Week-block bootstrap differences do not account for all team dependence or
experiment selection. The live prediction model is not automatically replaced.

```powershell
python scripts/prepare_lineup_improvements.py
python scripts/prepare_dynamic_lineups.py
python scripts/test_lineup_improvements.py
python scripts/compare_lineup_improvements.py
python scripts/verify_lineup_improvements.py
```

Open `results/lineup_improvements/report.html` for the comparison. Models,
predictions, coefficients, matching/coverage audits, monthly fit diagnostics,
feature manifests, selection scores and paired differences are saved alongside
it. Saved outcome bundles use `predict` from `compare_positional_matchups.py`
with the corresponding prepared feature rows. The saved player state reflects
the final prediction month's cutoff and requires a new past-only fit when
refreshing for a later month.

## Results

Training-only selection retained the original feature baseline for both model
families. Every Poisson candidate selected alpha 0.1 from the validation grid;
the previous experiment used a fixed 0.01, so its Poisson values are not the same
baseline comparison. On the identical 1,065 test matches:

| Model | Baseline log loss | All four changes | Baseline accuracy | All four changes |
|---|---:|---:|---:|---:|
| Poisson (selected alpha 0.1) | 0.98710 | 0.98741 | 53.90% | 53.80% |
| Random forest | 0.98262 | 0.98350 | 53.33% | 53.24% |

Lower log loss is better. Improved matching alone reduced the RF positional
model's test loss from 0.98354 to 0.98303, but the original RF baseline remained
better. Learned separate weights produced a small RF improvement over compact
fixed gaps (0.98303 to 0.98294). Lineup deviations did not improve either family
on the test set. Learned player corrections improved the Poisson prior-control
model from 0.98868 to 0.98741, but worsened the corresponding RF model.

All test-set paired intervals cross zero, and the audit-season comparisons are
mixed. These data support the identity-coverage improvement, not a reliable
predictive advantage for the new features. The experiment does not select a
model based on the best-looking test result.

Validation completed: eight unit tests, saved-model reload parity, all 34,080
forecast rows reconciled to metrics, 92 monthly player-fit cutoffs and
convergence checks, and identical fixture samples with unique starter IDs.
