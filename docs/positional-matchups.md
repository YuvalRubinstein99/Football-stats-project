# Position-based prediction experiment

This experiment reuses the existing player identity matches to recover each
starter's first listed FIFA position, then compares opposing positional units.
It estimates preferred-position groups; it does not recover actual match-day
formations or track individual player duels.

| Unit | Preferred positions | Compared attributes |
|---|---|---|
| Attack | ST, CF, LF, RF, LW, RW, LS, RS | Overall, shooting, dribbling, pace, physicality |
| Midfield | CAM, CM, CDM, LM, RM and left/right central variants | Overall, passing, dribbling, defending, physicality |
| Defense | CB, LB, RB, LWB, RWB and left/right center backs | Overall, defending, pace, physicality |
| Goalkeeper | GK | Overall |

Every accepted starter belongs to one unit. Multiple preferred positions do not
duplicate a player. Left/right wingers count as attackers, wide midfielders as
midfielders, and wingbacks as defenders. Unknown positions remain unknown.

For each side, the model receives unit means and coverage counts, plus:

- Attack overall minus opposing defense overall.
- Attack shooting/dribbling minus opposing defense defending.
- Attack pace/physicality minus opposing defense pace/physicality.
- Attack overall minus opposing goalkeeper overall.
- Home midfield minus away midfield for each midfield attribute.
- Home goalkeeper overall minus away goalkeeper overall.

For example, home attackers' mean shooting of 82 and away defenders' mean
defending of 75 produce a home shooting-versus-defense gap of +7. This is a
predictor, not a fixed probability or a claim that different attributes share
the same calibration. Models estimate its association with goals during training.

Identity matching retains the existing name/club rules and original-overall
agreement check. Duplicate players invalidate the fixture's positional features.
Source hashes are verified before reusing audit row identifiers. Means require
at least eight accepted starters on that side, at least one attacker/goalkeeper
or two midfielders/defenders with valid ratings. Incomplete groups are estimates;
coverage counts accompany them. Missing means propagate into missing gaps, then
training-only median imputation and indicators handle them inside each model.

The comparison retains the exact 8,521 training and 1,065 test matches from the
existing attribute experiment (Spain, Germany, Italy). No candidate gets a
different sample. Four feature sets retain the same original team identities,
overall ratings, form, odds and matchweek:

1. Original baseline.
2. Existing whole-team attributes and comparisons.
3. Original baseline plus positional means, counts and matchups.
4. Whole-team attributes plus positional means, counts and matchups.

Separate home/away Poisson regressions (fixed alpha 0.01) and a joint goal random
forest (500 trees by default, depth 5) predict goal means. Independent Poisson
score distributions convert the means to home/draw/away probabilities. Each
family selects its candidate using mean log loss across 2019–20 and 2020–21
forward validation seasons. Choices are saved before the 2021–22 audit and
2022–23 test evaluation. All four candidates are reported for transparency.
Calendar-week block bootstrap differences measure the incremental effect of
positions against the corresponding baseline.

The audit and test seasons have been inspected in earlier experiments. FIFA
publication timestamps are unverified, and preferred positions need not match
the role played that day. Results remain exploratory. This experiment saves
models and historical predictions; it does not replace the live model.

Run from the repository root:

```powershell
python scripts/prepare_positional_features.py
python scripts/test_positional_matchups.py
python scripts/compare_positional_matchups.py
```

Preparation requires the existing `results/team_attributes` data and matching
audit; recreate them with `python scripts/prepare_team_attributes.py` if absent.
Outputs go to `results/positional_matchups/`: `report.html`, `metrics.csv`,
`predictions.csv`, `paired_differences.csv`, `coverage.csv`, `position_audit.csv`,
the feature and selection manifests, and one fitted model per family/candidate.
Saved bundles can be loaded with joblib and passed with a prepared feature frame
to `predict` in `scripts/compare_positional_matchups.py`.

## Initial results (500 trees)

The validation seasons selected the original baseline in both families. On the
1,065-match 2022–23 test sample:

| Model | Baseline log loss | With positional features | Baseline accuracy | With positional features |
|---|---:|---:|---:|---:|
| Poisson | 0.98627 | 0.98875 | 52.96% | 51.83% |
| Random forest | 0.98262 | 0.98354 | 53.33% | 53.52% |

Lower log loss is better. The random forest gets two additional outcomes right,
but its probability quality is slightly worse. Adding positions to the existing
whole-team attributes improves RF log loss from 0.98355 to 0.98290, still above
the original 0.98262 baseline. All reported paired 95% week-bootstrap intervals
cross zero. There is no clear improvement from positions in this experiment.

All matchup features are observed for 4,362/8,521 training matches and 567/1,065
test matches; other rows require some imputation. Only 75 test matches have all
22 starters matched with recognized positions. Better identity coverage and
observed match-day positions remain possible directions, not tested findings.
