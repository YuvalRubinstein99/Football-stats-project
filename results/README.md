# Saved experiment results

Open [the main notebook](../notebooks/prediction_bivariate.ipynb) for the complete analysis, or download this directory and open `report.html` locally. PNG figures preview directly on GitHub; SVG versions are available for export.

| Files | Contents |
|---|---|
| `model_metrics.csv` | Goal errors and outcome-prediction metrics |
| `strategy_results.csv` | Bets, wins, losses, net profit, and ROI for every model/rule |
| `match_predictions.csv` | Test-row identifiers, actual scores, odds, predicted means/rates, and outcome probabilities |
| `run_details.json` | Input hashes, versions, feature names, seed, and fitted parameters |
| `error_slices.csv`, `team_errors.csv`, `worst_predictions.csv` | Failure analysis |
| `outcome_calibration.csv` | Per-outcome probability calibration |
| `probability_gap_winnings.csv` | Probability-gap bins and average realized winnings |
| `home_*_strategy.csv` | Binary decision-rule results, including square-root-profit variants |
| `*_confusion.csv` | Actual-outcome rows versus predicted-outcome columns |

Run `python run.py report` from the repository root to rebuild the report and plots from saved predictions. Detailed betting ledgers are reconstructed locally and ignored by Git. `python run.py train` regenerates the predictions and the ignored trained model.

Do not interpret a profitable rule in these saved results as independently validated: the same test set was inspected repeatedly. See [methodology](../docs/methodology.md) for the split, probability assumptions, and synthetic double-chance payouts.
