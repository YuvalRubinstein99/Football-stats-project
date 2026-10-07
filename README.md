# Football score prediction and betting backtests

Predict home and away football goals, convert the score predictions into match-outcome probabilities, and compare betting decision rules on recorded odds.

The current experiment compares two separate random forests, a single **2,000-tree multi-output random forest**, and a bivariate Poisson calibration layer. It also examines where predictions fail and whether probability-based betting rules produce positive historical returns.

## Start here

- [Main notebook](notebooks/prediction_bivariate.ipynb): saved results, model code, plots, and error analysis.
- [HTML report](results/report.html): download and open locally with the other files in `results/`; GitHub displays its source.
- [Model metrics](results/model_metrics.csv) and [strategy results](results/strategy_results.csv).
- [Methodology and limitations](docs/methodology.md).

![Where the models fail](results/failure_diagnostics.png)

## Current experiment

The existing split contains **11,561 usable training matches** and **1,447 test matches** after removing seven incomplete training rows. All models use the same test matches. The vector forest uses squared-error loss, 2,000 trees, depth 5, `max_features=0.3`, and `max_samples=0.5`.

| Model | Correct outcomes | Accuracy | Outcome log loss ↓ |
|---|---:|---:|---:|
| Separate random forests | 777 / 1,447 | 53.70% | 0.97879 |
| Bivariate Poisson, exact probabilities | 777 / 1,447 | 53.70% | 0.97883 |
| Single vector-output forest | 779 / 1,447 | 53.84% | 0.97884 |

The RF-based models never select a draw as their most likely outcome in this test set, despite assigning nonzero draw probabilities. Closely matched teams are also harder to predict. The notebook includes confusion matrices, calibration, team-level errors, goal errors, and individual wrong predictions.

### Betting results are exploratory

Each backtest stakes one unit per match unless the rule explicitly skips a bet. Reported profit is net of the stake. For the vector forest:

| Binary decision rule | Most-likely-side profit | Square-root-profit-weighted profit |
|---|---:|---:|
| Home wins vs. does not win | −75.32 units | +78.44 units |
| Home loses vs. does not lose | −62.62 units | −188.93 units |

Square-root-profit weighting chooses the largest `probability × sqrt(decimal_odds − 1)`. Double-chance payouts are **synthetic split-stake payouts**, not quoted market prices. A synthetic payout at or below one receives zero weight in that rule.

These strategies were explored repeatedly on the same test set. Positive returns are not independent confirmation of a profitable strategy. A fresh, chronologically held-out evaluation is needed before making that claim.

## Setup

Use **Python 3.12**. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

To open notebooks interactively:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

Open `notebooks/prediction_bivariate.ipynb`. Its embedded outputs can also be viewed on GitHub without installing anything. The saved model binary is optional: when absent, the notebook displays committed per-match predictions unless you explicitly choose to retrain.

## Reproduce

```bash
# Rebuild tables, plots, report, and notebook from saved predictions; no training:
python run.py report

# Refit models on data/train.csv and evaluate data/test.csv, then rebuild the report:
python run.py train

# Check Python/notebook syntax, probabilities, metrics, and strategy accounting:
python run.py check
```

The report command reconstructs the ignored betting ledgers from `results/match_predictions.csv`. The train command writes a local `results/vector_forest.joblib`; the model and detailed ledgers are intentionally ignored by Git. The report command does **not** evaluate newly edited model code—run `train` for that.

## Repository layout

```text
data/                 Existing CSV inputs and historical league tables
docs/                 Methodology, data notes, and legacy workflow notes
models/legacy/        Original saved estimators and encoders
notebooks/
  prediction_bivariate.ipynb   Current experiment
  legacy/             Original exploratory notebooks
results/              Curated metrics, predictions, figures, and report
scripts/              Project verification
src/                  Training, evaluation, reporting, and shared helpers
run.py                Train / report / check entry point
requirements.txt      Current workflow dependencies
```

Historical data and model files were preserved during cleanup. The repository still contains about 152 MB of pre-existing tracked material; restructuring does not reduce Git history. See [data notes](docs/data.md) before redistributing the datasets, and [legacy notes](docs/legacy.md) before running older notebooks. No software or dataset license has been selected by this cleanup.
