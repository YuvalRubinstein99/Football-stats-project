# Data inventory and provenance

The cleanup preserved the existing CSV contents. File names and schemas suggest football match tables, recorded Bet365 decimal odds, and player ratings; the repository does not currently contain sufficient attribution or licensing records to establish every upstream source.

| Location | Purpose |
|---|---|
| `data/train.csv`, `data/test.csv` | Prepared inputs used by the current experiment |
| `data/players_15.csv` through `data/players_23.csv` | Existing player-rating tables |
| `data/fifa_season.csv` | Existing season player-rating input |
| `data/bl_data/`, `data/epl_data(fully_proccessed)/`, `data/laliga_data/`, `data/seriea_data/`, `data/ucl_data/`, `data/season_data/` | Historical league and intermediate tables, retaining their original names |
| `data/X_train.csv`, `data/X_test.csv`, `data/Y_train.csv`, `data/Y_test.csv` | Historical exported feature/label files; not the current training entry point |
| `data/format.csv`, `data/predictions.csv` | Historical inference example/input and output |

Prepared targets are `home_score` and `away_score`. Team names, player-rating summaries, historical form variables, matchweek, and `B365A`, `B365D`, `B365H` odds are inputs. The current script fits its encoder using training data only and uses a common feature order for test data.

The prepared files have no date column, and the old notebook expects a `league` column that these files do not have. The current workflow handles the absent league column; the historical notebook has not been repaired beyond portable paths.

Before public redistribution, add the original source links and applicable permissions for the datasets. No dataset license is inferred or granted by this repository cleanup. Large historical files were retained, not silently deleted or removed from Git history.
