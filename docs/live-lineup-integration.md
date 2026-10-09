# Live lineup integration

Start the local interface with `python run.py app` and choose **Lineups + EA ratings (saved RF)**.
Install dependencies first with `python -m pip install -r requirements-live.txt`.
The desktop development setup also recognizes an optional `.lineup_deps/` directory.

```bash
python run.py predict --model lineup_rf --leagues E0
python run.py predict --model lineup_rf --leagues E0 SP1 D1 I1 F1
python run.py predict --model lineup_rf --leagues E0 --offline
python scripts/test_lineup_integration.py
```

The default CLI model is now `lineup_rf`. Use `--model poisson` for the separate
results-only baseline. The lineup mode uses the current season for form regardless
of `--seasons`; the saved forest was fitted on the original historical training set.
When no saved `results/vector_forest.joblib` is available, run `python run.py train`
to create it and the associated experiment metadata.

## Input sources

- [RotoWire](https://www.rotowire.com/soccer/lineups.php): public predicted or confirmed
  starting XIs for the five leagues. Injury-list entries are excluded from the XI;
  injury flags on selected starters are retained. A fixture must match by league,
  date, home club, and away club. Incomplete or unavailable XIs withhold predictions.
- [EA player ratings](https://www.ea.com/games/ea-sports-fc/ratings): the current
  edition's launch overall ratings, read from public page data. Only men's club
  groups and players are used. Rosters are cached for seven days. The downloaded
  edition is recorded; the old local FIFA 2023/24 files are not silently reused.
- [Football-Data](https://www.football-data.co.uk/data.php): current-season results,
  upcoming fixtures, and the explicit `B365H`, `B365D`, `B365A` odds. Another
  bookmaker's odds are never relabeled as Bet365. Missing odds withhold predictions.

No login or paid subscription is needed for the verified public pages. Availability
and HTML schemas may change; failed reads are reported. Blocked sources are not
bypassed. Requests save source URLs, download timestamps and hashes in
`results/live/predictions.json`. Offline mode requires previously cached files;
lineup caches older than 24 hours are not used for predictions.
RotoWire kickoff times are converted from US Eastern time. Fixtures already
started are withheld; today's fixtures with unverifiable kickoff times are also
withheld.

## Matching and features

Clubs use explicit aliases in `src/lineup_sources.py`. Players are matched within
their club by normalized exact name or unique full-name tokens. There is no
fuzzy-name maximum or replacement rating for an unknown starter. When a player
is missing from the club's EA roster, a public full-name search can match a unique,
exact name across clubs; the result is prominently flagged as a club mismatch.
This handles launch-roster transfers but still warrants review. Ambiguous or
unresolved players withhold the fixture.

Each side requires 11 distinct EA player IDs. Their rating summaries reproduce
the legacy definitions: maximum, minimum, sample standard deviation (`ddof=1`),
mean, **mean of logs**, **mean of square roots**, third highest, third lowest,
and maximum times minimum. Log of the mean would be a different feature.

Form uses current-season league results before the run's UTC date: cumulative
goal difference and points, plus totals over the last five completed matches.
All today's scores are excluded. The original repository lacks the original
form-generation code, so these date-based definitions are explicit; postponed
match treatment may differ from the historical prepared tables.

The fixture feed has no official matchweek. Equal home/away games played permits
an explicitly labeled `games played + 1` estimate. Unequal counts require an
override. Fixtures preceded by another scheduled fixture involving either team
are withheld until their form can be refreshed. Estimates can still differ from
official round numbers after postponements and must be checked.

The saved encoder is reconstructed from the unchanged historical training data.
Its complete feature order, training-data hash, scikit-learn version, and model
dimensions are checked against the saved run metadata. Tests also reproduce
committed goal predictions from held-out prepared inputs. Unseen teams are flagged
and use the original encoder's unknown-category behavior. Ligue 1 was absent from
the saved training teams; its predictions therefore require particular caution.

The random forest supplies the two expected goal counts. Independent Poisson
distributions then convert those counts to scoreline and outcome probabilities,
as in the existing RF-vector experiment. This is not the results-only Poisson
strength model and not the separately calibrated bivariate model.

## Review and corrections

Expand a match in the interface to see its starters, ratings, source edition,
injury flags and matching caveats. Fixtures without verified inputs appear under
**Fixtures needing attention**. No baseline substitution occurs. The JSON report
includes every successful feature row and player identity, as well as withheld
fixtures and reasons. CSV exports contain successful predictions only.

`data/live_overrides.json` allows explicit identity and matchweek corrections:

```json
{
  "player_ids": {"ROTOWIRE_PLAYER_ID": "EA_PLAYER_ID"},
  "matchweeks": {"E0|YYYY-MM-DD|HomeTeam|AwayTeam": 6}
}
```

Use the fixture names exactly as shown by Football-Data. Player ID overrides must
refer to a player in the downloaded club squad; an ID absent from that roster is
rejected. Values above are a schema example, not actual player mappings.

This workflow has input and inference checks, not a prospective accuracy result.
Historical training used actual lineups and older FIFA ratings; predicted XIs
and current EA launch ratings introduce a distribution change. The interface
does not repurpose the baseline's walk-forward scores as RF validation.
