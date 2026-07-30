# FPL Predictions

An end-to-end project for predicting Fantasy Premier League player and squad
performance, then expressing position-specific and overall squad strength as
percentile ratings.

The current implementation provides the reproducible data and training-table
foundation:

- a configurable client for the official FPL JSON API;
- retries, timeouts, HTTP handling, and response-shape validation;
- normalized player, club, position, gameweek, and fixture tables;
- timestamped, unchanged raw JSON responses;
- timestamped Parquet tables and a snapshot manifest;
- a command-line refresh workflow;
- deadline-validated, season/gameweek prediction snapshots;
- immutable ingestion of finalized official live gameweek statistics;
- a DuckDB catalog over snapshot and historical Parquet data;
- past-only rolling features, snapshot-time fixture features, and future
  1/3/5-gameweek labels;
- historical-mean and recent-points benchmarks;
- regularized regression and random-forest point predictors;
- purged rolling-origin validation, model comparison, and calibration;
- versioned, schema-validated model artifacts and a reusable prediction service;
- tests that use small saved responses rather than the live API.

The official endpoints are undocumented and can change. A failed or structurally
unexpected response stops the refresh with an error; the application never
substitutes invented data.

## Setup

Python 3.12 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional settings are documented in `.env.example`. The application reads these
variables from the process environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FPL_API_BASE_URL` | `https://fantasy.premierleague.com/api/` | API root |
| `FPL_API_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `FPL_DATA_DIR` | `data` | Local snapshot root |

The project does not automatically load `.env`; export values in your shell or
use your preferred environment loader.

## Refresh current data

After installation, run either:

```bash
fpl-refresh
# or
python scripts/fetch_current_data.py
```

Useful command options:

```bash
fpl-refresh --help
fpl-refresh --data-dir data --timeout 30 --verbose
```

Each successful refresh creates one UTC-stamped directory in both locations:

```text
data/
├── raw/<snapshot-id>/
│   ├── bootstrap-static.json
│   └── fixtures.json
└── processed/<snapshot-id>/
    ├── players.parquet
    ├── clubs.parquet
    ├── positions.parquet
    ├── gameweeks.parquet
    ├── fixtures.parquet
    └── manifest.json
```

Raw and processed data are ignored by Git. The player table includes stable FPL
and Premier League identifiers, names, API-derived club and position mappings,
price in millions, photo reference and URL, ownership, status and availability,
and all additional current fields supplied in each player record.
Nested JSON values supplied by the API are stored as JSON strings in Parquet;
their unchanged structures remain available in the raw response.

## Collect prediction snapshots

Collect each snapshot before its official gameweek deadline:

```bash
fpl-snapshot-gameweek --season 2026-27 --gameweek 1
```

The command derives the deadline from `bootstrap-static` and rejects a
post-deadline collection. It adds these fields to every normalized row:

- `season`;
- `snapshot_gameweek`;
- `snapshot_timestamp`;
- `snapshot_deadline_time`.

The season label is explicit because the API does not expose a stable canonical
season identifier. A missed historical deadline cannot be reconstructed from
the current live API; retain scheduled snapshots for future training.

## Ingest finalized gameweek history

After a gameweek is marked finished by the API:

```bash
fpl-ingest-history --season 2026-27 --through-gameweek 6
```

This saves exact live responses plus normalized `player_gameweek_stats`,
fixtures, gameweeks, and clubs under an immutable ingestion ID. Repeated runs
are retained. DuckDB resolves duplicate season/gameweek/player records to the
most recently ingested copy.

The official live endpoint aggregates a player's gameweek total. A blank is
therefore an explicit zero-point gameweek record, while a double gameweek is one
record containing the combined points from both fixtures.

## Build the DuckDB catalog and training data

```bash
fpl-build-catalog
fpl-build-training-data --recent-window 3 --horizons 1 3 5
```

By default the catalog is `data/fpl.duckdb`. Important views include:

- `snapshot_registry`;
- `player_snapshots`;
- `fixture_snapshots`;
- `player_gameweek_stats`;
- `historical_fixtures`.

Training rows are keyed by season, snapshot gameweek, and player. Features use
only snapshot-time fields and finalized gameweeks strictly before the target.
Labels begin at the snapshot gameweek and remain null until every gameweek in
the requested horizon is finalized. A missing player/gameweek record is not
silently treated as zero.

Upcoming fixture features are computed from the saved fixture snapshot. Their
counts represent blanks and doubles without hard-coded gameweek IDs.

## Train and evaluate player-point models

Once enough labeled snapshot gameweeks have accumulated:

```bash
fpl-train-models \
  --horizons 1 3 5 \
  --min-train-gameweeks 4 \
  --calibration-bins 10
```

Use `--training-data path/to/training.parquet` to select a table explicitly.
Otherwise, the newest table under `data/processed/training` is used.

Each horizon compares:

- the historical label mean;
- recent mean points projected across the horizon;
- regularized linear regression;
- random forest.

Validation is chronological. For a validation snapshot at gameweek `G`, a
same-season training row is eligible only if its entire future label ended
before `G`. For example, validation at GW8 with a three-gameweek target may use
a GW5 row, but not GW6 or GW7. Previous seasons remain eligible when their
timestamps are earlier.

Reports include MAE, RMSE, Pearson correlation, within-gameweek Spearman ranking
correlation, and calibration by predicted-point quantile. The candidate with
the lowest rolling-validation MAE is fitted to every finalized row for that
horizon.

Artifacts are stored under:

```text
models/<run-id>/horizon-<n>/
├── artifact/
│   ├── model.joblib
│   ├── metadata.json
│   └── feature_schema.json
├── fold_predictions.parquet
├── calibration.parquet
├── leaderboard.parquet
└── evaluation.json
```

Metadata records the target horizon, selected model, exact ordered features,
training cutoff, season/gameweek coverage, source training table, validation
results, and random seed. Player IDs, names, timestamps, paths, and target
columns are never model inputs. This supports newly added players as long as
their current snapshot supplies the saved feature schema.

Models are not retrained automatically. A future scheduled workflow will run
snapshot collection, finalized-history ingestion, training-data construction,
and retraining at appropriate points during the season.

## Historical backfill

The initial historical adapter targets the
[Vaastav Fantasy Premier League archive](https://github.com/vaastav/Fantasy-Premier-League).
The adapter downloads only `merged_gw.csv`, `fixtures.csv`, and `teams.csv` for
each season. Downloads require a complete Git commit SHA and record source URLs,
byte sizes, and SHA-256 checksums.

The real-data smoke test used revision:

```text
f2090d378ebd1b0c3d14884770dde95f38c50a0d
```

Download three archived seasons:

```bash
fpl-download-history \
  --seasons 2022-23 2023-24 2024-25 \
  --revision f2090d378ebd1b0c3d14884770dde95f38c50a0d
```

The command prints the resulting source root. Use it to build training rows:

```bash
fpl-build-historical-training \
  --source-root data/raw/external/vaastav__Fantasy-Premier-League/<revision> \
  --seasons 2022-23 2023-24 2024-25 \
  --horizons 1 3 5
```

This is deliberately described as a historical reconstruction, not an exact
deadline snapshot:

- unshifted `xP` is always excluded because the source documents uncertain
  post-match timing;
- same-gameweek performance is used only as a label;
- cumulative, expected-stat, and recent-form features use earlier gameweeks;
- prices after GW1 use the previous archived gameweek;
- archived final fixture scheduling may include hindsight about postponements;
- historical injury news and availability are not present.

A gameweek missing from both player rows and fixtures is treated as an explicit
league-wide blank. A player missing from an otherwise populated gameweek
remains unknown rather than being assigned zero.

Each backfill writes:

```text
data/processed/historical_backfill/<ingestion-id>/
├── training.parquet
├── player_gameweek_stats.parquet
├── audit.json
└── manifest.json
```

The audit reports seasons, rows, players, gameweek coverage, duplicate keys,
missing gameweeks, league-wide blanks, and coverage for every enriched feature.

Historical and current-season tables can be combined during model training:

```bash
fpl-train-models \
  --training-data \
    data/processed/historical_backfill/<id>/training.parquet \
    data/processed/training/training-<id>.parquet \
  --horizons 1 3 5 \
  --max-validation-folds 12 \
  --minimum-feature-coverage 0.5
```

Overlapping season/gameweek/player rows are rejected rather than silently
deduplicated. Missing columns remain missing. A feature is enabled only when it
meets the configured non-null training coverage, and numeric missingness
indicators let models distinguish an unavailable value from a real zero.

## Generate current player predictions

Apply a completed model run to the latest saved deadline snapshot:

```bash
fpl-predict-players \
  --models-run models/<run-id> \
  --season 2026-27 \
  --gameweek 1
```

This builds recent and fixture features without requiring current-season future
labels, loads each horizon artifact, and writes identified player predictions
under `outputs/predictions`.

The current historical models do not learn injury status because reliable
archived availability snapshots are unavailable. Status and news are retained
in the prediction output for later availability adjustment and warnings.
Preseason recent-form features may also be missing. These outputs are raw
player forecasts, not transfer recommendations or 0–100 squad ratings.

## Initial real-data benchmark

A local benchmark was run against the pinned three-season backfill using 12
evenly distributed purged rolling origins per horizon:

| Horizon | Selected model | Selected MAE | Recent-points MAE |
| --- | --- | ---: | ---: |
| 1 GW | Random forest | 1.028 | 1.064 |
| 3 GWs | Random forest | 2.534 | 2.846 |
| 5 GWs | Random forest | 3.786 | 4.422 |

This establishes that the modelling pipeline can beat its simple baselines on
the sampled origins. It is not yet a production-quality claim: fixture-history
limitations, calibration, availability adjustment, season-forward evaluation,
and uncertainty still require further work.

Example local query:

```bash
python - <<'PY'
import duckdb

with duckdb.connect("data/fpl.duckdb", read_only=True) as connection:
    print(connection.sql("SELECT * FROM snapshot_registry").df())
PY
```

## Test

```bash
pytest
```

Tests cover endpoint construction and error handling, response validation,
club/position mappings, price conversion, preservation of statistical fields,
unknown IDs, and JSON/Parquet snapshot persistence.

## Project layout

```text
src/fpl_predictions/
├── api/             # HTTP access and response validation
├── data/            # snapshots, history, features, catalog, and storage
├── modelling/       # baselines, temporal evaluation, artifacts, prediction
├── *_cli.py         # focused command-line workflows
└── config.py
scripts/             # runnable workflow wrappers
tests/               # isolated tests and compact saved API responses
misc/scripts/        # preserved exploratory notebooks
```

The exploratory `misc/scripts/practice_api.ipynb` is retained as useful early
work. Production code does not depend on notebook state.

## Roadmap

The original project goals remain in scope:

- reconstruct a squad through manual selection, manager import, or an official
  FPL screenshot;
- predict future points over configurable horizons with leakage-safe temporal
  validation;
- generate legal, useful reference squads rather than only uniform random ones;
- produce percentile-based goalkeeper, defence, midfield, forward, bench,
  captaincy, and overall ratings;
- warn clearly about invalid formations, availability uncertainty, and budget
  violations.

Phase 4 will validate and generate legal squads, aggregate player predictions
through the starting XI and captain rules, create reference populations, and
convert projected positional and overall points into 0–100 percentile ratings.
Screenshot OCR and the user interface remain separate later phases.
