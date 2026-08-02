# FPL Predictions

An end-to-end project for predicting Fantasy Premier League player and squad
performance, then expressing position-specific and overall squad strength as
percentile ratings.

The current implementation provides the reproducible data, modelling, and
squad-rating foundation:

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
- API-derived squad rules and strict squad legality validation;
- availability-aware XI, position, bench, and captain projections;
- deterministic legal reference squads and 0–100 percentile ratings;
- privacy-minimized, reproducible samples of public post-deadline manager squads;
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
counts represent blanks and doubles without hard-coded gameweek IDs. Completed
scores strictly before each snapshot also produce shrinkage-regularized club
attack and goals-conceded strengths. Upcoming opponents are summarized for
each horizon, including venue-adjusted attacking and defensive fixture factors.
Early-season and unseen clubs shrink to the neutral league-average prior.

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

When team-strength features are present, a random-forest ablation without
those features is also evaluated. This prevents the new fixture features from
being forced into a horizon where they do not improve chronological MAE.

The same command can train expected-minutes artifacts using `--target minutes`.
That comparison includes a hybrid which projects recent minutes when available
and uses the learned random forest as a cold-start fallback:

```bash
fpl-train-models \
  --training-data data/processed/historical_backfill/<id>/training.parquet \
  --horizons 1 3 5 \
  --min-train-gameweeks 8 \
  --target minutes
```

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
  --minutes-models-run models/<minutes-run-id> \
  --season 2026-27 \
  --gameweek 1
```

This builds recent and fixture features without requiring current-season future
labels, loads each horizon artifact, and writes identified point and optional
expected-minutes predictions under `outputs/predictions`. Expected minutes are
reported separately; they do not arbitrarily rescale the independently trained
point forecast.

The current historical models do not learn injury status because reliable
archived availability snapshots are unavailable. Status and news are retained
in the prediction output for later availability adjustment and warnings.
Preseason recent-form features may also be missing. These outputs are raw
player forecasts, not transfer recommendations or 0–100 squad ratings.

## Rate a squad

Create a JSON file containing all 15 player IDs, the starting XI and bench,
plus captain and vice-captain:

```json
{
  "player_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "starting_xi": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
  "bench": [2, 6, 7, 12],
  "captain": 15,
  "vice_captain": 14,
  "source": "manual"
}
```

Then rate it using a saved player-prediction file:

```bash
fpl-rate-squad \
  --squad my-squad.json \
  --predictions outputs/predictions/players-<id>.parquet \
  --horizon 3 \
  --strategy human_like \
  --reference-size 1000
```

This repository's saved 2026–27 GW1 data can be exercised immediately with:

```bash
python scripts/rate_squad.py \
  --squad examples/squad-2026-27-gw1.json \
  --predictions outputs/predictions/players-20260730T162418441941Z.parquet \
  --horizon 3 \
  --strategy human_like \
  --reference-size 100
```

The smaller reference size makes this a quick smoke test. Increase it to 1,000
for a more stable comparison distribution.

The command derives squad size, budget, club limit, positional quotas, and legal
formations from the matching saved FPL API response. It rejects duplicates,
stale IDs, invalid formations, invalid captaincy, club-limit violations, and
over-budget manual or generated squads.

The default `human_like` reference population corrects the upward bias caused
by comparing intentional £100m squads with broadly random, under-budget teams.
It samples complete squads with probability proportional to the product of
their players' current ownership percentages, conditioned on legality and a
£99–100m spend. A 0.1 percentage-point floor, matching the API ownership
resolution, keeps zero-rounded players possible. The saved point model then
selects each reference squad's best legal XI and captain and scores it exactly
like the submitted team.

This is an explicit independence approximation to observed human selection,
not a claim that the generated squads are actual managers. It uses current
ownership only to model which players humans select; all projected strength and
ratings remain derived from the saved point model. `--reference-budget-band`
changes the default £1m comparison band. Legacy `price_aware`,
`ownership_weighted`, and `broad_legal` strategies remain available for
diagnostics, but are not the default rating benchmark. The method, target cost,
band, seed, horizon, lineup policy, reference quantiles, and generated table are
saved with every report.

Reports preserve empirical midrank percentiles, so tied projections receive the
middle of their shared rank. They also present a school-style 0–100 score using
`75 + 8 × inverse-normal-CDF(smoothed percentile)`, capped at 99.9. This is a
monotonic display calibration: it cannot change squad ordering, projected
points, or the evidence behind the percentile.

| School score | Approximate comparative meaning |
| ---: | --- |
| 50 | Extremely poor; approximately the bottom 0.1% |
| 70 | Okay, but below the median reference squad |
| 75 | Median reference squad |
| 80 | Good; approximately the top 27% |
| 90 | Excellent; approximately the top 3% |
| 95 | World-class; approximately the top 0.6% |
| 99.9 | Display ceiling; a perfect 100 is not awarded |

Goalkeeper, defence, midfield, attack, bench, and captaincy are each scored
against their corresponding distribution. Overall is scored directly from
starting-XI points plus one captain bonus; it is not an average of component
scores. Bench points remain separate.
Active chips are rejected rather than silently applying normal scoring.
Automatic substitutions and vice-captain takeover are not yet simulated.

When the prediction file includes expected-minutes artifacts, every rating
report includes a 15-player `player_projections` section with raw points,
availability-adjusted points, expected minutes, lineup role, and captaincy.

When the current API supplies a chance-of-playing percentage, only the first
gameweek portion of a forecast is adjusted. Later gameweeks are preserved
because return timing is unknown. A non-available player with no numeric chance
keeps the raw forecast and produces a warning rather than an invented estimate.

## Collect real manager reference squads

After a gameweek deadline, collect a seeded sample from the public Overall
standings:

```bash
python scripts/snapshot_managers.py \
  --season 2026-27 \
  --gameweek 1 \
  --sample-size 250 \
  --seed 2026
```

The command discovers the season-specific Overall league from public manager
metadata; it does not hard-code a league ID. It samples standings pages and
entries reproducibly, downloads public picks, verifies 15 unique current player
IDs, positions 1–15, and exactly one captain and vice-captain, then retains
lineup multipliers and active-chip information unchanged.

Collection is deliberately rejected before the API deadline. For 2026–27 GW1,
the saved official deadline is `2026-08-21T17:30:00Z`; public picks are not
available before then. Known public entries can be collected explicitly by
repeating `--manager-id`:

```bash
python scripts/snapshot_managers.py \
  --season 2026-27 \
  --gameweek 1 \
  --manager-id 12345 \
  --manager-id 67890
```

Each immutable run writes:

```text
data/
├── raw/manager_samples/<season>/gw-<n>/<ingestion-id>/
│   ├── bootstrap-static.json
│   ├── sampling-frame.json
│   └── picks/manager-<id>.json
└── processed/manager_samples/<season>/gw-<n>/<ingestion-id>/
    ├── managers.parquet
    ├── manager_picks.parquet
    ├── sample_player_rates.parquet
    ├── audit.json
    └── manifest.json
```

Manager and team names are intentionally not stored. The processed data keeps
only public entry IDs, sampling ranks, gameweek history, chips, and picks. The
DuckDB catalog exposes deduplicated `manager_gameweek_samples` and
`manager_gameweek_picks` views, plus `manager_sample_player_rates`.

The audit records formation and current-cost distributions plus the correlation
and mean absolute gap between sampled selection rates and the API's population
ownership. `sample_player_rates.parquet` retains selection, starter, captain,
and vice-captain rates by player ID.

These samples are not automatically used as the rating population yet. They
must first accumulate and pass coverage/bias audits. Until then, reports remain
explicitly labeled as synthetic human-like comparisons.

## Initial real-data benchmark

A local benchmark was run against the pinned three-season backfill using 12
evenly distributed purged rolling origins per horizon:

| Horizon | Selected model | Selected MAE | Recent-points MAE |
| --- | --- | ---: | ---: |
| 1 GW | Random forest, strength ablated | 1.028 | 1.064 |
| 3 GWs | Random forest with opponent strength | 2.530 | 2.846 |
| 5 GWs | Random forest with opponent strength | 3.748 | 4.422 |

The separately selected expected-minutes hybrids achieved MAE of 13.618,
41.092, and 66.281 minutes over the 1-, 3-, and 5-gameweek horizons. These are
playing-time expectations, not appearance probabilities; scenario-based
automatic substitutions still require a calibrated appearance classifier.

This establishes that the modelling pipeline can beat its simple baselines on
the sampled origins. Opponent strength provides modest gains rather than a
claimed breakthrough, and the benchmark remains an initial local validation
rather than a production-quality guarantee.

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

Tests cover endpoint handling, normalization, snapshot persistence, leakage-safe
features, temporal model validation, historical backfill, squad legality,
availability adjustment, projection aggregation, reference reproducibility, and
percentile ratings.

## Project layout

```text
src/fpl_predictions/
├── api/             # HTTP access and response validation
├── data/            # snapshots, history, features, catalog, and storage
├── modelling/       # baselines, temporal evaluation, artifacts, prediction
├── squads/          # legal squads, projections, references, and ratings
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

Phase 4 now validates and generates legal squads, aggregates player predictions
through the starting XI and captain rules, creates reproducible reference
populations, and converts projected positional and overall points into 0–100
percentile ratings.

The next larger increments are manager-team import, screenshot OCR, a user
interface, and a constrained recommendation optimizer for transfers, starting
XIs, benches, and captaincy. Recommendations will reuse the same current
snapshot, predictions, availability logic, and legal squad constraints, so they
can update throughout the season as new snapshots and models are produced.
