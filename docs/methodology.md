# Methodology notes

Short companion to the README. The dissertation is the authoritative source; this file is a navigation aid for the code.

## Known pitfalls fixed in the final pipeline

| # | Symptom | Root cause | Fix in this repo |
|---|---------|------------|------------------|
| 1 | No regional ESIOS series available | ESIOS does not publish regional demand; plant-level generation is private | Pivoted to peninsular totals; regional code removed |
| 2 | Open-Meteo HTTP 429 under single-shot downloads | API rate limit | Monthly batching + timed pauses + VPN IP rotation (`src/weather_data/`) |
| 3 | 78 h/station missing from each download | UTC→local-time boundary at month edges | Each monthly request extended by 1 day on each side and trimmed |
| 4 | 248 390 MW wind peak (impossible) | Sub-hourly intervals summed instead of averaged | Server-side averaging forced in `src/grid_data/` |
| 5 | Out-of-memory on 25 M+ row merge | Full-dataset join | Chunked monthly merges in `src/merges/` |
| 6 | Thousands of correlated weather columns | Site-level inputs with heavy spatial redundancy | K-Means spatial compression + per-cluster aggregation (`src/clustering/`); cross-*k* sensitivity in `src/modelling/` |

## Experimental setup (final study)

- Target variables: peninsular wind generation, peninsular solar generation, peninsular demand.
- Features: per-cluster weather aggregates (actual + day-ahead forecast), calendar features, holidays, GDP, population, installed capacity.
- Model: XGBoost regressor, time-series split (train / val / test, no leakage).
- *k* sweep: {8, 10, 12, 15}.
- Control: identical pipeline run with randomised feature labels (`ml_pipeline_random_control.py`).

## Figures

The `figures/` folder matches the dissertation figure numbers (`fig01`…`fig17`) and tables (`table01`…`table03`).
