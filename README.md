# Spain Renewable Forecasting

Short-term forecasting of **wind generation, solar generation, and electricity demand** on the Spanish peninsular grid using gradient-boosted trees (XGBoost) and K-Means spatial feature compression.

Final-year undergraduate individual project (ECM3175), MEng Mechanical Engineering, University of Exeter.

> The written dissertation is **not** included in this repository. It remains in the author's private folders. This repository contains code, methodology and result artefacts only.

---

## Project aims

Variable renewables are hard to integrate into power systems because their output depends on weather and must be matched to demand at every moment. This project asks whether an off-the-shelf machine-learning method (XGBoost) can give operationally useful hour-ahead forecasts of the three quantities a system operator needs to balance:

1. Wind generation
2. Solar generation
3. Electricity demand (load)

for the Spanish peninsular system (REE control area), using only publicly available data.

## Data

| Source | Via | Content | Coverage |
| --- | --- | --- | --- |
| ESIOS (Red Eléctrica de España) | Public API, authenticated | 15-minute wind generation, solar generation, demand, day-ahead price | 2022–2025, peninsular total |
| Open-Meteo Historical Weather API | Public API | Reanalysis weather (temperature, wind speed/direction at 10 m / 100 m / 180 m, shortwave radiation, cloud cover, etc.) | 2022–2025 |
| Open-Meteo Forecast Archive | Public API | Day-ahead forecast weather (same variables) | 2022–2025 |
| INE | File download | Spanish provincial GDP and population | Annual |
| REE capacity registry | Public download | Installed capacity per wind farm | 2022–2025 |

The weather data covers **2,811 geographic points**: 843 wind farms, 1,829 solar sites, and 139 demand centres (cities ≥ 50 000 inhabitants on the peninsula). Over the four-year window this is roughly **700 million raw weather observations**.

Raw data (parquet, csv) is **not** committed. It can be reproduced from scratch by running the download scripts described below; expect this to take several days end-to-end because of the Open-Meteo rate limit.

## Method

### Pipeline (final, working version)

1. **Grid targets** — `src/grid_data/` pulls 15-minute wind, solar and demand series from ESIOS. The server is forced to return hourly averages (not sums) to avoid a unit-aggregation error that otherwise produced a physically impossible 248 GW wind peak.
2. **Weather data** — `src/weather_data/` pulls actual and forecast weather from Open-Meteo. The final downloader runs in **monthly batches with timed pauses and VPN-based IP rotation** to stay inside the rate limit, and extends each monthly request by one day on either side before trimming, to eliminate a UTC-to-local-time boundary bug that was dropping 78 hours of data per station.
3. **Supporting features** — `src/features/` downloads GDP, population, holidays, and wind-farm installed capacity.
4. **Chunked merges** — `src/merges/` joins weather + grid data **one monthly batch at a time**. An earlier single-shot merge of 25 M+ rows ran out of memory; the chunked architecture keeps peak memory bounded.
5. **K-Means spatial compression** — `src/clustering/` groups the 2 811 locations into *k* regional clusters (final study tests *k* ∈ {8, 10, 12, 15}) and aggregates weather features per cluster. This cuts input dimensionality by two orders of magnitude without discarding spatially distinct weather regimes. An earlier attempt that fed thousands of raw site-level columns straight into XGBoost suffered from signal redundancy and was abandoned.
6. **Modelling** — `src/modelling/ml_pipeline.py` trains three XGBoost regressors (wind, solar, load) with a time-series cross-validation split and feature routing per target.
7. **Random-features control** — `src/modelling/ml_pipeline_random_control.py` reruns the same pipeline with randomised feature labels to confirm that measured skill is not an artefact of the fitting procedure.
8. **Cross-*k* sensitivity** — the pipeline is executed for each *k* and compared on error, R², and compute cost. See `figures/table02_cross_k_comparison.*` and `figures/fig06_cross_k_mae.png` / `fig07_cross_k_r2.png` / `fig08_cross_k_cost.png`.
9. **Reporting** — `src/modelling/results_analysis.py` generates all dissertation figures and tables (saved under `figures/`).

### Scope note

The original design was *regional* (per-city demand, per-plant generation). This was abandoned because ESIOS does not publish regional demand series and plant-level generation data is privately held. The project pivoted to **peninsular-total** targets; the regional-only draft code is **not** in this repository.

## Repository layout

```
spain-renewable-forecasting/
├── README.md                 ← this file
├── requirements.txt          ← Python packages used by the final pipeline
├── .gitignore
├── src/
│   ├── grid_data/            ← ESIOS downloaders (targets + day-ahead price)
│   ├── weather_data/
│   │   ├── load/             ← Open-Meteo for the 139 demand cities
│   │   ├── wind/             ← Open-Meteo for the 843 wind farms (batched + rescue)
│   │   └── solar/            ← Open-Meteo for the 1 829 solar sites
│   ├── coordinates/          ← build the location tables used by the downloaders
│   ├── features/             ← GDP, holidays, population, installed capacity
│   ├── merges/               ← chunked, per-domain merge of weather + grid data
│   ├── clustering/           ← K-Means spatial compression + per-cluster aggregation
│   └── modelling/            ← macro-table assembly, XGBoost training, result analysis
├── tools/                    ← audit/verification scripts used during development
├── figures/                  ← final dissertation figures + metric tables
└── docs/                     ← methodology notes (optional)
```

## How to reproduce

### Prerequisites

- Python 3.10 or newer
- A free ESIOS API token (request one at <https://www.esios.ree.es/es/pagina/api>)
- Several hundred GB of free disk (raw weather parquet is bulky)
- Patience; Open-Meteo throttles unauthenticated clients, so the full download takes days

### Setup

```bash
pip install -r requirements.txt
export ESIOS_API_KEY="your_token_here"      # macOS/Linux
setx ESIOS_API_KEY "your_token_here"        # Windows (new terminal)
```

Every script that touches ESIOS reads the token from the `ESIOS_API_KEY` environment variable; the hardcoded placeholder `YOUR_ESIOS_API_KEY_HERE` is a safety stub, not a working key.

### Pipeline order

1. `src/coordinates/` — generate location tables for the three domains.
2. `src/grid_data/` — download ESIOS targets.
3. `src/weather_data/{load,wind,solar}/` — download Open-Meteo actual + forecast weather in monthly batches (expect to rotate your IP during this).
4. `src/features/` — download GDP, holidays, population, capacity.
5. `src/merges/` — chunked merges per domain.
6. `src/clustering/kmeans_clustering.py` — cluster the 2 811 locations.
7. `src/clustering/spatial_dimensioner_{load,wind,solar}.py` — aggregate weather features per cluster.
8. `src/modelling/macro_table_merger.py` — assemble the macro training table.
9. `src/modelling/ml_pipeline.py` — train XGBoost models for each target. Repeat for each *k*.
10. `src/modelling/ml_pipeline_random_control.py` — random-feature control run.
11. `src/modelling/results_analysis.py` — regenerate figures and tables.

## Results (summary)

Final figures and metric tables live in `figures/`. In short, XGBoost with K-Means-compressed weather features gives usable hour-ahead forecasts for all three targets; the cross-*k* study picks the operating point that balances error against compute cost. See the dissertation for the full write-up (not distributed with this code).

## Licence and credit

Code: author's own work. If you reuse it, a citation is appreciated.

Data belongs to the respective providers (ESIOS/REE, Open-Meteo, INE) and is subject to their own terms of use.
