"""
================================================================================
PHASE 6: MACHINE LEARNING FORECASTING PIPELINE
Spanish Peninsular Power System — Day-Ahead XGBoost Forecasting
================================================================================
Architecture:
  - ONE unified macro-level master dataset per k-value
  - THREE separate XGBoost regressors (Wind / Solar / Load)
  - ONE unified pipeline function, callable per k-value
  - Net Load derived post-hoc: Load_pred - (Wind_pred + Solar_pred)
  - Cross-k comparison: pipeline runs for k=8, 10, 12, 15 and aggregates results

Cross-Model Feature Linkage (Renewable Models Only):
  Wind model receives:
    - All WEM_ prefixed features (own namespace)
    - Lagged national demand   (LDM_demand_lag_24/168, rollmean_24/168)
    - Lagged solar generation  (SEM_solar_gen_lag_24/168, rollmean_24/168)
    - 9 shared global calendar features

  Solar model receives:
    - All SEM_ prefixed features (own namespace)
    - Lagged national demand   (LDM_demand_lag_24/168, rollmean_24/168)
    - Lagged wind generation   (WEM_wind_gen_lag_24/168, rollmean_24/168)
    - 9 shared global calendar features

  Physical rationale:
    Metered renewable output is bounded by system absorption capacity and
    subject to REE curtailment orders during low-demand/high-generation
    periods. Lagged demand informs models of the historical grid absorption
    regime. Lagged cross-renewable generation informs models of the shared
    curtailment environment during periods of combined high renewable output.
    Only lagged values are used — no target-hour forecasts — to avoid leakage.

  Load model receives:
    - All LDM_ prefixed features (own namespace, already contains demand lags)
    - 9 shared global calendar features
    - No cross-model additions (demand lags already primary features here)

Temporal Split (chronological, no shuffle, no leakage):
  - Train      : 2022-01-01 to 2024-12-31
  - Validation : 2025-01-01 to 2025-09-30  (early stopping + tuning)
  - Test       : 2025-10-01 to 2025-12-31  (final evaluation only)

Author : [Your Name]
Date   : 2025
================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
#     Edit ONLY this block. Nothing else in the script needs touching.
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = (
    r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3"
    r"\Individual Project\ML Models\Macro Table for Training"
)

K_DATASET_PATHS = {
    8:  os.path.join(BASE_DIR, "k=8",  "macro_master_dataset_k8.parquet"),
    10: os.path.join(BASE_DIR, "k=10", "macro_master_dataset_k10.parquet"),
    12: os.path.join(BASE_DIR, "k=12", "macro_master_dataset_k12.parquet"),
    15: os.path.join(BASE_DIR, "k=15", "macro_master_dataset_k15.parquet"),
}

COMPARISON_OUTPUT_DIR = os.path.join(BASE_DIR, "cross_k_comparison")

# ── Target column names ───────────────────────────────────────────────────────
TARGETS = {
    "wind":  "WEM_target_national_wind_gen",
    "solar": "SEM_target_national_solar_gen",
    "load":  "LDM_target_national_demand",
}

# ── Column prefixes for dynamic domain-feature routing ───────────────────────
PREFIXES = {
    "wind":  "WEM_",
    "solar": "SEM_",
    "load":  "LDM_",
}

# ── Global calendar features (no prefix) — shared by ALL three models ─────────
GLOBAL_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend", "is_holiday", "is_non_working_day",
]

# ── Cross-model demand lags → added to BOTH wind and solar models ─────────────
# Physical rationale: metered renewable output is bounded by system demand via
# curtailment. Lagged demand captures the historical grid absorption regime
# without leaking any target-hour information.
DEMAND_LAG_FEATURES = [
    "LDM_demand_lag_24",
    "LDM_demand_lag_168",
    "LDM_demand_rollmean_24",
    "LDM_demand_rollmean_168",
]

# ── Cross-model solar lags → added to the WIND model only ─────────────────────
# Physical rationale: during periods of combined high renewable generation,
# curtailment is applied across technologies simultaneously. Lagged solar
# output informs the wind model of the shared curtailment environment.
# Note: SEM_solar_gen_lag/rollmean live in SEM_ namespace so are NOT already
# captured by the wind model's WEM_ prefix routing — explicit addition needed.
SOLAR_LAG_FEATURES = [
    "SEM_solar_gen_lag_24",
    "SEM_solar_gen_lag_168",
    "SEM_solar_gen_rollmean_24",
    "SEM_solar_gen_rollmean_168",
]

# ── Cross-model wind lags → added to the SOLAR model only ─────────────────────
# Physical rationale: symmetric to the above. Lagged wind output informs the
# solar model of the shared curtailment environment.
# Note: WEM_wind_gen_lag/rollmean live in WEM_ namespace so are NOT already
# captured by the solar model's SEM_ prefix routing — explicit addition needed.
WIND_LAG_FEATURES = [
    "WEM_wind_gen_lag_24",
    "WEM_wind_gen_lag_168",
    "WEM_wind_gen_rollmean_24",
    "WEM_wind_gen_rollmean_168",
]

# ── Chronological split boundaries ───────────────────────────────────────────
SPLIT = {
    "val_start":  "2025-01-01",
    "test_start": "2025-10-01",
}

early_stopping=75

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "gamma":            0.0,
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "early_stopping_rounds": "early_stopping",   # ← add this line
}

# ── Physical reference constants for normalised MAE metrics ──────────────────
INSTALLED_WIND_MW    = 32_000   # installed wind capacity 2025 (MW)
INSTALLED_SOLAR_MW   = 47_000   # installed solar capacity 2025 (MW)
PEAK_DEMAND_MW       = 38_043   # peak demand confirmed from test data (MW)
MEAN_LOAD_MW         = 27_334   # mean load confirmed from test data (MW)
SOLAR_MAPE_THRESH_MW = 50       # solar obs below this excluded from MAPE

CLIP_RENEWABLES      = True


# ──────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_columns(df: pd.DataFrame, required: list, context: str) -> None:
    """
    Fail fast with a descriptive error if any required column is absent.
    Catches naming mismatches immediately rather than silently dropping features.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"\n[COLUMN ERROR - {context}]\n"
            f"Missing columns:\n  {missing}\n"
            f"Check parquet file and prefix conventions."
        )


def route_features(df: pd.DataFrame, prefix: str, target: str,
                   extra_cols: list = None) -> list:
    """
    Build the complete feature list for one model using three sources:

      1. Domain features  — all columns matching `prefix`, excluding `target`
      2. Global features  — 9 shared calendar/time columns (no prefix)
      3. Cross-model extras — columns from other models' namespaces,
                              explicitly passed via `extra_cols`

    Deduplication is applied at each stage to prevent any column appearing
    twice in the feature list, regardless of namespace overlap.

    Parameters
    ----------
    df         : master dataframe (used to resolve prefix matches)
    prefix     : column prefix for this model's domain (e.g. 'WEM_')
    target     : target column to exclude from inputs
    extra_cols : optional list of cross-model columns to append
    """
    domain_cols  = [c for c in df.columns if c.startswith(prefix) and c != target]
    seen         = set(domain_cols)

    global_extra = [c for c in GLOBAL_FEATURES if c not in seen]
    seen.update(global_extra)

    cross_extra  = []
    for c in (extra_cols or []):
        if c not in seen:
            cross_extra.append(c)
            seen.add(c)

    return domain_cols + global_extra + cross_extra


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str, target_type: str = "") -> dict:
    """
    Full dissertation-grade regression metric suite.

    Standard metrics
    ────────────────
    MAE    Mean Absolute Error (MW) — primary reported metric
    RMSE   Root Mean Squared Error (MW) — penalises large peak errors
    MAPE   Mean Absolute Percentage Error (%)
    R²     Coefficient of Determination

    Solar MAPE handling
    ───────────────────
    Observations where actual solar < SOLAR_MAPE_THRESH_MW (50 MW) are
    excluded to avoid night-hour division instability.

    Normalised MAE metrics (one per target, NaN for all others)
    ────────────────────────────────────────────────────────────
    nMAE_mean-load_%       load only      MAE / MEAN_LOAD_MW × 100
    cNMAE_wind_%           wind only      MAE / INSTALLED_WIND_MW × 100
    cNMAE_solar_%          solar only     MAE / INSTALLED_SOLAR_MW × 100
    pMAE_peak-demand_%     load only      MAE / PEAK_DEMAND_MW × 100
    pMAE_net-load_peak_%   net_load only  MAE / PEAK_DEMAND_MW × 100

    Parameters
    ----------
    y_true      : array of actual values (MW)
    y_pred      : array of predicted values (MW)
    label       : string label for this row (e.g. 'test_wind')
    target_type : one of 'wind', 'solar', 'load', 'net_load'
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # ── MAPE ──────────────────────────────────────────────────────────────────
    if target_type == "solar":
        mask = y_true > SOLAR_MAPE_THRESH_MW
        mape = (np.mean(np.abs((y_true[mask] - y_pred[mask])
                               / y_true[mask])) * 100
                if mask.sum() > 0 else np.nan)
    else:
        mask = np.abs(y_true) > 1e-6
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask])
                               / y_true[mask])) * 100

    # ── Normalised MAE metrics ────────────────────────────────────────────────
    nMAE_mean_load   = round(mae / MEAN_LOAD_MW    * 100, 3) if target_type == "load"     else np.nan
    cNMAE_wind       = round(mae / INSTALLED_WIND_MW  * 100, 3) if target_type == "wind"     else np.nan
    cNMAE_solar      = round(mae / INSTALLED_SOLAR_MW * 100, 3) if target_type == "solar"    else np.nan
    pMAE_peak_demand = round(mae / PEAK_DEMAND_MW  * 100, 3) if target_type == "load"     else np.nan
    pMAE_net_load    = round(mae / PEAK_DEMAND_MW  * 100, 3) if target_type == "net_load" else np.nan

    return {
        "model":                 label,
        "MAE":                   round(mae,  2),
        "RMSE":                  round(rmse, 2),
        "MAPE_%":                round(mape, 3) if not np.isnan(mape) else np.nan,
        "R2":                    round(r2,   5),
        "nMAE_mean-load_%":      nMAE_mean_load,
        "cNMAE_wind_%":          cNMAE_wind,
        "cNMAE_solar_%":         cNMAE_solar,
        "pMAE_peak-demand_%":    pMAE_peak_demand,
        "pMAE_net-load_peak_%":  pMAE_net_load,
    }

def print_metrics_block(m: dict) -> None:
    """Print a single metrics dict to the console in a readable format."""
    print(f"    {'MAE':<22} {m['MAE']:>10.2f}  MW")
    print(f"    {'RMSE':<22} {m['RMSE']:>10.2f}  MW")
    print(f"    {'MAPE':<22} {m['MAPE_%']:>10.3f}  %")
    print(f"    {'R2':<22} {m['R2']:>12.5f}")
    # Print whichever normalised metric is populated for this target
    if not np.isnan(m.get("nMAE_mean-load_%",    np.nan)):
        print(f"    {'nMAE_mean-load':<22} {m['nMAE_mean-load_%']:>10.3f}  %")
    if not np.isnan(m.get("cNMAE_wind_%",         np.nan)):
        print(f"    {'cNMAE_wind':<22} {m['cNMAE_wind_%']:>10.3f}  %")
    if not np.isnan(m.get("cNMAE_solar_%",        np.nan)):
        print(f"    {'cNMAE_solar':<22} {m['cNMAE_solar_%']:>10.3f}  %")
    if not np.isnan(m.get("pMAE_peak-demand_%",   np.nan)):
        print(f"    {'pMAE_peak-demand':<22} {m['pMAE_peak-demand_%']:>10.3f}  %")
    if not np.isnan(m.get("pMAE_net-load_peak_%", np.nan)):
        print(f"    {'pMAE_net-load_peak':<22} {m['pMAE_net-load_peak_%']:>10.3f}  %")

def fit_model(model, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """
    Fit one XGBoost regressor with validation-set early stopping.
    Reports the best iteration and validation RMSE achieved.
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    print(f"    Best iteration : {model.best_iteration}")
    print(f"    Best val RMSE  : {model.best_score:.4f}")
    return model


def get_feature_importance(model: xgb.XGBRegressor,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Gain-based feature importances — average loss improvement per split.
    Most informative importance type for dissertation reporting and for
    identifying which cross-model features were actually used by the model.
    """
    scores = model.get_booster().get_score(importance_type="gain")
    df_imp = (
        pd.DataFrame(scores.items(), columns=["feature", "gain"])
          .sort_values("gain", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )
    df_imp.insert(0, "rank", df_imp.index + 1)
    return df_imp


def print_feature_routing_summary(label: str, features: list) -> None:
    """
    Print a breakdown of what went into each model's feature set,
    split by namespace. Useful for verifying cross-model linkage is correct.
    """
    wem = [f for f in features if f.startswith("WEM_")]
    sem = [f for f in features if f.startswith("SEM_")]
    ldm = [f for f in features if f.startswith("LDM_")]
    glb = [f for f in features if f in GLOBAL_FEATURES]
    print(f"      {label}")
    if wem: print(f"        WEM domain    : {len(wem)} cols")
    if sem: print(f"        SEM domain    : {len(sem)} cols")
    if ldm: print(f"        LDM domain    : {len(ldm)} cols")
    if glb: print(f"        Global cal.   : {len(glb)} cols")
    print(f"        TOTAL         : {len(features)} cols")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SINGLE-K PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_k(k: int, data_path: str) -> dict:
    """
    Execute the full ML forecasting pipeline for one k-value dataset.

    Parameters
    ----------
    k         : int — spatial cluster count (8, 10, 12, or 15)
    data_path : str — absolute path to the k-specific parquet file

    Returns
    -------
    summary : dict — test-set metrics and metadata for the cross-k table

    Output files written to BASE_DIR/k={k}/pipeline_outputs/
        metrics_summary.csv        val + test metrics, all four targets
        test_predictions.csv       hourly predictions with actuals
        feature_importances.csv    top-20 per model (gain-based)
        feature_routing.csv        full feature list per model (audit trail)
        model_wind.pkl             saved trained wind model
        model_solar.pkl            saved trained solar model
        model_load.pkl             saved trained load model
    """
    out_dir = os.path.join(BASE_DIR, f"k={k}", "pipeline_outputs")
    make_dir(out_dir)
    pipeline_start = time.time()

    print()
    print("=" * 72)
    print(f"  PIPELINE RUN  |  k = {k} spatial clusters")
    print(f"  Data : {data_path}")
    print("=" * 72)

    # ── STAGE 1: Load ─────────────────────────────────────────────────────────
    print(f"\n[1/9] Loading dataset (k={k})...")
    df = pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"      Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"      Span  : {df['time'].min().date()} to {df['time'].max().date()}")

    nan_count = df.drop(columns=["time"]).isna().sum().sum()
    print(f"      NaNs  : {nan_count} (expected 0)")
    if nan_count > 0:
        raise ValueError(
            f"Dataset k={k} contains {nan_count} NaN values. "
            f"Resolve in data engineering before running ML pipeline."
        )

    # ── STAGE 2: Chronological Split ─────────────────────────────────────────
    print(f"\n[2/9] Applying chronological split...")
    val_start  = pd.to_datetime(SPLIT["val_start"])
    test_start = pd.to_datetime(SPLIT["test_start"])

    train_df = df[df["time"] <  val_start].copy()
    val_df   = df[(df["time"] >= val_start) & (df["time"] < test_start)].copy()
    test_df  = df[df["time"] >= test_start].copy()

    total = len(df)
    print(f"      Train (2022-01 to 2024-12) : {len(train_df):>6,} rows "
          f"({100*len(train_df)/total:.1f}%)")
    print(f"      Val   (2025-01 to 2025-09) : {len(val_df):>6,} rows "
          f"({100*len(val_df)/total:.1f}%)")
    print(f"      Test  (2025-10 to 2025-12) : {len(test_df):>6,} rows "
          f"({100*len(test_df)/total:.1f}%)")

    assert train_df["time"].max() < val_df["time"].min(),  "LEAKAGE: Train/Val overlap"
    assert val_df["time"].max()   < test_df["time"].min(), "LEAKAGE: Val/Test overlap"

    # ── STAGE 3: Feature Routing ──────────────────────────────────────────────
    # Wind  : own WEM_ domain + demand lags + solar gen lags + global
    # Solar : own SEM_ domain + demand lags + wind gen lags  + global
    # Load  : own LDM_ domain + global (demand lags already in LDM_ namespace)
    print(f"\n[3/9] Routing features (k={k} cluster columns auto-included)...")

    wind_feats = route_features(
        df, PREFIXES["wind"], TARGETS["wind"],
        extra_cols=DEMAND_LAG_FEATURES + SOLAR_LAG_FEATURES
    )
    solar_feats = route_features(
        df, PREFIXES["solar"], TARGETS["solar"],
        extra_cols=DEMAND_LAG_FEATURES + WIND_LAG_FEATURES
    )
    load_feats = route_features(
        df, PREFIXES["load"], TARGETS["load"]
    )

    print_feature_routing_summary("Wind model (WEM)  :", wind_feats)
    print_feature_routing_summary("Solar model (SEM) :", solar_feats)
    print_feature_routing_summary("Load model (LDM)  :", load_feats)

    # ── STAGE 4: Column Validation ────────────────────────────────────────────
    print(f"\n[4/9] Validating all routed columns exist in dataset...")
    validate_columns(df, wind_feats  + [TARGETS["wind"]],  "Wind Model")
    validate_columns(df, solar_feats + [TARGETS["solar"]], "Solar Model")
    validate_columns(df, load_feats  + [TARGETS["load"]],  "Load Model")
    validate_columns(df, DEMAND_LAG_FEATURES, "Cross-model demand lags")
    validate_columns(df, SOLAR_LAG_FEATURES,  "Cross-model solar lags")
    validate_columns(df, WIND_LAG_FEATURES,   "Cross-model wind lags")
    print(f"      All columns validated successfully.")

    # Save a full feature routing audit trail for the dissertation appendix
    routing_records = (
        [{"model": "WEM", "feature": f} for f in wind_feats]  +
        [{"model": "SEM", "feature": f} for f in solar_feats] +
        [{"model": "LDM", "feature": f} for f in load_feats]
    )
    pd.DataFrame(routing_records).to_csv(
        os.path.join(out_dir, "feature_routing.csv"), index=False
    )

    # ── STAGE 5: Model Initialisation ────────────────────────────────────────
    print(f"\n[5/9] Initialising three XGBoost regressors...")
    model_wind  = xgb.XGBRegressor(**XGB_PARAMS)
    model_solar = xgb.XGBRegressor(**XGB_PARAMS)
    model_load  = xgb.XGBRegressor(**XGB_PARAMS)

    # ── STAGE 6: Training ─────────────────────────────────────────────────────
    print(f"\n[6/9] Training (early stopping patience = 75 rounds)...")

    print(f"\n  -- Wind Energy Model (WEM) --")
    print(f"     Inputs: WEM domain + demand lags + solar gen lags + calendar")
    t0 = time.time()
    model_wind = fit_model(
        model_wind,
        train_df[wind_feats],  train_df[TARGETS["wind"]],
        val_df[wind_feats],    val_df[TARGETS["wind"]],
    )
    print(f"    Training time  : {time.time()-t0:.1f}s")

    print(f"\n  -- Solar Energy Model (SEM) --")
    print(f"     Inputs: SEM domain + demand lags + wind gen lags + calendar")
    t0 = time.time()
    model_solar = fit_model(
        model_solar,
        train_df[solar_feats], train_df[TARGETS["solar"]],
        val_df[solar_feats],   val_df[TARGETS["solar"]],
    )
    print(f"    Training time  : {time.time()-t0:.1f}s")

    print(f"\n  -- Load Demand Model (LDM) --")
    print(f"     Inputs: LDM domain + calendar")
    t0 = time.time()
    model_load = fit_model(
        model_load,
        train_df[load_feats],  train_df[TARGETS["load"]],
        val_df[load_feats],    val_df[TARGETS["load"]],
    )
    print(f"    Training time  : {time.time()-t0:.1f}s")

    # ── STAGE 7: Validation-Set Evaluation ───────────────────────────────────
    print(f"\n[7/9] Validation-set evaluation (2025-01 to 2025-09)...")

    vw = model_wind.predict(val_df[wind_feats])
    vs = model_solar.predict(val_df[solar_feats])
    vl = model_load.predict(val_df[load_feats])
    # Save validation predictions for the results analysis script
    

    if CLIP_RENEWABLES:
        vw = np.clip(vw, 0, None)
        vs = np.clip(vs, 0, None)

    val_net_actual = (val_df[TARGETS["load"]].values
                      - (val_df[TARGETS["wind"]].values
                         + val_df[TARGETS["solar"]].values))
    val_net_pred   = vl - (vw + vs)

    # Save validation predictions for the results analysis script
    val_out = val_df.copy()
    val_out["pred_wind"]       = vw
    val_out["pred_solar"]      = vs
    val_out["pred_load"]       = vl
    val_out["pred_net_load"]   = val_net_pred
    val_out["actual_net_load"] = val_net_actual
    val_out.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

    print("\n  [VAL] Wind Generation:")
    vm_wind  = compute_metrics(val_df[TARGETS["wind"]].values,  vw, "val_wind",     target_type="wind")
    print_metrics_block(vm_wind)
    print("\n  [VAL] Solar Generation:")
    vm_solar = compute_metrics(val_df[TARGETS["solar"]].values, vs, "val_solar",    target_type="solar")
    print_metrics_block(vm_solar)
    print("\n  [VAL] Load Demand:")
    vm_load  = compute_metrics(val_df[TARGETS["load"]].values,  vl, "val_load",     target_type="load")
    print_metrics_block(vm_load)
    print("\n  [VAL] Net Load (derived: Load - Wind - Solar):")
    vm_net   = compute_metrics(val_net_actual, val_net_pred,        "val_net_load", target_type="net_load")
    print_metrics_block(vm_net)

    # ── STAGE 8: Test-Set Evaluation ─────────────────────────────────────────
    print(f"\n[8/9] FINAL TEST-SET EVALUATION (2025-10 to 2025-12)")
    print("      This partition was NOT used during training or model selection.\n")

    tw = model_wind.predict(test_df[wind_feats])
    ts = model_solar.predict(test_df[solar_feats])
    tl = model_load.predict(test_df[load_feats])

    if CLIP_RENEWABLES:
        tw = np.clip(tw, 0, None)
        ts = np.clip(ts, 0, None)

    test_net_actual = (test_df[TARGETS["load"]].values
                       - (test_df[TARGETS["wind"]].values
                          + test_df[TARGETS["solar"]].values))
    test_net_pred   = tl - (tw + ts)

    print("  [TEST] Wind Generation:")
    tm_wind  = compute_metrics(test_df[TARGETS["wind"]].values,  tw, "test_wind",     target_type="wind")
    print_metrics_block(tm_wind)
    print("\n  [TEST] Solar Generation:")
    tm_solar = compute_metrics(test_df[TARGETS["solar"]].values, ts, "test_solar",    target_type="solar")
    print_metrics_block(tm_solar)
    print("\n  [TEST] Load Demand:")
    tm_load  = compute_metrics(test_df[TARGETS["load"]].values,  tl, "test_load",     target_type="load")
    print_metrics_block(tm_load)
    print("\n  [TEST] Net Load (Load - Wind - Solar, derived post-hoc):")
    tm_net   = compute_metrics(test_net_actual, test_net_pred,        "test_net_load", target_type="net_load")
    print_metrics_block(tm_net)

    # ── STAGE 9: Export ───────────────────────────────────────────────────────
    print(f"\n[9/9] Saving outputs to: {out_dir}")

    # 9a — Full metrics table (val + test, all four targets)
    all_metrics = [vm_wind, vm_solar, vm_load, vm_net,
                   tm_wind, tm_solar, tm_load, tm_net]
    metrics_df  = pd.DataFrame(all_metrics)
    metrics_df.insert(0, "k", k)
    metrics_df.insert(2, "partition", ["val"] * 4 + ["test"] * 4)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # 9b — Hourly test predictions with actuals (for dissertation plots)
    test_out = test_df.copy()
    test_out["pred_wind"]       = tw
    test_out["pred_solar"]      = ts
    test_out["pred_load"]       = tl
    test_out["pred_net_load"]   = test_net_pred
    test_out["actual_net_load"] = test_net_actual
    test_out.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # 9c — Feature importances, top-20 per model (gain-based)
    # Cross-model features will appear here if the model found them useful —
    # this is key evidence for the dissertation's feature design argument.
    imp_wind  = get_feature_importance(model_wind);  imp_wind["model"]  = "WEM"
    imp_solar = get_feature_importance(model_solar); imp_solar["model"] = "SEM"
    imp_load  = get_feature_importance(model_load);  imp_load["model"]  = "LDM"
    imp_all   = pd.concat([imp_wind, imp_solar, imp_load], ignore_index=True)
    imp_all.insert(0, "k", k)
    imp_all.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    # 9d — Persist trained models
    joblib.dump(model_wind,  os.path.join(out_dir, "model_wind.pkl"))
    joblib.dump(model_solar, os.path.join(out_dir, "model_solar.pkl"))
    joblib.dump(model_load,  os.path.join(out_dir, "model_load.pkl"))

    total_time = time.time() - pipeline_start
    print(f"      metrics_summary.csv      saved")
    print(f"      test_predictions.csv     saved")
    print(f"      feature_importances.csv  saved")
    print(f"      feature_routing.csv      saved")
    print(f"      model_wind/solar/load    saved")
    print(f"\n  Pipeline for k={k} completed in {total_time:.1f}s")

    # Print a final test results banner for this k
    print(f"\n  {'Model':<16} {'MAE':>10} {'RMSE':>10} {'MAPE%':>9} {'R2':>10}")
    print(f"  {'-'*58}")
    for row in [tm_wind, tm_solar, tm_load, tm_net]:
        print(
            f"  {row['model']:<16} "
            f"{row['MAE']:>10.2f} "
            f"{row['RMSE']:>10.2f} "
            f"{row['MAPE_%']:>9.3f} "
            f"{row['R2']:>10.5f}"
        )

    return {
        "k":                  k,
        "n_wind_features":    len(wind_feats),
        "n_solar_features":   len(solar_feats),
        "n_load_features":    len(load_feats),
        "runtime_s":          round(total_time, 1),
        "wind_MAE":           tm_wind["MAE"],
        "wind_RMSE":          tm_wind["RMSE"],
        "wind_MAPE_%":        tm_wind["MAPE_%"],
        "wind_R2":            tm_wind["R2"],
        "solar_MAE":          tm_solar["MAE"],
        "solar_RMSE":         tm_solar["RMSE"],
        "solar_MAPE_%":       tm_solar["MAPE_%"],
        "solar_R2":           tm_solar["R2"],
        "load_MAE":           tm_load["MAE"],
        "load_RMSE":          tm_load["RMSE"],
        "load_MAPE_%":        tm_load["MAPE_%"],
        "load_R2":            tm_load["R2"],
        "net_load_MAE":       tm_net["MAE"],
        "net_load_RMSE":      tm_net["RMSE"],
        "net_load_MAPE_%":    tm_net["MAPE_%"],
        "net_load_R2":        tm_net["R2"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-K COMPARISON RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_k_comparison(k_paths: dict) -> pd.DataFrame:
    """
    Execute the full pipeline for every k-value and produce a cross-k
    comparison table. This is the primary dissertation evidence for the
    optimal spatial resolution argument.

    Parameters
    ----------
    k_paths : dict — {k_value (int): parquet_path (str)}

    Returns
    -------
    comparison_df : pd.DataFrame — one row per k, all test-set metrics as cols
    """
    make_dir(COMPARISON_OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("  CROSS-K COMPARISON RUN")
    print(f"  k-values to evaluate : {sorted(k_paths.keys())}")
    print("=" * 72)

    all_summaries = []
    failed_k      = []

    for k in sorted(k_paths.keys()):
        path = k_paths[k]
        if not os.path.exists(path):
            print(f"\n  [WARNING] k={k} dataset not found at:\n  {path}")
            print(f"  Skipping k={k}.\n")
            failed_k.append(k)
            continue
        try:
            summary = run_pipeline_for_k(k, path)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  [ERROR] Pipeline failed for k={k}:\n  {e}\n")
            failed_k.append(k)

    if not all_summaries:
        raise RuntimeError(
            "All k-value pipelines failed. "
            "Check data paths and column names before re-running."
        )

    comparison_df = pd.DataFrame(all_summaries).set_index("k")

    # ── Print cross-k comparison table ───────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  CROSS-K COMPARISON  |  TEST SET RESULTS SUMMARY")
    print("=" * 72)

    targets_display = [
        ("Wind Generation", "wind_MAE",     "wind_RMSE",     "wind_R2"),
        ("Solar Generation","solar_MAE",    "solar_RMSE",    "solar_R2"),
        ("Load Demand",     "load_MAE",     "load_RMSE",     "load_R2"),
        ("Net Load",        "net_load_MAE", "net_load_RMSE", "net_load_R2"),
    ]

    for label, mae_col, rmse_col, r2_col in targets_display:
        print(f"\n  -- {label} --")
        print(f"  {'k':<6} {'MAE (MW)':>12} {'RMSE (MW)':>12} {'R2':>10}")
        print(f"  {'-'*44}")
        for k_val, row in comparison_df.iterrows():
            print(
                f"  {k_val:<6} "
                f"{row[mae_col]:>12.2f} "
                f"{row[rmse_col]:>12.2f} "
                f"{row[r2_col]:>10.5f}"
            )

    print(f"\n  -- Feature counts and runtime --")
    print(f"  {'k':<6} {'Wind feats':>12} {'Solar feats':>13} "
          f"{'Load feats':>12} {'Runtime (s)':>13}")
    print(f"  {'-'*60}")
    for k_val, row in comparison_df.iterrows():
        print(
            f"  {k_val:<6} "
            f"{row['n_wind_features']:>12} "
            f"{row['n_solar_features']:>13} "
            f"{row['n_load_features']:>12} "
            f"{row['runtime_s']:>13.1f}"
        )

    print(f"\n  -- Optimal k by lowest test-set MAE --")
    for label, mae_col, _, _ in targets_display:
        best_k = comparison_df[mae_col].idxmin()
        best_v = comparison_df.loc[best_k, mae_col]
        print(f"  {label:<22} best k = {best_k}  (MAE = {best_v:.2f} MW)")

    if failed_k:
        print(f"\n  [NOTE] Skipped k-values: {failed_k}")

    comparison_path = os.path.join(COMPARISON_OUTPUT_DIR, "cross_k_comparison.csv")
    comparison_df.to_csv(comparison_path)
    print(f"\n  Cross-k table saved to: {comparison_path}")
    print("=" * 72)

    return comparison_df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EXECUTION BLOCK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── MODE A: Run ALL four k-values and compare (full dissertation run) ─────
    # Trains 12 models total (3 per k), saves all outputs, prints cross-k table.
    # Expected runtime: 20-60 minutes depending on CPU.
    comparison_results = run_cross_k_comparison(K_DATASET_PATHS)

    # ── MODE B: Single k-value only (faster — use this first to test) ─────────
    # Comment out Mode A and uncomment below to test on k=8 before the full run.
    # single_result = run_pipeline_for_k(k=8, data_path=K_DATASET_PATHS[8])