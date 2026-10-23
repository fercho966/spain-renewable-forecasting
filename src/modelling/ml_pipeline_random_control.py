"""
================================================================================
RANDOM SPLIT PIPELINE
Spanish Peninsular Power System — Day-Ahead XGBoost Forecasting
================================================================================
Identical to ml_pipeline_phase6.py in every respect except the train-
validation-test split. Instead of a fixed chronological cut, rows are
randomly shuffled before splitting by percentage.

Split proportions (matching phase6 percentages)
  Train : 74.9%
  Val   : 19.2%
  Test  :  6.0%

random_state=42 ensures the split is reproducible — running the script
twice produces identical partitions.

Note on data leakage
────────────────────
Because lag features are pre-computed and stored as columns in the dataset,
a randomly assigned test row may have lag values derived from rows that ended
up in the training set. This means the model has indirect access to some
test-set information during training. Metrics will therefore tend to be
slightly more optimistic than the chronological split results. This should
be acknowledged when comparing the two sets of results.

Output files (prefixed with random_ to avoid overwriting phase6 outputs)
────────────────────────────────────────────────────────────────────────
Per k-value, saved to BASE_DIR/k={k}/random_outputs/:
  random_metrics_summary.csv
  random_test_predictions.csv
  random_feature_importances.csv
  random_feature_routing.csv
  random_model_wind.pkl
  random_model_solar.pkl
  random_model_load.pkl

Cross-k comparison saved to BASE_DIR/random_cross_k_comparison/:
  random_cross_k_comparison.csv

Architecture (identical to phase6)
────────────────────────────────────────────────────────────────
  - ONE unified macro-level master dataset per k-value
  - THREE separate XGBoost regressors (Wind / Solar / Load)
  - Net Load derived post-hoc: Load_pred - (Wind_pred + Solar_pred)
  - Cross-model feature linkage preserved
  - Same hyperparameters as phase6

Author : Fernando Galbete Menendez
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
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

RANDOM_COMPARISON_DIR = os.path.join(BASE_DIR, "random_cross_k_comparison")

# ── Target column names ───────────────────────────────────────────────────────
TARGETS = {
    "wind":  "WEM_target_national_wind_gen",
    "solar": "SEM_target_national_solar_gen",
    "load":  "LDM_target_national_demand",
}

# ── Column prefixes for dynamic feature routing ───────────────────────────────
PREFIXES = {
    "wind":  "WEM_",
    "solar": "SEM_",
    "load":  "LDM_",
}

# ── Global calendar features shared by all three models ───────────────────────
GLOBAL_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend", "is_holiday", "is_non_working_day",
]

# ── Cross-model lag features ──────────────────────────────────────────────────
DEMAND_LAG_FEATURES = [
    "LDM_demand_lag_24",
    "LDM_demand_lag_168",
    "LDM_demand_rollmean_24",
    "LDM_demand_rollmean_168",
]
SOLAR_LAG_FEATURES = [
    "SEM_solar_gen_lag_24",
    "SEM_solar_gen_lag_168",
    "SEM_solar_gen_rollmean_24",
    "SEM_solar_gen_rollmean_168",
]
WIND_LAG_FEATURES = [
    "WEM_wind_gen_lag_24",
    "WEM_wind_gen_lag_168",
    "WEM_wind_gen_rollmean_24",
    "WEM_wind_gen_rollmean_168",
]

# ── Random split settings ─────────────────────────────────────────────────────
# Proportions match the phase6 chronological split percentages
RANDOM_STATE = 42      # fixed seed for reproducibility
TEST_SIZE    = 0.060   # 6.0% test
VAL_SIZE     = 0.192   # 19.2% validation (of full dataset)
# Val fraction of the remaining train+val pool after test is removed:
# 0.192 / (1 - 0.060) = 0.2043
VAL_OF_TRAINVAL = VAL_SIZE / (1.0 - TEST_SIZE)   # ≈ 0.2043

# ── XGBoost hyperparameters (identical to phase6) ────────────────────────────
XGB_PARAMS = {
    "n_estimators":          2000,
    "learning_rate":         0.03,
    "max_depth":             6,
    "min_child_weight":      5,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "gamma":                 0.0,
    "objective":             "reg:squarederror",
    "eval_metric":           "rmse",
    "tree_method":           "hist",
    "random_state":          RANDOM_STATE,
    "n_jobs":                -1,
    "early_stopping_rounds": 75,
}

CLIP_RENEWABLES = True

# ── Physical reference constants for normalised MAE metrics ──────────────────
INSTALLED_WIND_MW    = 32_000
INSTALLED_SOLAR_MW   = 47_000
PEAK_DEMAND_MW       = 38_043
MEAN_LOAD_MW         = 27_334
SOLAR_MAPE_THRESH_MW = 50


# ──────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY FUNCTIONS (identical to phase6)
# ──────────────────────────────────────────────────────────────────────────────

def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_columns(df: pd.DataFrame, required: list, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"\n[COLUMN ERROR — {context}]\n"
            f"Missing columns:\n  {missing}\n"
        )


def route_features(df: pd.DataFrame, prefix: str, target: str,
                   extra_cols: list = None) -> list:
    """Build the complete feature list for one model."""
    domain_cols = [c for c in df.columns if c.startswith(prefix) and c != target]
    seen        = set(domain_cols)

    global_extra = [c for c in GLOBAL_FEATURES if c not in seen]
    seen.update(global_extra)

    cross_extra = []
    for c in (extra_cols or []):
        if c not in seen:
            cross_extra.append(c)
            seen.add(c)

    return domain_cols + global_extra + cross_extra


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str, target_type: str = "",
                    partition: str = "") -> dict:
    """
    Compute the full dissertation metric suite for one set of predictions.

    Parameters
    ----------
    y_true      : array of actual values (MW)
    y_pred      : array of predicted values (MW)
    label       : string label for this row (e.g. 'test_wind')
    target_type : one of 'wind', 'solar', 'load', 'net_load'
    partition   : 'val' or 'test'
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — solar uses 50 MW threshold to exclude night hours
    if target_type == "solar":
        mask = y_true > SOLAR_MAPE_THRESH_MW
        mape = (np.mean(np.abs((y_true[mask] - y_pred[mask])
                               / y_true[mask])) * 100
                if mask.sum() > 0 else np.nan)
    else:
        mask = np.abs(y_true) > 1e-6
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100

    return {
        "model":                  label,
        "partition":              partition,
        "MAE":                    round(mae,  2),
        "RMSE":                   round(rmse, 2),
        "MAPE_%":                 round(mape, 3) if not np.isnan(mape) else np.nan,
        "R2":                     round(r2,   5),
        "nMAE_mean-load_%":       round(mae / MEAN_LOAD_MW    * 100, 3) if target_type == "load"     else np.nan,
        "cNMAE_wind_%":           round(mae / INSTALLED_WIND_MW  * 100, 3) if target_type == "wind"     else np.nan,
        "cNMAE_solar_%":          round(mae / INSTALLED_SOLAR_MW * 100, 3) if target_type == "solar"    else np.nan,
        "pMAE_peak-demand_%":     round(mae / PEAK_DEMAND_MW  * 100, 3) if target_type == "load"     else np.nan,
        "pMAE_net-load_peak_%":   round(mae / PEAK_DEMAND_MW  * 100, 3) if target_type == "net_load" else np.nan,
    }


def print_metrics_block(label: str, m: dict) -> None:
    """Print a single metrics dict to the console."""
    print(f"\n  [{label}]")
    print(f"    {'MAE':<28} {m['MAE']:>10.2f}  MW")
    print(f"    {'RMSE':<28} {m['RMSE']:>10.2f}  MW")
    print(f"    {'MAPE':<28} {m['MAPE_%']:>10.3f}  %")
    print(f"    {'R2':<28} {m['R2']:>12.5f}")
    if not np.isnan(m.get("nMAE_mean-load_%",   np.nan)):
        print(f"    {'nMAE_mean-load':<28} {m['nMAE_mean-load_%']:>10.3f}  %")
    if not np.isnan(m.get("cNMAE_wind_%",        np.nan)):
        print(f"    {'cNMAE_wind':<28} {m['cNMAE_wind_%']:>10.3f}  %")
    if not np.isnan(m.get("cNMAE_solar_%",       np.nan)):
        print(f"    {'cNMAE_solar':<28} {m['cNMAE_solar_%']:>10.3f}  %")
    if not np.isnan(m.get("pMAE_peak-demand_%",  np.nan)):
        print(f"    {'pMAE_peak-demand':<28} {m['pMAE_peak-demand_%']:>10.3f}  %")
    if not np.isnan(m.get("pMAE_net-load_peak_%", np.nan)):
        print(f"    {'pMAE_net-load_peak':<28} {m['pMAE_net-load_peak_%']:>10.3f}  %")


def get_feature_importance(model: xgb.XGBRegressor,
                           feature_names: list,
                           model_code: str) -> pd.DataFrame:
    """Gain-based feature importance, top-20 features."""
    scores = model.get_booster().get_score(importance_type="gain")
    df_imp = (
        pd.DataFrame(scores.items(), columns=["feature", "gain"])
          .sort_values("gain", ascending=False)
          .head(20)
          .reset_index(drop=True)
    )
    df_imp.insert(0, "rank",  df_imp.index + 1)
    df_imp.insert(0, "model", model_code)
    return df_imp


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SINGLE K PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_k(k: int, data_path: str) -> dict:
    """
    Execute the random-split pipeline for one k-value dataset.

    Stages
    ──────
    1  Load dataset
    2  Apply random train / validation / test split
    3  Route features per model
    4  Validate column existence
    5  Initialise XGBoost regressors
    6  Train with early stopping on validation set
    7  Evaluate on validation set
    8  Evaluate on test set
    9  Save outputs

    Parameters
    ----------
    k         : int — spatial cluster count
    data_path : str — path to the k-specific parquet file

    Returns
    -------
    summary : dict — test metrics for the cross-k comparison table
    """
    out_dir = os.path.join(BASE_DIR, f"k={k}", "random_outputs")
    make_dir(out_dir)
    pipeline_start = time.time()

    print()
    print("=" * 72)
    print(f"  RANDOM SPLIT PIPELINE  |  k = {k} spatial clusters")
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
        raise ValueError(f"Dataset k={k} contains {nan_count} NaN values.")

    # ── STAGE 2: Random Split ─────────────────────────────────────────────────
    print(f"\n[2/9] Applying random split (seed = {RANDOM_STATE})...")

    # Step 1: separate test set from the rest
    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    # Step 2: separate validation from training
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=VAL_OF_TRAINVAL,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    n_total = len(df)
    print(f"      Train : {len(train_df):>6,} rows ({100*len(train_df)/n_total:.1f}%)")
    print(f"      Val   : {len(val_df):>6,} rows ({100*len(val_df)/n_total:.1f}%)")
    print(f"      Test  : {len(test_df):>6,} rows ({100*len(test_df)/n_total:.1f}%)")
    print(f"      Note  : Rows randomly shuffled — no chronological ordering")

    # ── STAGE 3: Feature Routing ──────────────────────────────────────────────
    print(f"\n[3/9] Routing features (k={k} cluster columns auto-included)...")
    wind_feats  = route_features(df, PREFIXES["wind"],  TARGETS["wind"],
                                 extra_cols=DEMAND_LAG_FEATURES + SOLAR_LAG_FEATURES)
    solar_feats = route_features(df, PREFIXES["solar"], TARGETS["solar"],
                                 extra_cols=DEMAND_LAG_FEATURES + WIND_LAG_FEATURES)
    load_feats  = route_features(df, PREFIXES["load"],  TARGETS["load"])

    print(f"      Wind model (WEM)  : {len(wind_feats)} cols")
    print(f"      Solar model (SEM) : {len(solar_feats)} cols")
    print(f"      Load model (LDM)  : {len(load_feats)} cols")

    # Save feature routing record
    routing_records = (
        [{"model": "WEM", "feature": f} for f in wind_feats]
      + [{"model": "SEM", "feature": f} for f in solar_feats]
      + [{"model": "LDM", "feature": f} for f in load_feats]
    )
    pd.DataFrame(routing_records).to_csv(
        os.path.join(out_dir, "random_feature_routing.csv"), index=False
    )

    # ── STAGE 4: Column Validation ────────────────────────────────────────────
    print(f"\n[4/9] Validating all routed columns exist in dataset...")
    validate_columns(df, wind_feats  + [TARGETS["wind"]],  "Wind Model")
    validate_columns(df, solar_feats + [TARGETS["solar"]], "Solar Model")
    validate_columns(df, load_feats  + [TARGETS["load"]],  "Load Model")
    print(f"      All columns validated successfully.")

    # ── STAGE 5: Initialise Models ────────────────────────────────────────────
    print(f"\n[5/9] Initialising three XGBoost regressors...")
    model_wind  = xgb.XGBRegressor(**XGB_PARAMS)
    model_solar = xgb.XGBRegressor(**XGB_PARAMS)
    model_load  = xgb.XGBRegressor(**XGB_PARAMS)

    # ── Prepare arrays ────────────────────────────────────────────────────────
    X_train_w = train_df[wind_feats].values
    X_val_w   = val_df[wind_feats].values
    X_test_w  = test_df[wind_feats].values
    y_train_w = train_df[TARGETS["wind"]].values
    y_val_w   = val_df[TARGETS["wind"]].values
    y_test_w  = test_df[TARGETS["wind"]].values

    X_train_s = train_df[solar_feats].values
    X_val_s   = val_df[solar_feats].values
    X_test_s  = test_df[solar_feats].values
    y_train_s = train_df[TARGETS["solar"]].values
    y_val_s   = val_df[TARGETS["solar"]].values
    y_test_s  = test_df[TARGETS["solar"]].values

    X_train_l = train_df[load_feats].values
    X_val_l   = val_df[load_feats].values
    X_test_l  = test_df[load_feats].values
    y_train_l = train_df[TARGETS["load"]].values
    y_val_l   = val_df[TARGETS["load"]].values
    y_test_l  = test_df[TARGETS["load"]].values

    # ── STAGE 6: Training ─────────────────────────────────────────────────────
    print(f"\n[6/9] Training (early stopping patience = 75 rounds)...")

    print(f"\n  -- Wind Energy Model (WEM) --")
    t0 = time.time()
    model_wind.fit(
        X_train_w, y_train_w,
        eval_set=[(X_train_w, y_train_w), (X_val_w, y_val_w)],
        verbose=False,
    )
    print(f"    Best iteration : {model_wind.best_iteration}")
    print(f"    Best val RMSE  : {model_wind.best_score:.4f}")
    print(f"    Training time  : {time.time()-t0:.1f}s")

    print(f"\n  -- Solar Energy Model (SEM) --")
    t0 = time.time()
    model_solar.fit(
        X_train_s, y_train_s,
        eval_set=[(X_train_s, y_train_s), (X_val_s, y_val_s)],
        verbose=False,
    )
    print(f"    Best iteration : {model_solar.best_iteration}")
    print(f"    Best val RMSE  : {model_solar.best_score:.4f}")
    print(f"    Training time  : {time.time()-t0:.1f}s")

    print(f"\n  -- Load Demand Model (LDM) --")
    t0 = time.time()
    model_load.fit(
        X_train_l, y_train_l,
        eval_set=[(X_train_l, y_train_l), (X_val_l, y_val_l)],
        verbose=False,
    )
    print(f"    Best iteration : {model_load.best_iteration}")
    print(f"    Best val RMSE  : {model_load.best_score:.4f}")
    print(f"    Training time  : {time.time()-t0:.1f}s")

    # ── STAGE 7: Validation Evaluation ───────────────────────────────────────
    print(f"\n[7/9] Validation-set evaluation...")

    vw = model_wind.predict(X_val_w)
    vs = model_solar.predict(X_val_s)
    vl = model_load.predict(X_val_l)

    if CLIP_RENEWABLES:
        vw = np.clip(vw, 0, None)
        vs = np.clip(vs, 0, None)

    val_net_actual = y_val_l - (y_val_w + y_val_s)
    val_net_pred   = vl - (vw + vs)

    vm_wind  = compute_metrics(y_val_w,        vw,           "val_wind",     target_type="wind",     partition="val")
    vm_solar = compute_metrics(y_val_s,        vs,           "val_solar",    target_type="solar",    partition="val")
    vm_load  = compute_metrics(y_val_l,        vl,           "val_load",     target_type="load",     partition="val")
    vm_net   = compute_metrics(val_net_actual, val_net_pred, "val_net_load", target_type="net_load", partition="val")

    print_metrics_block("VAL Wind Generation",                vm_wind)
    print_metrics_block("VAL Solar Generation",               vm_solar)
    print_metrics_block("VAL Load Demand",                    vm_load)
    print_metrics_block("VAL Net Load (derived post-hoc)",    vm_net)

    # ── STAGE 8: Test Evaluation ──────────────────────────────────────────────
    print(f"\n[8/9] TEST-SET EVALUATION")
    print(f"      This partition was NOT used during training or model selection.")

    tw = model_wind.predict(X_test_w)
    ts = model_solar.predict(X_test_s)
    tl = model_load.predict(X_test_l)

    if CLIP_RENEWABLES:
        tw = np.clip(tw, 0, None)
        ts = np.clip(ts, 0, None)

    test_net_actual = y_test_l - (y_test_w + y_test_s)
    test_net_pred   = tl - (tw + ts)

    tm_wind  = compute_metrics(y_test_w,         tw,            "test_wind",     target_type="wind",     partition="test")
    tm_solar = compute_metrics(y_test_s,         ts,            "test_solar",    target_type="solar",    partition="test")
    tm_load  = compute_metrics(y_test_l,         tl,            "test_load",     target_type="load",     partition="test")
    tm_net   = compute_metrics(test_net_actual,  test_net_pred, "test_net_load", target_type="net_load", partition="test")

    print_metrics_block("TEST Wind Generation",                tm_wind)
    print_metrics_block("TEST Solar Generation",               tm_solar)
    print_metrics_block("TEST Load Demand",                    tm_load)
    print_metrics_block("TEST Net Load (derived post-hoc)",    tm_net)

    # ── STAGE 9: Save Outputs ─────────────────────────────────────────────────
    print(f"\n[9/9] Saving outputs to: {out_dir}")

    # Metrics summary
    metrics_df = pd.DataFrame([vm_wind, vm_solar, vm_load, vm_net,
                                tm_wind, tm_solar, tm_load, tm_net])
    metrics_df.insert(0, "k", k)
    metrics_df.to_csv(os.path.join(out_dir, "random_metrics_summary.csv"),
                      index=False)
    print("      random_metrics_summary.csv      saved")

    # Test predictions
    test_pred_df = test_df[["time",
                             TARGETS["wind"],
                             TARGETS["solar"],
                             TARGETS["load"]]].copy()
    test_pred_df = test_pred_df.sort_values("time").reset_index(drop=True)
    # Realign predictions to sorted order
    test_sorted_idx = test_df["time"].argsort().values
    test_pred_df["pred_wind"]       = tw[test_sorted_idx]
    test_pred_df["pred_solar"]      = ts[test_sorted_idx]
    test_pred_df["pred_load"]       = tl[test_sorted_idx]
    test_pred_df["pred_net_load"]   = test_net_pred[test_sorted_idx]
    test_pred_df["actual_net_load"] = test_net_actual[test_sorted_idx]
    test_pred_df.to_csv(os.path.join(out_dir, "random_test_predictions.csv"),
                        index=False)
    print("      random_test_predictions.csv     saved")

    # Feature importances
    imp_wind  = get_feature_importance(model_wind,  wind_feats,  "WEM")
    imp_solar = get_feature_importance(model_solar, solar_feats, "SEM")
    imp_load  = get_feature_importance(model_load,  load_feats,  "LDM")
    imp_all   = pd.concat([imp_wind, imp_solar, imp_load], ignore_index=True)
    imp_all.insert(0, "k", k)
    imp_all.to_csv(os.path.join(out_dir, "random_feature_importances.csv"),
                   index=False)
    print("      random_feature_importances.csv  saved")

    # Save models
    joblib.dump(model_wind,  os.path.join(out_dir, "random_model_wind.pkl"))
    joblib.dump(model_solar, os.path.join(out_dir, "random_model_solar.pkl"))
    joblib.dump(model_load,  os.path.join(out_dir, "random_model_load.pkl"))
    print("      random_model_wind/solar/load    saved")

    total_time = time.time() - pipeline_start
    print(f"\n  Random-split pipeline for k={k} completed in {total_time:.1f}s")

    # Summary table row
    print(f"\n  {'Model':<22} {'MAE':>9} {'RMSE':>9} {'MAPE%':>8} {'R2':>9}")
    print(f"  {'-'*60}")
    for m in [tm_wind, tm_solar, tm_load, tm_net]:
        print(f"  {m['model']:<22} {m['MAE']:>9.2f} {m['RMSE']:>9.2f} "
              f"{m['MAPE_%']:>8.3f} {m['R2']:>9.5f}")

    return {
        "k":                    k,
        "n_wind_features":      len(wind_feats),
        "n_solar_features":     len(solar_feats),
        "n_load_features":      len(load_feats),
        "runtime_s":            round(total_time, 1),
        "wind_MAE":             tm_wind["MAE"],
        "wind_RMSE":            tm_wind["RMSE"],
        "wind_R2":              tm_wind["R2"],
        "solar_MAE":            tm_solar["MAE"],
        "solar_RMSE":           tm_solar["RMSE"],
        "solar_R2":             tm_solar["R2"],
        "load_MAE":             tm_load["MAE"],
        "load_RMSE":            tm_load["RMSE"],
        "load_R2":              tm_load["R2"],
        "net_load_MAE":         tm_net["MAE"],
        "net_load_RMSE":        tm_net["RMSE"],
        "net_load_R2":          tm_net["R2"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-K COMPARISON RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_k_comparison(k_paths: dict) -> pd.DataFrame:
    """
    Run the random-split pipeline for all k-values and produce a cross-k
    comparison table.
    """
    make_dir(RANDOM_COMPARISON_DIR)

    print("\n" + "=" * 72)
    print("  RANDOM SPLIT CROSS-K COMPARISON RUN")
    print(f"  k-values to evaluate : {sorted(k_paths.keys())}")
    print("=" * 72)

    all_summaries = []
    failed_k      = []

    for k in sorted(k_paths.keys()):
        path = k_paths[k]
        if not os.path.exists(path):
            print(f"\n  [WARNING] k={k} dataset not found: {path}")
            failed_k.append(k)
            continue
        try:
            summary = run_pipeline_for_k(k, path)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  [ERROR] Pipeline failed for k={k}:\n  {e}\n")
            failed_k.append(k)

    if not all_summaries:
        raise RuntimeError("All k-value pipelines failed.")

    comparison_df = pd.DataFrame(all_summaries).set_index("k")

    # Print cross-k summary
    print("\n\n" + "=" * 72)
    print("  CROSS-K COMPARISON  |  TEST SET RESULTS SUMMARY  |  RANDOM SPLIT")
    print("=" * 72)

    for label, mae_col, rmse_col, r2_col in [
        ("Wind Generation",  "wind_MAE",     "wind_RMSE",     "wind_R2"),
        ("Solar Generation", "solar_MAE",    "solar_RMSE",    "solar_R2"),
        ("Load Demand",      "load_MAE",     "load_RMSE",     "load_R2"),
        ("Net Load",         "net_load_MAE", "net_load_RMSE", "net_load_R2"),
    ]:
        print(f"\n  -- {label} --")
        print(f"  {'k':<6} {'MAE (MW)':>12} {'RMSE (MW)':>12} {'R2':>10}")
        print(f"  {'-'*44}")
        for k_val, row in comparison_df.iterrows():
            print(f"  {k_val:<6} {row[mae_col]:>12.2f} "
                  f"{row[rmse_col]:>12.2f} {row[r2_col]:>10.5f}")

    print(f"\n  -- Feature counts and runtime --")
    print(f"  {'k':<6} {'Wind feats':>12} {'Solar feats':>12} "
          f"{'Load feats':>12} {'Runtime (s)':>13}")
    print(f"  {'-'*59}")
    for k_val, row in comparison_df.iterrows():
        print(f"  {k_val:<6} {row['n_wind_features']:>12.0f} "
              f"{row['n_solar_features']:>12.0f} "
              f"{row['n_load_features']:>12.0f} "
              f"{row['runtime_s']:>13.1f}")

    # Best k by Net Load MAE
    best_k = comparison_df["net_load_MAE"].idxmin()
    print(f"\n  Best k by Net Load MAE : k={best_k}  "
          f"(MAE = {comparison_df.loc[best_k, 'net_load_MAE']:.2f} MW)")

    if failed_k:
        print(f"\n  [NOTE] Skipped k-values: {failed_k}")

    save_path = os.path.join(
        RANDOM_COMPARISON_DIR, "random_cross_k_comparison.csv"
    )
    comparison_df.to_csv(save_path)
    print(f"\n  Cross-k table saved to: {save_path}")
    print("=" * 72)

    return comparison_df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EXECUTION BLOCK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── MODE A: Run all four k-values (full dissertation run) ─────────────────
    comparison_results = run_cross_k_comparison(K_DATASET_PATHS)

    # ── MODE B: Single k=15 only (test before committing to full run) ─────────
    # single_result = run_pipeline_for_k(k=15, data_path=K_DATASET_PATHS[15])