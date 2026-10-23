"""
================================================================================
RANDOM SPLIT RESULTS ANALYSIS AND FIGURE GENERATION
Spanish Peninsular Power System — Day-Ahead Forecasting Dissertation
================================================================================
Reads the CSV outputs from ml_pipeline_random.py and produces all figures
and tables for the random-split results section of the dissertation.

All output files are prefixed with random_ and saved to a separate
dissertation_figures_random/ folder to avoid overwriting phase6 outputs.

Figures produced (16 total — identical structure to phase6 results)
────────────────────────────────────────────────────────────────────
 random_fig01  time_series_test.png
 random_fig02  scatter_plots.png
 random_fig03  residual_histograms.png
 random_fig04  diurnal_mae_profile.png
 random_fig05  diurnal_bias_profile.png
 random_fig06  cross_k_mae.png
 random_fig07  cross_k_r2.png
 random_fig08  cross_k_cost.png
 random_fig09  feature_importance_wind.png
 random_fig10  feature_importance_solar.png
 random_fig11  feature_importance_load.png
 random_fig12  ramp_rate_scatter.png
 random_fig13  ramp_rate_error_profile.png
 random_fig14  generation_regime_error.png
 random_fig15  weekly_error_profile.png
 random_fig16  val_vs_test_metrics.png

Tables produced (3 CSVs)
─────────────────────────
 random_table01_standard_metrics.csv
 random_table02_normalised_metrics.csv
 random_table03_cross_k.csv

Author : Fernando Galbete Menendez
Date   : 2025
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = (
    r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3"
    r"\Individual Project\ML Models\Macro Table for Training"
)

PRIMARY_K      = 15
FIG_DIR        = os.path.join(BASE_DIR, "dissertation_figures_random")
COMPARISON_DIR = os.path.join(BASE_DIR, "random_cross_k_comparison")
ALL_K          = [8, 10, 12, 15]

os.makedirs(FIG_DIR, exist_ok=True)

# ── Confirmed reference values ────────────────────────────────────────────────
PEAK_DEMAND_MW       = 38_043
MEAN_LOAD_MW         = 27_334
INSTALLED_WIND_MW    = 32_000
INSTALLED_SOLAR_MW   = 47_000
SOLAR_MAPE_THRESH_MW = 50

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
})

COL = {
    "wind":     "#1D9E75",
    "solar":    "#BA7517",
    "load":     "#185FA5",
    "net_load": "#993C1D",
    "actual":   "#2C2C2A",
    "val":      "#7F77DD",
}
LABELS = {
    "wind":     "Wind Generation",
    "solar":    "Solar Generation",
    "load":     "Load Demand",
    "net_load": "Net Load",
}
TARGETS = ["wind", "solar", "load", "net_load"]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def actual_col(t):
    return {
        "wind":     "WEM_target_national_wind_gen",
        "solar":    "SEM_target_national_solar_gen",
        "load":     "LDM_target_national_demand",
        "net_load": "actual_net_load",
    }[t]

def pred_col(t):
    return f"pred_{t}"

def mw_formatter(x, _):
    return f"{x:,.0f}"

def save_figure(fig, filename):
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_predictions(k):
    """Load test predictions from the random-split pipeline outputs."""
    path = os.path.join(BASE_DIR, f"k={k}", "random_outputs",
                        "random_test_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Random predictions not found: {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["hour"]    = df["time"].dt.hour
    df["weekday"] = df["time"].dt.dayofweek
    return df

def load_pipeline_metrics(k):
    path = os.path.join(BASE_DIR, f"k={k}", "random_outputs",
                        "random_metrics_summary.csv")
    return pd.read_csv(path)

def load_feature_importances(k):
    path = os.path.join(BASE_DIR, f"k={k}", "random_outputs",
                        "random_feature_importances.csv")
    return pd.read_csv(path)

def load_cross_k():
    path = os.path.join(COMPARISON_DIR, "random_cross_k_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Random cross-k comparison not found: {path}")
    return pd.read_csv(path, index_col="k")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  METRICS CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(df):
    """
    Compute the full dissertation metric suite from the random-split test
    predictions CSV. Identical logic to phase7 results analysis.
    """
    rows = []
    for target in TARGETS:
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            print(f"  [WARNING] {col_a} not found — skipping {target}")
            continue

        y_true = df[col_a].values
        y_pred = df[col_p].values

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        if target == "solar":
            mask       = y_true > SOLAR_MAPE_THRESH_MW
            n_excluded = int((~mask).sum())
            mape = (np.mean(np.abs((y_true[mask] - y_pred[mask])
                                   / y_true[mask])) * 100
                    if mask.sum() > 0 else np.nan)
            mape_note = (f"Excludes {n_excluded} obs < "
                         f"{SOLAR_MAPE_THRESH_MW} MW")
        else:
            mask  = np.abs(y_true) > 1e-6
            mape  = np.mean(
                np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            ) * 100
            mape_note = ""

        rows.append({
            "Target":                    LABELS[target],
            "MAE (MW)":                  round(mae,  2),
            "RMSE (MW)":                 round(rmse, 2),
            "MAPE (%)":                  round(mape, 3) if not np.isnan(mape) else np.nan,
            "R²":                        round(r2,   5),
            "nMAE_mean-load (%)":        round(mae / MEAN_LOAD_MW    * 100, 3) if target == "load"     else np.nan,
            "cNMAE_wind (%)":            round(mae / INSTALLED_WIND_MW  * 100, 3) if target == "wind"     else np.nan,
            "cNMAE_solar (%)":           round(mae / INSTALLED_SOLAR_MW * 100, 3) if target == "solar"    else np.nan,
            "pMAE_peak-demand (%)":      round(mae / PEAK_DEMAND_MW  * 100, 3) if target == "load"     else np.nan,
            "pMAE_net-load_peak (%)":    round(mae / PEAK_DEMAND_MW  * 100, 3) if target == "net_load" else np.nan,
            "_mape_note":                mape_note,
        })

    return pd.DataFrame(rows)


def print_metrics_table(mdf):
    print("\n" + "=" * 72)
    print("  RANDOM SPLIT METRICS  |  Test Set  |  k = 15")
    print("=" * 72)
    std_cols = ["Target", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]
    print("\n  Standard metrics:")
    print(mdf[std_cols].to_string(index=False))
    norm_cols = ["Target", "nMAE_mean-load (%)", "cNMAE_wind (%)",
                 "cNMAE_solar (%)", "pMAE_peak-demand (%)",
                 "pMAE_net-load_peak (%)"]
    print("\n  Normalised MAE metrics:")
    print(mdf[norm_cols].to_string(index=False))
    for _, row in mdf.iterrows():
        if row["_mape_note"]:
            print(f"\n  Solar MAPE note: {row['_mape_note']}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE FUNCTIONS (identical structure to phase7, random_ prefix on files)
# ──────────────────────────────────────────────────────────────────────────────

def plot_time_series(df, k):
    """
    Figure 01: Actual vs predicted for the test set.
    Note: because the test set is randomly sampled, the time series will
    appear discontinuous — this is expected and should be noted in the report.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Test Set Forecasts vs Actuals  |  k = {k}  |  Random Split\n"
        f"Note: test rows are randomly sampled — timeline is not continuous",
        fontsize=11, y=0.98
    )
    for ax, target in zip(axes, TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df["time"], df[col_a], color=COL["actual"],
                linewidth=0.6, alpha=0.85, label="Actual")
        ax.plot(df["time"], df[col_p], color=COL[target],
                linewidth=0.6, alpha=0.75, label="Predicted")
        ax.set_ylabel("MW", fontsize=9)
        ax.set_title(LABELS[target], fontsize=10, loc="left", pad=3)
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.legend(loc="upper right", framealpha=0.7, ncol=2)
    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "random_fig01_time_series_test.png")


def plot_scatter(df, pipeline_metrics, k):
    """Figure 02: Predicted vs actual scatter plots with R² annotation."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        f"Predicted vs Actual  |  Test Set  |  k = {k}  |  Random Split",
        fontsize=12
    )
    test_metrics = pipeline_metrics[pipeline_metrics["partition"] == "test"]
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue
        y_actual    = df[col_a].values
        y_predicted = df[col_p].values
        ax.scatter(y_actual, y_predicted, color=COL[target],
                   alpha=0.25, s=2, rasterized=True)
        mn, mx = min(y_actual.min(), y_predicted.min()), max(y_actual.max(), y_predicted.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.6, label="1:1 line")
        row = test_metrics[test_metrics["model"].str.contains(target)]
        if not row.empty:
            ax.text(0.04, 0.95,
                    f"R² = {row['R2'].values[0]:.4f}\n"
                    f"MAE = {row['MAE'].values[0]:,.0f} MW",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              alpha=0.8, ec="grey"))
        ax.set_xlabel("Actual (MW)")
        ax.set_ylabel("Predicted (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "random_fig02_scatter_plots.png")


def plot_residuals(df, k):
    """Figure 03: Residual distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Residual Distributions  (Predicted − Actual)  |  "
        f"k = {k}  |  Random Split",
        fontsize=12
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue
        residuals = df[col_p].values - df[col_a].values
        ax.hist(residuals, bins=80, color=COL[target], alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--",
                   label="Zero error")
        ax.axvline(residuals.mean(), color="red", linewidth=1.2,
                   label=f"Mean = {residuals.mean():+,.0f} MW")
        ax.set_xlabel("Error (MW)")
        ax.set_ylabel("Frequency")
        ax.set_title(LABELS[target], fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "random_fig03_residual_histograms.png")


def plot_diurnal_mae(df, k):
    """Figure 04: MAE by hour of day."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Mean Absolute Error by Hour of Day  |  k = {k}  |  Random Split",
        fontsize=11
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue
        d = df.copy()
        d["abs_err"] = np.abs(d[col_p] - d[col_a])
        hr = d.groupby("hour")["abs_err"].mean()
        ax.bar(range(24), hr.values, color=COL[target], alpha=0.8,
               edgecolor="white", linewidth=0.4)
        ax.axhline(hr.mean(), color="black", ls="--", lw=1, alpha=0.6,
                   label=f"Mean = {hr.mean():,.0f} MW")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "random_fig04_diurnal_mae_profile.png")


def plot_diurnal_bias(df, k):
    """Figure 05: Mean bias by hour of day."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Mean Bias by Hour of Day  |  k = {k}  |  Random Split",
        fontsize=11
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue
        d = df.copy()
        d["bias"] = d[col_p] - d[col_a]
        bias = d.groupby("hour")["bias"].mean()
        bar_colors = [COL[target] if v >= 0 else "#993C1D" for v in bias.values]
        ax.bar(range(24), bias.values, color=bar_colors, alpha=0.8,
               edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Bias (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "random_fig05_diurnal_bias_profile.png")


def plot_cross_k(ck):
    """Figures 06-08: Cross-k comparison charts."""
    k_vals = ck.index.tolist()
    x, w   = np.arange(len(k_vals)), 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, target in enumerate(TARGETS):
        col = f"{target}_MAE"
        if col in ck.columns:
            ax.bar(x + i*w, ck[col].values, w, label=LABELS[target],
                   color=COL[target], alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([f"k = {k}" for k in k_vals])
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Test-Set MAE by Spatial Cluster Count (k)  |  Random Split")
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "random_fig06_cross_k_mae.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, target in enumerate(TARGETS):
        col = f"{target}_R2"
        if col in ck.columns:
            ax.bar(x + i*w, ck[col].values, w, label=LABELS[target],
                   color=COL[target], alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([f"k = {k}" for k in k_vals])
    r2_cols = [f"{t}_R2" for t in TARGETS if f"{t}_R2" in ck.columns]
    ax.set_ylim([max(0, ck[r2_cols].min().min() - 0.02), 1.0])
    ax.set_ylabel("R²")
    ax.set_title("Test-Set R² by Spatial Cluster Count (k)  |  Random Split")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "random_fig07_cross_k_r2.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Computational Cost vs Spatial Resolution  |  Random Split",
                 fontsize=11)
    for col, lbl, clr in [("n_wind_features",  "Wind",  COL["wind"]),
                           ("n_solar_features", "Solar", COL["solar"]),
                           ("n_load_features",  "Load",  COL["load"])]:
        if col in ck.columns:
            a1.plot(k_vals, ck[col].values, marker="o", lw=2,
                    color=clr, label=lbl)
    a1.set_xlabel("k"); a1.set_ylabel("Input features")
    a1.set_title("Feature Count vs k"); a1.set_xticks(k_vals); a1.legend()
    if "runtime_s" in ck.columns:
        a2.bar(k_vals, ck["runtime_s"].values / 60, color="#888780",
               alpha=0.8, width=0.8)
        a2.set_xlabel("k"); a2.set_ylabel("Runtime (minutes)")
        a2.set_title("Training Runtime vs k"); a2.set_xticks(k_vals)
    fig.tight_layout()
    save_figure(fig, "random_fig08_cross_k_cost.png")


def plot_feature_importance(imp_df, k):
    """Figures 09-11: Feature importance for each model."""
    model_config = [
        ("WEM", "wind",  "Wind Energy Model",  9),
        ("SEM", "solar", "Solar Energy Model", 10),
        ("LDM", "load",  "Load Demand Model",  11),
    ]
    for model_code, target, model_name, fig_num in model_config:
        sub = (imp_df[imp_df["model"] == model_code]
               .sort_values("gain", ascending=False).head(15))
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(sub)), sub["gain"].values,
                color=COL[target], alpha=0.85)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(
            [f.replace("WEM_","").replace("SEM_","").replace("LDM_","")
             for f in sub["feature"].values], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(
            f"Top-15 Feature Importances  |  {model_name}  |  k = {k}  "
            f"|  Random Split"
        )
        fig.tight_layout()
        save_figure(fig,
                    f"random_fig{fig_num:02d}_feature_importance_{target}.png")


def plot_ramp_rate(df, k):
    """Figures 12-13: Net load ramp rate analysis."""
    col_a, col_p = "actual_net_load", "pred_net_load"
    if col_a not in df.columns:
        print("  [SKIP] Ramp rate — net load columns not found.")
        return
    d = df.sort_values("time").copy()
    d["ramp_actual"]    = d[col_a].diff()
    d["ramp_predicted"] = d[col_p].diff()
    d["ramp_abs_error"] = np.abs(d["ramp_predicted"] - d["ramp_actual"])
    d = d.dropna(subset=["ramp_actual"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(d["ramp_actual"], d["ramp_predicted"],
               color=COL["net_load"], alpha=0.2, s=3, rasterized=True)
    mn = min(d["ramp_actual"].min(), d["ramp_predicted"].min())
    mx = max(d["ramp_actual"].max(), d["ramp_predicted"].max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, alpha=0.7, label="1:1 line")
    corr = np.corrcoef(d["ramp_actual"], d["ramp_predicted"])[0, 1]
    ax.text(0.04, 0.95,
            f"Ramp MAE = {d['ramp_abs_error'].mean():,.0f} MW/h\n"
            f"Correlation = {corr:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="grey"))
    ax.set_xlabel("Actual Ramp Rate (MW/h)")
    ax.set_ylabel("Predicted Ramp Rate (MW/h)")
    ax.set_title(f"Net Load Ramp Rate  |  k = {k}  |  Random Split")
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(mw_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "random_fig12_ramp_rate_scatter.png")

    hr = d.groupby("hour")["ramp_abs_error"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(24), hr.values, color=COL["net_load"], alpha=0.85,
           edgecolor="white", linewidth=0.4)
    ax.axhline(hr.mean(), color="black", ls="--", lw=1, alpha=0.6,
               label=f"Mean = {hr.mean():,.0f} MW/h")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Ramp Rate MAE (MW/h)")
    ax.set_title(f"Net Load Ramp Rate Error by Hour  |  k = {k}  |  Random Split")
    ax.set_xticks(range(0, 24, 3))
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "random_fig13_ramp_rate_error_profile.png")


def plot_regime_error(df, k):
    """Figure 14: MAE by renewable output regime."""
    cw, cs = actual_col("wind"), actual_col("solar")
    if cw not in df.columns or cs not in df.columns:
        print("  [SKIP] Regime error."); return
    d = df.copy()
    d["ren"] = d[cw] + d[cs]
    p33, p66 = d["ren"].quantile(0.33), d["ren"].quantile(0.66)

    def reg(v):
        if v < p33:   return "Low\n(< 33rd pctile)"
        elif v < p66: return "Medium\n(33–66th pctile)"
        else:         return "High\n(> 66th pctile)"

    d["regime"] = d["ren"].apply(reg)
    order  = ["Low\n(< 33rd pctile)", "Medium\n(33–66th pctile)",
              "High\n(> 66th pctile)"]
    colors = ["#9FE1CB", "#5DCAA5", "#0F6E56"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"MAE by Renewable Output Regime  |  k = {k}  |  Random Split",
        fontsize=11
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a, col_p = actual_col(target), pred_col(target)
        if col_a not in d.columns:
            ax.set_visible(False); continue
        d["abs_err"] = np.abs(d[col_p] - d[col_a])
        mae_r = d.groupby("regime")["abs_err"].mean().reindex(order)
        bars = ax.bar(range(3), mae_r.values, color=colors,
                      alpha=0.85, edgecolor="white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(order, fontsize=8.5)
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        for bar, val in zip(bars, mae_r.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val*1.01,
                        f"{val:,.0f}", ha="center", fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "random_fig14_generation_regime_error.png")


def plot_weekly_error(df, k):
    """Figure 15: MAE by day of week."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"MAE by Day of Week  |  k = {k}  |  Random Split", fontsize=11
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a, col_p = actual_col(target), pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False); continue
        d = df.copy()
        d["abs_err"] = np.abs(d[col_p] - d[col_a])
        wk = d.groupby("weekday")["abs_err"].mean()
        ax.bar(range(7), wk.values, color=COL[target], alpha=0.85,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.axvline(4.5, color="grey", ls="--", lw=1, alpha=0.6,
                   label="Weekday/Weekend")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "random_fig15_weekly_error_profile.png")


def plot_val_vs_test(pipeline_metrics, k):
    """Figure 16: Validation vs test metric comparison."""
    vm = pipeline_metrics[pipeline_metrics["partition"] == "val"]
    tm = pipeline_metrics[pipeline_metrics["partition"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Validation vs Test Performance  |  k = {k}  |  Random Split",
        fontsize=11
    )
    for ax, metric, ylabel in zip(axes, ["MAE", "R2"], ["MAE (MW)", "R²"]):
        x, w = np.arange(4), 0.35
        vv = [vm[vm["model"].str.contains(t)][metric].values[0]
              if not vm[vm["model"].str.contains(t)].empty else 0
              for t in TARGETS]
        tv = [tm[tm["model"].str.contains(t)][metric].values[0]
              if not tm[tm["model"].str.contains(t)].empty else 0
              for t in TARGETS]
        ax.bar(x - w/2, vv, w, label="Validation",
               color=COL["val"],    alpha=0.85)
        ax.bar(x + w/2, tv, w, label="Test",
               color=COL["actual"], alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[t] for t in TARGETS], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        if metric == "MAE":
            ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "random_fig16_val_vs_test_metrics.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  TABLE EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_report_tables(mdf, ck):
    """Export three clean CSV tables ready for the dissertation."""
    # Table 01: Standard metrics
    t1 = mdf[["Target", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]].copy()
    t1.to_csv(os.path.join(FIG_DIR, "random_table01_standard_metrics.csv"),
              index=False)
    print("  Saved: random_table01_standard_metrics.csv")

    # Table 02: Normalised metrics
    norm_cols = ["Target", "nMAE_mean-load (%)", "cNMAE_wind (%)",
                 "cNMAE_solar (%)", "pMAE_peak-demand (%)",
                 "pMAE_net-load_peak (%)"]
    mdf[norm_cols].to_csv(
        os.path.join(FIG_DIR, "random_table02_normalised_metrics.csv"),
        index=False
    )
    print("  Saved: random_table02_normalised_metrics.csv")

    # Table 03: Cross-k
    keep = [c for c in [
        "n_wind_features", "n_solar_features", "n_load_features", "runtime_s",
        "wind_MAE", "solar_MAE", "load_MAE", "net_load_MAE",
        "wind_R2",  "solar_R2",  "load_R2",  "net_load_R2",
    ] if c in ck.columns]
    ck[keep].to_csv(
        os.path.join(FIG_DIR, "random_table03_cross_k.csv")
    )
    print("  Saved: random_table03_cross_k.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 72)
    print(f"  RANDOM SPLIT RESULTS ANALYSIS  |  k = {PRIMARY_K}")
    print(f"  Output → {FIG_DIR}")
    print("=" * 72)

    print(f"\n[1] Loading test predictions for k={PRIMARY_K}...")
    test_df = load_predictions(PRIMARY_K)
    print(f"    Test rows: {len(test_df):,}")

    print(f"\n[2] Loading pipeline metrics...")
    pipeline_metrics = load_pipeline_metrics(PRIMARY_K)

    print(f"\n[3] Loading feature importances...")
    imp_df = load_feature_importances(PRIMARY_K)

    print(f"\n[4] Loading cross-k comparison table...")
    try:
        ck             = load_cross_k()
        cross_k_loaded = True
        print(f"    Loaded: {len(ck)} k-values")
    except FileNotFoundError as e:
        print(f"    [WARNING] {e}")
        cross_k_loaded = False

    print(f"\n[5] Computing metrics...")
    mdf = compute_all_metrics(test_df)
    print_metrics_table(mdf)

    print(f"\n[6] Generating figures...")
    plot_time_series(test_df, PRIMARY_K)
    plot_scatter(test_df, pipeline_metrics, PRIMARY_K)
    plot_residuals(test_df, PRIMARY_K)
    plot_diurnal_mae(test_df, PRIMARY_K)
    plot_diurnal_bias(test_df, PRIMARY_K)
    if cross_k_loaded:
        plot_cross_k(ck)
    plot_feature_importance(imp_df, PRIMARY_K)
    plot_ramp_rate(test_df, PRIMARY_K)
    plot_regime_error(test_df, PRIMARY_K)
    plot_weekly_error(test_df, PRIMARY_K)
    plot_val_vs_test(pipeline_metrics, PRIMARY_K)

    print(f"\n[7] Exporting tables...")
    if cross_k_loaded:
        export_report_tables(mdf, ck)
    else:
        print("    [WARNING] Cross-k table skipped.")

    n_files = len(os.listdir(FIG_DIR))
    print(f"\n{'=' * 72}")
    print(f"  COMPLETE — {n_files} files written to:")
    print(f"  {FIG_DIR}")
    print("=" * 72)