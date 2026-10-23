"""
================================================================================
PHASE 7: RESULTS ANALYSIS AND FIGURE GENERATION
Spanish Peninsular Power System — Day-Ahead Forecasting Dissertation
================================================================================
Reads the CSV outputs from the ML pipeline and produces all figures and tables
needed for the dissertation results section.

No retraining is required. All calculations are performed on the saved
prediction CSVs produced by ml_pipeline_phase6.py.

Figures produced (16 total)
────────────────────────────
 fig01  time_series_test.png         Actual vs predicted, all targets, test period
 fig02  scatter_plots.png            Predicted vs actual scatter with R² annotation
 fig03  residual_histograms.png      Error distribution for all targets
 fig04  diurnal_mae_profile.png      Mean absolute error by hour of day
 fig05  diurnal_bias_profile.png     Mean signed error (bias) by hour of day
 fig06  cross_k_mae.png              Cross-k MAE comparison bar chart
 fig07  cross_k_r2.png               Cross-k R² comparison bar chart
 fig08  cross_k_cost.png             Feature count and runtime vs k
 fig09  feature_importance_wind.png  Top-15 gain-based features, Wind model
 fig10  feature_importance_solar.png Top-15 gain-based features, Solar model
 fig11  feature_importance_load.png  Top-15 gain-based features, Load model
 fig12  ramp_rate_scatter.png        Actual vs predicted Net Load ramp rates
 fig13  ramp_rate_error_profile.png  Ramp rate error by hour of day
 fig14  generation_regime_error.png  MAE by renewable output regime
 fig15  weekly_error_profile.png     MAE by day of week
 fig16  val_vs_test_metrics.png      Validation vs test metric comparison

Tables produced (3 CSVs)
─────────────────────────
 table01_standard_metrics.csv    MAE, RMSE, MAPE, R² for all targets
 table02_normalised_metrics.csv  All five normalised MAE variants
 table03_cross_k.csv             Cross-k performance and cost summary

Metrics reported
─────────────────
Standard
  MAE (MW)   Mean Absolute Error — primary metric, directly interpretable
  RMSE (MW)  Root Mean Squared Error — penalises large peak errors
  MAPE (%)   Mean Absolute Percentage Error
  R²         Coefficient of Determination

MAPE notes
  Wind / Load / Net Load : standard MAPE, excluding |y| ≤ 1e-6 MW
  Solar                  : excludes observations where actual < 50 MW
                           to avoid night-hour division instability
                           (50 MW ≈ 0.1% of 47 GW installed capacity)

Normalised MAE metrics (replaces old single nMAE column)
  nMAE_mean-load (%)         Load model only
                             MAE / mean actual load × 100
                             mean load = 27,334 MW (from test data)
                             Measures error relative to average system demand

  cNMAE_wind (%)             Wind model only
                             MAE / installed wind capacity × 100
                             installed wind = 32,000 MW (2025)
                             Preferred over mean-normalised for intermittent
                             sources — avoids exaggerating error during
                             low-generation periods

  cNMAE_solar (%)            Solar model only
                             MAE / installed solar capacity × 100
                             installed solar = 47,000 MW (2025)
                             Same rationale as cNMAE_wind

  pMAE_peak-demand (%)       Load model only
                             MAE / peak demand × 100
                             peak demand = 38,043 MW (from test data)
                             System-scale metric showing error as fraction
                             of worst-case grid loading event

  pMAE_net-load_peak (%)     Net Load only
                             Net Load MAE / peak demand × 100
                             Same denominator as above for consistency

Reference values (confirmed from test data, Oct–Dec 2025)
  Peak demand   : 38,043 MW  (38.04 GW)
  Peak net load : 33,988 MW  (33.99 GW)
  Mean load     : 27,334 MW  (27.33 GW)

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
# 0.  CONFIGURATION  — edit only this block
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = (
    r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3"
    r"\Individual Project\ML Models\Macro Table for Training"
)

# k-value to use for all single-k detailed figures.
# Set to whichever k gave the best Net Load MAE in the cross-k comparison.
PRIMARY_K = 15

# Output directory for all figures and tables
FIG_DIR = os.path.join(BASE_DIR, "dissertation_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Cross-k comparison directory
COMPARISON_DIR = os.path.join(BASE_DIR, "cross_k_comparison")

# All k-values evaluated (for cross-k figures)
ALL_K = [8, 10, 12, 15]

# ── Confirmed reference values (from test data, Oct–Dec 2025) ─────────────────
PEAK_DEMAND_MW       = 38_043   # peak demand in test set (MW)
MEAN_LOAD_MW         = 27_334   # mean load in test set (MW)
INSTALLED_WIND_MW    = 32_000   # installed wind capacity 2025 (MW)
INSTALLED_SOLAR_MW   = 47_000   # installed solar capacity 2025 (MW)

# Solar MAPE threshold: observations where actual solar < this value are
# excluded from the MAPE calculation to avoid night-hour division instability.
# 50 MW ≈ 0.1% of 47 GW installed capacity — represents genuine low/no-output.
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

# ── Colour palette ────────────────────────────────────────────────────────────
COL = {
    "wind":     "#1D9E75",   # teal green
    "solar":    "#BA7517",   # amber
    "load":     "#185FA5",   # blue
    "net_load": "#993C1D",   # coral red
    "actual":   "#2C2C2A",   # near-black
    "val":      "#7F77DD",   # purple
}
LABELS = {
    "wind":     "Wind Generation",
    "solar":    "Solar Generation",
    "load":     "Load",
    "net_load": "Net Load",
}
TARGETS = ["wind", "solar", "load", "net_load"]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def actual_col(target: str) -> str:
    """Return the name of the actual (measured) value column for a given target."""
    mapping = {
        "wind":     "WEM_target_national_wind_gen",
        "solar":    "SEM_target_national_solar_gen",
        "load":     "LDM_target_national_demand",
        "net_load": "actual_net_load",
    }
    return mapping[target]


def pred_col(target: str) -> str:
    """Return the name of the predicted value column for a given target."""
    return f"pred_{target}"


def mw_formatter(x, _):
    """Format y-axis tick labels as MW with thousands separator."""
    return f"{x:,.0f}"


def save_figure(fig, filename: str) -> None:
    """Save a figure to the dissertation figures directory and close it."""
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_predictions(k: int, partition: str = "test") -> pd.DataFrame:
    """
    Load test or validation predictions for a given k-value.

    Parameters
    ----------
    k         : int — spatial cluster count
    partition : str — 'test' or 'val'

    Returns
    -------
    pd.DataFrame with time, actual, and predicted columns plus
    derived hour and weekday columns for groupby operations.
    """
    fname = "test_predictions.csv" if partition == "test" else "val_predictions.csv"
    path  = os.path.join(BASE_DIR, f"k={k}", "pipeline_outputs", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    # Derive time-grouping columns used by diurnal and weekly error plots
    df["hour"]    = df["time"].dt.hour
    df["weekday"] = df["time"].dt.dayofweek   # 0 = Monday
    return df


def load_pipeline_metrics(k: int) -> pd.DataFrame:
    """Load the metrics_summary.csv produced by the ML pipeline for a given k."""
    path = os.path.join(BASE_DIR, f"k={k}", "pipeline_outputs", "metrics_summary.csv")
    return pd.read_csv(path)


def load_feature_importances(k: int) -> pd.DataFrame:
    """Load the feature_importances.csv produced by the ML pipeline for a given k."""
    path = os.path.join(BASE_DIR, f"k={k}", "pipeline_outputs", "feature_importances.csv")
    return pd.read_csv(path)


def load_cross_k_comparison() -> pd.DataFrame:
    """Load the cross-k comparison table produced by run_cross_k_comparison()."""
    path = os.path.join(COMPARISON_DIR, "cross_k_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cross-k comparison file not found: {path}")
    return pd.read_csv(path, index_col="k")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  METRICS CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full dissertation metric suite from the test predictions dataframe.
    No retraining required — reads directly from saved prediction CSVs.

    Standard metrics
    ────────────────
    MAE   Mean Absolute Error (MW)
    RMSE  Root Mean Squared Error (MW)
    MAPE  Mean Absolute Percentage Error (%)
    R²    Coefficient of Determination

    Solar MAPE handling
    ───────────────────
    Observations where actual solar < SOLAR_MAPE_THRESH_MW are excluded to
    avoid division instability during night hours when actual generation is
    near zero. The number of excluded observations is printed to the console.

    Normalised MAE metrics
    ──────────────────────
    nMAE_mean-load (%)        Load model only.  MAE / MEAN_LOAD_MW × 100
    cNMAE_wind (%)            Wind model only.  MAE / INSTALLED_WIND_MW × 100
    cNMAE_solar (%)           Solar model only. MAE / INSTALLED_SOLAR_MW × 100
    pMAE_peak-demand (%)      Load model only.  MAE / PEAK_DEMAND_MW × 100
    pMAE_net-load_peak (%)    Net load only.    MAE / PEAK_DEMAND_MW × 100

    Parameters
    ----------
    df : pd.DataFrame — test predictions dataframe from load_predictions()

    Returns
    -------
    pd.DataFrame with one row per target and all metric columns
    """
    rows = []

    for target in TARGETS:
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            print(f"  [WARNING] Column {col_a} not found — skipping {target}")
            continue

        y_true = df[col_a].values
        y_pred = df[col_p].values

        # ── Standard metrics ──────────────────────────────────────────────────
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        # ── MAPE ──────────────────────────────────────────────────────────────
        if target == "solar":
            # Exclude near-zero and night-hour observations from solar MAPE
            mape_mask  = y_true > SOLAR_MAPE_THRESH_MW
            n_excluded = int((~mape_mask).sum())
            n_included = int(mape_mask.sum())
            if n_included > 0:
                mape = np.mean(
                    np.abs((y_true[mape_mask] - y_pred[mape_mask])
                           / y_true[mape_mask])
                ) * 100
            else:
                mape = np.nan
            mape_note = (
                f"Excludes {n_excluded} obs where actual < "
                f"{SOLAR_MAPE_THRESH_MW} MW ({100*n_excluded/len(y_true):.1f}% "
                f"of test set); MAPE computed on {n_included} active-generation obs"
            )
        else:
            # Standard MAPE for non-solar targets, excluding numerical zeros
            mask  = np.abs(y_true) > 1e-6
            mape  = np.mean(
                np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            ) * 100
            mape_note = ""

        # ── Normalised MAE metrics ────────────────────────────────────────────
        # Each metric is only computed for the target it is defined for.
        # All other targets receive NaN, which appears as a blank in the CSV table.

        nMAE_mean_load   = (round(mae / MEAN_LOAD_MW    * 100, 3)
                            if target == "load"     else np.nan)

        cNMAE_wind       = (round(mae / INSTALLED_WIND_MW  * 100, 3)
                            if target == "wind"     else np.nan)

        cNMAE_solar      = (round(mae / INSTALLED_SOLAR_MW * 100, 3)
                            if target == "solar"    else np.nan)

        pMAE_peak_demand = (round(mae / PEAK_DEMAND_MW  * 100, 3)
                            if target == "load"     else np.nan)

        pMAE_net_load    = (round(mae / PEAK_DEMAND_MW  * 100, 3)
                            if target == "net_load" else np.nan)

        rows.append({
            "Target":                    LABELS[target],
            "MAE (MW)":                  round(mae,  2),
            "RMSE (MW)":                 round(rmse, 2),
            "MAPE (%)":                  round(mape, 3) if not np.isnan(mape) else np.nan,
            "R²":                        round(r2,   5),
            "nMAE_mean-load (%)":        nMAE_mean_load,
            "cNMAE_wind (%)":            cNMAE_wind,
            "cNMAE_solar (%)":           cNMAE_solar,
            "pMAE_peak-demand (%)":      pMAE_peak_demand,
            "pMAE_net-load_peak (%)":    pMAE_net_load,
            "_mape_note":                mape_note,   # internal — not exported to tables
        })

    return pd.DataFrame(rows)


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Print the full metrics table to the Spyder console in a readable format."""
    print("\n" + "=" * 72)
    print("  FULL METRICS TABLE  |  Test Set  |  k = 15")
    print(f"  Peak demand        : {PEAK_DEMAND_MW:,} MW  ({PEAK_DEMAND_MW/1000:.2f} GW)")
    print(f"  Mean load          : {MEAN_LOAD_MW:,} MW  ({MEAN_LOAD_MW/1000:.2f} GW)")
    print(f"  Installed wind     : {INSTALLED_WIND_MW:,} MW  ({INSTALLED_WIND_MW/1000:.0f} GW)")
    print(f"  Installed solar    : {INSTALLED_SOLAR_MW:,} MW  ({INSTALLED_SOLAR_MW/1000:.0f} GW)")
    print(f"  Solar MAPE thresh  : {SOLAR_MAPE_THRESH_MW} MW")
    print("=" * 72)

    std_cols = ["Target", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]
    print("\n  Standard metrics:")
    print(metrics_df[std_cols].to_string(index=False))

    norm_cols = [
        "Target",
        "nMAE_mean-load (%)",
        "cNMAE_wind (%)",
        "cNMAE_solar (%)",
        "pMAE_peak-demand (%)",
        "pMAE_net-load_peak (%)",
    ]
    print("\n  Normalised MAE metrics:")
    print(metrics_df[norm_cols].to_string(index=False))

    # Print solar MAPE note
    for _, row in metrics_df.iterrows():
        if row["_mape_note"]:
            print(f"\n  Solar MAPE note: {row['_mape_note']}")

    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, k: int) -> None:
    """
    Figure 01: Four-panel time series of actual vs predicted for the full
    test period (Oct–Dec 2025). Gives examiners an immediate visual sense of
    model tracking quality, seasonal patterns, and systematic biases.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Test Period Forecasts vs Actuals  |  k = {k}  (Oct–Dec 2025, hourly)",
        fontsize=12, y=0.98
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
    save_figure(fig, "fig01_time_series_test.png")


def plot_scatter(df: pd.DataFrame, pipeline_metrics: pd.DataFrame,
                 k: int) -> None:
    """
    Figure 02: Scatter plots of predicted vs actual for all four targets.
    Perfect prediction lies on the 1:1 diagonal. Deviations reveal systematic
    biases — e.g. solar overprediction at night, wind underprediction at high
    output. R² and MAE are annotated on each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"Predicted vs Actual  |  Test Set  |  k = {k}", fontsize=12)
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

        # 1:1 reference line
        mn = min(y_actual.min(), y_predicted.min())
        mx = max(y_actual.max(), y_predicted.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.6, label="1:1 line")

        # Annotate R² and MAE from pipeline metrics
        row = test_metrics[test_metrics["model"].str.contains(target)]
        if not row.empty:
            r2_val  = row["R2"].values[0]
            mae_val = row["MAE"].values[0]
            ax.text(0.04, 0.95,
                    f"R² = {r2_val:.4f}\nMAE = {mae_val:,.0f} MW",
                    transform=ax.transAxes, fontsize=8.5,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="grey"))

        ax.set_xlabel("Actual (MW)")
        ax.set_ylabel("Predicted (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "fig02_scatter_plots.png")


def plot_residuals(df: pd.DataFrame, k: int) -> None:
    """
    Figure 03: Residual (error = predicted - actual) distribution histograms.
    A well-calibrated model shows a near-zero mean residual and approximately
    symmetric distribution. Skew reveals systematic over- or under-prediction.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Residual Distributions  (Predicted − Actual)  |  Test Set  |  k = {k}",
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
        ax.axvline(residuals.mean(), color="red", linewidth=1.2, linestyle="-",
                   label=f"Mean = {residuals.mean():+,.0f} MW")
        ax.set_xlabel("Error (MW)")
        ax.set_ylabel("Frequency")
        ax.set_title(LABELS[target], fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "fig03_residual_histograms.png")


def plot_diurnal_mae(df: pd.DataFrame, k: int) -> None:
    """
    Figure 04: Mean Absolute Error by hour of day.
    This is the most operationally significant figure for the grid stability
    argument. It shows when the forecast is least reliable, which is when
    the system operator is most exposed to frequency regulation risk.

    Expected patterns:
    - Solar: peaks at morning ramp-up (06:00–09:00) and evening ramp-down
    - Wind:  relatively flat, elevated during overnight low-signal periods
    - Load:  peaks at morning and evening demand peaks
    - Net Load: worst during mid-morning solar ramp with coincident demand peak
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Mean Absolute Error by Hour of Day  |  Test Set  |  k = {k}",
        fontsize=11
    )
    hours = np.arange(24)

    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue

        df_copy = df.copy()
        df_copy["abs_err"] = np.abs(df_copy[col_p] - df_copy[col_a])
        diurnal_mae = df_copy.groupby("hour")["abs_err"].mean()

        ax.bar(hours, diurnal_mae.values, color=COL[target], alpha=0.8,
               edgecolor="white", linewidth=0.4)
        ax.axhline(diurnal_mae.mean(), color="black", linestyle="--",
                   linewidth=1, alpha=0.6,
                   label=f"Daily mean = {diurnal_mae.mean():,.0f} MW")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "fig04_diurnal_mae_profile.png")


def plot_diurnal_bias(df: pd.DataFrame, k: int) -> None:
    """
    Figure 05: Mean signed error (bias) by hour of day.
    Positive = model over-predicts. Negative = model under-predicts.
    Systematic bias at specific hours indicates structural model limitations.
    Green bars = over-prediction, red bars = under-prediction.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Mean Bias (Predicted − Actual) by Hour of Day  |  Test Set  |  k = {k}",
        fontsize=11
    )
    hours = np.arange(24)

    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue

        df_copy = df.copy()
        df_copy["bias"] = df_copy[col_p] - df_copy[col_a]
        diurnal_bias = df_copy.groupby("hour")["bias"].mean()

        # Colour bars by sign: model colour for over-prediction, red for under
        bar_colors = [COL[target] if v >= 0 else "#993C1D"
                      for v in diurnal_bias.values]
        ax.bar(hours, diurnal_bias.values, color=bar_colors, alpha=0.8,
               edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Bias (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))

    fig.tight_layout()
    save_figure(fig, "fig05_diurnal_bias_profile.png")


def plot_cross_k(ck: pd.DataFrame) -> None:
    """
    Figures 06-08: Cross-k comparison charts.
    These provide empirical evidence for the spatial granularity argument —
    whether increasing k from 8 to 15 improves accuracy enough to justify
    the additional computational cost and feature space.
    """
    k_vals = ck.index.tolist()
    x      = np.arange(len(k_vals))
    width  = 0.2

    # ── Figure 06: Cross-k MAE comparison ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, target in enumerate(TARGETS):
        col = f"{target}_MAE"
        if col in ck.columns:
            ax.bar(x + i * width, ck[col].values, width,
                   label=LABELS[target], color=COL[target], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"k = {k}" for k in k_vals])
    ax.set_ylabel("Mean Absolute Error (MW)")
    ax.set_title(
        "Test-Set MAE by Spatial Cluster Count (k)\n"
        "Lower MAE = better forecast accuracy"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "fig06_cross_k_mae.png")

    # ── Figure 07: Cross-k R² comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, target in enumerate(TARGETS):
        col = f"{target}_R2"
        if col in ck.columns:
            ax.bar(x + i * width, ck[col].values, width,
                   label=LABELS[target], color=COL[target], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"k = {k}" for k in k_vals])
    ax.set_ylabel("R² (Coefficient of Determination)")
    ax.set_title(
        "Test-Set R² by Spatial Cluster Count (k)\n"
        "Higher R² = greater variance explained"
    )
    r2_cols = [f"{t}_R2" for t in TARGETS if f"{t}_R2" in ck.columns]
    ax.set_ylim([max(0, ck[r2_cols].min().min() - 0.02), 1.0])
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig07_cross_k_r2.png")

    # ── Figure 08: Feature count and runtime vs k ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Computational Cost vs Spatial Resolution", fontsize=11)

    feat_map = {
        "n_wind_features":  ("Wind",  COL["wind"]),
        "n_solar_features": ("Solar", COL["solar"]),
        "n_load_features":  ("Load",  COL["load"]),
    }
    for col, (label, color) in feat_map.items():
        if col in ck.columns:
            ax1.plot(k_vals, ck[col].values, marker="o", linewidth=2,
                     color=color, label=label)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Total input features")
    ax1.set_title("Feature Count vs k")
    ax1.set_xticks(k_vals)
    ax1.legend()

    if "runtime_s" in ck.columns:
        ax2.bar(k_vals, ck["runtime_s"].values / 60, color="#888780",
                alpha=0.8, width=0.8)
        ax2.set_xlabel("Number of clusters (k)")
        ax2.set_ylabel("Pipeline runtime (minutes)")
        ax2.set_title("Training Runtime vs k")
        ax2.set_xticks(k_vals)

    fig.tight_layout()
    save_figure(fig, "fig08_cross_k_cost.png")


def plot_feature_importance(imp_df: pd.DataFrame, k: int) -> None:
    """
    Figures 09-11: Gain-based feature importance for each model.
    Gain = average loss improvement per split — the most interpretable
    importance metric for reporting. These figures validate the physical
    reasoning behind the cross-model feature linkage: if demand lags appear
    high in the wind/solar model importance charts, that empirically confirms
    they were genuinely informative to the trained model.
    """
    model_config = [
        ("WEM", "wind",  "Wind Energy Model",   9),
        ("SEM", "solar", "Solar Energy Model",  10),
        ("LDM", "load",  "Load Demand Model",   11),
    ]
    for model_code, target, model_name, fig_num in model_config:
        sub = (imp_df[imp_df["model"] == model_code]
               .sort_values("gain", ascending=False)
               .head(15))
        if sub.empty:
            print(f"  [SKIP] No importance data found for {model_code}")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(sub)), sub["gain"].values,
                color=COL[target], alpha=0.85)
        ax.set_yticks(range(len(sub)))

        # Strip namespace prefixes from feature names for cleaner labels
        clean_names = [
            f.replace("WEM_", "").replace("SEM_", "").replace("LDM_", "")
            for f in sub["feature"].values
        ]
        ax.set_yticklabels(clean_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(
            f"Top-15 Feature Importances  |  {model_name}  |  k = {k}\n"
            f"Gain = average loss improvement per split using each feature"
        )
        fig.tight_layout()
        save_figure(fig, f"fig{fig_num:02d}_feature_importance_{target}.png")


def plot_ramp_rate(df: pd.DataFrame, k: int) -> None:
    """
    Figures 12-13: Net load ramp rate analysis.
    Ramp rate = change in net load from one hour to the next (MW/h).
    This is the primary metric for frequency stability risk — large ramps
    require fast-response dispatchable generation to balance the grid.
    Accurate ramp prediction reduces the reserve margin REE must maintain.
    """
    col_a = "actual_net_load"
    col_p = "pred_net_load"
    if col_a not in df.columns:
        print("  [SKIP] Net load ramp rate figures — net load columns not found.")
        return

    df_copy = df.copy()
    df_copy["ramp_actual"]      = df_copy[col_a].diff()
    df_copy["ramp_predicted"]   = df_copy[col_p].diff()
    df_copy["ramp_abs_error"]   = np.abs(
        df_copy["ramp_predicted"] - df_copy["ramp_actual"]
    )
    df_copy = df_copy.dropna(subset=["ramp_actual"])

    # ── Figure 12: Ramp rate scatter ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df_copy["ramp_actual"], df_copy["ramp_predicted"],
               color=COL["net_load"], alpha=0.2, s=3, rasterized=True)
    mn = min(df_copy["ramp_actual"].min(), df_copy["ramp_predicted"].min())
    mx = max(df_copy["ramp_actual"].max(), df_copy["ramp_predicted"].max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, alpha=0.7, label="1:1 line")

    ramp_mae  = df_copy["ramp_abs_error"].mean()
    corr_coef = np.corrcoef(df_copy["ramp_actual"],
                             df_copy["ramp_predicted"])[0, 1]
    ax.text(0.04, 0.95,
            f"Ramp MAE = {ramp_mae:,.0f} MW/h\nCorrelation = {corr_coef:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="grey"))
    ax.set_xlabel("Actual Net Load Ramp Rate (MW/h)")
    ax.set_ylabel("Predicted Net Load Ramp Rate (MW/h)")
    ax.set_title(
        f"Net Load Ramp Rate: Predicted vs Actual  |  k = {k}\n"
        f"Accurate ramp prediction reduces frequency regulation exposure"
    )
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(mw_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    fig.tight_layout()
    save_figure(fig, "fig12_ramp_rate_scatter.png")

    # ── Figure 13: Ramp rate error by hour ────────────────────────────────────
    diurnal_ramp_mae = df_copy.groupby("hour")["ramp_abs_error"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(24), diurnal_ramp_mae.values, color=COL["net_load"],
           alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axhline(diurnal_ramp_mae.mean(), color="black", linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Mean = {diurnal_ramp_mae.mean():,.0f} MW/h")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Ramp Rate Error (MW/h)")
    ax.set_title(
        f"Net Load Ramp Rate Forecast Error by Hour  |  k = {k}\n"
        f"Peak error hours indicate when rapid-response reserves are most needed"
    )
    ax.set_xticks(range(0, 24, 3))
    ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig13_ramp_rate_error_profile.png")


def plot_regime_error(df: pd.DataFrame, k: int) -> None:
    """
    Figure 14: MAE broken down by renewable output regime.
    Low / Medium / High = combined wind+solar output terciles.

    High-regime periods carry the greatest curtailment risk for the Spanish
    grid. Showing that model error differs across regimes is a physically
    meaningful finding tied directly to grid stability — if the model is
    less accurate during high renewable output, that is the worst-case
    operational scenario for REE.
    """
    col_wind_actual  = actual_col("wind")
    col_solar_actual = actual_col("solar")
    if col_wind_actual not in df.columns or col_solar_actual not in df.columns:
        print("  [SKIP] Regime error figure — target columns not found.")
        return

    df_copy = df.copy()
    df_copy["combined_renewable"] = (df_copy[col_wind_actual]
                                     + df_copy[col_solar_actual])
    p33 = df_copy["combined_renewable"].quantile(0.33)
    p66 = df_copy["combined_renewable"].quantile(0.66)

    def assign_regime(val):
        if val < p33:   return "Low\n(< 33rd pctile)"
        elif val < p66: return "Medium\n(33–66th pctile)"
        else:           return "High\n(> 66th pctile)"

    df_copy["regime"] = df_copy["combined_renewable"].apply(assign_regime)
    regime_order  = ["Low\n(< 33rd pctile)", "Medium\n(33–66th pctile)",
                     "High\n(> 66th pctile)"]
    regime_colors = ["#9FE1CB", "#5DCAA5", "#0F6E56"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"MAE by Renewable Output Regime  |  k = {k}\n"
        f"Low / Medium / High = combined wind+solar output terciles",
        fontsize=11
    )
    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df_copy.columns:
            ax.set_visible(False)
            continue

        df_copy["abs_err"] = np.abs(df_copy[col_p] - df_copy[col_a])
        mae_by_regime = (df_copy.groupby("regime")["abs_err"]
                                .mean()
                                .reindex(regime_order))

        bars = ax.bar(range(3), mae_by_regime.values, color=regime_colors,
                      alpha=0.85, edgecolor="white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(regime_order, fontsize=8.5)
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))

        # Annotate bar values
        for bar, val in zip(bars, mae_by_regime.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                        f"{val:,.0f}", ha="center", fontsize=8.5)

    fig.tight_layout()
    save_figure(fig, "fig14_generation_regime_error.png")


def plot_weekly_error(df: pd.DataFrame, k: int) -> None:
    """
    Figure 15: MAE by day of week for each target.
    Demand exhibits strong weekday/weekend patterns due to industrial load
    profiles. Wind shows little day-of-week variation. Solar shows subtle
    variation due to cloud climatology. The vertical dashed line separates
    weekdays from the weekend.
    """
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes  = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"MAE by Day of Week  |  Test Set  |  k = {k}", fontsize=11)

    for ax, target in zip(axes.flatten(), TARGETS):
        col_a = actual_col(target)
        col_p = pred_col(target)
        if col_a not in df.columns:
            ax.set_visible(False)
            continue

        df_copy = df.copy()
        df_copy["abs_err"] = np.abs(df_copy[col_p] - df_copy[col_a])
        weekly_mae = df_copy.groupby("weekday")["abs_err"].mean()

        ax.bar(range(7), weekly_mae.values, color=COL[target], alpha=0.85,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_labels)
        ax.set_ylabel("MAE (MW)")
        ax.set_title(LABELS[target], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))
        ax.axvline(4.5, color="grey", linestyle="--", linewidth=1,
                   alpha=0.6, label="Weekday/Weekend")
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "fig15_weekly_error_profile.png")


def plot_val_vs_test(pipeline_metrics: pd.DataFrame, k: int) -> None:
    """
    Figure 16: Side-by-side comparison of validation and test metrics.
    If test metrics are significantly worse than validation metrics, this
    indicates overfitting to the validation set or distribution shift between
    the two periods — which would be a legitimate seasonal effect to discuss
    in the dissertation's conclusions section.
    """
    val_metrics  = pipeline_metrics[pipeline_metrics["partition"] == "val"].copy()
    test_metrics = pipeline_metrics[pipeline_metrics["partition"] == "test"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Validation vs Test Set Performance  |  k = {k}\n"
        f"Val: Jan–Sep 2025  |  Test: Oct–Dec 2025",
        fontsize=11
    )
    for ax, metric, ylabel in zip(axes, ["MAE", "R2"], ["MAE (MW)", "R²"]):
        x     = np.arange(4)
        width = 0.35

        val_values = [
            val_metrics[val_metrics["model"].str.contains(t)][metric].values[0]
            if not val_metrics[val_metrics["model"].str.contains(t)].empty else 0
            for t in TARGETS
        ]
        test_values = [
            test_metrics[test_metrics["model"].str.contains(t)][metric].values[0]
            if not test_metrics[test_metrics["model"].str.contains(t)].empty else 0
            for t in TARGETS
        ]

        ax.bar(x - width/2, val_values,  width, label="Validation",
               color=COL["val"], alpha=0.85)
        ax.bar(x + width/2, test_values, width, label="Test",
               color=COL["actual"], alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[t] for t in TARGETS], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        if metric == "MAE":
            ax.yaxis.set_major_formatter(FuncFormatter(mw_formatter))

    fig.tight_layout()
    save_figure(fig, "fig16_val_vs_test_metrics.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  TABLE EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_report_tables(metrics_df: pd.DataFrame,
                         cross_k_df: pd.DataFrame) -> None:
    """
    Export three clean CSV tables ready for the dissertation.

    table01_standard_metrics.csv
        MAE, RMSE, MAPE, R² for all four targets.
        Copy directly into dissertation as the main results table.

    table02_normalised_metrics.csv
        All five normalised MAE variants.
        NaN values appear as blank cells — correct for a dissertation table
        since each metric only applies to one specific target.

    table03_cross_k.csv
        Cross-k performance and computational cost summary.
        One row per k-value.
    """
    # ── Table 01: Standard metrics ────────────────────────────────────────────
    t1 = metrics_df[["Target", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]].copy()
    t1.to_csv(os.path.join(FIG_DIR, "table01_standard_metrics.csv"), index=False)
    print("  Saved: table01_standard_metrics.csv")

    # ── Table 02: Normalised MAE metrics ─────────────────────────────────────
    norm_cols = [
        "Target",
        "nMAE_mean-load (%)",
        "cNMAE_wind (%)",
        "cNMAE_solar (%)",
        "pMAE_peak-demand (%)",
        "pMAE_net-load_peak (%)",
    ]
    t2 = metrics_df[norm_cols].copy()
    t2.to_csv(os.path.join(FIG_DIR, "table02_normalised_metrics.csv"), index=False)
    print("  Saved: table02_normalised_metrics.csv")

    # ── Table 03: Cross-k comparison ──────────────────────────────────────────
    keep_cols = [c for c in [
        "n_wind_features", "n_solar_features", "n_load_features",
        "runtime_s",
        "wind_MAE",     "solar_MAE",     "load_MAE",     "net_load_MAE",
        "wind_RMSE",    "solar_RMSE",    "load_RMSE",    "net_load_RMSE",
        "wind_R2",      "solar_R2",      "load_R2",      "net_load_R2",
    ] if c in cross_k_df.columns]
    cross_k_df[keep_cols].to_csv(
        os.path.join(FIG_DIR, "table03_cross_k.csv")
    )
    print("  Saved: table03_cross_k.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 72)
    print(f"  PHASE 7: RESULTS ANALYSIS AND FIGURE GENERATION")
    print(f"  Primary k = {PRIMARY_K}  |  Output → {FIG_DIR}")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1] Loading test predictions for k={PRIMARY_K}...")
    test_df = load_predictions(PRIMARY_K, "test")
    print(f"    Test rows loaded : {len(test_df):,}")

    print(f"\n[2] Loading pipeline metrics for k={PRIMARY_K}...")
    pipeline_metrics = load_pipeline_metrics(PRIMARY_K)

    print(f"\n[3] Loading feature importances for k={PRIMARY_K}...")
    imp_df = load_feature_importances(PRIMARY_K)

    print(f"\n[4] Loading cross-k comparison table...")
    try:
        cross_k_df     = load_cross_k_comparison()
        cross_k_loaded = True
        print(f"    Cross-k table loaded : {len(cross_k_df)} k-values")
    except FileNotFoundError as e:
        print(f"    [WARNING] {e}")
        print("    Cross-k figures and table will be skipped.")
        cross_k_loaded = False

    # ── Compute all metrics ───────────────────────────────────────────────────
    print(f"\n[5] Computing full metrics table...")
    metrics_df = compute_all_metrics(test_df)
    print_metrics_table(metrics_df)

    # ── Generate all 16 figures ───────────────────────────────────────────────
    print(f"\n[6] Generating figures...")

    plot_time_series(test_df, PRIMARY_K)
    plot_scatter(test_df, pipeline_metrics, PRIMARY_K)
    plot_residuals(test_df, PRIMARY_K)
    plot_diurnal_mae(test_df, PRIMARY_K)
    plot_diurnal_bias(test_df, PRIMARY_K)

    if cross_k_loaded:
        plot_cross_k(cross_k_df)

    plot_feature_importance(imp_df, PRIMARY_K)
    plot_ramp_rate(test_df, PRIMARY_K)
    plot_regime_error(test_df, PRIMARY_K)
    plot_weekly_error(test_df, PRIMARY_K)
    plot_val_vs_test(pipeline_metrics, PRIMARY_K)

    # ── Export tables ─────────────────────────────────────────────────────────
    print(f"\n[7] Exporting report tables...")
    if cross_k_loaded:
        export_report_tables(metrics_df, cross_k_df)
    else:
        print("    [WARNING] Cross-k table skipped — run Mode A pipeline first.")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_files = len(os.listdir(FIG_DIR))
    print(f"\n{'=' * 72}")
    print(f"  COMPLETE — {n_files} files written to:")
    print(f"  {FIG_DIR}")
    print("=" * 72)