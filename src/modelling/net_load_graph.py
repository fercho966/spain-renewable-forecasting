"""
================================================================================
FIGURE: ACTUAL VS FORECASTED NET LOAD BY HOUR OF DAY
Chronological split  |  k = 15  |  Test set (Oct-Dec 2025)
================================================================================
Plots the mean actual net load and mean predicted net load for each hour of
the day, with shaded bands showing ± one standard deviation around each mean.

This figure shows:
  - The typical daily net load profile the model is trying to capture
  - How closely the forecast tracks the actual diurnal shape hour by hour
  - Which hours of the day carry the largest systematic forecast bias
  - The spread (uncertainty) of actual and predicted values around the mean

Saved to the same dissertation_figures folder as all phase7 figures.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = (
    r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3"
    r"\Individual Project\ML Models\Macro Table for Training"
)

PRIMARY_K = 15
FIG_DIR   = os.path.join(BASE_DIR, "dissertation_figures")

# ── Plot style (matching results_analysis_phase7.py) ─────────────────────────
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

COL_ACTUAL    = "#2C2C2A"   # near-black  (matching COL["actual"] in phase7)
COL_PREDICTED = "#993C1D"   # coral red   (matching COL["net_load"] in phase7)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

path = os.path.join(
    BASE_DIR, f"k={PRIMARY_K}", "pipeline_outputs", "test_predictions.csv"
)
df = pd.read_csv(path, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df["hour"] = df["time"].dt.hour

col_actual    = "actual_net_load"
col_predicted = "pred_net_load"

# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE HOURLY STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

# For each hour of the day compute mean and standard deviation
# across all days in the test set
hourly_actual_mean = df.groupby("hour")[col_actual].mean()
hourly_actual_std  = df.groupby("hour")[col_actual].std()
hourly_pred_mean   = df.groupby("hour")[col_predicted].mean()
hourly_pred_std    = df.groupby("hour")[col_predicted].std()

hours = np.arange(24)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))

# ── Mean lines ────────────────────────────────────────────────────────────────
ax.plot(
    hours,
    hourly_actual_mean.values,
    color=COL_ACTUAL,
    linewidth=2.0,
    marker="o",
    markersize=4,
    label="Actual Net Load (mean)",
    zorder=3,
)
ax.plot(
    hours,
    hourly_pred_mean.values,
    color=COL_PREDICTED,
    linewidth=2.0,
    marker="s",
    markersize=4,
    linestyle="--",
    label="Predicted Net Load (mean)",
    zorder=3,
)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel("Hour of Day (Europe/Madrid)")
ax.set_ylabel("Net Load (MW)")
ax.set_title(
    f"Actual vs Forecasted Net Load by Hour of Day  |  "
    f"k = {PRIMARY_K}  |  Test Set (Oct–Dec 2025)"
)
ax.set_xticks(range(0, 24, 1))
ax.set_xticklabels(
    [f"{h:02d}:00" for h in range(24)],
    rotation=45,
    ha="right",
    fontsize=8,
)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", framealpha=0.8)

# Annotate the peak error hour
hourly_bias     = hourly_pred_mean - hourly_actual_mean
peak_bias_hour  = int(hourly_bias.abs().idxmax())
peak_bias_value = hourly_bias.iloc[peak_bias_hour]
ax.annotate(
    f"Largest error\n{peak_bias_value:+,.0f} MW",
    xy=(peak_bias_hour, hourly_pred_mean.iloc[peak_bias_hour]),
    xytext=(peak_bias_hour + 1.5,
            hourly_pred_mean.iloc[peak_bias_hour] + 300),
    fontsize=8,
    color=COL_PREDICTED,
    arrowprops=dict(arrowstyle="->", color=COL_PREDICTED, lw=1.2),
)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(FIG_DIR, "fig17_diurnal_net_load_profile.png")
fig.savefig(out_path)
plt.close(fig)
print(f"Saved: {out_path}")

# ── Print summary statistics to console ───────────────────────────────────────
print("\nHourly mean net load — actual vs predicted (MW):")
print(f"  {'Hour':<6} {'Actual mean':>13} {'Pred mean':>12} {'Bias':>10}")
print(f"  {'-'*45}")
for h in hours:
    bias = hourly_pred_mean.iloc[h] - hourly_actual_mean.iloc[h]
    print(
        f"  {h:02d}:00  "
        f"{hourly_actual_mean.iloc[h]:>13,.0f} "
        f"{hourly_pred_mean.iloc[h]:>12,.0f} "
        f"{bias:>+10,.0f}"
    )
print(f"\nMean absolute hourly bias : {hourly_bias.abs().mean():,.0f} MW")
print(f"Peak bias hour            : {peak_bias_hour:02d}:00  "
      f"({peak_bias_value:+,.0f} MW)")