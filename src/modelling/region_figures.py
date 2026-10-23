import matplotlib
matplotlib.use('Agg')  # File output only — nothing appears in Spyder plots pane

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SPAIN MAINLAND POLYGON
# Traced clockwise from NW. Accurate enough for a regional overview map.
# No external library required.
# ==============================================================================
SPAIN_POLY = np.array([
    # Galicia — north coast
    (-8.87, 43.75), (-8.45, 43.78), (-8.03, 43.72), (-7.68, 43.78),
    (-7.30, 43.68), (-6.85, 43.57), (-6.30, 43.57), (-5.87, 43.67),
    (-5.35, 43.58), (-4.82, 43.41), (-4.20, 43.42), (-3.65, 43.52),
    (-3.25, 43.44), (-2.67, 43.31), (-1.98, 43.33), (-1.79, 43.37),
    # Pyrenees — France border
    (-1.50, 43.05), (-0.78, 42.87), (-0.14, 42.71),
    ( 0.67, 42.69), ( 1.42, 42.50), ( 1.72, 42.50),
    ( 2.02, 42.40), ( 2.55, 42.47), ( 3.17, 42.43),
    # East coast — Med, going south
    ( 3.31, 41.85), ( 3.13, 41.31), ( 2.19, 41.43),
    ( 1.52, 41.21), ( 0.87, 40.77), ( 0.23, 40.53),
    ( 0.05, 40.11), (-0.19, 39.79), (-0.38, 39.47),
    (-0.52, 38.97), (-0.37, 38.63), (-0.39, 38.30),
    (-0.58, 38.16), (-0.71, 37.63),
    # SE corner and south coast
    (-1.30, 37.56), (-1.59, 37.37), (-1.83, 36.74),
    (-2.11, 36.71), (-2.44, 36.73), (-2.95, 36.69),
    (-3.44, 36.68), (-4.01, 36.69), (-4.53, 36.70),
    (-5.01, 36.10), (-5.35, 36.01), (-5.60, 36.02),
    # Gibraltar / Trafalgar / Huelva — Atlantic south
    (-5.92, 36.19), (-6.19, 36.38), (-6.47, 36.60),
    (-6.72, 37.01), (-7.08, 37.03), (-7.41, 37.17),
    # Portugal border — north
    (-7.41, 37.62), (-7.28, 38.01), (-7.08, 38.18),
    (-7.04, 38.53), (-7.18, 38.90), (-7.08, 39.12),
    (-6.97, 39.47), (-6.99, 39.73), (-6.85, 40.18),
    (-6.60, 40.56), (-6.53, 41.03), (-6.68, 41.26),
    (-6.85, 41.55), (-6.55, 41.72), (-6.36, 41.90),
    (-6.62, 41.94), (-6.93, 41.97), (-7.40, 41.99),
    (-7.94, 42.05), (-8.20, 42.10), (-8.62, 42.06),
    (-8.87, 41.86), (-9.05, 41.75),
    # Galicia west coast back to start
    (-9.29, 42.17), (-9.29, 42.89), (-8.87, 43.75),
])

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = (r"C:\Users\ferna\OneDrive - University of Exeter"
            r"\MechEng 3\Individual Project\ML Models")

PATHS = {
    "Wind":  BASE_DIR + r"\Wind Energy Generation\ULTIMATE_WIND_MASTER_ByLocation.parquet",
    "Solar": BASE_DIR + r"\Solar Energy Generation\ULTIMATE_SOLAR_MASTER_by_Cluster.parquet",
    "Load":  BASE_DIR + r"\Load Prediction Model\master_training_dataset_location_sorted.parquet",
}
CONFIG = {
    "Wind":  {"loc": "GEM location ID", "weight": "active_capacity"},
    "Solar": {"loc": "cluster_name",    "weight": "total_capacity_mw"},
    "Load":  {"loc": "city",            "weight": "dynamic_population"},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
K            = 15
RANDOM_STATE = 42
PALETTE = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#6A4C93",
    "#F72585", "#4CC9F0", "#80B918", "#FFBE0B", "#FB5607",
    "#3A86FF", "#8338EC", "#06D6A0", "#118AB2", "#EF476F",
]
LON_MIN, LON_MAX = -9.5,  4.5
LAT_MIN, LAT_MAX = 35.8, 44.0

# ==============================================================================
# HELPERS
# ==============================================================================

def find_col(df, candidates):
    col_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")


def cluster_locations(path, cfg, k=15):
    df       = pd.read_parquet(path)
    lat_col  = find_col(df, ["latitude", "Latitude"])
    lon_col  = find_col(df, ["longitude", "Longitude"])
    wgt_col  = find_col(df, [cfg["weight"]])
    loc_col  = cfg["loc"]

    locs = df.groupby(loc_col).agg(
        lat    = (lat_col, "first"),
        lon    = (lon_col, "first"),
        weight = (wgt_col, "mean"),
    ).reset_index()

    # Clip to peninsula
    locs = locs[
        (locs["lon"] >= LON_MIN) & (locs["lon"] <= LON_MAX) &
        (locs["lat"] >= LAT_MIN) & (locs["lat"] <= LAT_MAX)
    ].copy()

    # Spherical longitude scaling — identical to pipelines
    mean_lat_rad = np.radians(locs["lat"].mean())
    locs["scaled_lon"] = locs["lon"] * np.cos(mean_lat_rad)

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    locs["region"] = km.fit_predict(
        locs[["lat", "scaled_lon"]],
        sample_weight=locs["weight"]
    )
    # NOTE: centroids returned here are only used as fallback.
    # The true visual label positions are computed from the meshgrid after
    # KMeans prediction, so they sit at the centre of the coloured area,
    # not the mean of the sparse location points.
    centroids = locs.groupby("region")[["lat", "lon"]].mean()
    return locs, centroids, km, mean_lat_rad


def make_spain_clip(ax):
    """Return a PathPatch that clips artwork to the Spain polygon."""
    codes = ([Path.MOVETO] +
             [Path.LINETO] * (len(SPAIN_POLY) - 2) +
             [Path.CLOSEPOLY])
    path  = Path(SPAIN_POLY, codes)
    patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    return patch


# ==============================================================================
# PLOTTING — one PNG per model
# ==============================================================================

model_configs = [
    ("Wind",  "WEM — Wind Energy Model",        "^",  65, "wind_k15_regions.png"),
    ("Solar", "SEM — Solar Energy Model",        "D",  55, "solar_k15_regions.png"),
    ("Load",  "LDM — Electricity Demand Model",  "o",  40, "load_k15_regions.png"),
]

cmap = mcolors.ListedColormap(PALETTE[:K])
norm = mcolors.BoundaryNorm(np.arange(-0.5, K + 0.5, 1), cmap.N)

# Fine meshgrid for region fill + border contours
RES      = 600
lon_grid = np.linspace(LON_MIN, LON_MAX, RES)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, RES)
LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)

for key, subtitle, marker, msize, fname in model_configs:
    print(f"Generating {fname} ...")

    locs, centroids, km, mean_lat_rad = cluster_locations(
        PATHS[key], CONFIG[key], k=K
    )

    # Predict region for every grid cell
    SCALED_LON_G = LON_G * np.cos(mean_lat_rad)
    grid_pts     = np.column_stack([LAT_G.ravel(), SCALED_LON_G.ravel()])
    GRID_LABELS  = km.predict(grid_pts).reshape(LON_G.shape).astype(float)

    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("#C8DCEC")   # sea colour

    # 1. Clip patch (invisible — just used as clip_path below)
    clip = make_spain_clip(ax)

    # 2. Filled region colours — clipped to Spain outline
    cf = ax.contourf(
        LON_G, LAT_G, GRID_LABELS,
        levels=np.arange(-0.5, K + 0.5, 1),
        colors=PALETTE[:K],
        alpha=0.72,
        zorder=2,
    )
    cf.set_clip_path(clip)

    # 3. Region boundary lines (white) — clipped to Spain outline
    cb = ax.contour(
        LON_G, LAT_G, GRID_LABELS,
        levels=np.arange(0.5, K - 0.5, 1),
        colors='white',
        linewidths=1.2,
        zorder=3,
    )
    cb.set_clip_path(clip)

    # 4. Spain coastline / border drawn on top
    ax.plot(
        SPAIN_POLY[:, 0], SPAIN_POLY[:, 1],
        color="#1a1a1a", linewidth=1.6, zorder=5,
        solid_capstyle='round', solid_joinstyle='round'
    )
    # Close the polygon visually
    ax.plot(
        [SPAIN_POLY[-1, 0], SPAIN_POLY[0, 0]],
        [SPAIN_POLY[-1, 1], SPAIN_POLY[0, 1]],
        color="#1a1a1a", linewidth=1.6, zorder=5
    )

    # 5. Location dots
    colours = [PALETTE[int(r) % K] for r in locs["region"]]
    ax.scatter(
        locs["lon"], locs["lat"],
        c=colours, s=msize, marker=marker,
        edgecolors="white", linewidths=0.6,
        zorder=6, alpha=0.95,
    )

    # 6. Region number labels — positioned at the visual centre of each
    #    coloured area on the map, not the mean of the sparse location points.
    #    We compute the median lon/lat of all meshgrid cells assigned to each
    #    region. The median is more robust than the mean for irregular shapes.
    spain_path = Path(SPAIN_POLY,
                      [Path.MOVETO] + [Path.LINETO] * (len(SPAIN_POLY) - 2) + [Path.CLOSEPOLY])
    grid_lons  = LON_G.ravel()
    grid_lats  = LAT_G.ravel()
    grid_regs  = GRID_LABELS.ravel().astype(int)

    # Only use grid cells that fall inside the Spain polygon
    inside_mask = spain_path.contains_points(
        np.column_stack([grid_lons, grid_lats])
    )

    for reg in range(K):
        mask = inside_mask & (grid_regs == reg)
        if mask.sum() == 0:
            continue
        label_lon = np.median(grid_lons[mask])
        label_lat = np.median(grid_lats[mask])
        ax.text(
            label_lon, label_lat, str(reg),
            fontsize=13, fontweight="bold",
            ha="center", va="center", color="white",
            bbox=dict(
                boxstyle="circle,pad=0.35",
                facecolor=PALETTE[reg % K],
                edgecolor="white", linewidth=1.3, alpha=0.95,
            ),
            zorder=7,
        )

    # 7. Legend removed — region numbers shown directly on map as labels

    # 8. Axes
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude", fontsize=10, color="#444444")
    ax.set_ylabel("Latitude",  fontsize=10, color="#444444")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.grid(True, linestyle="--", linewidth=0.3, color="#AAAAAA", alpha=0.5, zorder=1)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Spain — k = 15 Spatial Regions\n{subtitle}",
        fontsize=13, fontweight="500", pad=10, color="#1a1a1a"
    )

    # 9. Save
    out_path = os.path.join(SCRIPT_DIR, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

print("\nDone. Three PNG files saved to:", SCRIPT_DIR)