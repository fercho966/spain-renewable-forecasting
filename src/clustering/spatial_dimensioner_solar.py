import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

# Silence the Windows MKL Memory Leak Warning
os.environ["OMP_NUM_THREADS"] = "1"

# ==============================================================================
# PHASE 1: SOLAR PRE-PROCESSING (DO ONCE TO SAVE MEMORY)
# ==============================================================================

print("Loading Solar data...")
base_df = pd.read_parquet(
    r'C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project'
    r'\ML Models\Solar Energy Generation\ULTIMATE_SOLAR_MASTER_by_Cluster.parquet'
)

# Fix Timezone — convert from UTC to naive local time
base_df['time'] = pd.to_datetime(base_df['time'], utc=True).dt.tz_localize(None)

# Rename columns first so all subsequent code uses consistent names
base_df = base_df.rename(columns={
    'cluster_name': 'location_id',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'target_generation_mwh': 'target_national_solar_gen'
})

# ==============================================================================
# DST SAFETY — source level (applied after rename, before groupby):
#
# After stripping timezone info a DST clock-back produces a duplicate naive
# timestamp (e.g. 02:00 appears twice). Deduplicating here prevents the
# duplicate from propagating into every downstream aggregation.
# Must come after rename so 'location_id' and 'time' both exist.
# ==============================================================================
base_df = base_df[~base_df.duplicated(subset=['time', 'location_id'], keep='first')]

# 1. Solar Collinearity Cuts & Feature Drops
cols_to_drop = [
    'direct_radiation_forecast', 'diffuse_radiation_forecast',
    'direct_radiation_actual', 'diffuse_radiation_actual',
    'cloud_cover_low_forecast', 'cloud_cover_mid_forecast', 'cloud_cover_high_forecast',
    'cloud_cover_low_actual', 'cloud_cover_mid_actual', 'cloud_cover_high_actual'
]
base_df = base_df.drop(columns=[c for c in cols_to_drop if c in base_df.columns])

# 2. Engineering Flex: Photovoltaic Thermal Degradation Penalty (Threshold: 25°C)
base_df['pv_thermal_penalty_forecast'] = np.maximum(base_df['temperature_2m_forecast'] - 25.0, 0)
base_df['pv_thermal_penalty_actual'] = np.maximum(base_df['temperature_2m_actual'] - 25.0, 0)

# 3. Spherical Scaling
mean_lat_rad = np.radians(base_df['latitude'].mean())
base_df['scaled_longitude'] = base_df['longitude'] * np.cos(mean_lat_rad)

# 4. Pre-calculate Global Targets, Lags & Cyclical Time
print("Building Global Autoregressive Targets & Time Vectors...")
target_df = base_df.groupby('time')['target_national_solar_gen'].max().reset_index()
target_df = target_df.set_index('time')

# ==============================================================================
# DST SAFETY — target_df level (Steps 1-3):
#
# Step 1: Drop any remaining index duplicates (belt-and-braces).
# Step 2: asfreq('h') enforces a gapless hourly grid. Any clock-back hour
#         removed by Step 1 becomes a NaN row here.
# Step 3: ffill(limit=1) fills that single missing hour with the previous
#         hour's value. Without this, solar_gen_rollmean_168 (window=168)
#         propagates the 1-hour NaN across 168 rows, and dropna() silently
#         removes 7 days + 1 hour of data.
#         limit=1 ensures only genuine 1-hour DST gaps are filled; real
#         multi-hour outages remain NaN and are caught by dropna().
# ==============================================================================
target_df = target_df[~target_df.index.duplicated(keep='first')]
target_df = target_df.asfreq('h')
target_df = target_df.ffill(limit=1)

# Cyclical Time Vectors (computed after asfreq so the index is clean)
target_df['hour_sin'] = np.sin(2 * np.pi * target_df.index.hour / 24.0)
target_df['hour_cos'] = np.cos(2 * np.pi * target_df.index.hour / 24.0)
target_df['dow_sin'] = np.sin(2 * np.pi * target_df.index.dayofweek / 7.0)
target_df['dow_cos'] = np.cos(2 * np.pi * target_df.index.dayofweek / 7.0)
target_df['month_sin'] = np.sin(2 * np.pi * target_df.index.month / 12.0)
target_df['month_cos'] = np.cos(2 * np.pi * target_df.index.month / 12.0)

# Solar Power Lags — freq= makes these clock-time-aware, not positional
target_df['solar_gen_lag_24'] = target_df['target_national_solar_gen'].shift(freq='24h')
target_df['solar_gen_lag_168'] = target_df['target_national_solar_gen'].shift(freq='168h')

# Rolling windows are positional, but the index is now a clean regular grid
# so each position is exactly one hour — safe to use window sizes directly.
target_df['solar_gen_rollmean_24'] = target_df['solar_gen_lag_24'].rolling(window=24).mean()
target_df['solar_gen_rollmean_168'] = target_df['solar_gen_lag_24'].rolling(window=168).mean()


# ==============================================================================
# PHASE 2: SPATIAL HYPERPARAMETER LOOP (k = 8, 10, 12, 15)
# ==============================================================================

exclude_cols = [
    'time', 'location_id', 'latitude', 'longitude', 'total_capacity_mw',
    'scaled_longitude', 'region_id', 'target_national_solar_gen'
]
weather_features = [c for c in base_df.columns if c not in exclude_cols]

# Extract unique Solar Farms for K-Means
farms = base_df.groupby('location_id').agg({
    'latitude': 'first', 'scaled_longitude': 'first', 'total_capacity_mw': 'max'
}).reset_index()

k_values = [8, 10, 12, 15]

for k in k_values:
    print(f"\n--- Processing k={k} Solar Regions ---")

    # 1. Capacity-Weighted K-Means Clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    farms['region_id'] = kmeans.fit_predict(
        farms[['latitude', 'scaled_longitude']],
        sample_weight=farms['total_capacity_mw']
    )

    df_mapped = base_df.merge(farms[['location_id', 'region_id']], on='location_id', how='left')

    # -----------------------------------------------------------------------
    # 2. CAPACITY-WEIGHTED WEATHER AGGREGATION
    # -----------------------------------------------------------------------
    features_to_weigh = [c for c in weather_features]

    for feat in features_to_weigh:
        df_mapped[f'{feat}_weighted'] = df_mapped[feat] * df_mapped['total_capacity_mw']

    sum_dict = {f'{feat}_weighted': 'sum' for feat in features_to_weigh}
    sum_dict['total_capacity_mw'] = 'sum'

    # Unweighted dispersion (std) only for irradiance/radiation to detect turbulence
    irradiance_cols = [c for c in features_to_weigh if 'radiation' in c or 'irradiance' in c]
    for irr_feat in irradiance_cols:
        sum_dict[irr_feat] = 'std'

    regional_sums = df_mapped.groupby(['time', 'region_id']).agg(sum_dict).reset_index()

    regional_data = pd.DataFrame()
    regional_data['time'] = regional_sums['time']
    regional_data['region_id'] = regional_sums['region_id']
    regional_data['total_capacity_mw_sum'] = regional_sums['total_capacity_mw']

    for feat in features_to_weigh:
        regional_data[f'{feat}_mean'] = (
            regional_sums[f'{feat}_weighted'] / regional_sums['total_capacity_mw']
        )
        if feat in irradiance_cols:
            regional_data[f'{feat}_std'] = regional_sums[feat]

    del df_mapped
    # -----------------------------------------------------------------------

    # 3. The Wide Pivot
    pivot_df = regional_data.pivot(index='time', columns='region_id')
    pivot_df.columns = [f"reg_{reg}_{feat}" for feat, reg in pivot_df.columns]
    pivot_df = pivot_df.sort_index()

    # ==============================================================================
    # DST SAFETY (regional pivot): same three-step treatment as target_df.
    # The ffill(limit=1) prevents the single DST NaN from cascading through
    # the actual_cols shift and into the final join.
    # ==============================================================================
    pivot_df = pivot_df[~pivot_df.index.duplicated(keep='first')]
    pivot_df = pivot_df.asfreq('h')
    pivot_df = pivot_df.ffill(limit=1)

    # 4. Regional Lags — clock-time-aware shift, not positional
    actual_cols = [c for c in pivot_df.columns if 'actual' in c]
    pivot_df[actual_cols] = pivot_df[actual_cols].shift(freq='24h')

    rename_mapping = {c: c.replace('_actual_', '_actual_lag_24h_') for c in actual_cols}
    pivot_df = pivot_df.rename(columns=rename_mapping)

    # 5. Join with Targets & Drop NaNs
    final_ml_dataset = pivot_df.join(target_df, how='inner')
    final_ml_dataset = final_ml_dataset.dropna()

    # --- Strict Column Organization ---
    time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

    reg_cols = [c for c in final_ml_dataset.columns if c.startswith('reg_')]
    reg_cols.sort(key=lambda x: (int(x.split('_')[1]), x))

    global_cols = [
        'solar_gen_lag_24', 'solar_gen_lag_168',
        'solar_gen_rollmean_24', 'solar_gen_rollmean_168'
    ]
    target_col = ['target_national_solar_gen']

    final_ml_dataset = final_ml_dataset[time_cols + reg_cols + global_cols + target_col]

    # 6. DATA SAFETY VALIDATION BLOCK
    print("Running Data Safety Diagnostics...")
    nan_count = final_ml_dataset.isna().sum().sum()
    assert nan_count == 0, f"DATA SAFETY ALERT: Found {nan_count} NaN values!"
    inf_count = np.isinf(final_ml_dataset.select_dtypes(include=np.number)).values.sum()
    assert inf_count == 0, f"DATA SAFETY ALERT: Found {inf_count} Infinite values!"
    assert final_ml_dataset.index.is_monotonic_increasing, \
        "DATA SAFETY ALERT: Time index is not perfectly chronological!"

    # DST CONTINUITY CHECK — any gap > 1h means real missing data worth investigating
    time_diffs = pd.Series(final_ml_dataset.index).diff().dropna()
    max_gap = time_diffs.max()
    if max_gap > pd.Timedelta('1h'):
        gap_locs = time_diffs[time_diffs > pd.Timedelta('1h')].index
        gap_starts = final_ml_dataset.index[gap_locs]
        print(f"  WARNING: Largest time gap = {max_gap}. Starts at: {gap_starts.tolist()}")
    else:
        print("  Continuity check PASSED: no unexpected time gaps detected.")

    print(
        f"Safety Checks PASSED! Shape: {final_ml_dataset.shape} "
        f"| Columns: {len(final_ml_dataset.columns)}"
    )

    # 7. EXPORT
    parquet_filename = f'SOLAR_PILLAR_{k}_REGIONS.parquet'
    csv_filename = f'SOLAR_PILLAR_{k}_REGIONS.csv'

    final_ml_dataset.to_parquet(parquet_filename)
    final_ml_dataset.to_csv(csv_filename)

    print(f"Saved -> {parquet_filename}")
    print(f"Saved -> {csv_filename}")

print("\nPipeline Complete. Solar Pillar successfully engineered and verified.")