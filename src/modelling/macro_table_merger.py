import pandas as pd
import pyarrow.parquet as pq
import os


def build_macro_dataset(wind_path, solar_path, load_path, output_parquet, output_csv, k_val):
    print(f"\n[{k_val}] Initializing Macro-Merge Pipeline for k={k_val}...")

    # 1. Load the three datasets
    print(f"[{k_val}] Loading Parquet files into memory...")
    try:
        df_wind  = pd.read_parquet(wind_path)
        df_solar = pd.read_parquet(solar_path)
        df_load  = pd.read_parquet(load_path)

        # --- PARQUET INDEX NORMALISATION ---
        # The pillar scripts save with a DatetimeIndex named 'time'.
        # After pd.read_parquet the index is restored with that name.
        # reset_index() brings it back as a regular column called 'time'.
        # We do this explicitly rather than checking column names, so the
        # behaviour is always predictable regardless of how the file was saved.
        def enforce_time_col(df):
            if 'time' not in df.columns:
                df = df.reset_index()
                # After reset_index the former index uses whatever name it had.
                # Rename it to 'time' if it came back under a different label.
                if 'time' not in df.columns:
                    # The index was unnamed — pandas calls it 'index'
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'time'})
                    else:
                        raise ValueError(
                            "Could not locate a 'time' column after reset_index(). "
                            "Check that the pillar parquet files have a DatetimeIndex named 'time'."
                        )
            return df

        df_wind  = enforce_time_col(df_wind)
        df_solar = enforce_time_col(df_solar)
        df_load  = enforce_time_col(df_load)

    except FileNotFoundError as e:
        print(f"[{k_val}] ERROR: Could not find file. Please check naming convention: {e}")
        return None

    # 2. Define global temporal and calendar columns
    global_cols = [
        'time', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'is_holiday', 'is_non_working_day'
    ]

    # 3. Prefixing Function
    def prefix_dataframe(df, prefix, global_columns):
        df = df.set_index('time')
        new_cols = {col: f"{prefix}_{col}" for col in df.columns if col not in global_columns}
        df = df.rename(columns=new_cols)
        cols_to_drop = [c for c in global_columns if c in df.columns and c != 'time']
        df = df.drop(columns=cols_to_drop)
        return df

    print(f"[{k_val}] Applying physical prefixes (SEM_, WEM_, LDM_)...")
    df_solar_clean = prefix_dataframe(df_solar, "SEM", global_cols)
    df_wind_clean  = prefix_dataframe(df_wind,  "WEM", global_cols)

    df_load = df_load.set_index('time')
    new_load_cols = {col: f"LDM_{col}" for col in df_load.columns if col not in global_cols}
    df_load_clean = df_load.rename(columns=new_load_cols)

    # 4. The Master Merge
    print(f"[{k_val}] Executing strict inner join on temporal index...")
    macro_df = df_load_clean.join(df_wind_clean, how='inner').join(df_solar_clean, how='inner')
    macro_df = macro_df.reset_index()

    # 5. Engineering Safety Checks
    total_nans = macro_df.isna().sum().sum()
    print(f"[{k_val}] Merge Complete. Dimensions: {macro_df.shape[0]} rows x {macro_df.shape[1]} columns")
    if total_nans == 0:
        print(f"[{k_val}] Integrity Check Passed: 0 NaNs. Data is clean.")
    else:
        print(f"[{k_val}] WARNING: {total_nans} NaNs introduced during merge. Check temporal alignment.")

    # Continuity check — flag any unexpected time gaps in the merged output
    time_diffs = macro_df['time'].diff().dropna()
    max_gap = time_diffs.max()
    if max_gap > pd.Timedelta('1h'):
        print(f"[{k_val}] WARNING: Largest time gap in merged output = {max_gap}.")
    else:
        print(f"[{k_val}] Continuity Check Passed: no unexpected time gaps.")

    # 6. Create output directory if it does not already exist
    # FIX: the script previously assumed these folders existed. If they don't,
    # to_parquet() and to_csv() raise FileNotFoundError. os.makedirs with
    # exist_ok=True creates the folder (and any parents) only when needed.
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

    # 7. Export to Parquet and CSV
    print(f"[{k_val}] Exporting to Parquet and CSV...")
    macro_df.to_parquet(output_parquet, engine='pyarrow', compression='snappy', index=False)
    macro_df.to_csv(output_csv, index=False)

    print(f"[{k_val}] Pipeline complete for k={k_val}.")
    return macro_df


# ==============================================================================
# --- Execution Block & File Paths Setup ---
# ==============================================================================

# Data Input Directories
base_dir  = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models"
wind_dir  = os.path.join(base_dir, r"Wind Energy Generation\Spatially Dimensioned Data Table")
solar_dir = os.path.join(base_dir, r"Solar Energy Generation\Spatially Dimensioned Data Table")
load_dir  = os.path.join(base_dir, r"Load Prediction Model\Spatially Dimensioned Data Table Load")

# Output anchored to the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# EXECUTION: k = 8
# ---------------------------------------------------------
out_dir_k8 = os.path.join(script_dir, "k=8")
build_macro_dataset(
    wind_path       = os.path.join(wind_dir,  "WIND_PILLAR_8_REGIONS.parquet"),
    solar_path      = os.path.join(solar_dir, "SOLAR_PILLAR_8_REGIONS.parquet"),
    load_path       = os.path.join(load_dir,  "DEMAND_PILLAR_8_REGIONS.parquet"),
    output_parquet  = os.path.join(out_dir_k8, "macro_master_dataset_k8.parquet"),
    output_csv      = os.path.join(out_dir_k8, "macro_master_dataset_k8.csv"),
    k_val           = "8"
)

# ---------------------------------------------------------
# EXECUTION: k = 10
# ---------------------------------------------------------
out_dir_k10 = os.path.join(script_dir, "k=10")
build_macro_dataset(
    wind_path       = os.path.join(wind_dir,  "WIND_PILLAR_10_REGIONS.parquet"),
    solar_path      = os.path.join(solar_dir, "SOLAR_PILLAR_10_REGIONS.parquet"),
    load_path       = os.path.join(load_dir,  "DEMAND_PILLAR_10_REGIONS.parquet"),
    output_parquet  = os.path.join(out_dir_k10, "macro_master_dataset_k10.parquet"),
    output_csv      = os.path.join(out_dir_k10, "macro_master_dataset_k10.csv"),
    k_val           = "10"
)

# ---------------------------------------------------------
# EXECUTION: k = 12
# ---------------------------------------------------------
out_dir_k12 = os.path.join(script_dir, "k=12")
build_macro_dataset(
    wind_path       = os.path.join(wind_dir,  "WIND_PILLAR_12_REGIONS.parquet"),
    solar_path      = os.path.join(solar_dir, "SOLAR_PILLAR_12_REGIONS.parquet"),
    load_path       = os.path.join(load_dir,  "DEMAND_PILLAR_12_REGIONS.parquet"),
    output_parquet  = os.path.join(out_dir_k12, "macro_master_dataset_k12.parquet"),
    output_csv      = os.path.join(out_dir_k12, "macro_master_dataset_k12.csv"),
    k_val           = "12"
)

# ---------------------------------------------------------
# EXECUTION: k = 15
# ---------------------------------------------------------
out_dir_k15 = os.path.join(script_dir, "k=15")
build_macro_dataset(
    wind_path       = os.path.join(wind_dir,  "WIND_PILLAR_15_REGIONS.parquet"),
    solar_path      = os.path.join(solar_dir, "SOLAR_PILLAR_15_REGIONS.parquet"),
    load_path       = os.path.join(load_dir,  "DEMAND_PILLAR_15_REGIONS.parquet"),
    output_parquet  = os.path.join(out_dir_k15, "macro_master_dataset_k15.parquet"),
    output_csv      = os.path.join(out_dir_k15, "macro_master_dataset_k15.csv"),
    k_val           = "15"
)

print("\nAll spatial resolutions successfully merged and logically routed to their respective script directories.")