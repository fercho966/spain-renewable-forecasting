import pandas as pd
from pathlib import Path
import gc
import os

# ==========================================
# CONFIGURATION
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation")

# Update these to match the exact locations of your files
ACTUALS_FILE = BASE_DIR / "Actual Weather Data" / "solar_actual_weather_master.parquet"
FORECAST_FILE = BASE_DIR / "Weather Forecast Data" / "solar_forecast_weather_master.parquet"
TARGET_FILE = BASE_DIR / "Outputs" / "esios_solar_pv_generation_hourly_2022_2025.parquet"
CAPACITY_FILE = BASE_DIR / "Coordinates" / "processed_data" / "solar_api_extraction_targets.csv"

# Final Output Paths 
OUTPUT_BY_CLUSTER = SCRIPT_DIR / "ULTIMATE_SOLAR_MASTER_by_Cluster.parquet"
OUTPUT_BY_TIME = SCRIPT_DIR / "ULTIMATE_SOLAR_MASTER_by_Time.parquet"

def build_ultimate_solar_master():
    print("==================================================")
    print("ASSEMBLING THE ULTIMATE SOLAR ENERGY DATASET")
    print("==================================================")
    
    # ---------------------------------------------------------
    # STEP 1: LOAD ESIOS TARGET DATA
    # ---------------------------------------------------------
    print("1. Loading the ESIOS Peninsular Generation data...")
    if not TARGET_FILE.exists():
        print(f"[ERROR] Cannot find Target file at:\n{TARGET_FILE}")
        return
        
    df_gen = pd.read_parquet(TARGET_FILE)
    df_gen['time'] = pd.to_datetime(df_gen['time'], utc=True)
    print(f"   -> Found {len(df_gen):,} hours of generation targets.")

    # ---------------------------------------------------------
    # STEP 1.5: LOAD CLUSTER CAPACITY DATA (DICTIONARY METHOD)
    # ---------------------------------------------------------
    print("\n1.5 Loading Solar Cluster Capacity data...")
    if not CAPACITY_FILE.exists():
        print(f"[ERROR] Cannot find Capacity file at:\n{CAPACITY_FILE}")
        return
    
    df_cap = pd.read_csv(CAPACITY_FILE)
    # CRITICAL FIX: Force uniqueness. If a cluster appears twice, keep the first one.
    df_cap = df_cap.drop_duplicates(subset=['cluster_name'])
    
    # Create a hyper-fast dictionary map: {'Cluster_A': 50.5, 'Cluster_B': 12.0}
    capacity_map = dict(zip(df_cap['cluster_name'], df_cap['total_capacity_mw']))
    print(f"   -> Created memory-safe mapping for {len(capacity_map)} distinct solar clusters.")
    
    # ---------------------------------------------------------
    # STEP 2: LOAD & PREP ACTUAL WEATHER DATA
    # ---------------------------------------------------------
    print("\n2. Loading Actuals Weather Master (This may take a moment)...")
    if not ACTUALS_FILE.exists():
        print(f"[ERROR] Cannot find Actuals file at:\n{ACTUALS_FILE}")
        return
        
    df_actuals = pd.read_parquet(ACTUALS_FILE)
    df_actuals['time'] = pd.to_datetime(df_actuals['time'], utc=True)
    
    merge_keys = ['time', 'cluster_name', 'latitude', 'longitude']
    
    a_cols = {c: f"{c}_actual" for c in df_actuals.columns if c not in merge_keys}
    df_actuals = df_actuals.rename(columns=a_cols)
    print(f"   -> Actuals loaded and renamed. Rows: {len(df_actuals):,}")

    # ---------------------------------------------------------
    # STEP 3: LOAD & PREP FORECAST WEATHER DATA
    # ---------------------------------------------------------
    print("\n3. Loading Forecast Weather Master...")
    if not FORECAST_FILE.exists():
        print(f"[ERROR] Cannot find Forecast file at:\n{FORECAST_FILE}")
        return
        
    df_forecast = pd.read_parquet(FORECAST_FILE)
    df_forecast['time'] = pd.to_datetime(df_forecast['time'], utc=True)
    
    f_cols = {c: f"{c}_forecast" for c in df_forecast.columns if c not in merge_keys}
    df_forecast = df_forecast.rename(columns=f_cols)
    print(f"   -> Forecasts loaded and renamed. Rows: {len(df_forecast):,}")

    # ---------------------------------------------------------
    # STEP 4: THE MEGA-MERGE
    # ---------------------------------------------------------
    print("\n4. Executing Spatial-Temporal Merge (Actuals + Forecasts)...")
    df_final = pd.merge(df_forecast, df_actuals, on=merge_keys, how='inner')
    
    del df_actuals
    del df_forecast
    gc.collect()
    
    # ---------------------------------------------------------
    # STEP 5: ADD TARGETS AND CAPACITIES
    # ---------------------------------------------------------
    print("5. Snapping Target Generation onto Weather Rows...")
    df_final = pd.merge(df_final, df_gen, on='time', how='inner')

    print("5.5 Mapping Cluster Capacities onto Weather Rows...")
    # CRITICAL FIX: Use .map() instead of pd.merge to avoid Memory Explosions
    df_final['total_capacity_mw'] = df_final['cluster_name'].map(capacity_map)
    
    # Ensure output directory exists
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 6: SORTING & EXPORTING FORMAT 1 (BY CLUSTER)
    # ---------------------------------------------------------
    print("\n6. Sorting chronologically by Solar Cluster (Location-First)...")
    df_final = df_final.sort_values(by=['cluster_name', 'time']).reset_index(drop=True)
    
    print(f"   -> Saving Format 1 to: {OUTPUT_BY_CLUSTER.name}")
    df_final.to_parquet(OUTPUT_BY_CLUSTER, index=False)
    
    # ---------------------------------------------------------
    # STEP 7: SORTING & EXPORTING FORMAT 2 (BY TIME)
    # ---------------------------------------------------------
    print("\n7. Sorting spatially by Time (Time-First)...")
    df_final = df_final.sort_values(by=['time', 'cluster_name']).reset_index(drop=True)
    
    print(f"   -> Saving Format 2 to: {OUTPUT_BY_TIME.name}")
    df_final.to_parquet(OUTPUT_BY_TIME, index=False)
    
    print("\n==================================================")
    print("SUCCESS! BOTH FORMATS ARE SECURED ON YOUR DRIVE.")
    print("==================================================")

if __name__ == "__main__":
    build_ultimate_solar_master()