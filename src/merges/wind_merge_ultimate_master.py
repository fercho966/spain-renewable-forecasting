import pandas as pd
from pathlib import Path
import sys
import re

# ==========================================
# CONFIGURATION: THE PROVEN PATHS
# ==========================================
FORECAST_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Weather Forecast Data\Weather_forecast_FINAL_100m")

ACTUALS_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Actual Weather Data\Actual_weather_data_merged")

# THE FIX: Removed the accidental "\Wind Energy Generation\" from this path!
CAPACITY_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Capacity Wind Energy 15MW")

GENERATION_FILE = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Outputs\esios_wind_generation_hourly_2022_2025.parquet")

OUTPUT_FILE_LOC = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\ULTIMATE_WIND_MASTER_ByLocation.parquet")
OUTPUT_FILE_TIME = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\ULTIMATE_WIND_MASTER_ByTime.parquet")

def build_ultimate_master():
    print("==================================================")
    print("ASSEMBLING THE ULTIMATE WIND ENERGY DATASET")
    print("==================================================")
    
    # ---------------------------------------------------------
    # STEP 1: PREP THE GENERATION DATA (TARGETS)
    # ---------------------------------------------------------
    print("1. Loading the ESIOS Peninsular Generation data...")
    if not GENERATION_FILE.exists():
        print(f"[ERROR] Cannot find generation file at:\n{GENERATION_FILE}")
        return
        
    df_gen = pd.read_parquet(GENERATION_FILE)
    df_gen['time'] = pd.to_datetime(df_gen['time'], utc=True)
    print(f"   -> Found {len(df_gen):,} hours of generation targets.")
    
    # ---------------------------------------------------------
    # STEP 2: BUILD FILE DICTIONARIES (WITH REGEX)
    # ---------------------------------------------------------
    print("\n2. Scanning directories and creating file maps...")
    
    def get_batch_id(file_path):
        match = re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}_batch_\d{2})", file_path.name)
        return match.group(1) if match else None

    dict_forecast = {get_batch_id(f): f for f in FORECAST_DIR.rglob("*.parquet") if get_batch_id(f)}
    dict_actuals = {get_batch_id(f): f for f in ACTUALS_DIR.rglob("*.parquet") if get_batch_id(f)}
    dict_capacity = {get_batch_id(f): f for f in CAPACITY_DIR.rglob("*.parquet") if get_batch_id(f)}
    
    total_batches = len(dict_forecast)
    if total_batches == 0:
        print("[ERROR] No forecast files found in FORECAST_DIR.")
        return
        
    # Let's make sure we see 1440 for all three!
    print(f"   -> Mapped {total_batches} Forecast files.")
    print(f"   -> Mapped {len(dict_actuals)} Actual files.")
    print(f"   -> Mapped {len(dict_capacity)} Capacity files.")
    
    # ---------------------------------------------------------
    # STEP 3: BATCH PROCESSING
    # ---------------------------------------------------------
    print("\n3. Stitching batches together...")
    dfs_master = []
    processed_count = 0
    
    for batch_id, f_forecast in dict_forecast.items():
        f_actual = dict_actuals.get(batch_id)
        f_cap = dict_capacity.get(batch_id)
        
        if not f_actual or not f_cap:
            continue
            
        # Load the 3 matching batches
        df_f = pd.read_parquet(f_forecast)
        df_a = pd.read_parquet(f_actual)
        df_c = pd.read_parquet(f_cap)
        
        # Timezone Alignment
        df_f['time'] = pd.to_datetime(df_f['time'], utc=True)
        df_a['time'] = pd.to_datetime(df_a['time'], utc=True)
        df_c['time'] = pd.to_datetime(df_c['time'], utc=True)
        
        # Smart Renaming
        merge_keys = ['time', 'GEM location ID', 'Latitude', 'Longitude']
        
        f_cols = {c: f"{c}_forecast" for c in df_f.columns if c not in merge_keys}
        df_f = df_f.rename(columns=f_cols)
        
        a_cols = {c: f"{c}_actual" for c in df_a.columns if c not in merge_keys}
        df_a = df_a.rename(columns=a_cols)
        
        # The Batch Merge
        df_batch = pd.merge(df_f, df_a, on=merge_keys, how='inner')
        df_batch = pd.merge(df_batch, df_c, on=['time', 'GEM location ID'], how='inner')
        df_batch = pd.merge(df_batch, df_gen, on='time', how='inner')
        
        dfs_master.append(df_batch)
        processed_count += 1
        
        sys.stdout.write(f"\r   -> [MERGING] {processed_count}/{total_batches} batches perfectly locked together.")
        sys.stdout.flush()

    if not dfs_master:
        print("\n[ERROR] All files failed to merge. Dictionaries did not match.")
        return

    # ---------------------------------------------------------
    # STEP 4: FINAL COMPILATION & DUAL EXPORT
    # ---------------------------------------------------------
    print("\n\n4. Zipping all batches into the final massive dataset...")
    df_final = pd.concat(dfs_master, ignore_index=True)
    
    print("\n5a. Sorting chronologically by Wind Farm ID (Location-First)...")
    df_loc = df_final.sort_values(by=['GEM location ID', 'time']).reset_index(drop=True)
    print(f"    -> Saving {OUTPUT_FILE_LOC.name}...")
    df_loc.to_parquet(OUTPUT_FILE_LOC, index=False)
    
    print("\n5b. Sorting geographically by Hour (Time-First)...")
    df_time = df_final.sort_values(by=['time', 'GEM location ID']).reset_index(drop=True)
    print(f"    -> Saving {OUTPUT_FILE_TIME.name}...")
    df_time.to_parquet(OUTPUT_FILE_TIME, index=False)
    
    print("\n==================================================")
    print("SUCCESS! BOTH MASTER DATASETS ARE READY FOR MACHINE LEARNING.")
    print("==================================================")

if __name__ == "__main__":
    build_ultimate_master()