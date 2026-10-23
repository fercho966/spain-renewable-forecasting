import pandas as pd
from pathlib import Path
import os
import gc

# ==========================================
# 0. FILE PATHS & CONFIGURATION
# ==========================================
# Get the directory where this script is physically located
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation")

# Input Directories (Explicitly mapping to your forecast folders)
DIR_ATMOSPHERIC = SCRIPT_DIR / "Weather Forecast 1" / "raw"
DIR_OPTICAL = SCRIPT_DIR / "Weather Forecast 2" / "raw"

# Output Destination (Now locked to the script's directory)
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = OUTPUT_DIR / "solar_forecast_weather_master.parquet"

def merge_forecast_weather():
    print("--- INITIATING FORECAST MASTER MERGE ---")
    
    atm_files = list(DIR_ATMOSPHERIC.rglob("*.parquet"))
    opt_files = list(DIR_OPTICAL.rglob("*.parquet"))
    
    print(f"Found {len(atm_files)} Atmospheric chunks.")
    print(f"Found {len(opt_files)} Optical chunks.")
    
    if len(atm_files) == 0 or len(opt_files) == 0:
        raise FileNotFoundError("Missing Parquet files. Check your directory paths.")

    print("\nLoading Atmospheric Forecasts into memory...")
    df_atm = pd.concat([pd.read_parquet(f) for f in atm_files], ignore_index=True)
    print(f"Atmospheric Data Loaded: {len(df_atm):,} rows.")

    print("Loading Optical Forecasts into memory...")
    df_opt = pd.concat([pd.read_parquet(f) for f in opt_files], ignore_index=True)
    print(f"Optical Data Loaded: {len(df_opt):,} rows.")

    print("\nExecuting exact inner join on spatial-temporal keys...")
    join_keys = ['time', 'cluster_name', 'latitude', 'longitude']
    
    df_master = pd.merge(df_atm, df_opt, on=join_keys, how='inner')
    
    print(f"Merge Complete! Final Master Dimension: {len(df_master):,} rows × {len(df_master.columns)} columns.")
    
    # Force garbage collection to prevent memory spike during export
    del df_atm
    del df_opt
    gc.collect()

    print(f"\nExporting Forecast Master Dataset to PyArrow Parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_master.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
    print(f"SUCCESS: Forecast Master file locked and saved to -> {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_forecast_weather()