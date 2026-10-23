import pandas as pd
from pathlib import Path
import os
import gc

# ==========================================
# 0. FILE PATHS & CONFIGURATION
# ==========================================
# Base directories
BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation")

try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# Input Directories
# Assuming your raw folders are inside the respective Data 1 and Data 2 folders
DIR_ATMOSPHERIC = SCRIPT_DIR / "Actual Weather Data 1" / "raw"
DIR_OPTICAL = SCRIPT_DIR / "Actual Weather Data 2" / "raw"

# Output Destination
OUTPUT_DIR = BASE_DIR / "Processed Master Datasets"
OUTPUT_FILE = OUTPUT_DIR / "solar_actual_weather_master.parquet"

def merge_actual_weather():
    print("--- INITIATING MASTER MERGE ---")
    
    # 1. Locate all parquet files recursively
    atm_files = list(DIR_ATMOSPHERIC.rglob("*.parquet"))
    opt_files = list(DIR_OPTICAL.rglob("*.parquet"))
    
    print(f"Found {len(atm_files)} Atmospheric chunks.")
    print(f"Found {len(opt_files)} Optical chunks.")
    
    if len(atm_files) == 0 or len(opt_files) == 0:
        raise FileNotFoundError("Missing Parquet files. Please check the directory paths.")

    # ==========================================
    # 2. LOAD ATMOSPHERIC DATA
    # ==========================================
    print("\nLoading Atmospheric Dataset into memory...")
    # List comprehension + pd.concat is highly memory-efficient for parquet
    df_atm = pd.concat([pd.read_parquet(f) for f in atm_files], ignore_index=True)
    print(f"Atmospheric Data Loaded: {len(df_atm):,} rows.")

    # ==========================================
    # 3. LOAD OPTICAL DATA
    # ==========================================
    print("Loading Optical Dataset into memory...")
    df_opt = pd.concat([pd.read_parquet(f) for f in opt_files], ignore_index=True)
    print(f"Optical Data Loaded: {len(df_opt):,} rows.")

    # ==========================================
    # 4. EXECUTE INNER JOIN
    # ==========================================
    print("\nExecuting exact inner join on spatial-temporal keys...")
    join_keys = ['time', 'cluster_name', 'latitude', 'longitude']
    
    # Perform the merge
    df_master = pd.merge(df_atm, df_opt, on=join_keys, how='inner')
    
    print(f"Merge Complete! Final Master Dimension: {len(df_master):,} rows × {len(df_master.columns)} columns.")
    
    # Memory management: Delete the individual dataframes from RAM before saving the massive master file
    del df_atm
    del df_opt
    gc.collect()

    # ==========================================
    # 5. EXPORT MASTER DATASET
    # ==========================================
    print(f"\nExporting Master Dataset to PyArrow Parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save using the PyArrow engine
    df_master.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
    print(f"SUCCESS: Master file locked and saved to -> {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_actual_weather()