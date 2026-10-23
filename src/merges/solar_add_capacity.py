import pandas as pd
from pathlib import Path
import gc

# ==========================================
# CONFIGURATION
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation")

# The two Master files you already successfully generated
FILE_BY_CLUSTER = BASE_DIR / "ULTIMATE_SOLAR_MASTER_by_Cluster.parquet"
FILE_BY_TIME = BASE_DIR / "ULTIMATE_SOLAR_MASTER_by_Time.parquet"

# UPDATE THIS PATH: Point it to your Parquet capacity file!
CAPACITY_FILE = BASE_DIR / "Coordinates" / "processed_data" / "solar_clustered_metadata.parquet" 

def patch_capacity_to_masters():
    print("--- INITIATING CAPACITY PATCH VIA COORDINATES ---")

    # ==========================================
    # 1. BUILD THE COORDINATE DICTIONARY
    # ==========================================
    print("\n1. Loading Capacity Data from Parquet...")
    if not CAPACITY_FILE.exists():
        print(f"[ERROR] Could not find Capacity file at: {CAPACITY_FILE}")
        return

    df_cap = pd.read_parquet(CAPACITY_FILE)
    print(f"   -> Loaded {len(df_cap)} total rows from Capacity file.")
    
    # CAPACITY FILE uses 'optimal_latitude'
    df_cap['coord_key'] = df_cap['optimal_latitude'].round(4).astype(str) + "_" + df_cap['optimal_longitude'].round(4).astype(str)
    
    # Drop duplicates based on the exact coordinates
    df_cap = df_cap.drop_duplicates(subset=['coord_key'])
    print(f"   -> Found {len(df_cap)} UNIQUE coordinate locations after filtering.")
    
    # Create the dictionary: {'40.4168_-3.7038': 50.5}
    capacity_map = dict(zip(df_cap['coord_key'], df_cap['total_capacity_mw']))

    # ==========================================
    # 2. PATCH FORMAT 1 (LOCATION-FIRST)
    # ==========================================
    print("\n2. Loading Master Format 1 (By Cluster)...")
    if FILE_BY_CLUSTER.exists():
        df_cluster = pd.read_parquet(FILE_BY_CLUSTER)
        
        print("   -> Generating Coordinate Keys for mapping...")
        # WEATHER FILE uses standard 'latitude'
        df_cluster['coord_key'] = df_cluster['latitude'].round(4).astype(str) + "_" + df_cluster['longitude'].round(4).astype(str)
        
        print("   -> Stamping capacity onto all rows...")
        df_cluster['total_capacity_mw'] = df_cluster['coord_key'].map(capacity_map)
        
        print("   -> Cleaning up temporary keys and Overwriting Parquet file...")
        df_cluster = df_cluster.drop(columns=['coord_key'])
        df_cluster.to_parquet(FILE_BY_CLUSTER, index=False)
        print("   -> Format 1 successfully patched!")
        
        del df_cluster
        gc.collect()
    else:
        print(f"[ERROR] Could not find {FILE_BY_CLUSTER.name}")

    # ==========================================
    # 3. PATCH FORMAT 2 (TIME-FIRST)
    # ==========================================
    print("\n3. Loading Master Format 2 (By Time)...")
    if FILE_BY_TIME.exists():
        df_time = pd.read_parquet(FILE_BY_TIME)
        
        print("   -> Generating Coordinate Keys for mapping...")
        # WEATHER FILE uses standard 'latitude'
        df_time['coord_key'] = df_time['latitude'].round(4).astype(str) + "_" + df_time['longitude'].round(4).astype(str)
        
        print("   -> Stamping capacity onto all rows...")
        df_time['total_capacity_mw'] = df_time['coord_key'].map(capacity_map)
        
        print("   -> Cleaning up temporary keys and Overwriting Parquet file...")
        df_time = df_time.drop(columns=['coord_key'])
        df_time.to_parquet(FILE_BY_TIME, index=False)
        print("   -> Format 2 successfully patched!")
        
        del df_time
        gc.collect()
    else:
        print(f"[ERROR] Could not find {FILE_BY_TIME.name}")

    print("\n--- COORDINATE PATCH COMPLETE ---")

if __name__ == "__main__":
    patch_capacity_to_masters()