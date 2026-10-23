import pandas as pd
from pathlib import Path
import sys

# ==========================================
# CONFIGURATION: UPDATE THESE FOLDER NAMES!
# ==========================================
# Use absolute pathing based on your script location
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

# 1. FORECAST FOLDERS
FORECAST_FOLDERS = [
    SCRIPT_DIR / "Weather Forecast Data" / "Weather Forecast Data 1" / "data_weather_raw" ,  # Replace with your actual folder names
    SCRIPT_DIR / "Weather Forecast Data" / "Weather Forecast Data 2" / "data_weather_raw" ,
    SCRIPT_DIR / "Weather Forecast Data" / "Weather Forecast Data 3" / "data_weather_raw" 
]
FORECAST_OUTPUT = SCRIPT_DIR / "Weather Forecast Data" / "Weather Forecast Merged"

# 2. ACTUALS FOLDERS
ACTUALS_FOLDERS = [
    SCRIPT_DIR / "Actual Weather Data" / "Actual Weather Data 1" / "data_weather_raw" ,    # Replace with your actual folder names
    SCRIPT_DIR / "Actual Weather Data" / "Actual Weather Data 2" / "data_weather_raw" 
]
ACTUALS_OUTPUT = SCRIPT_DIR / "Actual Weather Data" / "Actual Weather Data Merged"

# The columns that must perfectly match to zip the files together
MERGE_KEYS = ["time", "GEM location ID", "Latitude", "Longitude"]

# ==========================================
# MERGE FUNCTION
# ==========================================
def merge_dataset_folders(input_folders, output_folder, dataset_name):
    print(f"\n========================================")
    print(f"STARTING MERGE: {dataset_name.upper()}")
    print(f"========================================")
    
    # Ensure all input folders exist before starting
    for folder in input_folders:
        if not folder.exists():
            print(f"[ERROR] Cannot find folder: {folder}")
            return

    output_folder.mkdir(parents=True, exist_ok=True)
    base_folder = input_folders[0]
    
    # Find all parquet files in the first folder
    base_files = list(base_folder.rglob("*.parquet"))
    print(f"Found {len(base_files)} files in the base folder to merge.\n")

    saved_count = 0
    error_count = 0

    for base_file in base_files:
        # Get the relative path (e.g., '2022-01/weather_...batch_001.parquet')
        rel_path = base_file.relative_to(base_folder)
        
        dfs = []
        missing_file = False
        
        # Load the matching file from EVERY folder in the group
        for folder in input_folders:
            target_file = folder / rel_path
            if not target_file.exists():
                print(f"[MISSING] Could not find {target_file.name} in {folder.name}. Skipping this batch.")
                missing_file = True
                error_count += 1
                break
            
            # Read the parquet file
            df = pd.read_parquet(target_file)
            dfs.append(df)

        if missing_file:
            continue

        # Zipper them together!
        try:
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                # Inner merge guarantees we only keep rows where the times and locations align perfectly
                merged_df = pd.merge(merged_df, next_df, on=MERGE_KEYS, how="inner")
            
            # Create output sub-folder (e.g., the '2022-01' folder) and save
            out_file = output_folder / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(out_file, index=False)
            
            saved_count += 1
            sys.stdout.write(f"\r[MERGED] {saved_count}/{len(base_files)} files processed successfully.")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n[MERGE ERROR] Failed on {rel_path.name}: {e}")
            error_count += 1

    print(f"\n\n{dataset_name} Merge Complete! Saved: {saved_count} | Errors: {error_count}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Merge the 3 Forecast folders into 1
    merge_dataset_folders(FORECAST_FOLDERS, FORECAST_OUTPUT, "Forecast Data")
    
    # 2. Merge the 2 Actuals folders into 1
    merge_dataset_folders(ACTUALS_FOLDERS, ACTUALS_OUTPUT, "Actuals Data")
    
    print("\nALL MERGING FINISHED. Ready for physics calculations!")