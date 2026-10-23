import pandas as pd
from pathlib import Path
import sys

# ==========================================
# CONFIGURATION
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

# Point these to your newly merged master folders!
FORECAST_DIR = SCRIPT_DIR / "Weather Forecast Data" / "Weather_forecast_merged"
ACTUALS_DIR = SCRIPT_DIR / "Actual Weather Data" / "Actual_weather_data_merged"

# The essential base columns every file must have
BASE_COLS = ["time", "GEM location ID", "Latitude", "Longitude"]

# The exact variables you downloaded
FORECAST_VARS = [
    "temperature_2m", "wind_speed_80m", "wind_speed_120m", "wind_direction_80m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "wind_direction_120m",
    "relative_humidity_2m", "surface_pressure", "pressure_msl"
]

ACTUALS_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_100m", "wind_direction_100m"
]

# ==========================================
# AUDIT FUNCTION
# ==========================================
def audit_dataset(directory: Path, expected_vars: list, dataset_name: str):
    print(f"\n==================================================")
    print(f"STARTING DEEP SCAN AUDIT: {dataset_name.upper()}")
    print(f"==================================================")

    if not directory.exists():
        print(f"[ERROR] Cannot find directory: {directory}")
        return

    expected_columns = set(BASE_COLS + expected_vars)
    parquet_files = list(directory.rglob("*.parquet"))
    
    total_files = len(parquet_files)
    if total_files == 0:
        print("[WARNING] No parquet files found in this directory.")
        return

    total_rows = 0
    files_with_missing_cols = 0
    files_with_nans = 0

    print(f"Scanning {total_files} files...\n")

    for i, file_path in enumerate(parquet_files, 1):
        try:
            df = pd.read_parquet(file_path)
            total_rows += len(df)
            
            # 1. Check for Missing Columns
            actual_columns = set(df.columns)
            missing_cols = expected_columns - actual_columns
            if missing_cols:
                files_with_missing_cols += 1
                print(f"[COLUMN ERROR] {file_path.name} is missing: {missing_cols}")

            # 2. Check for NaN / Null values
            if df.isna().any().any():
                files_with_nans += 1
                nan_counts = df.isna().sum()
                bad_cols = nan_counts[nan_counts > 0].to_dict()
                print(f"[NaN ERROR] {file_path.name} contains missing data: {bad_cols}")

            # Terminal progress update
            sys.stdout.write(f"\r[SCANNING] {i}/{total_files} files checked.")
            sys.stdout.flush()

        except Exception as e:
            print(f"\n[READ ERROR] Could not open {file_path.name}: {e}")

    print(f"\n\n--- AUDIT RESULTS FOR {dataset_name.upper()} ---")
    print(f"Total Files Scanned: {total_files}")
    print(f"Total Rows of Data:  {total_rows:,}")
    
    if files_with_missing_cols == 0 and files_with_nans == 0:
        print("HEALTH STATUS: PERFECT [ALL CLEAR]")
    else:
        print(f"HEALTH STATUS: FAILED")
        print(f" -> Files with missing columns: {files_with_missing_cols}")
        print(f" -> Files with NaN (blank) values: {files_with_nans}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    audit_dataset(FORECAST_DIR, FORECAST_VARS, "Forecast Data")
    audit_dataset(ACTUALS_DIR, ACTUALS_VARS, "Actual Data")
    
    print("\nAudit complete.")
