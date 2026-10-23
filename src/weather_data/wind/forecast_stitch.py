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

LOCATIONS_FILE = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Wind Farms Coordinates\wind_farm_unique_coordinates.csv"

# The source of the rescued 148-batch files
RESCUE_DIR = SCRIPT_DIR / "weather_forecast_rescue" / "data_weather_raw"

# The destination where we will put the perfectly formatted 30-batch files
FORECAST_DIR = SCRIPT_DIR / "weather_forecast_merged"

START_DATE = "2022-01-01"
END_DATE = "2022-11-30"

LOCATION_BATCH_SIZE = 25

# ==========================================
# HELPER FUNCTIONS (To match your original layout)
# ==========================================
def load_locations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GEM location ID"] = df["GEM location ID"].astype(str).str.strip()
    return df.drop_duplicates(subset=["GEM location ID"]).reset_index(drop=True)

def month_chunks(start_date: str, end_date: str):
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def location_batches(locations_df: pd.DataFrame, batch_size: int):
    n = len(locations_df)
    for i in range(0, n, batch_size):
        yield locations_df.iloc[i:i+batch_size].copy(), (i // batch_size) + 1

# ==========================================
# MAIN EXECUTION
# ==========================================
def rebatch_data():
    locations = load_locations(LOCATIONS_FILE)
    print(f"Loaded master coordinate list to guarantee perfect ordering.")

    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        month_str = date_start[:7]
        print(f"\n--- Re-batching Month: {month_str} ---")
        
        rescue_month_dir = RESCUE_DIR / month_str
        target_month_dir = FORECAST_DIR / month_str
        
        # 1. Load all the rescue files for this month into one massive pool
        rescue_files = list(rescue_month_dir.glob("*.parquet"))
        if not rescue_files:
            print(f"[WARNING] No rescue files found for {month_str}. Skipping.")
            continue
            
        df_month = pd.concat([pd.read_parquet(f) for f in rescue_files], ignore_index=True)
        
        # Make sure the target directory exists
        target_month_dir.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        # 2. Deal the data back out into chunks of 25 based on the original CSV order
        for batch_df, batch_idx in location_batches(locations, LOCATION_BATCH_SIZE):
            # Get the exact IDs for this specific batch (e.g., Batch 1 has IDs 1-25)
            batch_ids = batch_df['GEM location ID'].tolist()
            
            # Filter the massive pool for ONLY these 25 locations
            df_batch = df_month[df_month['GEM location ID'].isin(batch_ids)]
            
            if df_batch.empty:
                print(f"[ERROR] Batch {batch_idx:02d} came up empty!")
                continue
                
            # 3. Save it with the exact original naming convention
            out_filename = f"weather_{date_start}_to_{date_end}_batch_{batch_idx:02d}.parquet"
            out_file = target_month_dir / out_filename
            
            # Sort it chronologically and by ID just to be perfectly clean
            df_batch = df_batch.sort_values(by=['GEM location ID', 'time']).reset_index(drop=True)
            df_batch.to_parquet(out_file, index=False)
            
            saved += 1
            sys.stdout.write(f"\r[CREATED] {out_filename} (Rows: {len(df_batch)})")
            sys.stdout.flush()
            
        print(f"\nSuccessfully created {saved} perfectly formatted files for {month_str}.")

    print("\n==========================================")
    print("   RE-BATCHING COMPLETE")
    print("==========================================")

if __name__ == "__main__":
    rebatch_data()