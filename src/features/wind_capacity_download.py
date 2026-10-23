import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

INPUT_CSV = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Wind Farms Coordinates\spanish-peninsula-windfarms.csv"

# --- THE FIX: Absolute Pathing (Borrowed from your ESIOS script!) ---
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

OUTPUT_DIR = SCRIPT_DIR / "data_cap"
DATASET_NAME = "targets"

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Madrid"
LOCATION_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# =========================
def month_chunks(start_date: str, end_date: str):
    """Yields clean calendar-month start and end dates."""
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def location_batches(unique_ids: list, batch_size: int):
    """Yields batches of unique GEM location IDs."""
    n = len(unique_ids)
    for i in range(0, n, batch_size):
        yield unique_ids[i:i+batch_size], (i // batch_size) + 1

def chunk_path(date_start: str, date_end: str, batch_idx: int) -> Path:
    """Creates the folder structure and parquet file path."""
    folder = OUTPUT_DIR / DATASET_NAME / date_start[:7]  # e.g., 2022-01
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"capacity_{date_start}_to_{date_end}_batch_{batch_idx:02d}.parquet"

# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading and cleaning raw infrastructure data...")
    # Read with latin1 to handle Spanish characters perfectly
    df_wind = pd.read_csv(INPUT_CSV, encoding='latin1')
    
    # 1. Clean Capacity
    df_wind['Capacity (MW)'] = pd.to_numeric(df_wind['Capacity (MW)'], errors='coerce')
    
    # 2. Handle missing Start Dates (Assume pre-2022)
    df_wind['Start year'] = pd.to_numeric(df_wind['Start year'], errors='coerce').fillna(2020).astype(int)
    
    # 3. Handle missing Retired Dates (Assume active forever)
    df_wind['Retired year'] = pd.to_numeric(df_wind['Retired year'], errors='coerce').fillna(2099).astype(int)
    
    # Extract the exact list of 739 unique location IDs
    unique_location_ids = df_wind['GEM location ID'].dropna().unique().tolist()
    
    total_chunks = saved = skipped = 0
    
    print(f"\nStarting batch generation for {len(unique_location_ids)} locations...")
    
    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        
# Generate the timezone-aware hourly index for this specific month
        time_idx = pd.date_range(start=f"{date_start} 00:00:00", 
                                 end=f"{date_end} 23:00:00", 
                                 freq='h', tz=TIMEZONE)
        
        df_time = pd.DataFrame({'time': time_idx})
        # Ensure it is natively stored as Europe/Madrid in the Parquet file
        df_time['time'] = df_time['time'].dt.tz_convert(TIMEZONE)
        
        # ADD THIS LINE BACK IN: We need the year for the step-function!
        df_time['year'] = df_time['time'].dt.year
        
        for batch_ids, batch_idx in location_batches(unique_location_ids, LOCATION_BATCH_SIZE):
            total_chunks += 1
            out_file = chunk_path(date_start, date_end, batch_idx)
            
            # Resume capability: skip if file already exists
            if out_file.exists():
                skipped += 1
                continue
                
            # Filter the raw dataframe to ONLY include phases belonging to this batch of IDs
            batch_phases = df_wind[df_wind['GEM location ID'].isin(batch_ids)]
            
            # Cross merge time with the batch phases
            df_merged = df_time.merge(batch_phases[['GEM location ID', 'Capacity (MW)', 'Start year', 'Retired year']], how='cross')
            
            # Apply the active capacity step function
            df_merged['active_capacity'] = np.where(
                (df_merged['year'] >= df_merged['Start year']) & (df_merged['year'] < df_merged['Retired year']), 
                df_merged['Capacity (MW)'], 
                0.0
            )
            
            # Group by time and location to collapse expansions into cumulative capacity
            batch_timeline = df_merged.groupby(['time', 'GEM location ID'])['active_capacity'].sum().reset_index()
            
            # Ensure strict column order and format (No coordinates!)
            batch_timeline = batch_timeline[['time', 'GEM location ID', 'active_capacity']]
            
            # Save directly to Parquet
            batch_timeline.to_parquet(out_file, index=False)
            saved += 1
            print(f"[SAVED] {out_file.name} (Rows: {len(batch_timeline)})")

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Chunks: {total_chunks} | Saved: {saved} | Skipped: {skipped}")

if __name__ == "__main__":
    main()