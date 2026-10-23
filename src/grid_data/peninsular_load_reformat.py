import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "esios_demand_hourly_2022_2025_LOCALTIME.csv"

try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

OUTPUT_DIR = SCRIPT_DIR / "data_demand"
TIMEZONE = "Europe/Madrid"

# =========================
# HELPER FUNCTION
# =========================
def month_chunks(start_date: str, end_date: str):
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# =========================
# MAIN PIPELINE
# =========================
def main():
    print(f"Loading raw 15-minute demand data from {INPUT_FILE}...")
    
    # 1. Load the compressed CSV
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Convert string timestamps back to actual timezone-aware datetime objects
    # We use utc=True first to ensure a clean parse, then convert to Madrid time
    df['time'] = pd.to_datetime(df['datetime_local'], utc=True).dt.tz_convert(TIMEZONE)
    
    print("Resampling from 15-minute to 1-Hour averages...")
    # 3. Resample to Hourly (Taking the average MW of the four 15-min blocks)
    df = df.set_index('time')
    df_hourly = df[['value']].resample('1h').mean().reset_index()
    
    # 4. Rename column to our standard ML target variable name
    df_hourly.rename(columns={'value': 'actual_demand'}, inplace=True)
    
    # Find the min and max dates in the dataset to drive our loop
    start_date = df_hourly['time'].min().strftime('%Y-%m-%d')
    end_date = df_hourly['time'].max().strftime('%Y-%m-%d')
    
    print(f"Data range: {start_date} to {end_date}. Generating monthly Parquets...")
    
    total_months = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 5. Slice into our standard monthly files
    for d_start, d_end in month_chunks(start_date, end_date):
        total_months += 1
        
        # Define exact local boundaries for the month
        start_local = pd.to_datetime(f"{d_start} 00:00:00").tz_localize(TIMEZONE)
        end_local = pd.to_datetime(f"{d_end} 23:59:59").tz_localize(TIMEZONE)
        
        # Slice the month
        df_month = df_hourly[(df_hourly['time'] >= start_local) & (df_hourly['time'] <= end_local)].copy()
        
        if df_month.empty:
            continue
            
        # Create folder and save
        folder = OUTPUT_DIR / d_start[:7]
        folder.mkdir(parents=True, exist_ok=True)
        
        out_file = folder / f"demand_{d_start}_to_{d_end}.parquet"
        df_month.to_parquet(out_file, index=False)
        print(f"[SAVED] {out_file.name} (Rows: {len(df_month)})")

    print("\n--- DEMAND FORMATTING COMPLETE ---")
    print(f"Total Months Processed: {total_months}")

if __name__ == "__main__":
    main()