import pandas as pd
import holidays
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
# 1. Save exactly where this script is located
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    # Fallback if running directly in Spyder console without saving first
    SCRIPT_DIR = Path.cwd() 

OUTPUT_DIR = SCRIPT_DIR / "data_holidays"

# 2. Point to your clean CSV
CITIES_FILE = r"C:\Users\fg384\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model\City Coordinates\spanishpeninsula_cities_over_50k_lat&long.csv"

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Madrid"
CITY_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# =========================
def month_chunks(start_date: str, end_date: str):
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def city_batches(cities_df: pd.DataFrame, batch_size: int):
    n = len(cities_df)
    for i in range(0, n, batch_size):
        yield cities_df.iloc[i:i+batch_size].copy(), (i // batch_size) + 1

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load valid cities
    cities_df = pd.read_csv(CITIES_FILE, encoding='latin1')
    cities_df["name"] = cities_df["name"].astype(str).str.strip()
    
    # Pre-load holiday dictionaries for speed
    print("Loading Spanish Holiday definitions...")
    es_national = holidays.Spain(years=[2022, 2023, 2024, 2025])
    region_holidays = {}
    for region in cities_df["region_code"].dropna().unique():
        region_holidays[region] = holidays.Spain(subdiv=region, years=[2022, 2023, 2024, 2025])

    total_chunks = 0
    
    # Loop identically to the weather script
    for d_start, d_end in month_chunks(START_DATE, END_DATE):
        
        # Generate the exact 24-hour timeline for this specific month
        # e.g., 2022-01-01 00:00 to 2022-01-31 23:00
        month_times = pd.date_range(start=f"{d_start} 00:00:00", end=f"{d_end} 23:00:00", freq="h", tz=TIMEZONE)
        
        # Create folder for the month (e.g., "2022-01")
        folder = OUTPUT_DIR / d_start[:7]
        folder.mkdir(parents=True, exist_ok=True)
        
        for batch_df, batch_idx in city_batches(cities_df, CITY_BATCH_SIZE):
            total_chunks += 1
            out_file = folder / f"holidays_{d_start}_to_{d_end}_batch_{batch_idx:02d}.parquet"
            
            # Build the data for this specific batch
            batch_data = []
            
            for _, row in batch_df.iterrows():
                city_name = row["name"]
                region_code = row["region_code"]
                
                # Setup city dataframe
                df_city = pd.DataFrame({"time": month_times, "city": city_name})
                df_city["date_only"] = df_city["time"].dt.date
                
                # Check Weekends (0-4 are Mon-Fri, 5-6 are Sat-Sun)
                df_city["is_weekend"] = (df_city["time"].dt.dayofweek >= 5).astype(int)
                
                # Check Holidays
                df_city["is_holiday"] = df_city["date_only"].apply(
                    lambda d: 1 if (d in es_national) or (d in region_holidays.get(region_code, {})) else 0
                )
                
                # Combined feature
                df_city["is_non_working_day"] = ((df_city["is_weekend"] == 1) | (df_city["is_holiday"] == 1)).astype(int)
                
                batch_data.append(df_city.drop(columns=["date_only"]))
            
            # Combine the cities in this batch and save
            chunk_df = pd.concat(batch_data, ignore_index=True)
            chunk_df.to_parquet(out_file, index=False)
            print(f"[SAVED] {out_file.name} (Rows: {len(chunk_df)})")

    print(f"\n--- HOLIDAY PIPELINE COMPLETE ---")
    print(f"Total Chunks Generated: {total_chunks}")
    print(f"Data saved perfectly matched to your weather structure at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()