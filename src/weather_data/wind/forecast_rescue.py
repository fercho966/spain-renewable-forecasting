import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from pathlib import Path
import time
import sys

# =========================
# CONFIGURATION
# =========================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

LOCATIONS_FILE = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Wind Farms Coordinates\wind_farm_unique_coordinates.csv"

# THE RESCUE FOLDER: Keeps this data safe from your perfect ICON data
OUTPUT_DIR = SCRIPT_DIR / "data_weather_raw"

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# THE TARGET: Only the exact 11 months that ICON failed to provide
START_DATE = "2022-01-01"
END_DATE = "2022-11-30"  

# ALL 11 VARIABLES AT ONCE
HOURLY_VARS = [
    "temperature_2m", "wind_speed_80m", "wind_speed_120m", "wind_direction_80m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "wind_direction_120m",
    "relative_humidity_2m", "surface_pressure", "pressure_msl"
]

API_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Europe/Madrid"

# Batch size 5 guarantees the server won't crash when asking for 11 variables
LOCATION_BATCH_SIZE = 5

# =========================
# HELPER FUNCTIONS
# =========================
def load_locations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GEM location ID"] = df["GEM location ID"].astype(str).str.strip()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
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

def chunk_path(date_start: str, date_end: str, batch_idx: int) -> Path:
    folder = OUTPUT_DIR / date_start[:7]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"weather_rescue_{date_start}_to_{date_end}_batch_{batch_idx:03d}.parquet"

def build_params(batch_df: pd.DataFrame, date_start: str, date_end: str):
    return {
        "latitude": batch_df["Latitude"].tolist(),
        "longitude": batch_df["Longitude"].tolist(),
        "start_date": date_start,
        "end_date": date_end,
        "hourly": HOURLY_VARS,
        "timezone": API_TIMEZONE,
        #"models": "gfs_seamless"  # THE LIFESAVER: Swapping to the American database!
    }

def process_response(response, loc_row: pd.Series) -> pd.DataFrame:
    hourly = response.Hourly()
    
    times_utc = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    times_local = times_utc.tz_convert(LOCAL_TIMEZONE)

    data = {
        "time": times_local,
        "GEM location ID": loc_row["GEM location ID"],
        "Latitude": float(loc_row["Latitude"]),
        "Longitude": float(loc_row["Longitude"]),
    }

    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data)

# =========================
# MAIN PIPELINE
# =========================
def main():
    locations = load_locations(LOCATIONS_FILE)
    print(f"Loaded {len(locations)} unique wind farm locations.")

    cache_session = requests_cache.CachedSession(".cache_rescue", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.8)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    total_chunks = skipped = saved = failed = 0

    print("\nStarting the GFS Targeted Rescue Mission...")

    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        print(f"\n--- Rescuing Month: {date_start[:7]} ---")
        
        for batch_df, batch_idx in location_batches(locations, LOCATION_BATCH_SIZE):
            total_chunks += 1
            out_file = chunk_path(date_start, date_end, batch_idx)

            if out_file.exists():
                skipped += 1
                continue

            params = build_params(batch_df, date_start, date_end)

            # Custom Retry Block
            max_retries = 3
            api_success = False
            
            for attempt in range(max_retries):
                try:
                    responses = openmeteo.weather_api(URL, params=params)
                    api_success = True
                    break 
                except Exception as e:
                    error_str = str(e)
                    if "timeoutReached" in error_str or "500" in error_str:
                        print(f"      [SERVER TIMEOUT] Batch {batch_idx} snag. Retrying {attempt + 1}/{max_retries}...")
                        time.sleep(3)
                    else:
                        print(f"[API ERROR] {date_start}..{date_end} Batch {batch_idx}: {e}")
                        break
            
            if not api_success:
                failed += 1
                continue

            dfs = []
            for resp, (_, row) in zip(responses, batch_df.iterrows()):
                try:
                    dfs.append(process_response(resp, row))
                except Exception as e:
                    print(f"[PARSE ERROR] Location: {row['GEM location ID']}, Date: {date_start}: {e}")

            if not dfs:
                failed += 1
                continue

            chunk_df = pd.concat(dfs, ignore_index=True)
            chunk_df.to_parquet(out_file, index=False)
            saved += 1
            
            sys.stdout.write(f"\r[RESCUED] {out_file.name} (Rows: {len(chunk_df)})")
            sys.stdout.flush()

            time.sleep(2)  # Polite pacing

    print("\n\n--- RESCUE COMPLETE ---")
    print(f"Total Chunks: {total_chunks} | Rescued: {saved} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()