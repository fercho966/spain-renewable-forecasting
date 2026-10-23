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
OUTPUT_DIR = SCRIPT_DIR / "data_weather_raw"
DATASET_NAME = "weather_forecast_data"

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# 1 Year chunks: Keeps the payload tiny so the server NEVER times out.
YEARS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31")
]

# We split the variables to halve the computational load on Open-Meteo
HOURLY_VARS_PART_1 = [
    "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "temperature_80m"
]

HOURLY_VARS_PART_2 = [
    "temperature_120m", "temperature_2m", "surface_pressure", 
    "relative_humidity_2m", "wind_gusts_10m", "dew_point_2m", "pressure_msl"
]

# Ask in UTC to prevent March DST 500 errors, then convert locally.
API_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Europe/Madrid"

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

def chunk_path(year_str: str, gem_id: str) -> Path:
    folder = OUTPUT_DIR / DATASET_NAME / year_str
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"weather_{year_str}_{gem_id}.parquet"

def build_params(lat: float, lon: float, date_start: str, date_end: str, vars_list: list):
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_start,
        "end_date": date_end,
        "hourly": vars_list,
        "timezone": API_TIMEZONE, 
    }

def fetch_safe(openmeteo_client, params: dict, gem_id: str):
    """A safe fetcher that rigorously respects the 1000/hr limit pacing."""
    try:
        responses = openmeteo_client.weather_api(URL, params=params)
        time.sleep(3.8)  # THE MAGIC PACING NUMBER: Ensures < 950 calls/hour
        return responses[0], True
    except Exception as e:
        print(f"\n[API ERROR] {gem_id}: {e}")
        time.sleep(3.8) # Keep the pacing even on failures
        return None, False

def process_response(response, loc_row: pd.Series, vars_list: list) -> pd.DataFrame:
    hourly = response.Hourly()
    
    # Process UTC array
    times_utc = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    # Convert safely to Madrid time
    times_local = times_utc.tz_convert(LOCAL_TIMEZONE)

    data = {
        "time": times_local,
        "GEM location ID": loc_row["GEM location ID"],
        "Latitude": float(loc_row["Latitude"]),
        "Longitude": float(loc_row["Longitude"]),
    }

    for idx, var in enumerate(vars_list):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data)

# =========================
# MAIN PIPELINE
# =========================
def main():
    locations = load_locations(LOCATIONS_FILE)
    print(f"Loaded {len(locations)} unique wind farm locations from CSV.")

    cache_session = requests_cache.CachedSession(".cache_forecast", expire_after=-1)
    retry_session = retry(cache_session, retries=2, backoff_factor=1)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    total_files = skipped = saved = failed = 0

    print("\nStarting the 'Overnight Slow-Burn' Download...")
    print("Pacing: ~947 API calls per hour (Safely under the 1,000 limit).")
    
    for start_date, end_date in YEARS:
        year_str = start_date[:4]
        print(f"\n--- Processing Year: {year_str} ---")
        
        for _, row in locations.iterrows():
            total_files += 1
            gem_id = row['GEM location ID']
            out_file = chunk_path(year_str, gem_id)

            # Skip if we already perfected this file
            if out_file.exists():
                skipped += 1
                continue

            # PHASE 1: Wind Variables
            params_1 = build_params(row['Latitude'], row['Longitude'], start_date, end_date, HOURLY_VARS_PART_1)
            resp_1, success_1 = fetch_safe(openmeteo, params_1, gem_id)
            if not success_1:
                failed += 1
                continue

            # PHASE 2: Surface Variables
            params_2 = build_params(row['Latitude'], row['Longitude'], start_date, end_date, HOURLY_VARS_PART_2)
            resp_2, success_2 = fetch_safe(openmeteo, params_2, gem_id)
            if not success_2:
                failed += 1
                continue

            # MERGE AND SAVE
            try:
                df_1 = process_response(resp_1, row, HOURLY_VARS_PART_1)
                df_2 = process_response(resp_2, row, HOURLY_VARS_PART_2)
                
                chunk_df = pd.merge(df_1, df_2, on=["time", "GEM location ID", "Latitude", "Longitude"])
                chunk_df.to_parquet(out_file, index=False)
                saved += 1
                
                # Dynamic terminal output to show it is alive without spamming
                sys.stdout.write(f"\r[SAVED] {year_str} | {gem_id} | Progress: {saved} new files.")
                sys.stdout.flush()
                
            except Exception as e:
                failed += 1
                print(f"\n[PARSE ERROR] {gem_id}: {e}")

    print("\n\n--- PIPELINE COMPLETE ---")
    print(f"Total Expected: {total_files} | Saved: {saved} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()