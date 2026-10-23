import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from pathlib import Path
import time


# =========================
# CONFIGURATION
# =========================
# Use absolute pathing to avoid Windows MAX_PATH limits
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

LOCATIONS_FILE = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Wind Energy Generation\Wind Farms Coordinates\wind_farm_unique_coordinates.csv"
OUTPUT_DIR = SCRIPT_DIR / "data_weather_raw"
DATASET_NAME = "weather_forecast_data"

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"  # 4 full years of data

# Your excellent hub-height physics variables!
HOURLY_VARS = [
    "relative_humidity_2m",
    "surface_pressure",
    "pressure_msl",
]

# THE FIX: Request UTC from API to prevent DST crashes, convert locally
API_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Europe/Madrid"
LOCATION_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# ========================
def load_locations(csv_path: Path) -> pd.DataFrame:
    """Loads and validates the unique wind farm coordinates CSV."""
    df = pd.read_csv(csv_path)
    required = {"GEM location ID", "Latitude", "Longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df["GEM location ID"] = df["GEM location ID"].astype(str).str.strip()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    
    return df.drop_duplicates(subset=["GEM location ID"]).reset_index(drop=True)

def month_chunks(start_date: str, end_date: str):
    """Yields clean calendar-month start and end dates."""
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def location_batches(locations_df: pd.DataFrame, batch_size: int):
    """Yields batches of locations to avoid hitting API URL length limits."""
    n = len(locations_df)
    for i in range(0, n, batch_size):
        yield locations_df.iloc[i:i+batch_size].copy(), (i // batch_size) + 1

def chunk_path(date_start: str, date_end: str, batch_idx: int) -> Path:
    """Creates the folder structure and parquet file path."""
    folder = OUTPUT_DIR / DATASET_NAME / date_start[:7]  # e.g., 2022-01
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"weather_{date_start}_to_{date_end}_batch_{batch_idx:02d}.parquet"

def build_params(batch_df: pd.DataFrame, date_start: str, date_end: str):
    """Constructs the API payload."""
    return {
        "latitude": batch_df["Latitude"].tolist(),
        "longitude": batch_df["Longitude"].tolist(),
        "start_date": date_start,
        "end_date": date_end,
        "hourly": HOURLY_VARS,
        "timezone": API_TIMEZONE,
        "models": "icon_seamless"  # THE FIX: Bypass the corrupted "best_match" stitching!
    }

def process_response(response, loc_row: pd.Series) -> pd.DataFrame:
    """Parses a single Open-Meteo response into a Pandas DataFrame."""
    hourly = response.Hourly()

# Build timezone-aware local time array directly based on API response
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

    # Extract variables deterministically
    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data)

# =========================
# MAIN PIPELINE
# =========================
def main():
    locations = load_locations(LOCATIONS_FILE)
    print(f"Loaded {len(locations)} unique wind farm locations from CSV.")

    # Setup caching and robust retries
    cache_session = requests_cache.CachedSession(".cache_forecast", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.8)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    total_chunks = skipped = saved = failed = 0

    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        for batch_df, batch_idx in location_batches(locations, LOCATION_BATCH_SIZE):
            total_chunks += 1
            out_file = chunk_path(date_start, date_end, batch_idx)

            # Resume capability: skip if file already exists
            if out_file.exists():
                skipped += 1
                continue

            params = build_params(batch_df, date_start, date_end)

            try:
                responses = openmeteo.weather_api(URL, params=params)
            except Exception as e:
                failed += 1
                print(f"[API ERROR] {date_start}..{date_end} Batch {batch_idx}: {e}")
                continue

            dfs = []
            for resp, (_, row) in zip(responses, batch_df.iterrows()):
                try:
                    dfs.append(process_response(resp, row))
                except Exception as e:
                    print(f"[PARSE ERROR] Location: {row['GEM location ID']}, Date: {date_start}: {e}")

            if not dfs:
                failed += 1
                print(f"[EMPTY DATA] {date_start}..{date_end} Batch {batch_idx}: No valid data parsed.")

                continue

            # Combine all locations in this batch and save directly to Parquet
            chunk_df = pd.concat(dfs, ignore_index=True)
            chunk_df.to_parquet(out_file, index=False)
            saved += 1
            print(f"[SAVED] {out_file.name} (Rows: {len(chunk_df)})")

            time.sleep(2.5)  # Be polite to Open-Meteo

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Chunks: {total_chunks} | Saved: {saved} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()