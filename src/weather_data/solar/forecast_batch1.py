import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from pathlib import Path
import time

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation")

try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

LOCATIONS_FILE = BASE_DIR / "Coordinates" / "processed_data" / "solar_api_extraction_targets.csv"

# Shortened paths to avoid Windows MAX_PATH limits
OUTPUT_DIR = SCRIPT_DIR / "raw"
DATASET_NAME = "atm_forecast_data"

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"  # 4 full years of data

# The Atmospheric features (Part 1)
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]
# Timezone and Batching constraints
API_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Europe/Madrid"
LOCATION_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# ========================
def load_locations(csv_path: Path) -> pd.DataFrame:
    """Loads and validates the clustered solar farm coordinates CSV."""
    df = pd.read_csv(csv_path)
    required = {"cluster_name", "optimal_latitude", "optimal_longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df["cluster_name"] = df["cluster_name"].astype(str).str.strip()
    df["optimal_latitude"] = pd.to_numeric(df["optimal_latitude"], errors="coerce")
    df["optimal_longitude"] = pd.to_numeric(df["optimal_longitude"], errors="coerce")
    df = df.dropna(subset=["optimal_latitude", "optimal_longitude"]).copy()
    
    # Drop duplicates based strictly on physical coordinates
    df = df.drop_duplicates(subset=["optimal_latitude", "optimal_longitude"]).reset_index(drop=True)
    return df

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
    folder = OUTPUT_DIR / DATASET_NAME / date_start[:7] 
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"atm_fcst_{date_start}_to_{date_end}_batch_{batch_idx:02d}.parquet"

def build_params(batch_df: pd.DataFrame, date_start: str, date_end: str):
    return {
        "latitude": batch_df["optimal_latitude"].tolist(),
        "longitude": batch_df["optimal_longitude"].tolist(),
        "start_date": date_start,
        "end_date": date_end,
        "hourly": HOURLY_VARS,
        "timezone": API_TIMEZONE,
        #"models": "icon_seamless"  # Europe's premier meteorological model
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
        "cluster_name": loc_row["cluster_name"],
        "latitude": float(loc_row["optimal_latitude"]),
        "longitude": float(loc_row["optimal_longitude"]),
    }

    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data)

# =========================
# MAIN PIPELINE
# =========================
def main():
    locations = load_locations(LOCATIONS_FILE)
    print(f"Loaded {len(locations)} unique solar farm clusters from CSV.")

    # Separate cache for forecasts
    cache_session = requests_cache.CachedSession(".cache_atm_forecasts", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.8)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    total_chunks = skipped = saved = failed = 0

    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        for batch_df, batch_idx in location_batches(locations, LOCATION_BATCH_SIZE):
            total_chunks += 1
            out_file = chunk_path(date_start, date_end, batch_idx)

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
                    print(f"[PARSE ERROR] Location: {row['cluster_name']}, Date: {date_start}: {e}")

            if not dfs:
                failed += 1
                print(f"[EMPTY DATA] {date_start}..{date_end} Batch {batch_idx}: No valid data parsed.")
                continue

            chunk_df = pd.concat(dfs, ignore_index=True)
            chunk_df.to_parquet(out_file, index=False)
            saved += 1
            print(f"[SAVED] {out_file.name} (Rows: {len(chunk_df)})")

            time.sleep(3)  

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Chunks: {total_chunks} | Saved: {saved} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()