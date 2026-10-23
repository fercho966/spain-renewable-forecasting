import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from pathlib import Path
import time


# =========================
# CONFIGURATION
# =========================
# Update this path to where your CSV actually lives
CITIES_FILE = r"C:\Users\fg384\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model\City Coordinates\spanishpeninsula_cities_over_50k_lat&long.csv"
#CITIES_FILE = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\Load Prediction Model\City Coordinates\spanishpeninsula_cities_over_50k_lat&long.csv"
OUTPUT_DIR = Path("data_weather_raw")
DATASET_NAME = "historical_weather"

URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"  # 4 full years of data

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "is_day",
    "shortwave_radiation",
]

TIMEZONE = "Europe/Madrid"
CITY_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# =========================
def load_cities(csv_path: str) -> pd.DataFrame:
    """Loads and validates the cities CSV."""
    df = pd.read_csv(csv_path, encoding='latin1')
    required = {"name", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df["name"] = df["name"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    
    return df.drop_duplicates(subset=["name"]).reset_index(drop=True)

def month_chunks(start_date: str, end_date: str):
    """Yields clean calendar-month start and end dates."""
    starts = pd.date_range(start_date, end_date, freq='MS')
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > pd.to_datetime(end_date):
            end = pd.to_datetime(end_date)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def city_batches(cities_df: pd.DataFrame, batch_size: int):
    """Yields batches of cities to avoid hitting API URL length limits."""
    n = len(cities_df)
    for i in range(0, n, batch_size):
        yield cities_df.iloc[i:i+batch_size].copy(), (i // batch_size) + 1

def chunk_path(date_start: str, date_end: str, batch_idx: int) -> Path:
    """Creates the folder structure and parquet file path."""
    folder = OUTPUT_DIR / DATASET_NAME / date_start[:7]  # e.g., 2022-01
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"weather_{date_start}_to_{date_end}_batch_{batch_idx:02d}.parquet"

def build_params(batch_df: pd.DataFrame, date_start: str, date_end: str):
    """Constructs the API payload."""
    return {
        "latitude": batch_df["latitude"].tolist(),
        "longitude": batch_df["longitude"].tolist(),
        "start_date": date_start,
        "end_date": date_end,
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
    }

def process_response(response, city_row: pd.Series) -> pd.DataFrame:
    """Parses a single Open-Meteo response into a Pandas DataFrame."""
    hourly = response.Hourly()

    # Build timezone-aware local time array directly based on API response
    times_utc = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    # Convert to Madrid time. Parquet natively stores this timezone awareness!
    times_local = times_utc.tz_convert(TIMEZONE)

    data = {
        "time": times_local,
        "city": city_row["name"],
        "latitude": float(city_row["latitude"]),
        "longitude": float(city_row["longitude"]),
    }

    if "population" in city_row.index:
        data["population"] = city_row["population"]

    # Extract variables deterministically
    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data)

# =========================
# MAIN PIPELINE
# =========================
def main():
    cities = load_cities(CITIES_FILE)
    print(f"Loaded {len(cities)} cities from CSV.")

    # Setup caching and robust retries
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.8)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    total_chunks = skipped = saved = failed = 0

    for date_start, date_end in month_chunks(START_DATE, END_DATE):
        for batch_df, batch_idx in city_batches(cities, CITY_BATCH_SIZE):
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
                    print(f"[PARSE ERROR] City: {row['name']}, Date: {date_start}: {e}")

            if not dfs:
                failed += 1
                print(f"[EMPTY DATA] {date_start}..{date_end} Batch {batch_idx}: No valid data parsed.")
                continue

            # Combine all cities in this batch and save directly to Parquet
            chunk_df = pd.concat(dfs, ignore_index=True)
            chunk_df.to_parquet(out_file, index=False)
            saved += 1
            print(f"[SAVED] {out_file.name} (Rows: {len(chunk_df)})")

            time.sleep(3)  # <--- PUT IT EXACTLY HERE!

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Chunks: {total_chunks} | Saved: {saved} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()