import pandas as pd
import requests
from pathlib import Path
import time

# =========================
# CONFIGURATION
# =========================
# ⚠️ Paste your token back here!
TOKEN = "YOUR_ESIOS_API_KEY_HERE"  
INDICATOR_ID = 600  # Day-Ahead Market Price
GEO_ID = 3       # Spanish Peninsula

try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

OUTPUT_DIR = SCRIPT_DIR / "data_price"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Madrid"

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN
}

# =========================
# HELPER FUNCTIONS
# =========================
def month_chunks(start_date: str, end_date: str):
    """Yields clean calendar-month start dates and end dates."""
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
    print(f"Starting ESIOS download for Indicator {INDICATOR_ID} (Hourly Price)...")
    
    URL = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"
    total_months = saved = failed = 0

    for d_start, d_end in month_chunks(START_DATE, END_DATE):
        total_months += 1
        
        # --- THE MAGIC FIX: TIMEZONE PADDING ---
        # We add/subtract 1 day to the API request to cast a wider net
        padded_start = (pd.to_datetime(d_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        padded_end = (pd.to_datetime(d_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        esios_start = f"{padded_start}T00:00:00Z"
        esios_end = f"{padded_end}T23:59:59Z"
        
        # Structure the output file path exactly like the weather data
        folder = OUTPUT_DIR / d_start[:7]
        folder.mkdir(parents=True, exist_ok=True)
        out_file = folder / f"price_{d_start}_to_{d_end}.parquet"
        
        # Resume capability
        if out_file.exists():
            print(f"[SKIPPED] {out_file.name} already exists.")
            continue

        params = {
            "start_date": esios_start,
            "end_date": esios_end,
            "geo_ids[]": GEO_ID,
            "time_trunc": "hour"  # <-- Hourly resolution!
        }
        
        try:
            r = requests.get(URL, headers=HEADERS, params=params, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"API failed with status {r.status_code}: {r.text}")
            
            data = r.json()
            values = data.get("indicator", {}).get("values", [])
            
            if not values:
                print(f"[EMPTY] No data for {d_start} to {d_end}")
                failed += 1
                continue
                
            df = pd.DataFrame(values)
            
            # Aligning with our ML format:
            # 1. Parse UTC datetime from ESIOS
            df["time_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            
            # 2. Convert to Europe/Madrid and name it 'time' to match weather EXACTLY
            df["time"] = df["time_utc"].dt.tz_convert(TIMEZONE)
            
            # 3. Rename 'value' and filter down to just the two columns we need
            df.rename(columns={"value": "day_ahead_price"}, inplace=True)
            df_clean = df[["time", "day_ahead_price"]].copy()
            
            # 4. Strict Local Trimming: Cut off the padded days and keep EXACTLY the requested month
            start_local = pd.to_datetime(f"{d_start} 00:00:00").tz_localize(TIMEZONE)
            end_local = pd.to_datetime(f"{d_end} 23:59:59").tz_localize(TIMEZONE)
            df_clean = df_clean[(df_clean["time"] >= start_local) & (df_clean["time"] <= end_local)]
            
            # Remove duplicates and sort chronologically
            df_clean = df_clean.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
            
            # Save to Parquet natively
            df_clean.to_parquet(out_file, index=False)
            saved += 1
            print(f"[SAVED] {out_file.name} (Rows: {len(df_clean)})")
            
            time.sleep(1) # Be polite to the ESIOS servers
            
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed {d_start} to {d_end}: {e}")

    print("\n--- ESIOS PRICE PIPELINE COMPLETE ---")
    print(f"Total Months: {total_months} | Saved: {saved} | Failed: {failed}")

if __name__ == "__main__":
    main()