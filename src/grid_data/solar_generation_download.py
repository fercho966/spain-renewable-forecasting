import os
import time
import requests
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

TOKEN = "YOUR_ESIOS_API_KEY_HERE"  
if not TOKEN:
    raise RuntimeError("Missing ESIOS_API_KEY environment variable. Set it and restart your terminal.")

INDICATOR_ID = 1295  # Generación T.Real Solar fotovoltaica
URL = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"  

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
    "Accept-Encoding": "gzip, deflate"  
}
    
# Updated to save as a Parquet file in the script's directory
OUTFILE = SCRIPT_DIR / "esios_solar_pv_generation_hourly_2022_2025.parquet"

# -----------------------
# GENERADOR DE RANGOS 
# -----------------------
date_ranges = []
for year in [2022, 2023, 2024, 2025]:
    for month in range(1, 13):
        start_m = f"{year}-{month:02d}-01T00:00:00Z"
        if month == 12:
            end_m = f"{year+1}-01-01T00:00:00Z"
        else:
            end_m = f"{year}-{month+1:02d}-01T00:00:00Z"
            
        if year == 2022 and month == 1:
            start_m = "2021-12-31T21:30:00Z"
            
        date_ranges.append((start_m, end_m))

# -----------------------
# DOWNLOAD LOOP
# -----------------------
parts = []  

for start_date, end_date in date_ranges:
    print(f"Requesting {start_date} -> {end_date}")

    params = {
        "start_date": start_date,            
        "end_date": end_date,                
    }

    r = requests.get(URL, headers=HEADERS, params=params, timeout=60)
    print("Status:", r.status_code)
    
    if r.status_code != 200:
        print("Body:", r.text)
        raise RuntimeError(f"API call failed when asked for the period {start_date}. Status:{r.status_code}.")

    data = r.json()
    values = data.get("indicator", {}).get("values", [])

    if not values:
         raise ValueError(f"No hay datos devueltos para el mes {start_date} -> {end_date}.")

    df = pd.DataFrame(values)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["time"] = df["datetime_utc"].dt.tz_convert("Europe/Madrid") # Renamed for merge compatibility

    df = df[["time", "value"]]
    parts.append(df)
    
    time.sleep(1)

# ==========================================
# DATA AGGREGATION & EXPORT
# ==========================================
print("\nAggregating sub-hourly generation into 1-Hour Energy Blocks (MWh)...")
final_df = pd.concat(parts, ignore_index=True)
final_df = final_df.drop_duplicates(subset=["time"])
final_df = final_df.sort_values("time")

# Set 'time' as index to resample
final_df = final_df.set_index("time")

# Average the MW values over 1-hour intervals to yield Hourly MWh
final_df = final_df.resample("1h").mean().reset_index()

# Rename value to explicitly state the unit
final_df = final_df.rename(columns={"value": "target_generation_mwh"})

# Drop the inclusive boundary point if it exists
final_df = final_df[final_df["time"] < "2026-01-01 00:00:00+01:00"]

# Save to Parquet
final_df.to_parquet(OUTFILE, engine="pyarrow", index=False)

# -----------------------
# QUICK SANITY CHECKS
# -----------------------
print(f"\nSaved: {OUTFILE}")
print("Rows:", len(final_df))
print("Min local time:", final_df["time"].min())
print("Max local time:", final_df["time"].max())
print("Max MWh:", final_df["target_generation_mwh"].max())