import os
import time
import requests
import pandas as pd


# 551 Generación T.Real eólica
# 1745 Generación T.Real eólica SNP
# 2038 Generación T.Real eólica nacional

TOKEN = "YOUR_ESIOS_API_KEY_HERE"  # Your API token stored as an environment variable
if not TOKEN:
    raise RuntimeError("Missing ESIOS_API_KEY environment variable. Set it and restart your terminal.")

INDICATOR_ID = 551  
URL = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"  

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",  
    "Content-Type": "application/json",                               
    "x-api-key": TOKEN                                                
}
    
# FIX 1: Changed output format to Parquet to match the pipeline
OUTFILE = "esios_wind_generation_hourly_2022_2025.parquet"  

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
        "time_trunc": "hour",                # FIX 2: UNCOMMENTED AND SET TO 'hour'
        "time_agg": "average",
    }

    r = requests.get(URL, headers=HEADERS, params=params, timeout=60)
    print("Status:", r.status_code)
    
    if r.status_code != 200:
        print("Body:", r.text)
        raise RuntimeError(f"API call failed when asked for the period {start_date}. Status:{r.status_code}.")

    data = r.json()
    values = data.get("indicator", {}).get("values", [])

    if not values:
         raise ValueError(f"No hay datos devueltos para el mes {start_date} -> {end_date}. Revisa la disponibilidad en ESIOS.")

    df = pd.DataFrame(values)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert("Europe/Madrid")

    # FIX 3: Keep only local time and rename columns for the Master Merge!
    df = df[["datetime_local", "value"]].rename(columns={
        "datetime_local": "time",
        "value": "wind_gen_MW"
    })

    parts.append(df)
    time.sleep(1)

# Combine all years
final_df = pd.concat(parts, ignore_index=True)

# Remove duplicates
final_df = final_df.drop_duplicates(subset=["time"])

# Sort by local datetime
final_df = final_df.sort_values("time")

# FIX 4: Save directly to Parquet to preserve the complex timezone datatypes
final_df.to_parquet(OUTFILE, index=False)

# -----------------------
# QUICK SANITY CHECKS
# -----------------------
print(f"\nSaved: {OUTFILE}")
print("Rows:", len(final_df))
print("Min local time:", final_df["time"].min())
print("Max local time:", final_df["time"].max())
print("Max MW:", final_df["wind_gen_MW"].max())
print(final_df.head())