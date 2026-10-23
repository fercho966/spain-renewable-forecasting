import os
import requests
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
# Your API token stored as an environment variable
TOKEN = os.environ.get("ESIOS_API_KEY", "YOUR_ESIOS_API_KEY_HERE") 
if not TOKEN:
    raise RuntimeError("Missing ESIOS_API_KEY environment variable. Set it and restart your terminal.")

INDICATOR_ID = 1293  
URL = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"  

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json", 
    "Content-Type": "application/json",                               
    "x-api-key": TOKEN                                                
}

# Added padding chunks for 2021 and 2026 to catch the UTC timezone offsets!
DATE_RANGES = [
    ("2021-12-31T00:00:00Z", "2022-01-01T00:00:00Z"), # Padding Start
    ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ("2025-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), # Padding End
]

OUTFILE = "esios_demand_hourly_2022_2025_LOCALTIME.csv.gz"  

# -----------------------
# DOWNLOAD LOOP
# -----------------------
parts = []  

for start_date, end_date in DATE_RANGES:
    print(f"Requesting {start_date} -> {end_date}")

    params = {
        "start_date": start_date,            
        "end_date": end_date,                
        "time_trunc": "hour",     
        "time_agg": "average"                
    }

    r = requests.get(URL, headers=HEADERS, params=params, timeout=60)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("Body:", r.text)
        raise RuntimeError("API call failed.")

    data = r.json()
    
    # ESIOS sometimes returns empty sets for future padded dates, so we check first
    if "indicator" in data and "values" in data["indicator"]:
        values = data["indicator"]["values"]
        df = pd.DataFrame(values)

        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        df["datetime_local"] = df["datetime_utc"].dt.tz_convert("Europe/Madrid")
        df = df[["datetime_local", "value"]]
        parts.append(df)

final_df = pd.concat(parts, ignore_index=True)
final_df = final_df.drop_duplicates(subset=["datetime_local"])
final_df = final_df.sort_values("datetime_local")

# -----------------------
# STRICT LOCAL TIME TRIMMING
# -----------------------
print("\nTrimming dataset to exact local Madrid boundaries...")
start_bound = pd.to_datetime("2022-01-01 00:00:00").tz_localize("Europe/Madrid")
end_bound = pd.to_datetime("2025-12-31 23:59:59").tz_localize("Europe/Madrid")

final_df = final_df[(final_df["datetime_local"] >= start_bound) & (final_df["datetime_local"] <= end_bound)]

final_df.to_csv(OUTFILE, index=False, compression="gzip")

# -----------------------
# QUICK SANITY CHECKS
# -----------------------
print(f"\nSaved: {OUTFILE}")
print("Rows:", len(final_df))
print("Min local time:", final_df["datetime_local"].min())
print("Max local time:", final_df["datetime_local"].max())