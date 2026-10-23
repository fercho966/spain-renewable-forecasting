import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

OUTPUT_DIR = SCRIPT_DIR / "data_gdp"
CITIES_FILE = r"C:\Users\fg384\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model\City Coordinates\spanishpeninsula_cities_over_50k_lat&long.csv"

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Madrid"
CITY_BATCH_SIZE = 25

# Exact Total Regional GDP (in Millions of Euros) from INE Image
EXACT_GDP_DATA = {
    "AN": {2022: 184364, 2023: 199406, 2024: 212359},
    "AR": {2022: 43669,  2023: 46778,  2024: 49635},
    "AS": {2022: 26690,  2023: 28350,  2024: 30012},
    "IB": {2022: 37189,  2023: 41503,  2024: 44693},
    "CN": {2022: 48908,  2023: 53990,  2024: 58145},
    "CB": {2022: 15557,  2023: 16741,  2024: 17700},
    "CL": {2022: 65467,  2023: 69894,  2024: 73993},
    "CM": {2022: 50032,  2023: 52545,  2024: 55560},
    "CT": {2022: 258787, 2023: 283274, 2024: 302304},
    "VC": {2022: 127583, 2023: 139656, 2024: 148283},
    "EX": {2022: 23487,  2023: 25193,  2024: 26583},
    "GA": {2022: 71636,  2023: 77461,  2024: 81840},
    "MD": {2022: 267172, 2023: 295940, 2024: 316193},
    "MC": {2022: 37381,  2023: 39839,  2024: 42488},
    "NC": {2022: 23204,  2023: 24975,  2024: 26619},
    "PV": {2022: 80218,  2023: 86748,  2024: 91613},
    "RI": {2022: 9767,   2023: 10477,  2024: 11112},
    "CE": {2022: 1767,   2023: 1863,   2024: 1923},
    "ML": {2022: 1656,   2023: 1754,   2024: 1825}
}

# 2025 Growth Multipliers
GROWTH_2025 = {
    "AN": 1.033, "AR": 1.029, "AS": 1.028, "CB": 1.021, 
    "CL": 1.029, "CM": 1.029, "CT": 1.028, "VC": 1.027, 
    "EX": 1.021, "GA": 1.026, "MD": 1.033, "MC": 1.026, 
    "NC": 1.020, "PV": 1.020, "RI": 1.032,
    "IB": 1.028, "CN": 1.028, "CE": 1.028, "ML": 1.028 # Island/City fallbacks
}

# Pre-calculate 2025 exact values based on the multipliers
for region, data in EXACT_GDP_DATA.items():
    data[2025] = int(data[2024] * GROWTH_2025[region])

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
    for i in range(0, len(cities_df), batch_size):
        yield cities_df.iloc[i:i+batch_size].copy(), (i // batch_size) + 1

# =========================
# MAIN PIPELINE
# =========================
def main():
    cities_df = pd.read_csv(CITIES_FILE, encoding='latin1')
    cities_df["name"] = cities_df["name"].astype(str).str.strip()
    
    total_files = 0
    print("Generating Hourly GDP puzzle pieces using EXACT INE data...")

    for d_start, d_end in month_chunks(START_DATE, END_DATE):
        year = int(d_start[:4])
        
        month_times = pd.date_range(start=f"{d_start} 00:00:00", end=f"{d_end} 23:59:59", freq="h", tz=TIMEZONE)
        folder = OUTPUT_DIR / d_start[:7]
        folder.mkdir(parents=True, exist_ok=True)
        
        for batch_df, batch_idx in city_batches(cities_df, CITY_BATCH_SIZE):
            out_file = folder / f"gdp_{d_start}_to_{d_end}_batch_{batch_idx:02d}.parquet"
            batch_data = []
            
            for _, row in batch_df.iterrows():
                region = row["region_code"]
                
                # Fetch the exact year's value. If something is missing, fallback to 2024
                yearly_gdp = EXACT_GDP_DATA.get(region, {}).get(year, EXACT_GDP_DATA[region][2024])
                
                df_city = pd.DataFrame({
                    "time": month_times,
                    "city": row["name"],
                    "regional_gdp_millions": yearly_gdp
                })
                batch_data.append(df_city)
            
            pd.concat(batch_data, ignore_index=True).to_parquet(out_file, index=False)
            total_files += 1

    print(f"\n--- GDP DATA GENERATION COMPLETE ---")
    print(f"Total Files: {total_files}")
    print(f"Structure matched to weather/holidays/price in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()