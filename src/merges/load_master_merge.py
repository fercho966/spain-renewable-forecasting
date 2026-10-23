import pandas as pd
from pathlib import Path
import gc # Garbage collector to free up RAM

# =========================
# CONFIGURATION: YOUR EXACT FOLDER PATHS
# =========================
BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model")

PATHS = {
    "cities_csv": BASE_DIR / "City Coordinates" / "spanishpeninsula_cities_over_50k_lat&long.csv",
    "weather":    BASE_DIR / "Actual Weather Data" / "data_weather_raw" / "historical_weather",
    "forecast":   BASE_DIR / "Weather Forecast Data" / "data_weather_raw" / "weather_forecast_data", 
    "holidays":   BASE_DIR / "Holidays" / "data_holidays",
    "gdp":        BASE_DIR / "GDP" / "data_gdp",
    "population": BASE_DIR / "Population" / "data_population",
    "price":      BASE_DIR / "Day Ahead Market Price Elecrticity" / "data_price",
    "demand":     BASE_DIR / "Peninsular Load Outputs" / "data_demand"
}

# Define the TWO output files here
OUTPUT_FILE_TIME = BASE_DIR / "master_training_dataset_time_sorted.parquet"
OUTPUT_FILE_LOC  = BASE_DIR / "master_training_dataset_location_sorted.parquet"

def load_all_parquets(folder_path):
    files = list(Path(folder_path).rglob("*.parquet"))
    print(f"Loading {len(files)} files from {folder_path.name}...")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    print("--- INITIATING THE GRAND MERGE (CLEAN VERSION) ---")
    
    # 1. Load the Backbone (Weather Actuals)
    df_master = load_all_parquets(PATHS["weather"])
    df_master.drop(columns=["latitude", "longitude", "population"], errors="ignore", inplace=True)
    df_master.rename(columns={c: f"{c}_actual" for c in df_master.columns if c not in ["time", "city"]}, inplace=True)
    
    # 2. Merge Weather Forecasts
    df_forecast = load_all_parquets(PATHS["forecast"])
    df_forecast.drop(columns=["latitude", "longitude", "population"], errors="ignore", inplace=True)
    df_forecast.rename(columns={c: f"{c}_forecast" for c in df_forecast.columns if c not in ["time", "city"]}, inplace=True)
    df_master = pd.merge(df_master, df_forecast, on=["time", "city"], how="left")
    del df_forecast
    gc.collect()
    
    # 3. Merge Holidays
    df_holidays = load_all_parquets(PATHS["holidays"])
    df_master = pd.merge(df_master, df_holidays, on=["time", "city"], how="left")
    del df_holidays
    gc.collect()
    
    # 4. Merge GDP
    df_gdp = load_all_parquets(PATHS["gdp"])
    df_master = pd.merge(df_master, df_gdp, on=["time", "city"], how="left")
    del df_gdp
    gc.collect()

    # 5. Merge Dynamic Population
    df_pop = load_all_parquets(PATHS["population"])
    df_master = pd.merge(df_master, df_pop, on=["time", "city"], how="left")
    del df_pop
    gc.collect()

    # 6. Merge Prices
    df_price = load_all_parquets(PATHS["price"])
    df_master = pd.merge(df_master, df_price, on="time", how="left")
    del df_price
    gc.collect()

    # 7. Merge Demand
    df_demand = load_all_parquets(PATHS["demand"])
    df_master = pd.merge(df_master, df_demand, on="time", how="left")
    del df_demand
    gc.collect()
    
    # 8. Merge Static City Data (Coordinates)
    df_cities = pd.read_csv(PATHS["cities_csv"], encoding='latin1')
    df_cities["name"] = df_cities["name"].astype(str).str.strip()
    
    # Text Surgeon - Fix known Frankenstein text artifacts
    df_cities["name"] = df_cities["name"].replace({
        "A CoruÃ±a": "A Coruña",
        "LogroÃ±o": "Logroño",
        "MÃ¡laga": "Málaga",
        "CÃ¡diz": "Cádiz",
        "CastellÃ³n": "Castellón",
        "AlmerÃ\xada": "Almería",
        "JaÃ©n": "Jaén",
        "LeÃ³n": "León"
    }, regex=True)
    
    static_cols = df_cities[["name", "latitude", "longitude"]]
    df_master = pd.merge(df_master, static_cols, left_on="city", right_on="name", how="left")
    df_master.drop(columns=["name"], inplace=True)
    
    # 9. Column Reordering (Lat/Long right next to City)
    cols = df_master.columns.tolist()
    cols.remove('latitude')
    cols.remove('longitude')
    city_index = cols.index('city')
    cols.insert(city_index + 1, 'latitude')
    cols.insert(city_index + 2, 'longitude')
    df_master = df_master[cols]
    
    # 10. Sorting and Saving TWO versions
    print("\nSorting Master Dataset by TIME...")
    df_master.sort_values(by=["time", "city"], inplace=True)
    df_master.reset_index(drop=True, inplace=True)
    print(f"Saving Time-Sorted Dataset ({len(df_master):,} rows)...")
    df_master.to_parquet(OUTPUT_FILE_TIME, index=False)
    
    print("\nSorting Master Dataset by LOCATION...")
    df_master.sort_values(by=["city", "time"], inplace=True)
    df_master.reset_index(drop=True, inplace=True)
    print(f"Saving Location-Sorted Dataset ({len(df_master):,} rows)...")
    df_master.to_parquet(OUTPUT_FILE_LOC, index=False)
    
    print("\n--- PIPELINE COMPLETE ---")
    print("Final Pristine Columns:")
    for col in df_master.columns:
        print(f" - {col}")

if __name__ == "__main__":
    main()