import pandas as pd
from pathlib import Path
import difflib
import unicodedata

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(r"C:\Users\fg384\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model\Population")

# ⚠️ Make sure your file extension matches exactly what you saved! (.xlsx or .csv)
INE_FILE = BASE_DIR / "INE_population.xlsx" 
CITIES_FILE = r"C:\Users\fg384\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model\City Coordinates\spanishpeninsula_cities_over_50k_lat&long.csv"
OUTPUT_DIR = BASE_DIR / "Population" / "data_population"

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Madrid"
CITY_BATCH_SIZE = 25

# =========================
# HELPER FUNCTIONS
# =========================
def normalize_text(text):
    """Removes accents, hidden spaces, and converts to uppercase for perfect matching"""
    if pd.isna(text):
        return ""
    text = str(text).upper().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

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
    print("1. Loading and cleaning INE Population data...")
    
    # Read the clean Excel file. (If you saved as CSV, use pd.read_csv(INE_FILE) instead)
    df_ine = pd.read_excel(INE_FILE)
    
    # Apply our bulletproof normalizer to your new 'Ciudades' column
    df_ine['clean_ine_name'] = df_ine['Ciudades'].apply(normalize_text)

    # Build lookup dictionary directly from your clean columns
    pop_lookup = {}
    for _, row in df_ine.iterrows():
        city = row['clean_ine_name']
        if city not in pop_lookup:
            pop_lookup[city] = {}
            
        # Safely grab the data from the year columns
        for year in [2022, 2023, 2024, 2025]:
            # Pandas sometimes reads headers as integers or strings, so we check both
            if year in df_ine.columns:
                pop_lookup[city][year] = int(row[year])
            elif str(year) in df_ine.columns:
                pop_lookup[city][year] = int(row[str(year)])

    print("2. Mapping INE names to your 138 Cities...")
    cities_df = pd.read_csv(CITIES_FILE, encoding='latin1')
    cities_df["norm_name"] = cities_df["name"].apply(normalize_text)
    
    ine_city_names = list(pop_lookup.keys())
    final_mapping = {}
    missing_count = 0
    
    for _, row in cities_df.iterrows():
        my_city_norm = row["norm_name"]
        original_name = row["name"]
        
        if my_city_norm in pop_lookup:
            final_mapping[original_name] = my_city_norm
        else:
            matches = difflib.get_close_matches(my_city_norm, ine_city_names, n=1, cutoff=0.6)
            if matches:
                final_mapping[original_name] = matches[0]
            else:
                print(f"⚠️ WARNING: Could not find a match for: {original_name} (Normalized: {my_city_norm})")
                final_mapping[original_name] = None
                missing_count += 1

    if missing_count == 0:
        print("✅ SUCCESS: All 138 cities matched perfectly!")

    print("\n3. Generating Hourly Population puzzle pieces...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for d_start, d_end in month_chunks(START_DATE, END_DATE):
        year = int(d_start[:4])
        month_times = pd.date_range(start=f"{d_start} 00:00:00", end=f"{d_end} 23:59:59", freq="h", tz=TIMEZONE)
        folder = OUTPUT_DIR / d_start[:7]
        folder.mkdir(parents=True, exist_ok=True)
        
        for batch_df, batch_idx in city_batches(cities_df, CITY_BATCH_SIZE):
            out_file = folder / f"pop_{d_start}_to_{d_end}_batch_{batch_idx:02d}.parquet"
            batch_data = []
            
            for _, row in batch_df.iterrows():
                original_name = row["name"]
                ine_name_norm = final_mapping.get(original_name)
                pop_val = row["population"] # Fallback
                
                if ine_name_norm and ine_name_norm in pop_lookup:
                    if year in pop_lookup[ine_name_norm]:
                        pop_val = pop_lookup[ine_name_norm][year]
                    else:
                        pop_val = pop_lookup[ine_name_norm].get(2024, pop_val)

                df_city = pd.DataFrame({
                    "time": month_times,
                    "city": original_name,
                    "dynamic_population": int(pop_val)
                })
                batch_data.append(df_city)
            
            pd.concat(batch_data, ignore_index=True).to_parquet(out_file, index=False)
            total_files += 1

    print(f"\n--- POPULATION DATA GENERATION COMPLETE ---")
    print(f"Total Files: {total_files}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()