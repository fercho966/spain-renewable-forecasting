import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model")
MASTER_FILE = BASE_DIR / "master_training_dataset.parquet"
OUTPUT_CSV = BASE_DIR / "Missing_Data_Investigation.csv"

def main():
    print("--- INITIATING DATA DETECTIVE ---")
    print("Loading Master Dataset...")
    df = pd.read_parquet(MASTER_FILE)
    
    # Grab the name of the very first city in your dataset (probably A Coruna or Madrid)
    first_city = df['city'].iloc[0]
    print(f"Isolating timeline for {first_city} to pinpoint missing hours...")
    
    # Filter the dataset for just this one city to make it Excel-friendly (35,064 rows)
    df_single_city = df[df['city'] == first_city].copy()
    
    # Keep only the columns we care about right now
    df_diagnostic = df_single_city[['time', 'actual_demand', 'day_ahead_price']]
    
    # Let's find exactly WHERE the data is missing and print it to the terminal
    missing_demand = df_diagnostic[df_diagnostic['actual_demand'].isna()]
    missing_price = df_diagnostic[df_diagnostic['day_ahead_price'].isna()]
    
    print("\n🔍 FINDINGS:")
    print(f"Missing Demand Hour(s):")
    for t in missing_demand['time'].head(5): # Print the first few
        print(f" -> {t}")
        
    print(f"\nMissing Price Hour(s):")
    print(f" -> First missing hour: {missing_price['time'].min()}")
    print(f" -> Last missing hour:  {missing_price['time'].max()}")
    
    # Save the full timeline to CSV so you can scroll through it yourself
    print(f"\nSaving full timeline to CSV for manual Excel inspection...")
    df_diagnostic.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()