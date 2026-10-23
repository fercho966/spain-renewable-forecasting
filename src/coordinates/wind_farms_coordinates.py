import pandas as pd

# =========================
# CONFIGURATION
# =========================
INPUT_CSV = "spanish-peninsula-windfarms.csv"
OUTPUT_CSV = "wind_farm_unique_coordinates.csv"

def main():
    print("Loading raw wind farm data...")
    # Using latin1 to handle the Spanish accents safely
    df_wind = pd.read_csv(INPUT_CSV, encoding='latin1')
    
    print(f"Total individual phases loaded: {len(df_wind)}")

    # Extract the first occurrence of coordinates for each unique GEM location ID
    unique_coords = df_wind.groupby('GEM location ID')[['Latitude', 'Longitude']].first().reset_index()

    print(f"Extracted {len(unique_coords)} unique physical locations.")
    
    # Save to CSV
    unique_coords.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved successfully to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()