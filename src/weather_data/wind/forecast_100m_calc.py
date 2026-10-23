import pandas as pd
from pathlib import Path
import sys

# ==========================================
# CONFIGURATION
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

# Input: Your perfectly merged and rescued Forecast data
INPUT_DIR = SCRIPT_DIR / "weather_forecast_MERGED"

# Output: The final destination for the ML model
OUTPUT_DIR = SCRIPT_DIR / "weather_forecast_FINAL_100m"

def calculate_100m_physics():
    print("==================================================")
    print("STARTING 100m HUB-HEIGHT PHYSICS CALCULATIONS")
    print("==================================================")

    parquet_files = list(INPUT_DIR.rglob("*.parquet"))
    total_files = len(parquet_files)
    
    if total_files == 0:
        print("[ERROR] No files found to process.")
        return

    print(f"Processing {total_files} files...\n")
    
    for i, file_path in enumerate(parquet_files, 1):
        # 1. Read the file
        df = pd.read_parquet(file_path)
        
        # 2. Wind Speed Arithmetic Average
        df['wind_speed_100m'] = (df['wind_speed_80m'] + df['wind_speed_120m']) / 2.0
        
        # 3. Wind Direction Circular Average (Prevents 359° and 1° from averaging to 180°)
        diff = (df['wind_direction_120m'] - df['wind_direction_80m'] + 180) % 360 - 180
        df['wind_direction_100m'] = (df['wind_direction_80m'] + (diff / 2.0)) % 360
        
        # 4. Drop the raw 80m and 120m columns to match Actuals perfectly
        df = df.drop(columns=[
            'wind_speed_80m', 'wind_speed_120m', 
            'wind_direction_80m', 'wind_direction_120m'
        ])
        
        # 5. Save to the new directory while keeping the original folder structure
        rel_path = file_path.relative_to(INPUT_DIR)
        out_file = OUTPUT_DIR / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(out_file, index=False)
        
        # Progress update
        sys.stdout.write(f"\r[CALCULATED] {i}/{total_files} files processed.")
        sys.stdout.flush()

    print("\n\n--- CALCULATIONS COMPLETE ---")
    print(f"All 100m Forecast data securely saved to:\n{OUTPUT_DIR.name}")

if __name__ == "__main__":
    calculate_100m_physics()