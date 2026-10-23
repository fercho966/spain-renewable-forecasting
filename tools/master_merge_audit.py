import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Load Prediction Model")
FILE_PATH = BASE_DIR / "master_training_dataset.parquet"

def main():
    print("--- INITIATING POST-MERGE DATA AUDIT ---\n")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Row Count Check
    expected_rows = 4873896
    actual_rows = len(df)
    print(f"1. ROW COUNT CHECK:")
    print(f"   Expected Rows: {expected_rows:,}")
    print(f"   Actual Rows:   {actual_rows:,}")
    if actual_rows == expected_rows:
        print("   ✅ SUCCESS: Row count matches perfectly. No data was dropped or duplicated.")
    else:
        print(f"   ⚠️ WARNING: Row count mismatch! Difference: {actual_rows - expected_rows:,} rows.")

    # 2. Duplicate Check
    print("\n2. DUPLICATE CHECK:")
    duplicates = df.duplicated(subset=['time', 'city']).sum()
    if duplicates == 0:
        print("   ✅ SUCCESS: No duplicate time/city combinations found.")
    else:
        print(f"   ⚠️ WARNING: Found {duplicates:,} duplicate rows!")

    # 3. Missing Data (NaN) Scan
    print("\n3. MISSING DATA (NaN) SCAN:")
    print("   (This tells us if any dataset failed to join properly)")
    missing_data = df.isna().sum()
    missing_percentages = (missing_data / actual_rows) * 100
    
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"   ⚠️ {col}: {missing_data[col]:,} missing values ({missing_percentages[col]:.2f}%)")
        else:
            print(f"   ✅ {col}: 100% Complete")

    print("\n--- AUDIT COMPLETE ---")

if __name__ == "__main__":
    main()
