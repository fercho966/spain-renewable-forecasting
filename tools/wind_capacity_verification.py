import pyarrow.parquet as pq
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

# Ensure this matches the exact folder name you used in the generation script
TARGET_DIR = SCRIPT_DIR / "capacity_data"

def verify_total_rows():
    print(f"Scanning directory: {TARGET_DIR}")
    
    # Recursively find all parquet files in all month subfolders
    parquet_files = list(TARGET_DIR.rglob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found. Check your directory path.")
        return

    print(f"Found {len(parquet_files)} parquet files. Counting rows via metadata...")
    
    total_rows = 0
    for file in parquet_files:
        # PyArrow reads just the file metadata, bypassing RAM limits
        total_rows += pq.ParquetFile(file).metadata.num_rows

    print("-" * 40)
    print(f"Expected Rows: 25,912,296")
    print(f"Actual Rows:   {total_rows:,}")
    print("-" * 40)
    
    if total_rows == 25912296:
        print("✅ SUCCESS: Data matrix is mathematically perfect!")
    else:
        difference = abs(25912296 - total_rows)
        print(f"❌ MISMATCH: Difference of {difference:,} rows.")

if __name__ == "__main__":
    verify_total_rows()