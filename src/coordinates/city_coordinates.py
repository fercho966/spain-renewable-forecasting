import pandas as pd

# --- SETTINGS (edit only if ES.txt is not in the same folder) ---
INPUT_FILE = "ES/ES.txt"  # e.g. r"C:\Users\Fernando\Downloads\ES.txt" or "/Users/fernando/Downloads/ES.txt"
OUTPUT_FILE = "spanishpeninsula_cities_over_50k.csv"
MIN_POP = 50_000

# --- Load GeoNames ES.txt ---
cols = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1", "admin2",
    "admin3", "admin4", "population", "elevation", "dem", "timezone", "modification_date"
]

# --- Load (utf-8 fixes the weird characters) ---
df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    names=cols,
    encoding="utf-8",
    low_memory=False
)

# Clean/standardise key columns
df["feature_code"] = df["feature_code"].astype(str).str.strip()
df["population"] = pd.to_numeric(df["population"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

keep_codes = {"PPL", "PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5"}
df = df[(df["feature_class"] == "P") & (df["feature_code"].isin(keep_codes))]
df = df[df["population"] >= MIN_POP]

cities = df[["name", "latitude", "longitude", "population"]].dropna(subset=["latitude", "longitude"])
cities = cities.drop_duplicates(subset=["name"]).sort_values("population", ascending=False)

# Create "latitude,longitude" in a single column (no spaces)
cities["coords"] = cities["latitude"].map(lambda x: f"{x:.5f}") + "," + cities["longitude"].map(lambda x: f"{x:.5f}")

# Keep only what you want in the output (edit if you also want population)
cities = cities[["name", "coords", "population"]]

cities.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(cities)} cities to {OUTPUT_FILE}")