import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD AND PREP THE DATA
# ==========================================
# Load your current Excel/CSV
df_wind = pd.read_csv("spanish-peninsula-windfarms.csv", encoding='latin1')

# Ensure columns are numeric and fill blanks based on our previous logic
df_wind['Capacity (MW)'] = pd.to_numeric(df_wind['Capacity (MW)'], errors='coerce')
df_wind['Start year'] = pd.to_numeric(df_wind['Start year'], errors='coerce').fillna(2020).astype(int)

# Handle Retired year (if blank, we assume it's operating forever, so we set it to 2099)
df_wind['Retired year'] = pd.to_numeric(df_wind['Retired year'], errors='coerce').fillna(2099).astype(int)

print(f"Total individual phases (rows): {len(df_wind)}")
print(f"Total unique physical locations: {df_wind['GEM location ID'].nunique()}")

# Get the definitive coordinates for each location (taking the first phase's coords)
location_coords = df_wind.groupby('GEM location ID')[['Latitude', 'Longitude']].first().reset_index()

# ==========================================
# 2. GENERATE TIMELINE & AGGREGATE
# ==========================================
print("Generating 2022-2025 timeline...")

# Create the hourly timeline (Europe/Madrid localized naive timestamps)
time_idx = pd.date_range(start="2022-01-01 00:00:00", end="2025-12-31 23:00:00", freq='h')
df_time = pd.DataFrame({'time': time_idx})
df_time['year'] = df_time['time'].dt.year

print("Merging phases and calculating active capacities (Warning: High memory)...")
# Cross join timeline with ALL individual phases
df_merged = df_time.merge(df_wind[['GEM location ID', 'Capacity (MW)', 'Start year', 'Retired year']], how='cross')

# The Ultimate Step-Function: Active only if between Start Year and Retired Year
df_merged['active_capacity'] = np.where(
    (df_merged['year'] >= df_merged['Start year']) & (df_merged['year'] < df_merged['Retired year']), 
    df_merged['Capacity (MW)'], 
    0.0
)

print("Collapsing phases into distinct geographic locations...")
# Group by the hour and the unique location ID to sum the capacity of all phases
final_timeline = df_merged.groupby(['time', 'GEM location ID'])['active_capacity'].sum().reset_index()

# Merge the definitive coordinates back in so you have them for the weather download
final_timeline = final_timeline.merge(location_coords, on='GEM location ID', how='left')

# Reorder columns to make it clean
final_timeline = final_timeline[['time', 'GEM location ID', 'Latitude', 'Longitude', 'active_capacity']]

print("\n--- DONE ---")
print("Here is how your timeline handles an ampliación:")
print(final_timeline.head(10))

# Save this to parquet!
# final_timeline.to_parquet("wind_capacity_timeline_2022_2025.parquet", index=False)

# 1. Save the dynamic timeline matrix for your ML model
final_timeline.to_parquet("wind_capacity_timeline_2022_2025.parquet", index=False)

# 2. Save JUST the 739 unique coordinates to feed into your Open-Meteo script
unique_coords = final_timeline[['GEM location ID', 'Latitude', 'Longitude']].drop_duplicates()
unique_coords.to_csv("wind_farm_unique_coordinates.csv", index=False)

print("\nFiles successfully saved!")