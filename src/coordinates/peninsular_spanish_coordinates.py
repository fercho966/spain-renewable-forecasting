import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import os
import sys

print("--- Step 1: Loading Map Geometry (from local file) ---")

# Define the name of the shapefile
shapefile_name = "ne_110m_admin_0_countries.shp"

# Load the local shapefile from your project folder
try:
    world = gpd.read_file(shapefile_name)
except Exception as e:
    print(f"Error loading map: {e}")
    print(f"Please ensure '{shapefile_name}' and its related files (.shx, .dbf) are in the same folder as your script.")
    sys.exit()  # This stops the script if the file isn't found

# Filter for Spain
# .geometry.iloc[0] gets the shape information
spain_geometry = world[world['NAME'] == "Spain"].geometry.iloc[0]

print("Isolating peninsular Spain (largest polygon)...")

# Spain's geometry is a 'MultiPolygon' (mainland + islands).
# We want the largest polygon, which is the peninsula.
if spain_geometry.geom_type == 'MultiPolygon':
    # Get all polygons
    polygons = list(spain_geometry.geoms)
    # Find the one with the largest area
    peninsula_polygon = max(polygons, key=lambda p: p.area)
else:
    # Just in case it's a single polygon
    peninsula_polygon = spain_geometry

print("Peninsula geometry isolated.")

print("\n--- Step 2: Generating and Filtering Coordinate Grid ---")

# Get the bounding box of the peninsula
bounds = peninsula_polygon.bounds
minx, miny, maxx, maxy = bounds

# Define the resolution of our grid (in degrees)
# 0.25 is a good balance. 0.1 is higher resolution.
step = 0.25

# Create grid of points
lons = np.arange(minx, maxx, step)
lats = np.arange(miny, maxy, step)

valid_points = []
print(f"Checking {len(lons) * len(lats)} total points in the bounding box...")

# Loop through the grid and check if points are inside the polygon
for lon in lons:
    for lat in lats:
        point = Point(lon, lat)
        # This is the key check: is the point inside the peninsula?
        if peninsula_polygon.contains(point):
            # We store as (latitude, longitude)
            valid_points.append((lat, lon))

print(f"\n--- Step 3: Saving Results ---")
print(f"Found {len(valid_points)} valid coordinate points inside peninsular Spain.")

# Show user what we found for a quick check
print("\nFirst 10 points (latitude, longitude):")
for p in valid_points[:10]:
    print(f"  ({p[0]:.4f}, {p[1]:.4f})")

# Save all coordinates to a CSV file
# We use pandas to easily save to CSV
df_coords = pd.DataFrame(valid_points, columns=['latitude', 'longitude'])

output_file = "spanish_peninsula_coordinates.csv"
df_coords.to_csv(output_file, index=False)

print(f"\n✅ Success! All {len(df_coords)} coordinates saved to:")
# --- THIS IS THE CORRECTED LINE ---
print(f"   {os.path.abspath(output_file)}")