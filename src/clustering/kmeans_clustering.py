import pandas as pd
import numpy as np
import re
import os
from sklearn.cluster import DBSCAN

# ==========================================
# 0. FILE PATHS & CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\ferna\OneDrive - University of Exeter\MechEng 3\Individual Project\ML Models\Solar energy generation\Coordinates"

# Inputs
SOLAR_RAW_CSV = os.path.join(BASE_DIR, "spanish-peninsula-solar-parks.csv")

# Outputs
API_TARGETS_CSV = os.path.join(BASE_DIR, "processed_data", "solar_api_extraction_targets.csv")
CLUSTER_METADATA_PARQUET = os.path.join(BASE_DIR, "processed_data", "solar_clustered_metadata.parquet")

# Clustering constraints
MAX_RADIUS_KM = 4.0

def process_solar_spatial_compression():
    # ==========================================
    # 1. LOAD AND PREP THE DATA
    # ==========================================
    print(f"Loading raw solar data from: {SOLAR_RAW_CSV}")
    
    # FIX: Changed encoding to 'utf-8' to properly read Spanish accents (ñ, ú, etc.)
    df = pd.read_csv(SOLAR_RAW_CSV, encoding='utf-8')
    
    # --- HEADER MAPPING ---
    df = df.rename(columns={
        'Project Name': 'project_name',   
        'Capacity (MW)': 'capacity_mw',   
        'Latitude': 'latitude',           
        'Longitude': 'longitude'          
    })
    
    # Ensure numerics
    df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce').fillna(0.0)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Drop rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Initial Raw Solar Phases: {len(df)}")

    # ==========================================
    # 2. MATHEMATICAL CLUSTERING (DBSCAN + Haversine)
    # ==========================================
    print("Executing Haversine density clustering...")
    EARTH_RADIUS_KM = 6371.0
    coords_radians = np.radians(df[['latitude', 'longitude']].values)
    
    dbscan = DBSCAN(
        eps=MAX_RADIUS_KM / EARTH_RADIUS_KM, 
        min_samples=1, 
        algorithm='ball_tree', 
        metric='haversine'
    )
    df['cluster_id'] = dbscan.fit_predict(coords_radians)

    # ==========================================
    # 3. CAPACITY AGGREGATION & ANCHOR NAMING
    # ==========================================
    print("Calculating capacity-weighted centroids and anchor lineages...")
    df['lat_weighted'] = df['latitude'] * df['capacity_mw']
    df['lon_weighted'] = df['longitude'] * df['capacity_mw']
    
    anchor_indices = df.groupby('cluster_id')['capacity_mw'].idxmax()
    anchor_names = df.loc[anchor_indices, ['cluster_id', 'project_name']].set_index('cluster_id')
    
    clustered_df = df.groupby('cluster_id').agg(
        total_capacity_mw=('capacity_mw', 'sum'),
        sum_lat_weighted=('lat_weighted', 'sum'),
        sum_lon_weighted=('lon_weighted', 'sum'),
        sub_farm_lineage=('project_name', lambda x: ' | '.join(x.astype(str))),
        sub_farm_count=('project_name', 'count')
    )
    
    clustered_df = clustered_df.join(anchor_names)
    
    def clean_name(name):
        clean = re.sub(r'\s+([IVXLCDM]+|\d+)$', '', str(name), flags=re.IGNORECASE)
        return clean.strip() + "_Cluster"
        
    clustered_df['cluster_name'] = clustered_df['project_name'].apply(clean_name)
    clustered_df['optimal_latitude'] = clustered_df['sum_lat_weighted'] / clustered_df['total_capacity_mw']
    clustered_df['optimal_longitude'] = clustered_df['sum_lon_weighted'] / clustered_df['total_capacity_mw']
    
    final_cols = [
        'cluster_name', 'optimal_latitude', 'optimal_longitude', 
        'total_capacity_mw', 'sub_farm_count', 'sub_farm_lineage'
    ]
    final_df = clustered_df[final_cols].sort_values(by='total_capacity_mw', ascending=False).reset_index(drop=True)
    
    print(f"Total Grid Capacity Preserved: {final_df['total_capacity_mw'].sum():.2f} MW")
    print(f"Compressed {len(df)} phases into {len(final_df)} API Extraction Targets.")

    # ==========================================
    # 4. EXPORT ARTIFACTS
    # ==========================================
    # FIX: Added 'total_capacity_mw' to the target extraction list
    api_targets = final_df[['cluster_name', 'optimal_latitude', 'optimal_longitude', 'total_capacity_mw']]
    
    os.makedirs(os.path.dirname(API_TARGETS_CSV), exist_ok=True)
    
    # FIX: Added 'utf-8-sig' so Excel reads the Spanish characters flawlessly
    api_targets.to_csv(API_TARGETS_CSV, index=False, encoding='utf-8-sig')
    print(f"SUCCESS: Saved API Extraction Targets to -> {API_TARGETS_CSV}")
    
    final_df.to_parquet(CLUSTER_METADATA_PARQUET, index=False)
    print(f"SUCCESS: Saved Clustered Metadata to -> {CLUSTER_METADATA_PARQUET}")

if __name__ == "__main__":
    process_solar_spatial_compression()