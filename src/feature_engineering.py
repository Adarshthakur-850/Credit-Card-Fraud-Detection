import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def add_geospatial_features(df):
    print("Engineering geospatial features...")
    
    # Group by user to calculate diffs
    df['lat_prev'] = df.groupby('user_id')['latitude'].shift(1)
    df['lon_prev'] = df.groupby('user_id')['longitude'].shift(1)
    df['time_prev'] = df.groupby('user_id')['timestamp'].shift(1)
    
    # Distance
    df['dist_km'] = haversine_distance(
        df['latitude'], df['longitude'],
        df['lat_prev'], df['lon_prev']
    ).fillna(0)
    
    # Time diff in hours
    df['time_diff_h'] = (df['timestamp'] - df['time_prev']).dt.total_seconds() / 3600
    df['time_diff_h'] = df['time_diff_h'].replace(0, 0.001).fillna(0.1)
    
    # Speed (km/h)
    df['speed_kmh'] = df['dist_km'] / df['time_diff_h']
    
    # Cleanup
    df = df.drop(columns=['lat_prev', 'lon_prev', 'time_prev'])
    
    # Replace Inf and NaN
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df
