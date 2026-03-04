import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def load_data(filepath="data/transactions.csv"):
    if not os.path.exists(filepath):
        print(f"Generating synthetic data to {filepath}...")
        generate_synthetic_data(filepath)
    return pd.read_csv(filepath)

def generate_synthetic_data(filepath):
    np.random.seed(42)
    random.seed(42)
    
    n_users = 1000
    n_transactions = 20000
    
    # Base location (e.g., USA center)
    base_lat, base_lon = 39.8, -98.6
    
    data = []
    start_date = datetime.now() - timedelta(days=90)
    
    # User profiles (home location)
    user_profiles = {}
    for uid in range(n_users):
        user_profiles[uid] = {
            'lat': base_lat + np.random.normal(0, 5),
            'lon': base_lon + np.random.normal(0, 10)
        }
        
    for _ in range(n_transactions):
        uid = random.randint(0, n_users-1)
        is_fraud = 0
        
        # 1. Timestamps
        t_delta = random.randint(0, 90*24*60) # minutes
        timestamp = start_date + timedelta(minutes=t_delta)
        
        # 2. Location
        lat = user_profiles[uid]['lat'] + np.random.normal(0, 0.1)
        lon = user_profiles[uid]['lon'] + np.random.normal(0, 0.1)
        
        # 3. Amount
        amount = np.random.exponential(50)
        
        # Inject Fraud Scenarios
        if random.random() < 0.02: # 2% fraud rate
            is_fraud = 1
            scenario = random.choice(['remote', 'high_val', 'rapid'])
            
            if scenario == 'remote':
                # Jump to far location
                lat += np.random.normal(10, 2)
                lon += np.random.normal(10, 2)
            elif scenario == 'high_val':
                amount = np.random.exponential(500)
            
        data.append({
            'transaction_id': f"TXN{random.randint(100000, 999999)}",
            'user_id': uid,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': random.choice(['retail', 'food', 'travel', 'tech']),
            'latitude': lat,
            'longitude': lon,
            'city': 'Unknown',
            'country': 'US',
            'is_fraud': is_fraud
        })
        
    df = pd.DataFrame(data).sort_values('timestamp')
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
