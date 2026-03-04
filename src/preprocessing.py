import pandas as pd

def preprocess_data(df):
    print("Preprocessing data...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Sort by user and time for feature engineering
    df = df.sort_values(['user_id', 'timestamp'])
    
    return df
