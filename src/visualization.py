import matplotlib.pyplot as plt
import seaborn as sns

def plot_fraud_map(df, plots_dir="plots"):
    print("Generating Fraud Map...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot non-fraud as small grey dots
    non_fraud = df[df['is_fraud'] == 0]
    plt.scatter(non_fraud['longitude'], non_fraud['latitude'], 
                alpha=0.1, s=1, c='grey', label='Non-Fraud')
                
    # Plot fraud as red dots
    fraud = df[df['is_fraud'] == 1]
    plt.scatter(fraud['longitude'], fraud['latitude'], 
                alpha=0.6, s=10, c='red', label='Fraud')
                
    plt.title('Geospatial Distribution of Transactions')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.savefig(f"{plots_dir}/fraud_map.png")
    plt.close()
