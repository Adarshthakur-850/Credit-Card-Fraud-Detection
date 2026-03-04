import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(df, plots_dir="plots"):
    print("Running EDA...")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_fraud', data=df)
    plt.title('Fraud Class Distribution')
    plt.savefig(f"{plots_dir}/class_distribution.png")
    plt.close()
    
    # 2. Amount Distribution (Fraud vs Normal) - log scale
    plt.figure(figsize=(10, 6))
    sns.histplot(x='amount', hue='is_fraud', data=df, bins=50, log_scale=True)
    plt.title('Transaction Amount Distribution')
    plt.savefig(f"{plots_dir}/amount_distribution.png")
    plt.close()
    
    # 3. Fraud by Hour
    plt.figure(figsize=(10, 6))
    fraud_data = df[df['is_fraud'] == 1]
    sns.countplot(x='hour', data=fraud_data)
    plt.title('Fraud Transactions by Hour')
    plt.savefig(f"{plots_dir}/fraud_by_hour.png")
    plt.close()
