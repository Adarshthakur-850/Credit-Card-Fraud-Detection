from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import add_geospatial_features
from src.eda import run_eda
from src.visualization import plot_fraud_map
from src.model import train_models
from src.evaluation import evaluate_models
import joblib
import os

def main():
    print("Starting Credit Card Fraud Detection Pipeline...")
    
    # 1. Load Data
    df = load_data()
    print(f"Data Schema: {df.columns}")
    
    # 2. Preprocessing
    df = preprocess_data(df)
    
    # 3. Feature Engineering
    df = add_geospatial_features(df)
    
    # 4. EDA & Vis
    run_eda(df)
    plot_fraud_map(df)
    
    # 5. Model Training
    models, X_test, y_test = train_models(df)
    
    # 6. Evaluation
    results = evaluate_models(models, X_test, y_test)
    print("\nModel Comparison:")
    print(results)
    
    # 7. Save Best Model (e.g., XGBoost)
    if not os.path.exists("models"):
        os.makedirs("models")
        
    best_model = models.get('XGBoost') or models.get('Random Forest')
    joblib.dump(best_model, "models/fraud_model.pkl")
    print("\nPipeline completed. Model saved to models/fraud_model.pkl")

if __name__ == "__main__":
    main()
