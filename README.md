# Credit Card Fraud Detection

A production-quality system utilizing Transaction Patterns and Geospatial Anomalies to detect fraud.

## Project Structure
- `data/`: Synthetic dataset with geospatial injection.
- `models/`: Trained Fraud Detection Models.
- `plots/`: EDA and geospatial visualizations.
- `src/`: Core logic modules.
- `main.py`: Pipeline Orchestrator.

## Features
- **Geospatial Engineering**: Haversine distance, speed, and location jumps.
- **Pattern Analysis**: Transaction velocity and amount distribution.
- **Class Imbalance**: Handled using SMOTE.
- **Models**: Logistic Regression, Random Forest, XGBoost.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the pipeline:
```bash
python main.py
```
This will generate data, train models, and output performance metrics.
