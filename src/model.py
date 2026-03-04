from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train_models(df):
    print("Training models...")
    
    # 1. Prepare Data
    feature_cols = ['amount', 'hour', 'day', 'weekday', 'dist_km', 'time_diff_h', 'speed_kmh']
    X = df[feature_cols]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Handle Imbalance (SMOTE) only on training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    models = {}
    
    # 3. Logistic Regression
    print("Training Logistic Regression...")
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
    lr.fit(X_train_res, y_train_res)
    models['Logistic Regression'] = lr
    
    # 4. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    models['Random Forest'] = rf
    
    # 5. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_res, y_train_res)
    models['XGBoost'] = xgb
    
    return models, X_test, y_test
