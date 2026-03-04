from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd

def evaluate_models(models, X_test, y_test):
    print("Evaluating models...")
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            'Model': name,
            'Precision (Fraud)': report['1']['precision'],
            'Recall (Fraud)': report['1']['recall'],
            'F1-Score (Fraud)': report['1']['f1-score'],
            'ROC-AUC': auc
        })
        
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {auc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        
    return pd.DataFrame(results)
