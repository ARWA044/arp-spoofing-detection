import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_feature_importance(X_train_scaled, y_train, features):
    rf_temp = RandomForestClassifier(random_state=42)
    rf_temp.fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_temp.feature_importances_
    }).sort_values('Importance', ascending=True)
    return feature_importance

def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
