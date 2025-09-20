import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all detection models and return predictions.
    """
    models = {}
    predictions = {}

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    models['Random Forest'] = rf
    predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_proba}

    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    models['XGBoost'] = xgb
    predictions['XGBoost'] = {'pred': xgb_pred, 'proba': xgb_proba}

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)[:, 1]
    models['SVM'] = svm
    predictions['SVM'] = {'pred': svm_pred, 'proba': svm_proba}

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train)
    iso_pred_raw = iso.predict(X_test)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)
    models['Isolation Forest'] = iso
    predictions['Isolation Forest'] = {'pred': iso_pred, 'proba': None}

    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    autoencoder = MLPRegressor(hidden_layer_sizes=(3,), max_iter=500, random_state=42)
    autoencoder.fit(X_train_pca, X_train_pca)
    reconstructions = autoencoder.predict(X_test_pca)
    mse = np.mean((reconstructions - X_test_pca) ** 2, axis=1)
    threshold = np.percentile(mse, 99)
    ae_pred = (mse > threshold).astype(int)
    ae_proba = mse / np.max(mse)
    models['Autoencoder'] = {'model': autoencoder, 'pca': pca, 'threshold': threshold}
    predictions['Autoencoder'] = {'pred': ae_pred, 'proba': ae_proba}

    return models, predictions

