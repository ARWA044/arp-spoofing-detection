import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv('data/ARP_Spoofing_train.pcap.csv')
        test_df = pd.read_csv('data/ARP_Spoofing_test.pcap.csv')
        return train_df, test_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def preprocess_data(train_df, test_df):
    rate_threshold = train_df['Rate'].quantile(0.95)
    iat_threshold = train_df['IAT'].quantile(0.05)
    def label_row(row):
        if row['Rate'] > rate_threshold or row['ARP'] == 1 or row['IAT'] < iat_threshold:
            return 1
        return 0
    train_df['label'] = train_df.apply(label_row, axis=1)
    test_df['label'] = test_df.apply(label_row, axis=1)
    features = ['Rate', 'ARP', 'IAT']
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    y_test = test_df['label']
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

@st.cache_data
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPRegressor
    from sklearn.decomposition import PCA
    models = {}
    predictions = {}
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    models['Random Forest'] = rf
    predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_proba}
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    models['XGBoost'] = xgb
    predictions['XGBoost'] = {'pred': xgb_pred, 'proba': xgb_proba}
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]
    models['SVM'] = svm
    predictions['SVM'] = {'pred': svm_pred, 'proba': svm_proba}
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train_scaled)
    iso_pred_raw = iso.predict(X_test_scaled)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)
    models['Isolation Forest'] = iso
    predictions['Isolation Forest'] = {'pred': iso_pred, 'proba': None}
    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
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

