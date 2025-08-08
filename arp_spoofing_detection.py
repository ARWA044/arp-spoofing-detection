import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 1. Load Datasets
train_df = pd.read_csv('ARP_Spoofing_train.pcap.csv')
test_df = pd.read_csv('ARP_Spoofing_test.pcap.csv')


# 2. Label Data Using Heuristics

# Thresholds based on train data quantiles
rate_threshold = train_df['Rate'].quantile(0.95)
iat_threshold = train_df['IAT'].quantile(0.05)

def label_row(row):
    """Label packets as spoofed (1) or normal (0) based on feature thresholds."""
    if row['Rate'] > rate_threshold or row['ARP'] == 1 or row['IAT'] < iat_threshold:
        return 1
    return 0

train_df['label'] = train_df.apply(label_row, axis=1)
test_df['label'] = test_df.apply(label_row, axis=1)


# 3. Select Features and Labels
features = ['Rate', 'ARP', 'IAT']

X_train = train_df[features]
y_train = train_df['label']

X_test = test_df[features]
y_test = test_df['label']


# 4. Data Cleaning: Replace inf/-inf and drop missing
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

# Align labels with cleaned features
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]


# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Helper Function to Evaluate Models
def evaluate_model(name, y_true, y_pred, results):
    """Calculate and store classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": f"{acc*100:.1f}%",
        "Precision": f"{prec*100:.1f}%",
        "Recall": f"{rec*100:.1f}%",
        "F1-Score": f"{f1*100:.1f}%"
    })

results = []


# 7. Train and Evaluate Models

# 7.1 Random Forest (Supervised)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
evaluate_model("Random Forest", y_test, rf_pred, results)

# 7.2 XGBoost (Supervised)
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
evaluate_model("XGBoost", y_test, xgb_pred, results)

# 7.3 Support Vector Machine (Supervised)
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
evaluate_model("SVM", y_test, svm_pred, results)

# 7.4 Isolation Forest (Unsupervised)
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(X_train_scaled)
iso_pred_raw = iso.predict(X_test_scaled)
# Map Isolation Forest output: -1 (anomaly) → 1, 1 (normal) → 0
iso_pred = np.where(iso_pred_raw == -1, 1, 0)
evaluate_model("Isolation Forest", y_test, iso_pred, results)

# 7.5 Autoencoder using MLPRegressor (Unsupervised Anomaly Detection)
pca = PCA(n_components=3, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

autoencoder = MLPRegressor(hidden_layer_sizes=(3,), max_iter=500, random_state=42)
autoencoder.fit(X_train_pca, X_train_pca)
reconstructions = autoencoder.predict(X_test_pca)
mse = np.mean((reconstructions - X_test_pca) ** 2, axis=1)

# Threshold for anomaly detection (99th percentile)
threshold = np.percentile(mse, 99)
ae_pred = (mse > threshold).astype(int)
evaluate_model("Autoencoder", y_test, ae_pred, results)


# 8. Show Results in DataFrame
results_df = pd.DataFrame(results)
print(results_df)


# 9. Visualization Functions
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    plt.title(f"{model_name} Confusion Matrix")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = ['Normal', 'Spoofed']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


# 10. Plot Confusion Matrices and ROC Curves

# Random Forest
plot_confusion_matrix(y_test, rf_pred, "Random Forest")
plot_roc_curve(y_test, rf.predict_proba(X_test_scaled)[:, 1], "Random Forest")

# XGBoost
plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
plot_roc_curve(y_test, xgb.predict_proba(X_test_scaled)[:, 1], "XGBoost")

# SVM
plot_confusion_matrix(y_test, svm_pred, "SVM")
try:
    svm_scores = svm.decision_function(X_test_scaled)
    plot_roc_curve(y_test, svm_scores, "SVM")
except AttributeError:
    pass  # No decision_function available, skip ROC

# Isolation Forest (no ROC, only confusion matrix)
plot_confusion_matrix(y_test, iso_pred, "Isolation Forest")

# Autoencoder (no ROC, only confusion matrix)
plot_confusion_matrix(y_test, ae_pred, "Autoencoder")

# 11. t-SNE Visualization of Dataset

# Combine train and test data for visualization
X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))
y_all = pd.concat([y_train, y_test])

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_all_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[y_all == 0, 0], X_embedded[y_all == 0, 1], 
            label='Normal', alpha=0.5, s=20, color='blue')
plt.scatter(X_embedded[y_all == 1, 0], X_embedded[y_all == 1, 1], 
            label='Spoofed', alpha=0.5, s=20, color='red')
plt.legend()
plt.title("t-SNE Visualization of ARP Spoofing Dataset")
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.grid(True)
plt.show()
