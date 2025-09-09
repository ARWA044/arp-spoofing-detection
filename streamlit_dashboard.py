import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ARP Spoofing Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the datasets."""
    try:
        train_df = pd.read_csv('ARP_Spoofing_train.pcap.csv')
        test_df = pd.read_csv('ARP_Spoofing_test.pcap.csv')
        return train_df, test_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def preprocess_data(train_df, test_df):
    """Preprocess the data with labeling and scaling."""
    # Label data using heuristics
    rate_threshold = train_df['Rate'].quantile(0.95)
    iat_threshold = train_df['IAT'].quantile(0.05)
    
    def label_row(row):
        if row['Rate'] > rate_threshold or row['ARP'] == 1 or row['IAT'] < iat_threshold:
            return 1
        return 0
    
    train_df['label'] = train_df.apply(label_row, axis=1)
    test_df['label'] = test_df.apply(label_row, axis=1)
    
    # Select features
    features = ['Rate', 'ARP', 'IAT']
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    y_test = test_df['label']
    
    # Clean data
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Align labels
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

@st.cache_data
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train all models and return predictions."""
    models = {}
    predictions = {}
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    models['Random Forest'] = rf
    predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_proba}
    
    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    models['XGBoost'] = xgb
    predictions['XGBoost'] = {'pred': xgb_pred, 'proba': xgb_proba}
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]
    models['SVM'] = svm
    predictions['SVM'] = {'pred': svm_pred, 'proba': svm_proba}
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train_scaled)
    iso_pred_raw = iso.predict(X_test_scaled)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)
    models['Isolation Forest'] = iso
    predictions['Isolation Forest'] = {'pred': iso_pred, 'proba': None}
    
    # Autoencoder
    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    autoencoder = MLPRegressor(hidden_layer_sizes=(3,), max_iter=500, random_state=42)
    autoencoder.fit(X_train_pca, X_train_pca)
    reconstructions = autoencoder.predict(X_test_pca)
    mse = np.mean((reconstructions - X_test_pca) ** 2, axis=1)
    threshold = np.percentile(mse, 99)
    ae_pred = (mse > threshold).astype(int)
    ae_proba = mse / np.max(mse)  # Normalize for probability-like scores
    
    models['Autoencoder'] = {'model': autoencoder, 'pca': pca, 'threshold': threshold}
    predictions['Autoencoder'] = {'pred': ae_pred, 'proba': ae_proba}
    
    return models, predictions

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

def plot_confusion_matrix_plotly(y_true, y_pred, model_name):
    """Create interactive confusion matrix using Plotly."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal', 'Spoofed'],
        y=['Normal', 'Spoofed'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'{model_name} Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=400
    )
    
    return fig

def plot_roc_curve_plotly(y_true, y_scores, model_name):
    """Create interactive ROC curve using Plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        width=500,
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ ARP Spoofing Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        st.error("Failed to load data files. Please ensure ARP_Spoofing_train.pcap.csv and ARP_Spoofing_test.pcap.csv are in the current directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Overview", "🤖 Model Training", "📈 Visualizations", "🔮 Predictions", "📋 Model Comparison"]
    )
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data(train_df, test_df)
    
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", f"{len(train_df):,}")
        with col2:
            st.metric("Test Samples", f"{len(test_df):,}")
        with col3:
            st.metric("Features", len(features))
        with col4:
            st.metric("Spoofed Rate (Train)", f"{(y_train.sum() / len(y_train) * 100):.1f}%")
        
        # Data distribution
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Label distribution
            label_counts = pd.Series(y_train).value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=['Normal', 'Spoofed'],
                title="Training Data Label Distribution",
                color_discrete_map={'Normal': '#1f77b4', 'Spoofed': '#ff7f0e'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature distributions
            feature_data = pd.DataFrame(X_train_scaled, columns=features)
            fig = px.box(
                feature_data,
                title="Feature Distributions (Scaled)",
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        st.dataframe(train_df.describe(), use_container_width=True)
        
        # Feature importance (using Random Forest)
        st.subheader("Feature Importance")
        rf_temp = RandomForestClassifier(random_state=42)
        rf_temp.fit(X_train_scaled, y_train)
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf_temp.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Random Forest)",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("Train All Models", type="primary"):
            with st.spinner("Training models..."):
                models, predictions = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Store in session state
                st.session_state.models = models
                st.session_state.predictions = predictions
                
                st.success("Models trained successfully!")
        
        if 'models' in st.session_state:
            st.subheader("Model Performance")
            
            # Calculate metrics for all models
            results = []
            for model_name, pred_data in st.session_state.predictions.items():
                metrics = calculate_metrics(y_test, pred_data['pred'])
                results.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['Accuracy']*100:.1f}%",
                    'Precision': f"{metrics['Precision']*100:.1f}%",
                    'Recall': f"{metrics['Recall']*100:.1f}%",
                    'F1-Score': f"{metrics['F1-Score']*100:.1f}%"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Best model
            best_model = results_df.loc[results_df['Accuracy'].str.rstrip('%').astype(float).idxmax()]
            st.success(f"🏆 Best performing model: {best_model['Model']} with {best_model['Accuracy']} accuracy")
    
    elif page == "📈 Visualizations":
        st.header("📈 Model Visualizations")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first by going to the 'Model Training' page.")
            return
        
        model_choice = st.selectbox("Select Model:", list(st.session_state.predictions.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            pred_data = st.session_state.predictions[model_choice]
            cm_fig = plot_confusion_matrix_plotly(y_test, pred_data['pred'], model_choice)
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            # ROC Curve (if available)
            if pred_data['proba'] is not None:
                roc_fig = plot_roc_curve_plotly(y_test, pred_data['proba'], model_choice)
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("ROC curve not available for this model")
        
        # t-SNE Visualization
        st.subheader("t-SNE Visualization")
        if st.button("Generate t-SNE Plot"):
            with st.spinner("Computing t-SNE..."):
                # Combine train and test data
                X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))
                y_all = pd.concat([y_train, y_test])
                
                # Compute t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                X_embedded = tsne.fit_transform(X_all_scaled)
                
                # Create plot
                fig = px.scatter(
                    x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    color=y_all.map({0: 'Normal', 1: 'Spoofed'}),
                    title="t-SNE Visualization of ARP Spoofing Dataset",
                    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
                    color_discrete_map={'Normal': '#1f77b4', 'Spoofed': '#ff7f0e'}
                )
                fig.update_layout(width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Predictions":
        st.header("🔮 Real-time Predictions")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first by going to the 'Model Training' page.")
            return
        
        st.subheader("Input Network Traffic Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rate = st.number_input("Rate", min_value=0.0, value=100.0, step=0.1)
        with col2:
            arp = st.selectbox("ARP", [0, 1])
        with col3:
            iat = st.number_input("IAT (Inter-Arrival Time)", min_value=0.0, value=0.01, step=0.001)
        
        # Prepare input data
        input_data = np.array([[rate, arp, iat]])
        input_scaled = scaler.transform(input_data)
        
        # Make predictions
        st.subheader("Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Predictions:**")
            for model_name, model in st.session_state.models.items():
                if model_name == 'Autoencoder':
                    # Special handling for autoencoder
                    input_pca = model['pca'].transform(input_scaled)
                    reconstruction = model['model'].predict(input_pca)
                    mse = np.mean((reconstruction - input_pca) ** 2)
                    prediction = 1 if mse > model['threshold'] else 0
                    confidence = mse / model['threshold']
                else:
                    prediction = model.predict(input_scaled)[0]
                    if hasattr(model, 'predict_proba'):
                        confidence = model.predict_proba(input_scaled)[0][1]
                    else:
                        confidence = 0.5
                
                status = "🚨 SPOOFED" if prediction == 1 else "✅ NORMAL"
                st.write(f"**{model_name}:** {status} (Confidence: {confidence:.3f})")
        
        with col2:
            # Ensemble prediction
            predictions = []
            for model_name, model in st.session_state.models.items():
                if model_name == 'Autoencoder':
                    input_pca = model['pca'].transform(input_scaled)
                    reconstruction = model['model'].predict(input_pca)
                    mse = np.mean((reconstruction - input_pca) ** 2)
                    pred = 1 if mse > model['threshold'] else 0
                else:
                    pred = model.predict(input_scaled)[0]
                predictions.append(pred)
            
            ensemble_pred = 1 if sum(predictions) > len(predictions) / 2 else 0
            ensemble_status = "🚨 SPOOFED" if ensemble_pred == 1 else "✅ NORMAL"
            st.write(f"**Ensemble Prediction:** {ensemble_status}")
            st.write(f"**Voting:** {sum(predictions)}/{len(predictions)} models predict spoofing")
    
    elif page == "📋 Model Comparison":
        st.header("📋 Model Comparison")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first by going to the 'Model Training' page.")
            return
        
        # Calculate detailed metrics
        comparison_data = []
        for model_name, pred_data in st.session_state.predictions.items():
            metrics = calculate_metrics(y_test, pred_data['pred'])
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Convert to percentage for display
        display_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
        
        st.subheader("Performance Metrics Comparison")
        st.dataframe(display_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("Performance Comparison Chart")
        
        # Melt data for plotting
        plot_data = comparison_df.melt(
            id_vars=['Model'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            var_name='Metric',
            value_name='Score'
        )
        
        fig = px.bar(
            plot_data,
            x='Model',
            y='Score',
            color='Metric',
            title="Model Performance Comparison",
            barmode='group',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model recommendations
        st.subheader("Model Recommendations")
        
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        best_precision = comparison_df.loc[comparison_df['Precision'].idxmax()]
        best_recall = comparison_df.loc[comparison_df['Recall'].idxmax()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Best Overall Accuracy:** {best_accuracy['Model']} ({best_accuracy['Accuracy']*100:.1f}%)")
            st.info(f"**Best F1-Score:** {best_f1['Model']} ({best_f1['F1-Score']*100:.1f}%)")
        
        with col2:
            st.info(f"**Best Precision:** {best_precision['Model']} ({best_precision['Precision']*100:.1f}%)")
            st.info(f"**Best Recall:** {best_recall['Model']} ({best_recall['Recall']*100:.1f}%)")

if __name__ == "__main__":
    main()
