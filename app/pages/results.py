import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import load_data, preprocess_data
from components.charts import plot_confusion_matrix_plotly, plot_roc_curve_plotly

def show(prediction_mode=False, comparison_mode=False):
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        st.error("Failed to load data files. Please ensure ARP_Spoofing_train.pcap.csv and ARP_Spoofing_test.pcap.csv are in the correct directory.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data(train_df, test_df)

    if prediction_mode:
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
        input_data = np.array([[rate, arp, iat]])
        input_scaled = scaler.transform(input_data)
        st.subheader("Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Predictions:**")
            for model_name, model in st.session_state.models.items():
                if model_name == 'Autoencoder':
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
        return

    if comparison_mode:
        st.header("📋 Model Comparison")
        if 'models' not in st.session_state:
            st.warning("Please train the models first by going to the 'Model Training' page.")
            return
        comparison_data = []
        for model_name, pred_data in st.session_state.predictions.items():
            from utils.network_utils import calculate_metrics
            metrics = calculate_metrics(y_test, pred_data['pred'])
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score']
            })
        comparison_df = pd.DataFrame(comparison_data)
        display_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
        st.subheader("Performance Metrics Comparison")
        st.dataframe(display_df, use_container_width=True)
        st.subheader("Performance Comparison Chart")
        plot_data = comparison_df.melt(
            id_vars=['Model'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            var_name='Metric',
            value_name='Score'
        )
        import plotly.express as px
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
        return

    st.header("📈 Model Visualizations")
    if 'models' not in st.session_state:
        st.warning("Please train the models first by going to the 'Model Training' page.")
        return
    model_choice = st.selectbox("Select Model:", list(st.session_state.predictions.keys()),key="detection_mode_selectbox")
    col1, col2 = st.columns(2)
    with col1:
        pred_data = st.session_state.predictions[model_choice]
        cm_fig = plot_confusion_matrix_plotly(y_test, pred_data['pred'], model_choice)
        st.plotly_chart(cm_fig, use_container_width=True)
    with col2:
        if pred_data['proba'] is not None:
            roc_fig = plot_roc_curve_plotly(y_test, pred_data['proba'], model_choice)
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("ROC curve not available for this model")
    st.subheader("t-SNE Visualization")
    if st.button("Generate t-SNE Plot"):
        with st.spinner("Computing t-SNE..."):
            from sklearn.manifold import TSNE
            X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))
            y_all = pd.concat([y_train, y_test])
            tsne = TSNE(n_components=2, random_state=42)
            X_embedded = tsne.fit_transform(X_all_scaled)
            import plotly.express as px
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

