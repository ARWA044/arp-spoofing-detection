import streamlit as st
from utils.data_loader import load_data, preprocess_data, train_models
from utils.alerts import show_success
from components.metrics import show_metrics_table

def show():
    st.header("🤖 Model Training & Evaluation")
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        st.error("Failed to load data files. Please ensure ARP_Spoofing_train.pcap.csv and ARP_Spoofing_test.pcap.csv are in the correct directory.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data(train_df, test_df)

    if st.button("Train All Models", type="primary"):
        with st.spinner("Training models..."):
            models, predictions = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
            st.session_state.models = models
            st.session_state.predictions = predictions
            show_success("Models trained successfully!")

    if 'models' in st.session_state:
        st.subheader("Model Performance")
        results = []
        for model_name, pred_data in st.session_state.predictions.items():
            from utils.network_utils import calculate_metrics
            metrics = calculate_metrics(y_test, pred_data['pred'])
            results.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']*100:.1f}%",
                'Precision': f"{metrics['Precision']*100:.1f}%",
                'Recall': f"{metrics['Recall']*100:.1f}%",
                'F1-Score': f"{metrics['F1-Score']*100:.1f}%"
            })
        show_metrics_table(results)
        best_model = max(results, key=lambda x: float(x['Accuracy'].rstrip('%')))
        st.success(f"🏆 Best performing model: {best_model['Model']} with {best_model['Accuracy']} accuracy")

