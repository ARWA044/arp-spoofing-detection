import streamlit as st
import pandas as pd
from utils.data_loader import load_data, preprocess_data
from utils.network_utils import get_feature_importance

def show():
    st.header("📊 Dataset Overview")
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        st.error("Failed to load data files. Please ensure ARP_Spoofing_train.pcap.csv and ARP_Spoofing_test.pcap.csv are in the correct directory.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data(train_df, test_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{len(train_df):,}")
    with col2:
        st.metric("Test Samples", f"{len(test_df):,}")
    with col3:
        st.metric("Features", len(features))
    with col4:
        st.metric("Spoofed Rate (Train)", f"{(y_train.sum() / len(y_train) * 100):.1f}%")

    st.subheader("Data Distribution")
    col1, col2 = st.columns(2)
    with col1:
        label_counts = pd.Series(y_train).value_counts()
        import plotly.express as px
        fig = px.pie(
            values=label_counts.values,
            names=['Normal', 'Spoofed'],
            title="Training Data Label Distribution",
            color_discrete_map={'Normal': '#1f77b4', 'Spoofed': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        feature_data = pd.DataFrame(X_train_scaled, columns=features)
        fig = px.box(
            feature_data,
            title="Feature Distributions (Scaled)",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(train_df.describe(), use_container_width=True)

    st.subheader("Feature Importance")
    feature_importance = get_feature_importance(X_train_scaled, y_train, features)
    import plotly.express as px
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
