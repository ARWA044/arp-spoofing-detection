import streamlit as st
from components.sidebar import sidebar
from pages import dashboard, detection, results

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

st.set_page_config(
    page_title="ARP Spoofing Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

page = sidebar(key_suffix="main")

if page == "📊 Data Overview":
    dashboard.show()
elif page == "🤖 Model Training":
    detection.show()
elif page == "📈 Visualizations":
    results.show()
elif page == "🔮 Predictions":
    results.show(prediction_mode=True)
elif page == "📋 Model Comparison":
    results.show(comparison_mode=True)