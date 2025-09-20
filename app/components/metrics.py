import pandas as pd
import streamlit as st

def show_metrics_table(results):
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

