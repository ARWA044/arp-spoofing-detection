import streamlit as st


import uuid

def sidebar(key_suffix=None):
    st.sidebar.title("Navigation")
    # Use Streamlit session id if available, else fallback to a random uuid
    session_id = st.session_state.get('_sidebar_session_id', None)
    if session_id is None:
        session_id = str(uuid.uuid4())
        st.session_state['_sidebar_session_id'] = session_id
    selectbox_key = f"sidebar_selectbox_{session_id}"
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "📊 Data Overview",
            "🤖 Model Training",
            "📈 Visualizations",
            "🔮 Predictions",
            "📋 Model Comparison"
        ],
        key=selectbox_key
    )
    return page


