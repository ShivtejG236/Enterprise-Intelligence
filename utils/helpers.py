import streamlit as st

def apply_dark_theme():
    """
    Apply global custom CSS for a dark enterprise theme.
    """
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .css-1d391kg {
            background-color: #1E1E1E;
        }
        .stButton>button {
            background-color: #2E3B4E;
            color: white;
            border: 1px solid #4B5E7D;
        }
        .stButton>button:hover {
            background-color: #4B5E7D;
            border-color: #6C84A9;
        }
        .metric-card {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric-title {
            font-size: 0.9rem;
            color: #A0AAB2;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-value.warning {
            color: #FFC107;
        }
        .metric-value.danger {
            color: #F44336;
        }
        </style>
    """, unsafe_allow_html=True)
