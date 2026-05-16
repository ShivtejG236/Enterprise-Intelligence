import streamlit as st
import streamlit.components.v1 as components
import os

def render_global_kg():
    st.markdown("## 🕸️ Global Knowledge Graph")
    
    st.info("This graph is built from all consolidated enterprise documents.")
    
    # Check if the global graph HTML exists
    if os.path.exists("global_kg_network.html"):
        with open("global_kg_network.html", 'r', encoding='utf-8') as f:
            html = f.read()
        components.html(html, height=600)
    else:
        st.warning("No global knowledge graph found. Please ingest documents first.")
