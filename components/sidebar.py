import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.info("Geometry-Aware Platform Active")
        
        # We could add sliders for Theta here in a real implementation, 
        # but config.py is sufficient for the hackathon baseline.
        
        st.markdown("---")
        st.markdown("### System Status")
        if "rag_engine" in st.session_state and st.session_state.rag_engine.index is not None:
            st.success("✅ Index Loaded")
        else:
            st.warning("⚠️ No Index Loaded")
            
        if "demo_data_path" in st.session_state:
            st.success("✅ Time Series Data Ready")
        
        st.markdown("---")
        st.markdown("*Lablab.ai Hackathon*")
        st.markdown("*Track: Data & Intelligence*")
