import streamlit as st
from typing import Dict, Any

def render_geometry_metrics(metrics: Dict[str, Any]):
    """
    Renders the custom geometry metrics panel.
    """
    st.markdown("### 📐 Geometry-Aware Consolidation Metrics")
    
    if not metrics:
        st.info("No metrics available. Please ingest a document first.")
        return
        
    cols = st.columns(4)
    
    with cols[0]:
        val = f"{metrics.get('original_count', 0)} → {metrics.get('consolidated_count', 0)}"
        ratio = metrics.get('consolidation_ratio', 1.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Consolidation Ratio</div>
            <div class="metric-value">{val}</div>
            <div style="font-size: 0.8rem; color: #aaa;">{(1-ratio)*100:.1f}% reduction in nodes</div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        d_eff = metrics.get('d_eff', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Effective Dimension (d_eff)</div>
            <div class="metric-value">{d_eff:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[2]:
        spread = metrics.get('mean_spread', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Spread ($\\bar{{d}}$)</div>
            <div class="metric-value">{spread:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[3]:
        theta = metrics.get('theta_bound', 0.85)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Identity Error Bound ($\\theta$)</div>
            <div class="metric-value" style="color: #4CAF50;">≥ {theta}</div>
        </div>
        """, unsafe_allow_html=True)
