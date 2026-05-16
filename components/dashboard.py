import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

def render_risk_dashboard():
    st.markdown("## 📈 Risk & Anomaly")
    
    if "demo_data_path" not in st.session_state:
        st.info("Generating demo time-series data...")
        from utils.data_generator import generate_synthetic_timeseries
        st.session_state.demo_data_path = generate_synthetic_timeseries()
        st.rerun()
        
    df = pd.read_csv(st.session_state.demo_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Anomaly Detection using Isolation Forest
    features = ['Traffic', 'CPU_Usage', 'Error_Rate']
    clf = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = clf.fit_predict(df[features])
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    anomalies = df[df['Anomaly'] == 1]
    
    st.markdown("### Traffic Overview with Anomalies")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Traffic'], mode='lines', name='Traffic', line=dict(color='#4CAF50')))
    fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Traffic'], mode='markers', name='Anomaly', 
                             marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(
        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
        font_color='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### CPU Usage")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['CPU_Usage'], mode='lines', name='CPU', line=dict(color='#2196F3')))
        fig2.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['CPU_Usage'], mode='markers', name='Anomaly', 
                                  marker=dict(color='red', size=8, symbol='x')))
        fig2.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)
        
    with cols[1]:
        st.markdown("### Error Rate")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Error_Rate'], mode='lines', name='Error Rate', line=dict(color='#FF9800')))
        fig3.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Error_Rate'], mode='markers', name='Anomaly', 
                                  marker=dict(color='red', size=8, symbol='x')))
        fig3.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
        st.plotly_chart(fig3, use_container_width=True)
        
    if not anomalies.empty:
        st.warning(f"Detected {len(anomalies)} anomalies in the current period. Use the Agent Chat to investigate.")
