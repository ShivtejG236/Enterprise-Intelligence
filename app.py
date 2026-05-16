import streamlit as st
import os

from utils.helpers import apply_dark_theme
from backend.rag_engine import GeometryAwareRAGEngine
from backend.analytics_agents import MultiAgentOrchestrator
from backend.audit_logger import AuditLogger
from backend.knowledge_graph import build_knowledge_graph, generate_pyvis_html

from components.sidebar import render_sidebar
from components.geometry_metrics import render_geometry_metrics
from components.chat import render_chat_interface
from components.dashboard import render_risk_dashboard
from components.kg_visualizer import render_global_kg

st.set_page_config(
    page_title="Geometry-Aware Enterprise AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_dark_theme()

# Initialize Backend
@st.cache_resource
def init_backend():
    rag_engine = GeometryAwareRAGEngine()
    orchestrator = MultiAgentOrchestrator(rag_engine=rag_engine)
    audit_logger = AuditLogger()
    return rag_engine, orchestrator, audit_logger

rag_engine, orchestrator, audit_logger = init_backend()
st.session_state.rag_engine = rag_engine

render_sidebar()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Document Ingestion", 
    "💬 Intelligent Chat", 
    "📈 Risk Dashboard", 
    "🕸️ Knowledge Graph"
])

with tab1:
    st.markdown("## 📄 Document Ingestion")
    uploaded_files = st.file_uploader("Upload Enterprise Documents (PDF, TXT)", accept_multiple_files=True)
    
    if st.button("Process & Consolidate"):
        if uploaded_files:
            from llama_index.core import Document
            docs = []
            for f in uploaded_files:
                # Basic text extraction for demo purposes
                text = f.read().decode('utf-8', errors='ignore')
                docs.append(Document(text=text, metadata={"filename": f.name}))
            
            with st.spinner("Processing: Hierarchical Parsing → Gemini Embeddings → Geometry-Aware Consolidation..."):
                res = rag_engine.ingest_documents(docs)
                
                if res.get("status") == "success":
                    st.success("Documents Ingested and Consolidated Successfully!")
                    st.session_state.latest_metrics = res["metrics"]
                    
                    # Also build global KG in background
                    # Gather all consolidated nodes content
                    retriever = rag_engine.get_retriever(top_k=50) # Just get some nodes for demo global graph
                    try:
                        nodes = retriever.retrieve("enterprise") # dummy query to fetch top nodes
                        if nodes:
                            sg = build_knowledge_graph([n.get_content() for n in nodes])
                            generate_pyvis_html(sg, "global_kg_network.html")
                    except Exception as e:
                        print(f"Global KG building failed: {e}")
                else:
                    st.error(f"Error: {res.get('message')}")
        else:
            st.warning("Please upload files first.")
            
    if "latest_metrics" in st.session_state:
        render_geometry_metrics(st.session_state.latest_metrics)

with tab2:
    render_chat_interface(orchestrator, audit_logger)

with tab3:
    render_risk_dashboard()

with tab4:
    render_global_kg()
