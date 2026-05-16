import streamlit as st
import pandas as pd
from backend.knowledge_graph import build_knowledge_graph, generate_pyvis_html
import os
import streamlit.components.v1 as components

def render_chat_interface(orchestrator, audit_logger):
    st.markdown("## 💬 Intelligent Agent Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "validation" in message and message["validation"]:
                v = message["validation"]
                color = "green" if v.get("is_grounded") else "red"
                st.markdown(f"<span style='color:{color}; font-size:0.8rem;'>Confidence: {v.get('confidence_score', 0)*100:.2f} - {v.get('explanation', '')}</span>", unsafe_allow_html=True)
            if "trace" in message and message["trace"]:
                with st.expander("Agent Trace & Sourcing"):
                    for t in message["trace"]:
                        st.json(t)
            if "subgraph" in message and message["subgraph"]:
                with st.expander("Query-Specific Knowledge Graph"):
                    components.html(message["subgraph"], height=400)

    # React to user input
    if prompt := st.chat_input("Ask about the enterprise documents or anomalies..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Agents are analyzing..."):
                anomaly_context = ""
                if "demo_data_path" in st.session_state:
                    df = pd.read_csv(st.session_state.demo_data_path)
                    anomaly_context = df.tail(30).to_string() # Just passing tail for context
                
                try:
                    result = orchestrator.run_query(prompt, anomaly_context=anomaly_context)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                        st.error("⚠️ Gemini API rate limit reached. Please wait a minute and try again.")
                    else:
                        st.error(f"An unexpected error occurred: {err_str}")
                    st.stop()
                
                # Render response
                st.markdown(result["final_response"])
                
                val = result.get("validation", {})
                color = "green" if val.get("is_grounded") else "red"
                st.markdown(f"<span style='color:{color}; font-size:0.8rem;'>Confidence: {val.get('confidence_score', 0)*100:.2f} - {val.get('explanation', '')}</span>", unsafe_allow_html=True)
                
                with st.expander("Agent Trace & Sourcing"):
                    for t in result["trace"]:
                        st.json(t)
                        
                subgraph_html = ""
                if result.get("nodes_info"):
                    # Generate dynamic subgraph
                    # In a real app, we would get the full text from the nodes, here we will simulate with the trace
                    try:
                        nodes_content = []
                        if orchestrator.rag_engine:
                            # Just re-retrieve to get the content for subgraph
                            nodes = orchestrator.rag_engine.get_retriever().retrieve(prompt)
                            nodes_content = [n.get_content() for n in nodes]
                        
                        if nodes_content:
                            sg = build_knowledge_graph(nodes_content)
                            if len(sg.nodes) > 0:
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w') as tmp:
                                    tmp_path = tmp.name
                                path = generate_pyvis_html(sg, tmp_path)
                                with open(path, 'r', encoding='utf-8') as f:
                                    subgraph_html = f.read()
                                with st.expander("Query-Specific Knowledge Graph"):
                                    components.html(subgraph_html, height=400)
                    except Exception as e:
                        print(f"Failed to generate subgraph: {e}")

                # Log
                if orchestrator.rag_engine and orchestrator.rag_engine.gac_metrics:
                    metrics = orchestrator.rag_engine.gac_metrics[-1]
                else:
                    metrics = {}
                audit_logger.log_query(prompt, result["final_response"], result["trace"], metrics)

        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["final_response"],
            "validation": val,
            "trace": result["trace"],
            "subgraph": subgraph_html
        })
