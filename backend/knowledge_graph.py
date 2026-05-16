import networkx as nx
from pyvis.network import Network
from typing import List, Dict, Any
import json
import os

from backend.gemini_utils import generate_structured_response
import config

def extract_triplets_from_text(text: str) -> List[Dict[str, str]]:
    """
    Use Gemini Pro to extract Knowledge Graph triplets from text.
    """
    prompt = f"""
    You are an expert Data Extractor. Extract entity-relationship triplets from the following text.
    Focus on key business entities, concepts, and quantifiable metrics.
    
    Text:
    {text}
    
    Respond ONLY with a JSON array of objects in this format:
    [
        {{"source": "Entity 1", "relationship": "action or connection", "target": "Entity 2"}}
    ]
    If no meaningful relationships are found, return an empty array [].
    """
    response = generate_structured_response(prompt, model_name=config.GEMINI_REASONING_MODEL)
    if isinstance(response, list):
        return response
    return []

def build_knowledge_graph(nodes_content: List[str]) -> nx.Graph:
    """
    Builds a NetworkX graph from a list of text contents by extracting triplets.
    """
    G = nx.Graph()
    for text in nodes_content:
        # Avoid huge texts, take first 1500 chars for entity extraction to save latency/tokens
        triplets = extract_triplets_from_text(text[:1500])
        for t in triplets:
            src = t.get("source")
            rel = t.get("relationship")
            tgt = t.get("target")
            if src and tgt and rel:
                G.add_node(src, title=src)
                G.add_node(tgt, title=tgt)
                G.add_edge(src, tgt, label=rel)
    return G

def generate_pyvis_html(G: nx.Graph, output_path: str = "kg_network.html"):
    """
    Converts NetworkX graph to a PyVis HTML visualization.
    """
    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="white")
    # For Streamlit compatibility, notebook=False
    net.from_nx(G)
    
    # Physics settings to prevent crazy bouncing
    net.force_atlas_2based()
    
    # Try to save, ignoring the built-in Pyvis HTML warning
    try:
        net.save_graph(output_path)
    except Exception as e:
        print(f"Error saving PyVis graph: {e}")
        
    return output_path
