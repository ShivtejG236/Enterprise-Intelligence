import numpy as np
from typing import List, Tuple, Dict, Any
from .gac.strategies import GACConsolidator, CompressedStore
from .gac.clustering import cluster_kmeans
from .gac.theory import cluster_spread, d_eff as compute_d_eff

def consolidate_nodes(
    nodes: List[Any], 
    embeddings: np.ndarray, 
    theta: float = 0.85, 
    strategy: str = "gac",
    min_cluster_size: int = 5
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Consolidate a list of LlamaIndex nodes using Geometry-Aware Consolidation.
    
    Args:
        nodes: List of original LlamaIndex nodes (e.g. TextNode)
        embeddings: Numpy array of shape (N, D) containing embeddings for nodes.
        theta: GAC identity error bound threshold (default 0.85).
        strategy: "gac" or "centroid".
        min_cluster_size: Used to estimate K for KMeans.
        
    Returns:
        consolidated_nodes: List of new LlamaIndex nodes representing the compressed set.
        metrics: Dictionary containing d_eff, \bar d, and consolidation ratio.
    """
    N = len(nodes)
    if N <= 1:
        # Nothing to consolidate — return full metrics dict so callers never get a KeyError
        return nodes, {
            "original_count": N,
            "consolidated_count": N,
            "consolidation_ratio": 1.0,
            "d_eff": 0.0,
            "mean_spread": 0.0,
            "theta_bound": theta
        }
        
    # Estimate K based on min_cluster_size
    n_clusters = max(1, N // min_cluster_size)
    if n_clusters == 1:
        labels = np.zeros(N, dtype=np.int64)
    else:
        labels = cluster_kmeans(embeddings, n_clusters=n_clusters)
        
    if strategy == "gac":
        consolidator = GACConsolidator(theta=theta)
    else:
        from .gac.strategies import CentroidConsolidator
        consolidator = CentroidConsolidator()
        
    store = consolidator.fit_transform(embeddings, labels)
    
    # Generate new consolidated nodes
    consolidated_nodes = []
    
    # Calculate global metrics
    d_eff_list = []
    spread_list = []
    
    for k in np.unique(labels):
        if k == -1: continue
        X_k = embeddings[labels == k]
        if len(X_k) > 1:
            d_bar = cluster_spread(X_k)
            d_eff_val = compute_d_eff(X_k)
            spread_list.append(d_bar)
            d_eff_list.append(d_eff_val)
            
    avg_d_eff = np.mean(d_eff_list) if d_eff_list else 0.0
    avg_spread = np.mean(spread_list) if spread_list else 0.0
    
    # We will build consolidated TextNodes in the rag_engine, 
    # so we'll just return the store and let the caller create LlamaIndex nodes.
    # Wait, the interface should probably be cleaner. Let's return the store and metrics, 
    # and let rag_engine handle the LlamaIndex node creation.
    
    metrics = {
        "original_count": N,
        "consolidated_count": len(store.vectors),
        "consolidation_ratio": len(store.vectors) / N if N > 0 else 1.0,
        "d_eff": avg_d_eff,
        "mean_spread": avg_spread,
        "theta_bound": theta
    }
    
    return store, metrics
