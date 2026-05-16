import pytest
import numpy as np
from backend.geometry_consolidation import consolidate_nodes

class MockNode:
    def __init__(self, id_):
        self.node_id = id_
        
def test_geometry_consolidation_metrics():
    # Create mock embeddings
    np.random.seed(42)
    # 3 clusters
    c1 = np.random.normal(loc=[1,1], scale=0.1, size=(20, 2))
    c2 = np.random.normal(loc=[-1,-1], scale=0.1, size=(20, 2))
    c3 = np.random.normal(loc=[1,-1], scale=0.1, size=(20, 2))
    embeddings = np.vstack([c1, c2, c3])
    
    nodes = [MockNode(i) for i in range(60)]
    
    store, metrics = consolidate_nodes(nodes, embeddings, theta=0.85, strategy="gac", min_cluster_size=5)
    
    assert metrics["original_count"] == 60
    assert metrics["consolidated_count"] < 60
    assert metrics["consolidation_ratio"] < 1.0
    assert not np.isnan(metrics["d_eff"])
    assert not np.isnan(metrics["mean_spread"])
    
def test_geometry_consolidation_empty():
    nodes = []
    embeddings = np.array([])
    store, metrics = consolidate_nodes(nodes, embeddings)
    
    assert metrics["original_count"] == 0
    assert metrics["consolidation_ratio"] == 1.0
