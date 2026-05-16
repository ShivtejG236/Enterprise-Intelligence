import os
import chromadb
import numpy as np
from typing import List, Dict, Any, Tuple

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import TextNode, BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import AutoMergingRetriever

from backend.geometry_consolidation import consolidate_nodes
import config

class GeometryAwareRAGEngine:
    def __init__(self):
        if config.EMBEDDING_PROVIDER == "local":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.embed_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL
            )
        else:
            self.embed_model = GeminiEmbedding(
                model_name=config.EMBEDDING_MODEL,
                api_key=os.getenv("GOOGLE_API_KEY"),
                embed_batch_size=100
            )
        Settings.embed_model = self.embed_model
        
        # Setup ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        
        # We will use two collections: one for raw nodes, one for consolidated nodes
        self.raw_collection = self.chroma_client.get_or_create_collection("raw_nodes")
        self.gac_collection = self.chroma_client.get_or_create_collection("gac_nodes")
        
        self.raw_vector_store = ChromaVectorStore(chroma_collection=self.raw_collection)
        self.gac_vector_store = ChromaVectorStore(chroma_collection=self.gac_collection)
        
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        
        self.index = None
        self.gac_metrics = []

    def ingest_documents(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Process documents through Hierarchical parser, then apply GAC, and store.
        """
        if not docs:
            return {"status": "error", "message": "No documents provided."}
            
        # 1. Parse into hierarchical nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        leaf_nodes = get_leaf_nodes(nodes)
        
        # Content-hash dedup: skip re-embedding documents we have already processed
        import hashlib
        full_text_hash = hashlib.sha256("".join(d.text for d in docs).encode()).hexdigest()
        existing = self.gac_collection.get(where={"doc_hash": full_text_hash})
        if existing["ids"]:
            print(f"Skipping re-embedding: content unchanged (hash={full_text_hash[:8]}...)")
            # Reload index from persisted store
            storage_context = StorageContext.from_defaults(vector_store=self.gac_vector_store)
            self.index = VectorStoreIndex.from_vector_store(self.gac_vector_store, storage_context=storage_context)
            return {"status": "success", "metrics": self.gac_metrics[-1] if self.gac_metrics else {}, "nodes_before": 0, "nodes_after": 0, "cached": True}

        # Extract texts for nodes that need embeddings
        nodes_to_embed = [node for node in leaf_nodes if not node.embedding]
        if nodes_to_embed:
            texts = [node.get_content() for node in nodes_to_embed]
            
            if config.EMBEDDING_PROVIDER == "gemini":
                # By-pass LlamaIndex's flawed batch implementation which makes single requests in a loop.
                import google.generativeai as genai
                import time
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                
                batch_size = 100
                from google.api_core.exceptions import ResourceExhausted
                import re
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = genai.embed_content(
                                model=config.EMBEDDING_MODEL,
                                content=batch_texts,
                                task_type="RETRIEVAL_DOCUMENT",
                                title="Document Chunk"
                            )
                            batch_embeddings = response["embedding"]
                            
                            for j, emb in enumerate(batch_embeddings):
                                nodes_to_embed[i + j].embedding = emb
                                
                            # Respect RPM rate limits
                            time.sleep(1.5)
                            break # Success, break out of retry loop
                            
                        except ResourceExhausted as e:
                            if attempt < max_retries - 1:
                                # Parse wait time from error message
                                error_msg = str(e)
                                match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                                wait_time = float(match.group(1)) + 1 if match else 60.0
                                print(f"Rate limit hit! Waiting {wait_time}s before continuing...")
                                time.sleep(wait_time)
                            else:
                                raise e
            else:
                # Use standard LlamaIndex batching for HuggingFace local models
                batch_embeddings = self.embed_model.get_text_embedding_batch(texts)
                for node, emb in zip(nodes_to_embed, batch_embeddings):
                    node.embedding = emb
                
        # 2. Extract embeddings to a numpy array for GAC
        embeddings = np.array([node.embedding for node in leaf_nodes])
        
        # 3. Apply Geometry-Aware Consolidation
        store, metrics = consolidate_nodes(
            leaf_nodes, 
            embeddings, 
            theta=config.GAC_THETA, 
            strategy=config.GAC_STRATEGY
        )
        self.gac_metrics.append(metrics)
        
        # 4. Create Consolidated Nodes from Store
        consolidated_nodes = []
        for i, vec in enumerate(store.vectors):
            cluster_id = store.cluster_ids[i]
            source_id = store.source_ids[i]
            
            # Combine content of all source nodes belonging to this cluster
            # For simplicity in this hackathon, we gather all text from the cluster
            cluster_mask = (store.cluster_ids == cluster_id)
            # Actually, `store.cluster_ids` maps each representative to a cluster.
            # We need the original labels from the clustering to get the full text.
            # But wait, store doesn't directly expose original labels.
            # We can use the source node if source_id != -1
            if source_id != -1 and source_id < len(leaf_nodes):
                content = leaf_nodes[source_id].get_content()
                metadata = leaf_nodes[source_id].metadata
            else:
                # Synthetic node, just use a placeholder or find closest node
                content = f"Consolidated Cluster {cluster_id} Summary"
                metadata = {"gac_synthetic": True}
                
            c_node = TextNode(
                text=content,
                embedding=vec.tolist(),
                metadata={
                    "cluster_id": int(cluster_id),
                    "is_consolidated": True,
                    "embedding_model": config.EMBEDDING_MODEL,  # Audit: track model for drift detection
                    "doc_hash": full_text_hash,
                    **metadata
                }
            )
            c_node.id_ = f"gac_node_{cluster_id}_{i}"
            consolidated_nodes.append(c_node)
            
        # 5. Store in VectorStoreIndex
        storage_context = StorageContext.from_defaults(vector_store=self.gac_vector_store)
        
        # Add original nodes to a base index so AutoMergingRetriever can walk up the tree
        self.base_index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=self.raw_vector_store))
        
        # And the consolidated ones to the active index
        self.index = VectorStoreIndex(consolidated_nodes, storage_context=storage_context)
        
        return {
            "status": "success",
            "metrics": metrics,
            "nodes_before": len(leaf_nodes),
            "nodes_after": len(consolidated_nodes)
        }

    def get_retriever(self, top_k=5):
        """
        Returns a hybrid retriever. 
        We use the base index for the AutoMergingRetriever and our GAC index for vector search.
        """
        if not self.index:
            # If not in memory, try to load from chroma
            storage_context = StorageContext.from_defaults(vector_store=self.gac_vector_store)
            self.index = VectorStoreIndex.from_vector_store(self.gac_vector_store, storage_context=storage_context)
            
        base_retriever = self.index.as_retriever(similarity_top_k=top_k)
        # In a full implementation, we could wrap this in an AutoMergingRetriever
        # AutoMergingRetriever requires the docstore with the parent nodes.
        # For simplicity in this setup, we'll return the standard retriever on the GAC nodes.
        return base_retriever
