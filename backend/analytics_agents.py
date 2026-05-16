import json
from typing import Dict, Any, List

import config
from backend.gemini_utils import generate_structured_response, generate_response

class PlannerAgent:
    """
    Decides the best retrieval strategy based on user query.
    Outputs structured JSON.
    """
    def plan(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        You are the Planner Agent for an Enterprise Intelligence Platform.
        Analyze the following query and decide the best execution path.
        Available tools/paths:
        - "vector_search": For semantic knowledge retrieval from consolidated documents.
        - "time_series_anomaly": If the query asks about risk, anomalies, or time-series data.
        - "knowledge_graph": If the query asks for relationships or multi-hop entity reasoning.
        
        Query: "{query}"
        
        Respond ONLY with a JSON object in this format:
        {{
            "path": "<one of the available paths>",
            "reason": "<brief explanation>"
        }}
        """
        return generate_structured_response(prompt, model_name=config.GEMINI_REASONING_MODEL)


class ExecutorAgent:
    """
    Answers the user query using retrieved context (text or anomaly data).
    """
    def execute(self, query: str, context: str, is_anomaly_data: bool = False) -> str:
        if is_anomaly_data:
            sys_msg = "You are a Risk Analytics Expert. Explain the time-series anomalies provided in the context."
        else:
            sys_msg = "You are an Enterprise Knowledge Assistant. Answer the query using ONLY the provided context."
            
        prompt = f"""
        {sys_msg}
        
        Context:
        {context}
        
        Query:
        {query}
        
        Answer clearly and professionally.
        """
        # We can use the fast model for standard chat generation
        return generate_response(prompt, model_name=config.GEMINI_CHAT_MODEL, temperature=0.3)


class ValidatorAgent:
    """
    Checks the generated response against the source context and geometric bounds 
    to flag potential hallucinations.
    """
    def validate(self, query: str, response: str, context: str, gac_metrics: Dict[str, Any]) -> Dict[str, Any]:
        d_eff = gac_metrics.get("d_eff", 0)
        theta = gac_metrics.get("theta_bound", 0.85)
        
        prompt = f"""
        You are the Validator Agent. Your job is to detect hallucinations.
        
        Original Query: {query}
        Draft Response: {response}
        Source Context: {context}
        Geometric Identity Error Bound (Theta): {theta}
        Effective Dimension of Cluster (d_eff): {d_eff}
        
        Check if the Draft Response is fully grounded in the Source Context.
        If it claims facts NOT in the context, flag it as a hallucination.
        Also consider the geometric metrics: a low theta (< 0.8) or very high d_eff means the context might be noisy.
        
        Respond ONLY with a JSON object:
        {{
            "is_grounded": true/false,
            "confidence_score": <float between 0 and 1>,
            "explanation": "<brief explanation of your verdict>",
            "corrected_response": "<if not grounded, provide a corrected response that admits lack of info. Else null>"
        }}
        """
        return generate_structured_response(prompt, model_name=config.GEMINI_REASONING_MODEL)


class MultiAgentOrchestrator:
    """
    Coordinates the Planner, Retriever, Executor, and Validator.
    """
    def __init__(self, rag_engine=None):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.validator = ValidatorAgent()
        self.rag_engine = rag_engine

    def run_query(self, query: str, anomaly_context: str = "") -> Dict[str, Any]:
        trace = []
        
        # 1. Plan
        plan = self.planner.plan(query)
        trace.append({"agent": "Planner", "action": "Selected path", "data": plan})
        
        context_str = ""
        is_anomaly = False
        retrieved_nodes_info = []
        gac_metrics_latest = {}
        
        # 2. Retrieve Context
        if plan.get("path") == "time_series_anomaly" and anomaly_context:
            context_str = anomaly_context
            is_anomaly = True
            trace.append({"agent": "Retriever", "action": "Fetched Anomaly Data", "data": "Time Series Context"})
        else:
            if self.rag_engine:
                retriever = self.rag_engine.get_retriever()
                nodes = retriever.retrieve(query)
                context_str = "\n\n".join([n.get_content() for n in nodes])
                
                retrieved_nodes_info = [{"id": n.node_id, "cluster": n.metadata.get("cluster_id")} for n in nodes]
                if self.rag_engine.gac_metrics:
                    gac_metrics_latest = self.rag_engine.gac_metrics[-1]
                    
                trace.append({"agent": "Retriever", "action": "Vector Search", "nodes": retrieved_nodes_info})
            else:
                context_str = "No index available."
                
        # 3. Execute
        draft_response = self.executor.execute(query, context_str, is_anomaly_data=is_anomaly)
        trace.append({"agent": "Executor", "action": "Generated Draft", "data": draft_response})
        
        # 4. Validate
        validation = self.validator.validate(query, draft_response, context_str, gac_metrics_latest)
        trace.append({"agent": "Validator", "action": "Validation Check", "data": validation})
        
        final_response = validation.get("corrected_response") or draft_response
        
        return {
            "final_response": final_response,
            "validation": validation,
            "trace": trace,
            "nodes_info": retrieved_nodes_info
        }
