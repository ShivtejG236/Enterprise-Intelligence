import json
import os
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_dir: str = "./data"):
        self.log_file = os.path.join(log_dir, "audit_log.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        
    def log_query(self, query: str, response: str, trace: list, metrics: Dict[str, Any] = None):
        """
        Logs a query trace along with hallucination scores and geometric metrics.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "trace": trace,
            "gac_metrics": metrics or {}
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def get_logs(self) -> list:
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        return logs
