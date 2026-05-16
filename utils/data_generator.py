import pandas as pd
import numpy as np
import os
import config

def generate_synthetic_timeseries():
    """
    Generates synthetic server/financial metrics with some anomalies for the Risk Dashboard.
    """
    out_file = os.path.join(config.DEMO_DATA_DIR, "synthetic_timeseries.csv")
    if os.path.exists(out_file):
        return out_file
        
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    
    # Normal operations
    traffic = np.random.normal(loc=1000, scale=100, size=100)
    cpu_usage = np.random.normal(loc=40, scale=5, size=100)
    error_rate = np.random.normal(loc=0.01, scale=0.005, size=100)
    
    # Inject anomalies
    # Anomaly 1: Traffic Spike
    traffic[20:25] += 800
    cpu_usage[20:25] += 40
    
    # Anomaly 2: Silent Error Rate Creep
    error_rate[60:75] += np.linspace(0.01, 0.15, 15)
    
    # Anomaly 3: Traffic Drop
    traffic[85:90] -= 700
    
    df = pd.DataFrame({
        "Date": dates,
        "Traffic": traffic,
        "CPU_Usage": cpu_usage,
        "Error_Rate": np.clip(error_rate, 0, 1)
    })
    
    df.to_csv(out_file, index=False)
    return out_file
