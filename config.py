import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Models Configuration
# Priority: GEMINI_REASONING_MODEL > GEMINI_MODEL > hardcoded default
GEMINI_REASONING_MODEL = os.getenv("GEMINI_REASONING_MODEL") or "gemini-2.5-pro"
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL") or "gemini-2.5-flash"

# Embedding Configuration (Switch to 'gemini' for final deployment)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") 
# GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "models/gemini-embedding-2"

# GAC Parameters
GAC_THETA = float(os.getenv("GAC_THETA", "0.85"))
GAC_STRATEGY = os.getenv("GAC_STRATEGY", "gac")

# General Settings
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
DEMO_DATA_DIR = os.getenv("DEMO_DATA_DIR", "./demo_data")

# Setup directories if they don't exist
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DEMO_DATA_DIR, exist_ok=True)
