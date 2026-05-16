<div align="center">

# 🌌 Enterprise Intelligence Platform
### *Geometry-Aware Hierarchical RAG for Enterprise Documents*

[![lablab.ai Hackathon](https://img.shields.io/badge/lablab.ai-Hackathon%20Submission-blueviolet?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6bTAgOUwyIDExbDEwIDUgMTAtNXptMCA1TDIgMTZsMTAgNSAxMC01eiIvPjwvc3ZnPg==)](https://lablab.ai)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

</div>

---

## 📖 Overview

Most enterprise RAG systems embed documents, store them, and call it a day. This platform goes further.

We introduce **Geometry-Aware Consolidation (GAC)** — a mathematically-grounded approach that uses the *intrinsic geometry of the embedding space* to consolidate redundant document chunks before they ever reach the retrieval layer. The result is a **~55% reduction in stored nodes** while preserving semantic coverage, eliminating the "noisy neighbor" retrieval failure that plagues naive RAG systems.

Paired with a **multi-agent orchestration layer** (Planner → Retriever → Executor → Validator), the platform can reason over enterprise documents, detect anomalies in time-series data, and provide auditable, geometry-validated responses.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                            │
│                                                                 │
│  PDF/TXT ──► HierarchicalParser ──► Leaf Nodes                 │
│              [2048 / 512 / 128 tokens]                          │
│                        │                                        │
│                        ▼                                        │
│              Embedding Model (local / Gemini)                   │
│                        │                                        │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │  Geometry-Aware Consolidation│                        │
│         │  ─────────────────────────── │                        │
│         │  KMeans Clustering           │                        │
│         │  + Spectral Spread Analysis  │                        │
│         │  + Identity Error Bounding   │                        │
│         │  → ~55% node reduction       │                        │
│         └──────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│              ChromaDB (Persistent Vector Store)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                               │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  [Planner Agent] ──► Route: vector_search / anomaly / KG       │
│      │                                                          │
│      ▼                                                          │
│  [Retriever] ──► ChromaDB cosine search on GAC nodes           │
│      │                                                          │
│      ▼                                                          │
│  [Executor Agent] ──► Gemini Flash generates response           │
│      │                                                          │
│      ▼                                                          │
│  [Validator Agent] ──► Groundedness check + confidence score   │
│      │                  (informed by d_eff & theta bounds)      │
│      ▼                                                          │
│  Final Response + Audit Trace + Knowledge Graph                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔷 **Geometry-Aware Consolidation** | Reduces stored nodes ~55% using spectral geometry of embedding clusters |
| 🤖 **Multi-Agent Orchestration** | Planner → Retriever → Executor → Validator with full trace logging |
| 🕸️ **Knowledge Graph Extraction** | Per-query entity-relationship graphs extracted by Gemini |
| 📈 **Risk & Anomaly Dashboard** | Time-series anomaly detection with interactive Plotly charts |
| 🛡️ **Hallucination Validation** | Every response validated for groundedness with a confidence score |
| 📋 **Full Audit Trail** | All queries, responses, and GAC metrics logged for traceability |
| 🔄 **Dual Embedding Provider** | Local HuggingFace (`bge-small`) for dev, Gemini for production |
| ⚡ **Content-Hash Dedup** | Re-uploading unchanged docs skips re-embedding entirely |

---

## 🧮 The Math: Consolidation-Interference Duality

For a cluster $C_k$ with mean pairwise cosine distance $\bar{d}_k$, retrieval threshold $\theta$, and effective dimension $d_{\text{eff}}$, the **identity retrieval error bound** is:

$$\varepsilon_{\text{id}}(C_k, r) \geq 1 - c_1 \left(\frac{1 - \theta}{\bar{d}_k}\right)^{d_{\text{eff}}}$$

GAC uses this bound to decide whether a cluster is "safe to collapse" to a single representative, or whether its geometric spread is too high and it must be preserved. This is what produces principled compression rather than heuristic summarisation.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A Google AI Studio API key ([get one here](https://aistudio.google.com))

### Setup

```bash
# Clone the repository
git clone https://github.com/ShivtejG236/Enterprise-Intelligence.git
cd Enterprise-Intelligence

# Create and activate virtual environment
python3 -m venv .data_intel
source .data_intel/bin/activate  # On Windows: .data_intel\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

All configuration is driven by environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *required* | Google AI Studio API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (local) or `models/gemini-embedding-2` (production) |
| `EMBEDDING_PROVIDER` | `local` | `local` for development, unset for production |
| `GEMINI_REASONING_MODEL` | `gemini-2.5-pro` | Model for Planner & Validator agents |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | Model for Executor agent |
| `GAC_THETA` | `0.85` | Identity error bound threshold for consolidation |
| `GAC_STRATEGY` | `gac` | Consolidation strategy (`gac` or `centroid`) |

### Dev vs. Production

| Mode | `EMBEDDING_PROVIDER` | `EMBEDDING_MODEL` | Rate Limits |
|---|---|---|---|
| **Local Dev** | `local` (via `ENVIRONMENT=local`) | `BAAI/bge-small-en-v1.5` | None |
| **Production** | unset (None) | `models/gemini-embedding-2` | Gemini quotas apply |

---

## 📁 Project Structure

```
enterprise-intelligence/
├── app.py                      # Streamlit entrypoint
├── config.py                   # Centralized configuration
├── requirements.txt
│
├── backend/
│   ├── rag_engine.py           # GeometryAwareRAGEngine (core ingestion + retrieval)
│   ├── geometry_consolidation.py # GAC orchestration
│   ├── gemini_utils.py         # Gemini API wrapper with rate-limit handling
│   ├── analytics_agents.py     # Multi-agent orchestrator (Planner/Executor/Validator)
│   ├── knowledge_graph.py      # Entity-relationship triplet extraction + PyVis
│   ├── audit_logger.py         # Query & response audit logging
│   └── gac/
│       ├── theory.py           # d_eff, cluster_spread, spectral_bound
│       ├── strategies.py       # GACConsolidator, CentroidConsolidator
│       ├── clustering.py       # KMeans wrapper
│       └── metrics.py          # GAC metrics computation
│
├── components/
│   ├── chat.py                 # Intelligent Agent Chat tab
│   ├── dashboard.py            # Risk & Anomaly Dashboard tab
│   ├── kg_visualizer.py        # Global Knowledge Graph tab
│   ├── sidebar.py              # Configuration sidebar
│   └── geometry_metrics.py     # GAC metrics display cards
│
├── utils/
│   ├── helpers.py              # Dark theme, utilities
│   └── data_generator.py      # Demo time-series data generation
│
├── demo_data/                  # Sample enterprise documents for demo
├── tests/
│   └── test_gac.py             # GAC unit tests
└── .streamlit/
    └── config.toml             # Streamlit server configuration
```

---

## 🏆 Hackathon Track

**lablab.ai Enterprise Intelligence Hackathon** · Track: Data & Intelligence

This submission demonstrates:
- Novel application of geometric information theory to RAG compression
- Production-hardened dual-provider architecture (local ↔ Gemini)
- Full auditability via structured agent traces and confidence scores
- End-to-end Gemini integration (embeddings, reasoning, validation, KG extraction)

---

## 👥 Team

Built by **Shivtej** & **Shashank** · IIT Guwahati · 2025-26

---

<div align="center">

*"Don't just store documents. Understand their geometry."*

</div>
