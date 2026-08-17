"""
App configuration & Latency Budget definitions.
"""

import os
from backend.config import settings

# Retrieval Latency Budget for FAISS/In-Memory Hybrid Vector Search (in ms)
LATENCY_BUDGET_MS = float(os.getenv("LATENCY_BUDGET_MS", "50"))

# Pipeline SLA Budget (in ms)
PIPELINE_LATENCY_BUDGET_MS = float(os.getenv("PIPELINE_LATENCY_BUDGET_MS", "200"))
