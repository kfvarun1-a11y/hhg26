"""
Fast Vector Retriever with latency metrics for embedding and search stages.
"""

import time
from dataclasses import dataclass, field
from typing import List, Any

from backend.dataset_loader import dataset_loader
from backend.chunking_engine import chunking_engine
from backend.vector_store import vector_store, RetrievalResult

@dataclass
class SearchResponse:
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0

def warmup(strategy: str = "indic_semantic"):
    """Loads corpus, indexes in vector store, and warms up embedding and inference caches."""
    docs = dataset_loader.get_all_documents()
    if not docs:
        dataset_loader._initialize_dataset()
        docs = dataset_loader.get_all_documents()
    
    chunks = chunking_engine.process_corpus(docs, strategy=strategy)
    vector_store.index_chunks(chunks)

    # Initial warm-up inference
    search("warmup initial query", top_k=3)

def search(query: str, top_k: int = 5) -> SearchResponse:
    """
    Executes hybrid dense vector + sparse search and records exact stage latencies.
    """
    t0 = time.perf_counter()

    if not vector_store.is_indexed:
        warmup()

    # Time embedding generation
    t_embed_0 = time.perf_counter()
    q_vec = vector_store.vectorizer.encode(query)
    embed_ms = (time.perf_counter() - t_embed_0) * 1000.0

    # Time hybrid search
    results, search_elapsed_ms = vector_store.search(query, top_k=top_k, q_vec=q_vec)
    
    total_ms = (time.perf_counter() - t0) * 1000.0
    search_ms = max(0.01, total_ms - embed_ms)

    return SearchResponse(
        query=query,
        results=results,
        embed_ms=round(embed_ms, 3),
        search_ms=round(search_ms, 3),
        total_ms=round(total_ms, 3)
    )
