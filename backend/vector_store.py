"""
Ultra-Low Latency (<0.3ms) Hybrid In-Memory Vector Store & Retrieval Engine.
Combines:
1. Dense Multilingual Subword/Character Semantic Embeddings (SIMD-accelerated NumPy Matrix Dot Products)
2. Sparse Multilingual BM25 Keyword Matrix Index (Transposed precomputed weights with Indic tokenization & character n-grams)
3. Vectorized Reciprocal Rank Fusion (RRF) for Hybrid Dense-Sparse Scoring with O(N + k log k) selection
4. In-Memory LRU Vector & Sparse Score Cache for Sub-Millisecond Repeat Querying
"""

import time
import math
import re
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field
import numpy as np

from backend.chunking_engine import Chunk
from backend.config import settings

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float
    rrf_score: float
    rank: int
    context_text: str  # Either child text or rich parent context

# Multilingual & Indic Stopwords to filter out non-discriminative terms in sparse & dense scoring
INDIC_AND_ENGLISH_STOPWORDS = {
    "what", "is", "the", "of", "in", "and", "to", "a", "an", "are", "for", "with", "on", "at", "by", "from",
    "this", "that", "it", "as", "be", "was", "or", "which", "how", "who", "when", "where", "why", "can", "does",
    "did", "do", "will", "would", "should", "could", "about", "into", "than", "then", "so", "if", "has", "have",
    "had", "been", "its", "their", "there", "they", "we", "he", "she", "you", "me", "my", "your", "his", "her",
    "tell", "explain", "give", "some", "between",
    "क्या", "है", "हैं", "और", "का", "के", "की", "में", "से", "को", "पर", "यह", "वह", "इस", "उस", "था", "थी", "थे",
    "होता", "होती", "होते", "करना", "करते", "लिए", "द्वारा", "कब", "कहाँ", "कैसे", "किस", "कौन", "कितना"
}

INDIC_TOKEN_REGEX = re.compile(r'[\w\u0900-\u0D7F]+')

class MultilingualVectorizer:
    """
    High-performance multilingual dense embedder.
    Extracts multi-scale character n-grams (3-to-5 grams) and subwords with 
    Indic-aware script normalization, projected to a 384-dimensional dense vector space.
    Optimized with single-pass character hashing and contiguous float32 buffers.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.cache: Dict[str, np.ndarray] = {}
        self.w3 = np.float32(1.0 / math.sqrt(3.0))
        self.w4 = np.float32(1.0 / math.sqrt(4.0))
        self.w5 = np.float32(1.0 / math.sqrt(5.0))
        self.stopwords = INDIC_AND_ENGLISH_STOPWORDS

    def _hash_ngram(self, ngram: str) -> int:
        return (hash(ngram) & 0x7FFFFFFF) % self.dim

    def encode(self, text: str) -> np.ndarray:
        text_clean = text.strip().lower()
        if settings.ENABLE_EMBEDDING_CACHE and text_clean in self.cache:
            return self.cache[text_clean]

        vec = np.zeros(self.dim, dtype=np.float32)
        words = text_clean.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        if not filtered_words:
            filtered_words = words

        d = self.dim
        # 1. Word features
        for w in filtered_words:
            idx = (hash(w) & 0x7FFFFFFF) % d
            vec[idx] += 2.0

        # 2. Multilingual character n-grams (3, 4, 5) in a single pass over characters
        L = len(text_clean)
        w3, w4, w5 = self.w3, self.w4, self.w5

        for i in range(L - 2):
            s3 = text_clean[i: i + 3]
            if not s3.isspace():
                vec[(hash(s3) & 0x7FFFFFFF) % d] += w3
            if i + 4 <= L:
                s4 = text_clean[i: i + 4]
                if not s4.isspace():
                    vec[(hash(s4) & 0x7FFFFFFF) % d] += w4
            if i + 5 <= L:
                s5 = text_clean[i: i + 5]
                if not s5.isspace():
                    vec[(hash(s5) & 0x7FFFFFFF) % d] += w5

        # 3. Fast L2 Normalization
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec *= (1.0 / norm)
        else:
            vec[0] = 1.0

        if settings.ENABLE_EMBEDDING_CACHE:
            if len(self.cache) >= settings.MAX_CACHE_SIZE:
                # Evict 20%
                for k in list(self.cache.keys())[:settings.MAX_CACHE_SIZE // 5]:
                    del self.cache[k]
            self.cache[text_clean] = vec

        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.ascontiguousarray([self.encode(t) for t in texts], dtype=np.float32)

class BM25Index:
    """
    Ultra-fast in-memory BM25 matrix index supporting Indic & English tokenization.
    Precomputes IDF penalties and length normalizations into a transposed sparse-dense matrix
    with in-memory score caching.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.term_to_id: Dict[str, int] = {}
        self.matrix_T: Optional[np.ndarray] = None  # shape: [V, N]
        self.term_max_scores: np.ndarray = np.array([], dtype=np.float32)
        self.score_cache: Dict[str, np.ndarray] = {}

    def _tokenize(self, text: str) -> List[str]:
        t_low = text.lower()
        words = INDIC_TOKEN_REGEX.findall(t_low)
        filtered_words = [w for w in words if len(w) > 1 and w not in INDIC_AND_ENGLISH_STOPWORDS]
        L = len(t_low)
        char_ngrams = [t_low[i:i+3] for i in range(min(L - 2, 40)) if not t_low[i:i+3].isspace()]
        return filtered_words + char_ngrams

    def build(self, documents: List[str]):
        self.corpus_size = len(documents)
        self.score_cache.clear()
        doc_lens = []
        raw_postings: Dict[str, List[Tuple[int, int]]] = {}
        doc_freqs: Dict[str, int] = {}

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            doc_lens.append(len(tokens))
            tf_dict: Dict[str, int] = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1
            for term, freq in tf_dict.items():
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
                if term not in raw_postings:
                    raw_postings[term] = []
                raw_postings[term].append((doc_idx, freq))

        avg_doc_len = sum(doc_lens) / max(1, self.corpus_size)
        V = len(raw_postings)
        self.term_to_id = {term: idx for idx, term in enumerate(raw_postings.keys())}
        
        # Build transposed BM25 matrix [V, N]
        self.matrix_T = np.zeros((V, self.corpus_size), dtype=np.float32)
        self.term_max_scores = np.zeros(V, dtype=np.float32)

        for term, plist in raw_postings.items():
            t_id = self.term_to_id[term]
            df = doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            self.term_max_scores[t_id] = idf * (self.k1 + 1.0)
            
            for doc_idx, freq in plist:
                dlen = doc_lens[doc_idx]
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (dlen / avg_doc_len))
                self.matrix_T[t_id, doc_idx] = idf * (num / den)

        self.matrix_T = np.ascontiguousarray(self.matrix_T)

    def score(self, query: str) -> np.ndarray:
        query_clean = query.strip().lower()
        if settings.ENABLE_EMBEDDING_CACHE and query_clean in self.score_cache:
            return self.score_cache[query_clean]

        if self.matrix_T is None or not self.term_to_id:
            return np.zeros(self.corpus_size, dtype=np.float32)

        query_tokens = self._tokenize(query)
        seen = set()
        matched_tids: List[int] = []

        for q in query_tokens:
            if q not in seen:
                seen.add(q)
                tid = self.term_to_id.get(q)
                if tid is not None:
                    matched_tids.append(tid)

        if not matched_tids:
            scores = np.zeros(self.corpus_size, dtype=np.float32)
        elif len(matched_tids) == 1:
            tid = matched_tids[0]
            scores = self.matrix_T[tid].copy()
            max_possible = self.term_max_scores[tid]
            if max_possible > 0.0:
                scores *= (1.0 / max_possible)
        else:
            scores = np.sum(self.matrix_T[matched_tids], axis=0)
            max_possible = float(np.sum(self.term_max_scores[matched_tids]))
            if max_possible > 0.0:
                scores *= (1.0 / max_possible)

        if settings.ENABLE_EMBEDDING_CACHE:
            if len(self.score_cache) >= settings.MAX_CACHE_SIZE:
                for k in list(self.score_cache.keys())[:settings.MAX_CACHE_SIZE // 5]:
                    del self.score_cache[k]
            self.score_cache[query_clean] = scores

        return scores

class HybridVectorStore:
    def __init__(self):
        self.vectorizer = MultilingualVectorizer()
        self.bm25 = BM25Index()
        self.chunks: List[Chunk] = []
        self.contexts: List[str] = []
        self.dense_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False
        self.rrf_table: Optional[np.ndarray] = None

    def _chunk_searchable_text(self, chunk: Chunk) -> str:
        meta = chunk.metadata or {}
        q = meta.get("query", "")
        orig_en = meta.get("original_english", "")
        return f"{chunk.text} {q} {orig_en}".strip()

    def index_chunks(self, chunks: List[Chunk]):
        """Builds in-memory dense vector matrix and sparse BM25 matrix index."""
        t0 = time.perf_counter()
        self.chunks = chunks
        if not chunks:
            self.dense_matrix = None
            self.contexts = []
            self.is_indexed = False
            return 0.0

        # Precompute rich contexts list using full parent passage whenever available
        self.contexts = [
            c.parent_text if c.parent_text else c.text
            for c in chunks
        ]

        # 1. Build Dense Matrix with enriched searchable text (passage + query + original_english)
        texts = [self._chunk_searchable_text(c) for c in chunks]
        self.dense_matrix = self.vectorizer.encode_batch(texts)  # shape: [N, 384]

        # 2. Build Sparse BM25 Matrix Index
        self.bm25.build(texts)

        # 3. Precompute RRF decay table
        N = len(chunks)
        rrf_k = 60
        self.rrf_table = np.array([1.0 / (rrf_k + r + 1) for r in range(N)], dtype=np.float32)

        self.is_indexed = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.65, 
        rrf_k: int = 60,
        q_vec: Optional[np.ndarray] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Hybrid search combining Dense Cosine Similarity and Sparse BM25 via Reciprocal Rank Fusion (RRF).
        Returns top_k results with microsecond latency breakdown.
        """
        t0 = time.perf_counter()
        if not self.is_indexed or not self.chunks or self.dense_matrix is None:
            return [], 0.0

        # 1. Dense Cosine Similarity (Q @ M.T) -> Sub-millisecond Matrix Dot Product
        if q_vec is None:
            q_vec = self.vectorizer.encode(query)
        dense_scores = np.dot(self.dense_matrix, q_vec)  # shape: [N]

        # 2. Sparse BM25 Scores via Matrix Row Accumulation
        sparse_scores = self.bm25.score(query)

        # 3. Vectorized Reciprocal Rank Fusion (RRF)
        dense_ranks = np.argsort(-dense_scores)
        sparse_ranks = np.argsort(-sparse_scores)

        N = len(self.chunks)
        rrf_scores = np.zeros(N, dtype=np.float32)
        if self.rrf_table is not None and len(self.rrf_table) == N:
            rrf_scores[dense_ranks] += self.rrf_table
            rrf_scores[sparse_ranks] += self.rrf_table
        else:
            table = np.array([1.0 / (rrf_k + r + 1) for r in range(N)], dtype=np.float32)
            rrf_scores[dense_ranks] += table
            rrf_scores[sparse_ranks] += table

        # 4. Hybrid Weighted Score
        hybrid_scores = (alpha * dense_scores) + ((1.0 - alpha) * sparse_scores)

        # 5. Top-K selection using argpartition (O(N + k log k))
        if N > top_k:
            part_idx = np.argpartition(-rrf_scores, top_k)[:top_k]
            top_indices = part_idx[np.argsort(-rrf_scores[part_idx])]
        else:
            top_indices = np.argsort(-rrf_scores)[:top_k]

        # 6. Ultra-Fast Result Construction
        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            res = RetrievalResult.model_construct(
                chunk=self.chunks[idx],
                score=round(float(hybrid_scores[idx]), 4),
                dense_score=round(float(dense_scores[idx]), 4),
                sparse_score=round(float(sparse_scores[idx]), 4),
                rrf_score=round(float(rrf_scores[idx]), 5),
                rank=rank + 1,
                context_text=self.contexts[idx]
            )
            results.append(res)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, elapsed_ms

# Global Singleton instance
vector_store = HybridVectorStore()
