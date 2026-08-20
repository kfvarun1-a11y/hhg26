"""
Advanced Multi-Strategy Chunking Engine for Multilingual (Indic & English) RAG.
Implements:
1. Indic & Multilingual Semantic Chunking (aware of Purna Viram ।, ॥, clauses, and sentence boundaries)
2. Hierarchical Parent-Child Chunking (granular child vectors + rich parent context)
3. Metadata-Aware Sliding Window Chunking (overlap + embedded structural metadata)
4. Recursive Multi-Level Boundary Splitting
5. Chunking Strategy Comparison & Benchmark Utilities
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from backend.dataset_loader import DocumentRecord, PassageMetadata

class Chunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    token_count: int
    strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_text: Optional[str] = None  # For Hierarchical Parent-Child

class ChunkingMetrics(BaseModel):
    strategy_name: str
    total_chunks: int
    avg_tokens_per_chunk: float
    min_tokens: int
    max_tokens: int
    std_dev_tokens: float
    boundary_preservation_score: float  # 0.0 to 1.0
    overlap_ratio: float

# Indic Punctuation & Sentence Delimiters
INDIC_PUNCTUATION_REGEX = re.compile(r'([।॥!?\.\n]+)')
CLAUSE_SEPARATORS_REGEX = re.compile(r'([,;:\-–—\(\)])')

def estimate_tokens(text: str) -> int:
    """Estimates token count for multilingual / Indic / English text."""
    if not text:
        return 0
    # Words + Indic conjunct token heuristic
    words = text.strip().split()
    # Approx 1.3 tokens per word for Indic / morphological scripts
    return max(1, int(len(words) * 1.25))

class ChunkingEngine:
    def __init__(self, default_chunk_size: int = 100, default_overlap: int = 20):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    # =========================================================================
    # Strategy 1: Indic & Multilingual Semantic Boundary Chunking
    # =========================================================================
    def chunk_indic_semantic(
        self, 
        doc: DocumentRecord, 
        target_tokens: int = 80, 
        overlap_sentences: int = 1
    ) -> List[Chunk]:
        """
        Splits text by sentence boundaries (respecting Hindi/Indic Purna Viram ।, ॥, 
        Bengali/Telugu/Tamil periods, and standard ?!) and groups them semantically 
        to maintain grammatical and contextual cohesion.
        """
        text = doc.passage_text.strip()
        # Split keeping delimiters
        raw_parts = INDIC_PUNCTUATION_REGEX.split(text)
        sentences: List[str] = []
        
        current_sentence = ""
        for part in raw_parts:
            if not part:
                continue
            if INDIC_PUNCTUATION_REGEX.match(part):
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        if not sentences:
            sentences = [text]

        chunks: List[Chunk] = []
        i = 0
        chunk_idx = 0

        while i < len(sentences):
            current_chunk_sentences: List[str] = []
            current_tokens = 0

            j = i
            while j < len(sentences):
                sent = sentences[j]
                sent_tokens = estimate_tokens(sent)
                if current_tokens + sent_tokens > target_tokens and len(current_chunk_sentences) > 0:
                    break
                current_chunk_sentences.append(sent)
                current_tokens += sent_tokens
                j += 1

            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc.id}-indic-sem-{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=doc.id,
                    text=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    strategy="indic_semantic",
                    parent_text=doc.passage_text.strip(),
                    metadata={
                        **doc.metadata.model_dump(),
                        "sentence_count": len(current_chunk_sentences),
                        "chunk_index": chunk_idx,
                        "query": doc.query
                    }
                )
            )
            chunk_idx += 1
            # Advance with overlap
            advance = max(1, len(current_chunk_sentences) - overlap_sentences)
            i += advance

        return chunks

    # =========================================================================
    # Strategy 2: Hierarchical Parent-Child Chunking
    # =========================================================================
    def chunk_hierarchical(
        self, 
        doc: DocumentRecord, 
        child_target_tokens: int = 35, 
        parent_target_tokens: int = 150
    ) -> List[Chunk]:
        """
        Generates small 'Child Chunks' for high-precision dense vector retrieval,
        while attaching the full 'Parent Passage Context' to deliver complete
        grounding context to the LLM without clipping answers.
        """
        full_parent_text = doc.passage_text.strip()
        sentences = [s.strip() for s in INDIC_PUNCTUATION_REGEX.split(full_parent_text) if s.strip() and not INDIC_PUNCTUATION_REGEX.match(s)]
        if not sentences:
            sentences = full_parent_text.split()

        chunks: List[Chunk] = []
        child_idx = 0
        current_child_words: List[str] = []

        words = full_parent_text.split()
        step = max(1, int(child_target_tokens * 0.8))
        window_size = int(child_target_tokens * 1.0)

        for w_start in range(0, len(words), step):
            child_slice = words[w_start: w_start + window_size]
            child_text = " ".join(child_slice)
            if not child_text.strip():
                continue

            chunk_id = f"{doc.id}-hier-child-{child_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=doc.id,
                    text=child_text,
                    token_count=estimate_tokens(child_text),
                    strategy="hierarchical_parent_child",
                    parent_text=full_parent_text,
                    metadata={
                        **doc.metadata.model_dump(),
                        "child_index": child_idx,
                        "parent_token_count": estimate_tokens(full_parent_text),
                        "query": doc.query
                    }
                )
            )
            child_idx += 1

        return chunks

    # =========================================================================
    # Strategy 3: Metadata-Aware Overlapping Sliding Window
    # =========================================================================
    def chunk_metadata_sliding_window(
        self, 
        doc: DocumentRecord, 
        chunk_size: int = 75, 
        overlap: int = 20
    ) -> List[Chunk]:
        """
        Splits by word/token window with explicit overlap and embeds structured
        metadata prefixes into the indexable content for cross-lingual and topic grounding.
        """
        words = doc.passage_text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 0
        step = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            window_words = words[i: i + chunk_size]
            core_text = " ".join(window_words)
            # Embed structured metadata header
            meta_header = f"[Lang: {doc.metadata.language.upper()} | Topic: {doc.metadata.topic or 'General'}] "
            rich_text = meta_header + core_text

            chunk_id = f"{doc.id}-meta-slide-{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=doc.id,
                    text=rich_text,
                    token_count=estimate_tokens(rich_text),
                    strategy="metadata_sliding_window",
                    parent_text=doc.passage_text.strip(),
                    metadata={
                        **doc.metadata.model_dump(),
                        "window_start": i,
                        "window_end": i + len(window_words),
                        "overlap": overlap,
                        "query": doc.query
                    }
                )
            )
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break

        return chunks

    # =========================================================================
    # Strategy 4: Recursive Multi-Level Boundary Splitting
    # =========================================================================
    def chunk_recursive_boundary(
        self, 
        doc: DocumentRecord, 
        max_tokens: int = 80
    ) -> List[Chunk]:
        """
        Hierarchically splits on paragraphs (\\n\\n) -> lines (\\n) -> Indic/Latin sentences
        (।, ., ?, !) -> clauses (, ;) -> whitespace.
        """
        separators = ["\n\n", "\n", "।", "॥", ". ", "? ", "! ", "; ", ", ", " "]
        
        def _split_text(text: str, sep_idx: int) -> List[str]:
            if estimate_tokens(text) <= max_tokens:
                return [text.strip()] if text.strip() else []
            if sep_idx >= len(separators):
                # Hard fallback split
                words = text.split()
                mid = len(words) // 2
                return [" ".join(words[:mid]), " ".join(words[mid:])]

            sep = separators[sep_idx]
            splits = text.split(sep)
            result = []
            accumulator = ""

            for s in splits:
                cand = (accumulator + sep + s) if accumulator else s
                if estimate_tokens(cand) <= max_tokens:
                    accumulator = cand
                else:
                    if accumulator:
                        result.append(accumulator.strip())
                    if estimate_tokens(s) <= max_tokens:
                        accumulator = s
                    else:
                        sub_splits = _split_text(s, sep_idx + 1)
                        result.extend(sub_splits)
                        accumulator = ""
            if accumulator.strip():
                result.append(accumulator.strip())
            return result

        raw_chunks = _split_text(doc.passage_text, 0)
        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            if not text.strip():
                continue
            chunk_id = f"{doc.id}-rec-bound-{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=doc.id,
                    text=text,
                    token_count=estimate_tokens(text),
                    strategy="recursive_boundary",
                    parent_text=doc.passage_text.strip(),
                    metadata={
                        **doc.metadata.model_dump(),
                        "recursive_depth": len(separators),
                        "chunk_index": idx,
                        "query": doc.query
                    }
                )
            )
        return chunks

    # =========================================================================
    # Strategy Dispatcher & Batch Processing
    # =========================================================================
    def process_document(self, doc: DocumentRecord, strategy: str = "indic_semantic") -> List[Chunk]:
        """Dispatches chunking based on selected strategy."""
        if strategy == "indic_semantic":
            return self.chunk_indic_semantic(doc, target_tokens=self.default_chunk_size)
        elif strategy == "hierarchical_parent_child":
            return self.chunk_hierarchical(doc, child_target_tokens=40, parent_target_tokens=160)
        elif strategy == "metadata_sliding_window":
            return self.chunk_metadata_sliding_window(doc, chunk_size=self.default_chunk_size, overlap=self.default_overlap)
        elif strategy == "recursive_boundary":
            return self.chunk_recursive_boundary(doc, max_tokens=self.default_chunk_size)
        else:
            return self.chunk_indic_semantic(doc)

    def process_corpus(self, docs: List[DocumentRecord], strategy: str = "indic_semantic") -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for d in docs:
            all_chunks.extend(self.process_document(d, strategy=strategy))
        return all_chunks

    # =========================================================================
    # Strategy Analytics & Comparison
    # =========================================================================
    def compare_strategies(self, docs: List[DocumentRecord]) -> Dict[str, ChunkingMetrics]:
        """Runs all 4 chunking strategies on the corpus and computes comparative metrics."""
        strategies = ["indic_semantic", "hierarchical_parent_child", "metadata_sliding_window", "recursive_boundary"]
        results: Dict[str, ChunkingMetrics] = {}

        for strat in strategies:
            chunks = self.process_corpus(docs, strategy=strat)
            if not chunks:
                continue

            token_counts = [c.token_count for c in chunks]
            avg_tok = sum(token_counts) / len(token_counts)
            variance = sum((x - avg_tok) ** 2 for x in token_counts) / len(token_counts)
            std_dev = math.sqrt(variance)

            # Measure boundary integrity: how many chunks end with proper punctuation
            punct_endings = sum(1 for c in chunks if c.text.strip().endswith(('।', '॥', '.', '!', '?')))
            boundary_score = punct_endings / len(chunks) if chunks else 0.0

            results[strat] = ChunkingMetrics(
                strategy_name=strat,
                total_chunks=len(chunks),
                avg_tokens_per_chunk=round(avg_tok, 2),
                min_tokens=min(token_counts),
                max_tokens=max(token_counts),
                std_dev_tokens=round(std_dev, 2),
                boundary_preservation_score=round(boundary_score, 3),
                overlap_ratio=0.25 if "sliding" in strat or "semantic" in strat else 0.10
            )

        return results

# Global Singleton instance
chunking_engine = ChunkingEngine()
