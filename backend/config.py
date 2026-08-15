"""
Configuration settings for Voice-Enabled RAG System.
Handles API keys, dataset paths, chunking strategies, latency thresholds, and guardrails.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class Settings(BaseModel):
    # App Settings
    APP_NAME: str = "Voice-Enabled RAG System"
    APP_VERSION: str = "1.0.0"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # STT Settings (Sarvam AI or ElevenLabs)
    STT_PROVIDER: Literal["sarvam", "elevenlabs", "fallback"] = "sarvam"
    SARVAM_API_KEY: Optional[str] = os.getenv("SARVAM_API_KEY", "")
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY", "")
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    ELEVENLABS_STT_URL: str = "https://api.elevenlabs.io/v1/speech-to-text"
    SARVAM_LANGUAGE_CODE: str = "hi-IN"  # hi-IN, en-IN, te-IN, ta-IN, bn-IN, etc.

    # LLM Settings
    LLM_PROVIDER: Literal["groq", "gemini", "openai", "sarvam", "fast_synthesizer"] = "fast_synthesizer"
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.2

    # Dataset Settings (ai4bharat/MSMARCO-XI)
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_LANGUAGES: List[str] = ["hi", "en", "te", "ta", "bn"]
    LOCAL_CACHE_DIR: Path = DATA_DIR / "msmarco_cache"
    SAMPLE_SIZE: int = 200  # Default preloaded passage count for sub-millisecond retrieval

    # Chunking Strategies
    DEFAULT_CHUNKING_STRATEGY: Literal[
        "indic_semantic", 
        "hierarchical_parent_child", 
        "metadata_sliding_window", 
        "recursive_boundary"
    ] = "indic_semantic"
    CHUNK_SIZE: int = 128  # Target tokens / words
    CHUNK_OVERLAP: int = 24  # Tokens / words overlap

    # Latency & Performance Targets
    TARGET_LATENCY_MS: float = 200.0
    ENABLE_EMBEDDING_CACHE: bool = True
    MAX_CACHE_SIZE: int = 2048

    # Guardrail Thresholds
    MIN_SIMILARITY_THRESHOLD: float = 0.25  # Cosine similarity below this triggers ungrounded/off-topic fallback
    MAX_HALLUCINATION_SCORE: float = 0.45   # Hallucination score above this rejects answer
    OFF_TOPIC_RELEVANCE_FLOOR: float = 0.22

    # Standard Grounding Refusal Message when query is not related to dataset context
    DEFAULT_UNGROUNDED_RESPONSE: str = "I couldn’t find sufficient information in the provided dataset to answer this question."

settings = Settings()

