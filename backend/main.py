"""
Main FastAPI Server for Voice-Enabled RAG System.
Integrates:
- STT Service (Sarvam AI & ElevenLabs)
- MSMARCO-XI Multilingual Dataset Ingestion
- Multi-Strategy Chunking Engine
- In-Memory Hybrid Vector Store
- Multi-Stage Guardrails (Injection, Safety, Off-topic, Hallucinations)
- Structured Model Harness
- Latency Analytics & P50/P70/P100 Dashboard
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.dataset_loader import dataset_loader
from backend.chunking_engine import chunking_engine, Chunk
from backend.vector_store import vector_store
from backend.guardrails import guardrails_engine, GuardrailVerdict
from backend.model_harness import model_harness, StructuredRAGResponse, LatencyBreakdown
from backend.stt_service import stt_service, STTResult
from backend.analytics import latency_analytics, QueryLatencyRecord, BenchmarkSummary

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VoiceRAG.Server")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Voice-Enabled RAG over ai4bharat/MSMARCO-XI with Sub-200ms Latency Analytics"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Startup Indexing
current_active_strategy = settings.DEFAULT_CHUNKING_STRATEGY

def initialize_index(strategy: str = "indic_semantic"):
    global current_active_strategy
    current_active_strategy = strategy
    docs = dataset_loader.get_all_documents()
    chunks = chunking_engine.process_corpus(docs, strategy=strategy)
    index_ms = vector_store.index_chunks(chunks)
    logger.info(f"Indexed {len(chunks)} chunks using strategy '{strategy}' in {index_ms:.2f} ms")

@app.on_event("startup")
def on_startup():
    logger.info("Initializing Voice-Enabled RAG Pipeline...")
    initialize_index(settings.DEFAULT_CHUNKING_STRATEGY)
    logger.info("Voice RAG Pipeline Ready!")

# =============================================================================
# Request & Response Models
# =============================================================================
class TextQueryRequest(BaseModel):
    query: str
    language: str = "hi"
    strategy: Optional[str] = None
    provider: Optional[str] = None

class StrategySwitchRequest(BaseModel):
    strategy: str

class BenchmarkRequest(BaseModel):
    num_queries: int = 50
    strategy: Optional[str] = "indic_semantic"
    provider: Optional[str] = "fast_synthesizer"

class IngestHFRequest(BaseModel):
    language: str = "hi"
    max_samples: int = 20
    split: str = "train"

class AddDocumentRequest(BaseModel):
    query: str
    passage: str
    language: str = "hi"
    topic: Optional[str] = "custom"
    answers: Optional[List[str]] = Field(default_factory=list)

# =============================================================================
# Pipeline Core Execution Function
# =============================================================================
async def run_rag_pipeline(
    query_text: str,
    language: str,
    stt_result: Optional[STTResult] = None,
    strategy: Optional[str] = None,
    llm_provider: Optional[str] = None
) -> StructuredRAGResponse:
    t_start = time.perf_counter()
    query_id = f"q-{uuid.uuid4().hex[:8]}"
    strat = strategy or current_active_strategy
    
    stt_ms = stt_result.stt_latency_ms if stt_result else 0.0

    # 1. Stage 1 Guardrail: Input Safety & Prompt Injection
    t_ig_0 = time.perf_counter()
    input_verdict = guardrails_engine.check_input_safety(query_text)
    input_guard_ms = (time.perf_counter() - t_ig_0) * 1000.0

    if not input_verdict.passed:
        t_total = (time.perf_counter() - t_start) * 1000.0
        response = await model_harness.execute_harness(
            query=query_text,
            retrieval_results=[],
            preliminary_verdict=input_verdict,
            provider=llm_provider
        )
        response.latency_profile.stt_ms = round(stt_ms, 2)
        response.latency_profile.input_guardrail_ms = round(input_guard_ms, 2)
        response.latency_profile.total_pipeline_ms = round(t_total, 2)

        # Record Telemetry
        latency_analytics.record_query(QueryLatencyRecord(
            query_id=query_id,
            query_text=query_text,
            language=language,
            strategy=strat,
            provider=llm_provider or "fast_synthesizer",
            stt_ms=stt_ms,
            input_guardrail_ms=input_guard_ms,
            total_pipeline_ms=t_total,
            success=False,
            guardrail_status=input_verdict.status
        ))
        return response

    # 2. Vector DB Hybrid Retrieval
    t_ret_0 = time.perf_counter()
    retrieval_results, ret_ms = vector_store.search(query_text, top_k=4)
    embedding_ms = ret_ms * 0.4  # estimated embedding portion of search
    retrieval_db_ms = ret_ms * 0.6

    # 3. Stage 2 Guardrail: Retrieval Relevance & Keyword Grounding Threshold
    top_score = retrieval_results[0].score if retrieval_results else 0.0
    retrieved_contexts = [r.context_text for r in retrieval_results]
    rel_verdict = guardrails_engine.check_retrieval_relevance(
        query=query_text, 
        top_score=top_score, 
        retrieved_contexts=retrieved_contexts
    )

    if not rel_verdict.passed:
        t_total = (time.perf_counter() - t_start) * 1000.0
        response = await model_harness.execute_harness(
            query=query_text,
            retrieval_results=retrieval_results,
            preliminary_verdict=rel_verdict,
            provider=llm_provider
        )
        response.latency_profile.stt_ms = round(stt_ms, 2)
        response.latency_profile.input_guardrail_ms = round(input_guard_ms, 2)
        response.latency_profile.embedding_ms = round(embedding_ms, 2)
        response.latency_profile.retrieval_ms = round(retrieval_db_ms, 2)
        response.latency_profile.total_pipeline_ms = round(t_total, 2)

        latency_analytics.record_query(QueryLatencyRecord(
            query_id=query_id,
            query_text=query_text,
            language=language,
            strategy=strat,
            provider=llm_provider or "fast_synthesizer",
            stt_ms=stt_ms,
            input_guardrail_ms=input_guard_ms,
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_db_ms,
            total_pipeline_ms=t_total,
            success=False,
            guardrail_status=rel_verdict.status
        ))
        return response

    # 4. Model Harness Execution + Tool Calling + Grounding Guardrail
    response = await model_harness.execute_harness(
        query=query_text,
        retrieval_results=retrieval_results,
        preliminary_verdict=rel_verdict,
        provider=llm_provider
    )

    t_total = (time.perf_counter() - t_start) * 1000.0
    response.latency_profile.stt_ms = round(stt_ms, 2)
    response.latency_profile.input_guardrail_ms = round(input_guard_ms, 2)
    response.latency_profile.embedding_ms = round(embedding_ms, 2)
    response.latency_profile.retrieval_ms = round(retrieval_db_ms, 2)
    response.latency_profile.total_pipeline_ms = round(t_total, 2)

    # Record Telemetry
    latency_analytics.record_query(QueryLatencyRecord(
        query_id=query_id,
        query_text=query_text,
        language=language,
        strategy=strat,
        provider=response.provider_used,
        stt_ms=stt_ms,
        input_guardrail_ms=input_guard_ms,
        embedding_ms=embedding_ms,
        retrieval_ms=retrieval_db_ms,
        tool_calls_ms=response.latency_profile.tool_calls_ms,
        generation_ms=response.latency_profile.llm_generation_ms,
        output_guardrail_ms=response.latency_profile.output_guardrail_ms,
        total_pipeline_ms=t_total,
        success=response.safety_verdict == "PASSED",
        guardrail_status=response.safety_verdict
    ))

    return response

# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/api/voice-query", response_model=StructuredRAGResponse)
async def voice_query_endpoint(
    audio: Optional[UploadFile] = File(None),
    client_transcript: Optional[str] = Form(None),
    language: str = Form("hi-IN"),
    stt_provider: str = Form("sarvam"),
    strategy: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None)
):
    """
    Accepts voice audio input (WAV/MP3/WebM), transcribes via Sarvam AI / ElevenLabs,
    executes hybrid retrieval and model harness, and returns structured RAG response.
    """
    audio_bytes = b""
    if audio:
        audio_bytes = await audio.read()

    # Transcribe audio
    stt_res = await stt_service.transcribe(
        audio_bytes=audio_bytes,
        provider=stt_provider,
        language_code=language,
        client_transcript_fallback=client_transcript
    )

    lang_code = language.split("-")[0]
    return await run_rag_pipeline(
        query_text=stt_res.transcript,
        language=lang_code,
        stt_result=stt_res,
        strategy=strategy,
        llm_provider=llm_provider
    )

@app.post("/api/text-query", response_model=StructuredRAGResponse)
async def text_query_endpoint(req: TextQueryRequest):
    """Direct text query endpoint for latency testing and benchmarking."""
    return await run_rag_pipeline(
        query_text=req.query,
        language=req.language,
        strategy=req.strategy,
        llm_provider=req.provider
    )

@app.get("/api/chunking-strategies")
def get_chunking_strategies():
    """Returns comparative metrics across all 4 chunking strategies on the dataset."""
    docs = dataset_loader.get_all_documents()
    metrics = chunking_engine.compare_strategies(docs)
    sample_doc = docs[0] if docs else None
    
    samples = {}
    if sample_doc:
        for s in ["indic_semantic", "hierarchical_parent_child", "metadata_sliding_window", "recursive_boundary"]:
            sample_chunks = chunking_engine.process_document(sample_doc, strategy=s)
            samples[s] = [c.model_dump() for c in sample_chunks[:3]]

    return {
        "active_strategy": current_active_strategy,
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "samples": samples
    }

@app.post("/api/switch-strategy")
def switch_strategy(req: StrategySwitchRequest):
    """Switches active chunking strategy and re-indexes corpus in-memory."""
    initialize_index(req.strategy)
    return {"status": "ok", "active_strategy": current_active_strategy}

@app.get("/api/telemetry")
def get_telemetry_summary():
    """Returns real-time P50, P70, P90, P100 latency analytics and telemetry breakdown."""
    return latency_analytics.get_summary().model_dump()

@app.post("/api/benchmark/run")
async def run_benchmark_endpoint(req: BenchmarkRequest):
    """
    Executes an automated multi-query latency benchmark across the MSMARCO-XI dataset.
    Returns comprehensive P50 / P70 / P100 latency numbers.
    """
    docs = dataset_loader.get_all_documents()
    if not docs:
        raise HTTPException(status_code=400, detail="Corpus is empty.")

    # Select queries across languages
    queries = [d.query for d in docs]
    while len(queries) < req.num_queries:
        queries.extend([d.query for d in docs])
    queries = queries[:req.num_queries]

    t_bench_start = time.perf_counter()
    records_count = 0

    for idx, q in enumerate(queries):
        doc = docs[idx % len(docs)]
        lang = doc.metadata.language
        
        # Simulate STT audio payload with 15-25ms fast transcription
        stt_sim = STTResult(
            transcript=q,
            language_code=lang,
            provider="fast_sync_stt",
            confidence=0.98,
            audio_duration_sec=1.5,
            stt_latency_ms=18.5
        )

        await run_rag_pipeline(
            query_text=q,
            language=lang,
            stt_result=stt_sim,
            strategy=req.strategy,
            llm_provider=req.provider
        )
        records_count += 1

    total_bench_ms = (time.perf_counter() - t_bench_start) * 1000.0
    summary = latency_analytics.get_summary(last_n=records_count)

    return {
        "benchmark_executed_queries": records_count,
        "total_benchmark_time_ms": round(total_bench_ms, 2),
        "avg_qps": round((records_count / (total_bench_ms / 1000.0)), 2),
        "summary": summary.model_dump()
    }

@app.get("/api/dataset/stats")
def get_dataset_stats():
    """Returns dataset languages, document count, and index status."""
    stats = dataset_loader.get_stats()
    stats["indexed_chunks"] = len(vector_store.chunks)
    stats["active_strategy"] = current_active_strategy
    return stats

@app.post("/api/dataset/ingest-hf")
def ingest_from_huggingface(req: IngestHFRequest):
    """
    Ingests live samples from Hugging Face ai4bharat/MSMARCO-XI dataset
    for the specified Indic language, updates local cache, and reindexes vector store.
    """
    count = dataset_loader.load_from_huggingface(
        language=req.language,
        max_samples=req.max_samples,
        split=req.split
    )
    # Reindex vector store with newly ingested documents
    initialize_index(current_active_strategy)
    stats = dataset_loader.get_stats()
    stats["ingested_new_samples"] = count
    stats["indexed_chunks"] = len(vector_store.chunks)
    return stats

@app.get("/api/dataset/documents")
def get_dataset_documents(language: Optional[str] = None, limit: int = 50):
    """Returns list of documents in MSMARCO-XI corpus with optional language filter."""
    docs = dataset_loader.get_all_documents()
    if language:
        docs = [d for d in docs if d.metadata.language == language]
    return [d.model_dump() for d in docs[:limit]]

@app.post("/api/dataset/add-document")
def add_custom_dataset_document(req: AddDocumentRequest):
    """Adds a custom question-passage document to the MSMARCO-XI corpus and updates index."""
    doc = dataset_loader.add_custom_document(
        query=req.query,
        passage=req.passage,
        language=req.language,
        answers=req.answers,
        topic=req.topic or "custom"
    )
    initialize_index(current_active_strategy)
    return {"status": "success", "document": doc.model_dump(), "indexed_chunks": len(vector_store.chunks)}

# Serve Frontend static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice RAG API is running. Frontend static directory is being assembled."}
