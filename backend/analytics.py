"""
High-Precision Latency Telemetry & Analytics Suite for Voice RAG Pipeline.
Tracks stage-by-stage latencies, calculates P50, P70, P90, P100 percentiles, 
and executes multi-query automated benchmark runs across the MSMARCO-XI dataset.
"""

import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryLatencyRecord(BaseModel):
    query_id: str
    query_text: str
    language: str
    strategy: str
    provider: str
    stt_ms: float = 0.0
    input_guardrail_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    tool_calls_ms: float = 0.0
    generation_ms: float = 0.0
    output_guardrail_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    success: bool = True
    guardrail_status: str = "PASSED"
    timestamp: float = Field(default_factory=time.time)

class LatencyPercentiles(BaseModel):
    metric_name: str
    count: int
    p50: float
    p70: float
    p90: float
    p95: float
    p99: float
    p100: float
    min: float
    mean: float
    std_dev: float

class BenchmarkSummary(BaseModel):
    total_queries: int
    successful_queries: int
    failed_or_blocked_queries: int
    sub_200ms_compliance_rate: float  # Percentage of queries meeting < 200ms
    overall_pipeline: LatencyPercentiles
    stage_percentiles: Dict[str, LatencyPercentiles]
    recent_records: List[QueryLatencyRecord]

class LatencyAnalytics:
    def __init__(self, max_history: int = 5000):
        self.history: List[QueryLatencyRecord] = []
        self.max_history = max_history

    def record_query(self, record: QueryLatencyRecord):
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _calc_percentiles(self, values: List[float], metric_name: str) -> LatencyPercentiles:
        if not values:
            return LatencyPercentiles(
                metric_name=metric_name, count=0,
                p50=0.0, p70=0.0, p90=0.0, p95=0.0, p99=0.0, p100=0.0,
                min=0.0, mean=0.0, std_dev=0.0
            )

        arr = np.array(values, dtype=np.float64)
        count = len(arr)
        p50 = float(np.percentile(arr, 50))
        p70 = float(np.percentile(arr, 70))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        p100 = float(np.max(arr))
        min_v = float(np.min(arr))
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))

        return LatencyPercentiles(
            metric_name=metric_name,
            count=count,
            p50=round(p50, 2),
            p70=round(p70, 2),
            p90=round(p90, 2),
            p95=round(p95, 2),
            p99=round(p99, 2),
            p100=round(p100, 2),
            min=round(min_v, 2),
            mean=round(mean_v, 2),
            std_dev=round(std_v, 2)
        )

    def get_summary(self, last_n: Optional[int] = None) -> BenchmarkSummary:
        records = self.history[-last_n:] if last_n else self.history
        if not records:
            empty_p = self._calc_percentiles([], "Total Pipeline")
            return BenchmarkSummary(
                total_queries=0,
                successful_queries=0,
                failed_or_blocked_queries=0,
                sub_200ms_compliance_rate=0.0,
                overall_pipeline=empty_p,
                stage_percentiles={},
                recent_records=[]
            )

        total_latencies = [r.total_pipeline_ms for r in records]
        stt_latencies = [r.stt_ms for r in records if r.stt_ms > 0]
        input_guard_latencies = [r.input_guardrail_ms for r in records]
        embedding_latencies = [r.embedding_ms for r in records]
        retrieval_latencies = [r.retrieval_ms for r in records]
        generation_latencies = [r.generation_ms for r in records]
        output_guard_latencies = [r.output_guardrail_ms for r in records]

        successful = [r for r in records if r.guardrail_status == "PASSED"]
        sub_200 = sum(1 for l in total_latencies if l < 200.0)
        sub_200_rate = round((sub_200 / len(total_latencies)) * 100.0, 2)

        stages = {
            "stt": self._calc_percentiles(stt_latencies, "Speech-to-Text"),
            "input_guardrail": self._calc_percentiles(input_guard_latencies, "Input Guardrail"),
            "embedding": self._calc_percentiles(embedding_latencies, "Query Embedding"),
            "retrieval": self._calc_percentiles(retrieval_latencies, "Vector Retrieval (Hybrid)"),
            "generation": self._calc_percentiles(generation_latencies, "Model Generation"),
            "output_guardrail": self._calc_percentiles(output_guard_latencies, "Output Grounding Check")
        }

        return BenchmarkSummary(
            total_queries=len(records),
            successful_queries=len(successful),
            failed_or_blocked_queries=len(records) - len(successful),
            sub_200ms_compliance_rate=sub_200_rate,
            overall_pipeline=self._calc_percentiles(total_latencies, "Total End-to-End Pipeline"),
            stage_percentiles=stages,
            recent_records=records[-20:]
        )

# Global Singleton instance
latency_analytics = LatencyAnalytics()
