"""
Multi-Stage Guardrail Engine for Voice RAG Pipeline.
Enforces:
1. Input Safety & Prompt Injection / Jailbreak Detection
2. Out-of-Domain / Off-Topic Query Filtering
3. Vector Retrieval Relevance Floor
4. Post-Generation Grounding & Hallucination Verification
"""

import re
import time
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field

from backend.config import settings

class GuardrailVerdict(BaseModel):
    passed: bool
    stage: str  # "input_safety", "retrieval_relevance", "output_grounding"
    status: str  # "PASSED", "BLOCKED_INJECTION", "BLOCKED_TOXIC", "BLOCKED_OFF_TOPIC", "BLOCKED_UNGROUNDED"
    reason: str
    risk_score: float  # 0.0 (clean) to 1.0 (severe violation)
    hallucination_score: float = 0.0
    grounded_claims_ratio: float = 1.0
    latency_ms: float = 0.0

# Injection / Jailbreak Patterns
JAILBREAK_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
    r"(?i)\bdisregard\s+(the\s+)?system\s+prompt\b",
    r"(?i)\byou\s+are\s+now\s+in\s+(developer|dan|jailbreak)\s+mode\b",
    r"(?i)\breveal\s+(your\s+)?(system|internal)\s+(prompt|instructions|keys)\b",
    r"(?i)\bbypass\s+all\s+(filters|guardrails|safety)\b",
    r"(?i)\bexecute\s+arbitrary\s+code\b",
    r"(?i)\bdrop\s+database\b",
]

# Inappropriate / Toxic Keywords (English & Indic Romanized)
TOXIC_KEYWORDS = [
    "hack into", "make a bomb", "credit card dump", "ddos attack", 
    "steal password", "kill yourself", "hate speech", "exploit vulnerability"
]

class GuardrailsEngine:
    def __init__(self):
        self.injection_regexes = [re.compile(p) for p in JAILBREAK_PATTERNS]

    # =========================================================================
    # Stage 1: Input Safety & Prompt Injection Check
    # =========================================================================
    def check_input_safety(self, query: str) -> GuardrailVerdict:
        t0 = time.perf_counter()
        q_lower = query.lower().strip()

        # 1. Prompt Injection
        for regex in self.injection_regexes:
            if regex.search(q_lower):
                elapsed = (time.perf_counter() - t0) * 1000.0
                return GuardrailVerdict(
                    passed=False,
                    stage="input_safety",
                    status="BLOCKED_INJECTION",
                    reason="Potential prompt injection or jailbreak attempt detected.",
                    risk_score=0.95,
                    latency_ms=round(elapsed, 3)
                )

        # 2. Inappropriate Content
        for kw in TOXIC_KEYWORDS:
            if kw in q_lower:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return GuardrailVerdict(
                    passed=False,
                    stage="input_safety",
                    status="BLOCKED_TOXIC",
                    reason="Inappropriate, unsafe, or harmful request detected.",
                    risk_score=0.90,
                    latency_ms=round(elapsed, 3)
                )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return GuardrailVerdict(
            passed=True,
            stage="input_safety",
            status="PASSED",
            reason="Input query passed all safety and injection checks.",
            risk_score=0.05,
            latency_ms=round(elapsed, 3)
        )

    # =========================================================================
    # Stage 2: Post-Retrieval Relevance Floor & Keyword Verification Check
    # =========================================================================
    def check_retrieval_relevance(
        self, 
        query: str, 
        top_score: float, 
        retrieved_contexts: Optional[List[str]] = None
    ) -> GuardrailVerdict:
        t0 = time.perf_counter()
        
        # Extract query content keywords
        words = re.findall(r'[\w\u0900-\u0D7F]+', query.lower())
        stopwords = {
            "what", "is", "the", "of", "in", "and", "to", "a", "an", "are", "for", "with", "on", "at", "by", "from",
            "this", "that", "it", "as", "be", "was", "or", "which", "how", "who", "when", "where", "why", "can", "does",
            "did", "do", "will", "would", "should", "could", "about", "into", "than", "then", "so", "if", "has", "have",
            "had", "been", "its", "their", "there", "they", "we", "he", "she", "you", "me", "my", "your", "his", "her",
            "tell", "explain", "give", "some", "between",
            "क्या", "है", "हैं", "और", "का", "के", "की", "में", "से", "को", "पर", "यह", "वह", "इस", "उस", "था", "थी", "थे",
            "होता", "होती", "होते", "करना", "करते", "लिए", "द्वारा", "कब", "कहाँ", "कैसे", "किस", "कौन", "कितना"
        }
        q_kws = [w for w in words if len(w) > 1 and w not in stopwords]

        # Check keyword presence in top retrieved contexts
        if retrieved_contexts and q_kws:
            combined = " ".join(retrieved_contexts).lower()
            matched = sum(1 for kw in q_kws if kw in combined)
            ratio = matched / len(q_kws)
            
            # For off-topic rejection: If key subject nouns are missing from retrieved passages
            if len(q_kws) <= 2:
                is_off_topic = (ratio < 0.75 or top_score < 0.30)
            else:
                is_off_topic = (ratio < 0.50 or top_score < 0.28)

            if is_off_topic:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return GuardrailVerdict(
                    passed=False,
                    stage="retrieval_relevance",
                    status="BLOCKED_OFF_TOPIC",
                    reason=f"Query is not related to the dataset context (similarity: {top_score:.3f}, keyword alignment: {ratio:.2f}).",
                    risk_score=round(1.0 - top_score, 3),
                    latency_ms=round(elapsed, 3)
                )

        if top_score < settings.MIN_SIMILARITY_THRESHOLD:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GuardrailVerdict(
                passed=False,
                stage="retrieval_relevance",
                status="BLOCKED_OFF_TOPIC",
                reason=f"Top retrieved context similarity ({top_score:.3f}) is below confidence threshold ({settings.MIN_SIMILARITY_THRESHOLD}). Query is not related to the dataset.",
                risk_score=round(1.0 - top_score, 3),
                latency_ms=round(elapsed, 3)
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return GuardrailVerdict(
            passed=True,
            stage="retrieval_relevance",
            status="PASSED",
            reason="Retrieved context satisfies factual relevance threshold.",
            risk_score=round(1.0 - top_score, 3),
            latency_ms=round(elapsed, 3)
        )

    # =========================================================================
    # Stage 3: Grounding & Hallucination Check
    # =========================================================================
    def check_output_grounding(
        self, 
        generated_answer: str, 
        retrieved_contexts: List[str]
    ) -> GuardrailVerdict:
        """
        Evaluates whether key facts, entities, and assertions in generated_answer
        are grounded in the retrieved passages. Calculates Hallucination Score (0.0 to 1.0).
        """
        t0 = time.perf_counter()
        if not generated_answer or not retrieved_contexts:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GuardrailVerdict(
                passed=False,
                stage="output_grounding",
                status="BLOCKED_UNGROUNDED",
                reason="Empty answer or missing context for grounding.",
                risk_score=1.0,
                hallucination_score=1.0,
                grounded_claims_ratio=0.0,
                latency_ms=round(elapsed, 3)
            )

        combined_context = " ".join(retrieved_contexts).lower()
        
        # Tokenize answer clauses/sentences
        sentences = [s.strip() for s in re.split(r'[।॥\.\?!]+', generated_answer) if len(s.strip()) > 5]
        if not sentences:
            sentences = [generated_answer]

        grounded_count = 0
        for sent in sentences:
            words = [w.lower().strip(".,!?;:\"'") for w in sent.split() if len(w) > 2]
            if not words:
                continue
            # Match word presence in context
            matched_words = sum(1 for w in words if w in combined_context)
            ratio = matched_words / len(words)
            if ratio >= 0.40:
                grounded_count += 1

        grounded_ratio = grounded_count / len(sentences) if sentences else 1.0
        hallucination_score = round(1.0 - grounded_ratio, 3)

        elapsed = (time.perf_counter() - t0) * 1000.0

        if hallucination_score > settings.MAX_HALLUCINATION_SCORE:
            return GuardrailVerdict(
                passed=False,
                stage="output_grounding",
                status="BLOCKED_UNGROUNDED",
                reason=f"Generated answer failed factual grounding (Hallucination score: {hallucination_score:.2f} > limit {settings.MAX_HALLUCINATION_SCORE}).",
                risk_score=hallucination_score,
                hallucination_score=hallucination_score,
                grounded_claims_ratio=round(grounded_ratio, 3),
                latency_ms=round(elapsed, 3)
            )

        return GuardrailVerdict(
            passed=True,
            stage="output_grounding",
            status="PASSED",
            reason="Answer is verified and grounded in retrieved dataset passages.",
            risk_score=hallucination_score,
            hallucination_score=hallucination_score,
            grounded_claims_ratio=round(grounded_ratio, 3),
            latency_ms=round(elapsed, 3)
        )

# Global Singleton instance
guardrails_engine = GuardrailsEngine()
