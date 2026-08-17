"""
Dataset Ingestion and CLI Tool for ai4bharat/MSMARCO-XI.
Provides direct Hugging Face `load_dataset` ingestion across 14+ Indic languages
and populates the local Voice RAG vector store cache.

Usage:
    python ingest_dataset.py --stats
    python ingest_dataset.py --lang hi --samples 10
    python ingest_dataset.py --lang te --samples 10
    python ingest_dataset.py --all-langs
"""

import sys
import argparse
from pathlib import Path

# Fix console encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.dataset_loader import dataset_loader, MSMARCO_XI_LANG_PREFIX
from backend.chunking_engine import chunking_engine
from backend.vector_store import vector_store

def print_stats():
    stats = dataset_loader.get_stats()
    print("=" * 60)
    print(" ✦ ai4bharat/MSMARCO-XI Dataset Status ✦")
    print("=" * 60)
    print(f"Source:              {stats['source']}")
    print(f"Total Documents:     {stats['total_documents']}")
    print(f"Cache File:          {stats['cache_file']}")
    print("\nLanguage Breakdown:")
    for lang, count in stats["languages"].items():
        print(f"  • {lang:<6} : {count} documents")
    print("\nSupported Indic Languages:")
    print(f"  {', '.join(stats['supported_indic_languages'])}")
    print("=" * 60)

def ingest_language(lang: str, samples: int = 20, split: str = "train"):
    print(f"\n[Ingesting] ai4bharat/MSMARCO-XI for language '{lang}' (max {samples} samples)...")
    count = dataset_loader.load_from_huggingface(language=lang, max_samples=samples, split=split)
    
    # Reindex vector store
    docs = dataset_loader.get_all_documents()
    chunks = chunking_engine.process_corpus(docs, strategy="indic_semantic")
    idx_time = vector_store.index_chunks(chunks)
    
    print(f"✓ Successfully processed {count} records for '{lang}'.")
    print(f"✓ In-Memory Vector Store re-indexed: {len(chunks)} total chunks in {idx_time:.2f}ms.")

def display_samples(lang: str = "hi", limit: int = 3):
    docs = [d for d in dataset_loader.get_all_documents() if d.metadata.language == lang]
    if not docs:
        docs = dataset_loader.get_all_documents()[:limit]
    
    print(f"\n--- Sample Passages from ai4bharat/MSMARCO-XI ({lang}) ---")
    for idx, doc in enumerate(docs[:limit], 1):
        print(f"\n[{idx}] ID: {doc.id} | Topic: {doc.metadata.topic} | Lang: {doc.metadata.language}")
        print(f"Query:   {doc.query}")
        print(f"Passage: {doc.passage_text[:120]}...")
        if doc.answers:
            print(f"Answer:  {doc.answers[0]}")

def main():
    parser = argparse.ArgumentParser(description="Ingest and explore ai4bharat/MSMARCO-XI dataset for Voice RAG")
    parser.add_argument("--stats", action="store_true", help="Display current dataset stats")
    parser.add_argument("--lang", type=str, default="hi", help="Language code to ingest (e.g., hi, te, ta, bn, mr, kn, gu)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to ingest from Hugging Face")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Dataset split")
    parser.add_argument("--show-samples", action="store_true", help="Display sample entries from the dataset")
    parser.add_argument("--all-langs", action="store_true", help="Ingest samples for all supported Indic languages")

    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.show_samples:
        display_samples(lang=args.lang)
    elif args.all_langs:
        for lang in list(MSMARCO_XI_LANG_PREFIX.keys())[:5]:
            ingest_language(lang, samples=5, split=args.split)
        print_stats()
    else:
        ingest_language(args.lang, samples=args.samples, split=args.split)
        display_samples(lang=args.lang)

if __name__ == "__main__":
    main()
