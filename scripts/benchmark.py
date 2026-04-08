"""Benchmark script — measures latency and retrieval quality across sample queries.

Usage:
    python scripts/benchmark.py

Output is written to benchmark_results/results_<timestamp>.json
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.pipeline import Pipeline
from src.models.query import QueryInput

SAMPLE_QUERIES = [
    {"query": "What is photosynthesis?", "user_id": "bench_user", "grade": "5"},
    {"query": "Explain the water cycle", "user_id": "bench_user", "grade": "5"},
    {"query": "What is Newton's second law?", "user_id": "bench_user", "grade": "10"},
    {"query": "Define osmosis", "user_id": "bench_user", "grade": "10"},
    {"query": "What causes seasons on Earth?", "user_id": "bench_user", "grade": "5"},
]


async def run_query(pipeline: Pipeline, entry: dict) -> dict:
    query_input = QueryInput(
        query_text=entry["query"],
        user_id=entry["user_id"],
        session_id=str(uuid.uuid4()),
    )
    start = time.time()
    response = await pipeline.run(query_input)
    latency_ms = (time.time() - start) * 1000

    return {
        "query": entry["query"],
        "latency_ms": round(latency_ms, 2),
        "retrieval_quality": response.metadata.retrieval_quality_flag,
        "retrieval_score": response.metadata.retrieval_quality_score,
        "prompt_version": response.metadata.prompt_version,
        "answer_preview": response.answer_text[:120],
        "quiz_generated": not response.quiz_form.skipped,
    }


async def main():
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    pipeline = Pipeline()
    print(f"\nRunning {len(SAMPLE_QUERIES)} benchmark queries ...\n")

    results = []
    for entry in SAMPLE_QUERIES:
        print(f"  {entry['query'][:60]} ...", end=" ", flush=True)
        result = await run_query(pipeline, entry)
        results.append(result)
        print(f"{result['latency_ms']:.0f}ms  [{result['retrieval_quality']}]")

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    quality_counts: dict[str, int] = {}
    for r in results:
        quality_counts[r["retrieval_quality"]] = quality_counts.get(r["retrieval_quality"], 0) + 1

    summary = {
        "timestamp": timestamp,
        "total_queries": len(results),
        "avg_latency_ms": round(avg_latency, 2),
        "quality_distribution": quality_counts,
        "results": results,
    }

    out_path = f"benchmark_results/results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAvg latency: {avg_latency:.0f}ms")
    print(f"Quality distribution: {quality_counts}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
