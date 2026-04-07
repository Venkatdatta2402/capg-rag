"""Benchmark script — runs the same queries through both architectures and compares metrics.

Usage:
    python scripts/benchmark.py

Outputs a side-by-side comparison of:
- Quality: retrieval_quality_score, judge verdicts
- Performance: tokens used, LLM call count
- Latency: end-to-end time, per-stage timings
"""

import asyncio
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.query import QueryInput
from src.orchestrator.arch_a_pipeline import ArchAPipeline
from src.orchestrator.arch_b_pipeline import ArchBPipeline


SAMPLE_QUERIES = [
    QueryInput(
        query_text="what is M to cm?",
        user_id="bench_user_1",
        session_id="bench_session_1",
    ),
    QueryInput(
        query_text="how do plants make food?",
        user_id="bench_user_1",
        session_id="bench_session_2",
    ),
    QueryInput(
        query_text="what is evaporation?",
        user_id="bench_user_1",
        session_id="bench_session_3",
    ),
]


async def run_pipeline(pipeline, queries, arch_name):
    """Run all queries through a pipeline and collect metrics."""
    results = []
    for q in queries:
        start = time.time()
        try:
            response = await pipeline.run(q)
            elapsed = (time.time() - start) * 1000
            results.append({
                "query": q.query_text,
                "architecture": arch_name,
                "answer_preview": response.answer_text[:100],
                "prompt_version": response.metadata.prompt_version,
                "retrieval_quality_score": response.metadata.retrieval_quality_score,
                "retrieval_quality_flag": response.metadata.retrieval_quality_flag,
                "generation_model": response.metadata.generation_model,
                "latency_ms": round(elapsed, 2),
            })
        except Exception as e:
            results.append({
                "query": q.query_text,
                "architecture": arch_name,
                "error": str(e),
                "latency_ms": round((time.time() - start) * 1000, 2),
            })
    return results


async def main():
    print("=" * 70)
    print("CAPG-RAG Architecture Benchmark")
    print("=" * 70)

    # Run Architecture A
    print("\n--- Architecture A (Parallel) ---")
    try:
        arch_a = ArchAPipeline()
        results_a = await run_pipeline(arch_a, SAMPLE_QUERIES, "A")
    except Exception as e:
        print(f"Architecture A failed to initialize: {e}")
        results_a = [{"error": str(e), "architecture": "A"}]

    # Run Architecture B
    print("\n--- Architecture B (Sequential Precision) ---")
    try:
        arch_b = ArchBPipeline()
        results_b = await run_pipeline(arch_b, SAMPLE_QUERIES, "B")
    except Exception as e:
        print(f"Architecture B failed to initialize: {e}")
        results_b = [{"error": str(e), "architecture": "B"}]

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for ra, rb in zip(results_a, results_b):
        print(f"\nQuery: {ra.get('query', 'N/A')}")
        print(f"  Arch A - Latency: {ra.get('latency_ms', 'N/A')}ms | "
              f"Quality: {ra.get('retrieval_quality_flag', 'N/A')} "
              f"({ra.get('retrieval_quality_score', 'N/A')})")
        print(f"  Arch B - Latency: {rb.get('latency_ms', 'N/A')}ms | "
              f"Quality: {rb.get('retrieval_quality_flag', 'N/A')} "
              f"({rb.get('retrieval_quality_score', 'N/A')})")

    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    with open("benchmark_results/latest.json", "w") as f:
        json.dump({"arch_a": results_a, "arch_b": results_b}, f, indent=2, default=str)
    print(f"\nResults saved to benchmark_results/latest.json")


if __name__ == "__main__":
    asyncio.run(main())
