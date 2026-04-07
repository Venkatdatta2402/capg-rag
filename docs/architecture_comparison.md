# Architecture Comparison: A vs B

## Overview

This project supports two runtime architectures for the RAG pipeline. Both share the same infrastructure, data models, and prompt governance layer. Switch between them by setting `CAPG_ARCHITECTURE=A` or `CAPG_ARCHITECTURE=B` in your `.env` file.

## Architecture A — Parallel Context + Raw Retrieval

**Flow:**
1. Context & Rephrase Agent runs **in parallel** with raw query retrieval
2. When both complete, results are re-ranked using the enriched query
3. Prompt Service selects the versioned prompt
4. RAG Agent generates the answer

**Key characteristics:**
- Single combined agent handles both query rephrasing and context building
- Retrieval starts immediately on the raw query (lower latency start)
- Larger candidate set goes to cross-encoder re-ranking
- Separate judge model (different from generation model)
- Retry triggered via `retry_mode` flag

**When to prefer:**
- When latency is critical (parallel start reduces wall-clock time)
- When raw queries are already reasonably specific

## Architecture B — Sequential Precision-First

**Flow:**
1. Fetch learner profile
2. Query Transformation Agent enriches query with keyword pool
3. Hierarchical retrieval (chapter -> section -> paragraph) using enriched query
4. Cross-encoder re-ranking on smaller, higher-quality candidate set
5. Context Object Builder constructs context (parallel with step 3-4)
6. Prompt Service selects the versioned prompt
7. RAG Agent generates the answer

**Key characteristics:**
- Separate Query Transform Agent and Context Object Builder (modularity)
- No retrieval on raw query — enriched query drives everything
- Hierarchical narrowing produces fewer, better candidates
- Re-ranking is faster (fewer candidates to score)
- Judge reuses generation model (already has full context)
- Retry triggered by routing signals (UNDERSTOOD/NOT_UNDERSTOOD)

**When to prefer:**
- When retrieval quality matters more than initial latency
- When the knowledge base has clear hierarchical structure (chapters/sections)
- When raw queries are short and ambiguous

## Running the Benchmark

```bash
# Ensure your .env has valid API keys and Qdrant is running
python scripts/benchmark.py
```

The benchmark runs the same queries through both architectures and outputs:
- Retrieval quality scores
- End-to-end latency
- Answer previews

Results are saved to `benchmark_results/latest.json`.

## Switching Architectures

```bash
# In .env
CAPG_ARCHITECTURE=A   # or B

# The API automatically uses the correct pipeline
uvicorn api.main:app --reload
```

The `/health` endpoint shows which architecture is active.
