# CAPG-RAG

**Context-Aware Prompt Governance System for RAG-based AI Applications**

A dual-architecture RAG system with self-improving prompt governance. Supports two switchable runtime architectures (parallel vs sequential precision) for A/B comparison on quality, performance, and latency.

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate capg-rag

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 3. Seed initial prompts
python scripts/seed_prompts.py

# 4. Run the API
uvicorn api.main:app --reload

# 5. Test
curl http://localhost:8000/health
```

## Architecture

The system runs two loops:

- **Loop A (Runtime):** Handles live learner interactions — context enrichment, retrieval, generation, comprehension verification
- **Loop B (Governance):** Offline prompt improvement — analysis, risk assessment, experimentation, canary deployment

Two runtime architectures are supported:

| | Architecture A | Architecture B |
|---|---|---|
| Strategy | Parallel context + raw retrieval | Sequential precision-first |
| Query handling | Combined agent | Separate transform + context builder |
| Retrieval | Hybrid on raw query, then re-rank | Hierarchical (chapter->section->paragraph) |
| Re-ranking set | Larger (more candidates) | Smaller (higher precision) |
| Judge | Separate model | Reuses generation model |

Switch via `CAPG_ARCHITECTURE=A` or `B` in `.env`.

See [docs/architecture_comparison.md](docs/architecture_comparison.md) for details.

## Project Structure

```
config/          # Settings and architecture flags
src/models/      # Pydantic data models
src/llm/         # LLM provider abstraction (OpenAI, Gemini)
src/retrieval/   # Vector store, BM25, hybrid, hierarchical, reranker
src/agents/      # arch_a/, arch_b/, shared/ agent implementations
src/orchestrator/ # Pipeline orchestration and factory
src/prompt_service/ # Prompt registry, selector, canary routing
src/governance/  # Loop B stubs (analysis, risk, experiment, suggestion)
src/storage/     # PostgreSQL, Redis, feedback store
src/ingestion/   # PDF parsing, chunking, indexing
api/             # FastAPI application
tests/           # Test suite
scripts/         # Benchmark and seed scripts
```

## Running Tests

```bash
pytest tests/ -v
```

## Benchmarking

```bash
python scripts/benchmark.py
# Results saved to benchmark_results/latest.json
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Main interaction endpoint |
| POST | `/ingest` | Ingest PDF into knowledge base |
| POST | `/prompts/register` | Register a prompt version |
| GET | `/prompts/active` | List active prompts |
| GET | `/health` | Health check (shows active architecture) |
