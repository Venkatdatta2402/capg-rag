# PG-CARAG

**Context-Aware Prompt Governance RAG System for CBSE/NCERT**

A production-grade RAG system for CBSE/NCERT education with sequential precision retrieval, MCQ-based comprehension testing, tag-driven prompt selection, and an offline prompt governance loop.

---

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate pg-carag

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY (and optionally GOOGLE_API_KEY) to .env

# 3. Start Qdrant (vector database)
docker run -d -p 6333:6333 -v ~/.qdrant:/qdrant/storage qdrant/qdrant

# 4. Start Elasticsearch
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.13.0

# 5. Start PostgreSQL (learner profiles)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:16

# 6. Seed teaching-style prompts
python scripts/seed_prompts.py

# 7. Ingest NCERT PDFs
curl -X POST http://localhost:8000/ingest \
  -F "file=@ncert_grade10_science_ch6.pdf" \
  -F "grade=10" \
  -F "subject=Science" \
  -F "chapter_number=6" \
  -F "chapter_title=Life Processes"

# 8. Extract and save keywords from textbooks
python scripts/extract_keywords.py \
  --pdf ncert_grade10_science.pdf \
  --grade 10 \
  --subject Science

# 9. Run the API
uvicorn api.main:app --reload

# 10. Health check
curl http://localhost:8000/health
```

---

## How It Works

### Runtime Flow (per interaction)

```
Student query
  → fetch learner profile (PostgreSQL) + session state (Elasticsearch)
  → build ContextObject from profile + session  [built once, persisted in session]
  → lookup domain keywords (grade+subject keyword store)
  → query transform agent  (enriches query with NCERT keywords, detects topic/chapter/section)
  → hierarchical retrieval (chapter → section → subsection progressively narrows scope)
  → cross-encoder reranking
  → prompt selection       (LLM estimates topic strength; tag-match selects teaching style)
  → RAG generation         (answer + 3 MCQ quiz questions via tool call in ONE LLM pass)
  → interaction appended to session_interactions (ES) and session_memory (ES)
  → response returned to learner
```

### Quiz / Retry Flow

```
Learner submits MCQ answers → POST /quiz/submit
  → loads interaction from session_interactions (ES)
  → grades each answer deterministically (selected == correct)
  → wrong answers only: judge LLM generates retrieval_feedback
  → updates interaction in ES with student_response + quiz.status
  → updates session_memory (ES) with quiz score
  → if PASSED: verdicts returned, done
  → if FAILED:
      - session ContextObject.retry_mode set to True (persisted in session_memory)
      - judge retrieval_feedback aggregated into retrieval_hint
      - 3 best-fit prompt choices returned as MCQ

Learner picks a teaching style → POST /quiz/regenerate
  → full pipeline re-runs with retrieval hint appended to original query
  → chosen prompt version bypasses selector
  → fresh answer + new quiz form returned
```

### End-of-Session Review

```
POST /session/end  (fire-and-forget)
  → SessionReviewAgent (LLM) reads summary_of_past + recent_interactions from session_memory
  → updates long-term LearnerProfile (technically_strong/weak, softskills, learning_styles)
  → writes FeedbackRecord for governance pipeline
  → deletes session from session_memory (ES)
```

### Governance Loop (offline)

Accumulates feedback records from all interactions, mines patterns, assesses risk, generates prompt candidates, runs canary experiments, and promotes or rejects candidates automatically.

---

## Storage Architecture

| Store | Backend | Purpose |
|-------|---------|---------|
| `LearnerProfile` | PostgreSQL (JSONB) | Long-term learner profile — updated once per session end |
| `session_memory` | Elasticsearch | Runtime session state: recent 5 interactions + rolling `PastSummary` + `ContextObject` |
| `session_interactions` | Elasticsearch | Full interaction log: answers, MCQ quiz with correct keys, student responses |
| `PromptRegistry` | In-memory → PostgreSQL | Versioned teaching-style prompts |
| `FeedbackStore` | In-memory → PostgreSQL | Governance feedback records |
| Vector store | Qdrant | NCERT chunk embeddings with hierarchy metadata |

### Two Elasticsearch Indices

**`session_memory`** — short-term, deleted at session end
```json
{
  "session_id": "...",
  "retry_count": 0,
  "context_object": { "learner_grade": "Class 5", "retry_mode": false, "..." },
  "recent_interactions": [
    { "interaction_id": "...", "question": "...", "model_answer": "...",
      "topic": "...", "quiz_status": "submitted", "score": 2 }
  ],
  "summary_of_past": {
    "covered_topics": ["..."],
    "common_errors": ["..."],
    "key_misconceptions": ["..."],
    "performance_trend": "improving"
  }
}
```

**`session_interactions`** — persistent interaction log
```json
{
  "session_id": "...",
  "interactions": [
    {
      "interaction_id": "...",
      "question": "...",
      "model_answer": "...",
      "quiz": { "questions": [{ "question_id": "q1", "options": ["A)...","B)..."], "correct_answer": "B" }], "status": "submitted" },
      "student_response": { "answers": [...], "score": 2 },
      "context_used": [...],
      "meta": { "subject": "Science", "topic": "...", "difficulty": "medium" }
    }
  ]
}
```

---

## ContextObject Lifecycle

`ContextObject` is built **once per session** from `LearnerProfile` + `SessionState` and persisted in `session_memory`. It is reused on every subsequent pipeline run within the session.

The judge path is the only writer after creation:
- Quiz FAILED → `session_store.set_retry_mode()` → `context_object.retry_mode = True`
- Next pipeline run picks up `retry_mode=True` without rebuilding

---

## Prompt Selection (Tag-Based)

Each prompt version carries a set of tags. The selector builds a context tag set from the session's `ContextObject` and picks the prompt with maximum tag overlap.

**Tag namespaces:**

| Namespace | Example tags |
|-----------|-------------|
| `learnstyle:` | `visual`, `example_driven`, `step_by_step`, `guided`, `challenge_based`, `hint_sensitive` |
| `softskill:` | `decomposition`, `abstraction`, `working_memory`, `attention_control`, `reflection` |
| `topic:` | `topic:strong`, `topic:weak` (LLM-estimated per query) |
| Retry | `retry` |

---

## Project Structure

```
config/
  settings.py               # Pydantic Settings — all config from .env

src/
  agents/
    query_transform.py      # Rewrites raw query for retrieval (NCERT keyword injection)
    context_builder.py      # Builds ContextObject from profile + session (deterministic)
    rag_agent.py            # Generates answer + MCQ quiz via present_mcq tool call
    judge.py                # Grades MCQ deterministically; LLM only on wrong answers
    session_review.py       # End-of-session LLM review → updated learner profile
    base.py

  orchestrator/
    pipeline.py             # LangGraph StateGraph — wires all nodes end-to-end
    base.py
    retry.py                # Retry manager

  retrieval/
    vector_store.py         # Qdrant dense retrieval with NCERT hierarchy filters
    hierarchical.py         # Progressive narrowing: grade → chapter → section
    hybrid.py               # Dense + BM25 fused via Reciprocal Rank Fusion
    reranker.py             # Cross-encoder reranking (sentence-transformers)
    sparse.py               # BM25 retrieval

  prompt_service/
    registry.py             # Versioned prompt storage (in-memory → PostgreSQL)
    selector.py             # Tag-overlap scoring + LLM topic strength estimation
    canary.py               # 5%/95% canary routing

  storage/
    interaction_store.py    # ES session_interactions index — full interaction log
    session_memory.py       # ES session_memory index — recent window + summary + ContextObject
    user_profile.py         # Long-term learner profile (PostgreSQL)
    feedback_store.py       # Interaction outcome records for governance
    keyword_store.py        # Grade+subject → curated NCERT keyword lists

  tools/
    keyword_lookup.py       # LangGraph @tool + plain async wrapper for pipeline nodes
    present_mcq.py          # Tool schema for structured MCQ generation via LLM tool call

  models/
    learner.py              # LearnerProfile, SessionState, RecentInteraction, PastSummary
    query.py                # QueryInput, EnrichedQuery, ContextObject
    response.py             # GenerationResponse, QuizForm, QuizQuestion
    interaction.py          # SessionInteractionDocument, Interaction, QuizData, StudentResponse
    feedback.py             # JudgeVerdict, FeedbackRecord

  llm/                      # LLM abstraction (OpenAI, Gemini) with tool-call support
  ingestion/                # PDF parser (section detection), chunker, indexer
  governance/               # Offline loop stubs (analysis, risk, experiment, suggestion)

api/
  main.py                   # FastAPI app
  dependencies.py           # DI singletons
  routes/
    chat.py                 # POST /chat
    quiz.py                 # POST /quiz/submit, POST /quiz/regenerate
    session.py              # POST /session/end
    ingest.py               # POST /ingest
    keywords.py             # POST /keywords, GET /keywords/{grade}/{subject}
    prompts.py              # Prompt registry endpoints

scripts/
  seed_prompts.py           # Seeds teaching-style prompts into registry
  extract_keywords.py       # LLM-powered keyword extraction from NCERT PDFs
  benchmark.py              # Latency + quality benchmark across sample queries

tests/
  conftest.py
  test_agents/
  test_api/
  test_orchestrator/
  test_retrieval/
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Submit a student query; returns answer + MCQ quiz form |
| POST | `/quiz/submit` | Submit MCQ answers; deterministic grading + judge feedback on wrong answers |
| POST | `/quiz/regenerate` | Re-run pipeline with chosen teaching style + retrieval hint |
| POST | `/session/end` | Fire-and-forget end-of-session review; updates long-term profile |
| POST | `/ingest` | Ingest NCERT PDF chapter into vector DB |
| POST | `/keywords` | Upload curated keyword list for a grade+subject |
| GET | `/keywords/{grade}/{subject}` | Retrieve keyword list |
| GET | `/keywords` | List all registered grade+subject keys |
| POST | `/prompts/register` | Register a prompt version |
| GET | `/prompts/active` | List active prompts |
| GET | `/health` | Health check |

---

## Configuration

Key `.env` variables:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...          # optional, for Gemini models

GENERATION_MODEL=gpt-4.1
GENERATION_PROVIDER=openai

CONTEXT_MODEL=gpt-4o-mini
CONTEXT_PROVIDER=openai

JUDGE_MODEL=gpt-4o-mini
JUDGE_PROVIDER=openai

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=capg_knowledge

DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/pg_carag

ELASTICSEARCH_URL=http://localhost:9200

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CANARY_TRAFFIC_PERCENT=5
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
```

---

## Running Tests

```bash
pytest tests/ -v
```

## Benchmarking

```bash
python scripts/benchmark.py
# Results saved to benchmark_results/results_<timestamp>.json
```
