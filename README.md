# PG-CARAG

**Context-Aware Prompt Governance RAG System for CBSE/NCERT**

A production-grade RAG system for CBSE/NCERT education with two-pass hierarchical retrieval, chunk-level enrichment, MCQ-based comprehension testing, tag-driven adaptive prompt selection, and an offline prompt governance loop.

---

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate pg-carag

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY (and optionally GOOGLE_API_KEY) to .env

# 3. Start Qdrant (vector database — two collections: chunks + hierarchy nodes)
docker run -d -p 6333:6333 -v ~/.qdrant:/qdrant/storage qdrant/qdrant

# 4. Start Elasticsearch (session memory, interaction log, learner profiles, eval)
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.13.0

# 5. Seed teaching-style prompts
python scripts/seed_prompts.py

# 6. Ingest NCERT PDFs (chunks + hierarchy nodes built in one call)
curl -X POST http://localhost:8000/ingest \
  -F "file=@ncert_grade10_science_ch6.pdf" \
  -F "grade=10" \
  -F "subject=Science" \
  -F "unit=Unit II" \
  -F "chapter_number=6" \
  -F "chapter_title=Life Processes"

# 7. Run the API
uvicorn api.main:app --reload

# 8. Health check
curl http://localhost:8000/health
```

---

## How It Works

### Runtime Flow (per interaction)

```
Student query
  → fetch_profile       — LearnerProfileDocument (ES) + SessionState (ES)
  → initial_retrieve    — broad hybrid search (grade-scoped only), collect chunk signals
  → transform_query     — rewrite query using chunk keywords/concepts + session context
  → retrieve            — hierarchical: unit→chapter→section→subsection (top-3 per level)
     (parallel)           final: parallel dense per top-3 scope + BM25 sparse → RRF
  → select_prompt       — LLM reasons ContextObject; tag-overlap picks teaching style
  → rerank              — cross-encoder reranks retrieved chunks
  → generate            — answer + 3 MCQ questions (same LLM pass via present_mcq tool)
  → persist             — interaction → session_interactions (ES) + session_memory (ES)
  → response returned
```

### Quiz / Judge Flow

```
POST /quiz/submit
  → load interaction from session_interactions (ES)
  → deterministic grading: selected == correct_answer
  → wrong answers only: judge LLM generates retrieval_feedback per wrong answer
  → update ES: student_response + quiz.status = "submitted"
  → update session_memory with quiz score
  → PASSED (≥ 2/3): return verdicts
  → FAILED:
      judge reads session memory → generates follow-up question targeting knowledge gap
      set_retry_mode() → ContextObject.retry_mode = True (persisted in session_memory)
      full pipeline re-runs with judge question (session context skipped in transform_query)
      follow-up answer + quiz returned inline in the same response
```

### Ingestion Pipeline

```
POST /ingest (PDF upload)
  → PDFParser       — pages with section/subsection metadata (regex on decimal headings)
  → Chunker         — structure-aware splits: subsection > section > paragraph > word-limit
                      chunk_size=450 words, overlap=67 words
  → ChunkEnricher   — one LLM call per 8 chunks → JSON array of keywords (5-8) + concepts (2-4)
  → Indexer         — upserts enriched chunks into Qdrant capg_knowledge
  → HierarchyBuilder— one HierarchyNode per unique level scope (unit/chapter/section/subsection)
                      LLM generates 2-3 sentence summary per node
                      keywords + concepts aggregated from constituent chunks
  → HierarchyStore  — batch-embeds summaries → upserts into Qdrant capg_hierarchy
```

### End-of-Session Review

```
POST /session/end  (fire-and-forget, two background tasks)

SessionReviewAgent:
  → reads full interaction log (session_interactions)
  → LLM updates softskills, learning_style, technical_skills in LearnerProfileDocument
  → writes FeedbackRecord for governance pipeline
  → deletes session from session_memory

EvalAgent:
  → reads all interactions from session_interactions
  → per-interaction LLM evaluation: correctness, groundedness, answer_relevance,
    coherence, sufficiency, error_type
  → writes EvalResult to session_evaluations (ES)
```

### Governance Loop (offline)

Accumulates `FeedbackRecord`s, mines patterns, assesses prompt risk, generates candidates, runs 5% canary experiments, and promotes or rejects automatically.

---

## Storage Architecture

| Store | Backend | Purpose |
|-------|---------|---------|
| `capg_knowledge` | Qdrant | Enriched chunk embeddings with full NCERT hierarchy metadata |
| `capg_hierarchy` | Qdrant | HierarchyNode summary embeddings for scope traversal |
| `session_memory` | Elasticsearch | Short-term: recent 5 interactions + PastSummary + ContextObject |
| `session_interactions` | Elasticsearch | Persistent: full interaction log, MCQ correct answers, student responses |
| `learner_profiles` | Elasticsearch | Long-term learner profile — updated once per session end |
| `session_evaluations` | Elasticsearch | Per-interaction quality scores from EvalAgent |

---

## Retrieval Architecture

### Two-Pass Design

**Pass 1 — initial_retrieve (signal collection)**
Broad hybrid search (dense + BM25 RRF), scoped to grade only. Chunks returned here are not used for generation — only their `keywords` and `concepts` fields are pooled and fed into `transform_query` to inform query rewriting.

**Pass 2 — retrieve (hierarchical precision)**
`HierarchicalRetriever` narrows scope level by level using embedded `HierarchyNode` summaries stored in `capg_hierarchy`.

### Hierarchy Traversal

Each level scores candidates with:
```
score = 0.90 × vector_similarity
      + 0.05 × keyword_overlap      # |query_kw ∩ node_kw| / len(query_kw)
      + 0.03 × concept_overlap      # |query_ct ∩ node_ct| / len(query_ct)
      + 0.02 × title_match          # binary: any query token in title
```
Weights renormalised when `query_keywords` or `query_concepts` is empty.

```
Level 1 — unit:        no parent filter                       → top-3 units
Level 2 — chapter:     unit IN [u1, u2, u3]                  → top-3 chapters
Level 3 — section:     chapter_title IN [c1, c2, c3]         → top-3 sections
Level 4 — subsection:  section_number IN [s1, s2, s3]        → top-3 subsections
Final:    parallel dense search per scope × top-3 + BM25 sparse → RRF → top_k chunks
```

---

## ContextObject

Built by `PromptSelector` from `LearnerProfileDocument` + `EnrichedQuery`. Persisted in `session_memory`. Reused on every turn within the session.

```python
class ContextObject:
    grade: str
    learning_styles: list[str]    # learnstyle: tags
    softskills_strong: list[str]  # softskill: tags
    softskills_weak: list[str]    # softskill: tags
    topic_strength: str           # "topic:strong" | "topic:weak"
    retry_mode: bool              # set by judge path only
    retry_count: int              # set by judge path only
```

**Session memory access rules:**

| Agent | Reads |
|-------|-------|
| `query_transform` | `summary_of_past` + `recent_interactions` (only when quiz unattempted) |
| `judge` | `summary_of_past` + `recent_interactions` (only on quiz FAILED) |
| `select_prompt` | `context_object` only |

---

## Prompt Selection

Each prompt carries a set of tags. `PromptSelector` builds a context tag set from `ContextObject` and picks the prompt with maximum tag overlap.

**Tag namespaces:**

| Namespace | Tags |
|-----------|------|
| `learnstyle:` | `visual`, `textual`, `example_driven`, `abstract_first`, `guided`, `exploratory`, `step_by_step`, `challenge_based`, `immediate_feedback`, `delayed_reflection`, `hint_sensitive` |
| `softskill:` | `decomposition`, `abstraction`, `pattern_mapping`, `working_memory`, `attention_control`, `process_discipline`, `error_detection`, `confidence_calibration`, `reflection` |
| `topic:` | `topic:strong`, `topic:weak` |
| Retry | `retry` |

---

## Project Structure

```
config/
  settings.py                  # Pydantic Settings — all config from .env

src/
  agents/
    query_transform.py         # Rewrites query using chunk signals + session context
    rag_agent.py               # Generates answer + MCQ quiz via present_mcq tool call
    judge.py                   # Deterministic grading; LLM on wrong answers + follow-up question
    session_review.py          # End-of-session LLM review → updated learner profile
    eval_agent.py              # Per-interaction quality scoring at session end
    base.py

  orchestrator/
    pipeline.py                # LangGraph StateGraph — two-pass retrieval, parallel branches

  retrieval/
    vector_store.py            # Qdrant dense retrieval with NCERT hierarchy filters
    hierarchy_store.py         # Qdrant capg_hierarchy — embed summaries, composite scoring
    hierarchical.py            # Top-3-at-every-level traversal + parallel final retrieval
    hybrid.py                  # Dense + BM25 RRF (used for initial_retrieve)
    reranker.py                # Cross-encoder reranking
    sparse.py                  # BM25 retrieval

  ingestion/
    pdf_parser.py              # PDF → pages with section/subsection metadata
    chunker.py                 # Structure-aware chunker (450 words, 67-word overlap)
    chunk_enricher.py          # Batch LLM enrichment — keywords + concepts per chunk
    hierarchy_builder.py       # Builds HierarchyNodes with LLM summaries
    indexer.py                 # Upserts enriched chunks into capg_knowledge

  prompt_service/
    registry.py                # Versioned prompt storage
    selector.py                # LLM context analysis + tag-overlap prompt selection
    canary.py                  # 5%/95% canary routing

  storage/
    interaction_store.py       # ES session_interactions — full interaction log
    session_memory.py          # ES session_memory — recent window + PastSummary + ContextObject
    learner_profile_store.py   # ES learner_profiles — weighted skill update
    eval_store.py              # ES session_evaluations — eval results
    feedback_store.py          # Governance feedback records

  models/
    hierarchy.py               # HierarchyNode
    learner.py                 # SessionState, RecentInteraction, PastSummary
    query.py                   # QueryInput, EnrichedQuery, ContextObject
    response.py                # GenerationResponse, QuizForm, QuizQuestion
    interaction.py             # Interaction, QuizData, StudentResponse, ContextChunk
    feedback.py                # JudgeVerdict, FeedbackRecord
    eval.py                    # InteractionEval, EvalResult
    profile_document.py        # LearnerProfileDocument, SkillEntry
    retrieval.py               # RetrievalResult, RerankedChunk

  tools/
    present_mcq.py             # Tool schema for structured MCQ generation

  llm/                         # LLM abstraction (OpenAI, Gemini) with tool-call support
  governance/                  # Offline loop (analysis, risk, suggestion, experiment)

api/
  main.py                      # FastAPI app
  dependencies.py              # DI singletons
  routes/
    chat.py                    # POST /chat
    quiz.py                    # POST /quiz/submit
    session.py                 # POST /session/end
    ingest.py                  # POST /ingest
    prompts.py                 # Prompt registry endpoints

scripts/
  seed_prompts.py              # Seeds teaching-style prompts into registry
  benchmark.py                 # Latency + quality benchmark across sample queries
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Submit student query; returns answer + MCQ quiz form |
| POST | `/quiz/submit` | Grade MCQ answers; on FAILED, judge generates follow-up + pipeline re-runs inline |
| POST | `/session/end` | Fire-and-forget session end: profile review + eval scoring |
| POST | `/ingest` | Ingest NCERT PDF — chunks + hierarchy nodes built in one call |
| POST | `/prompts/register` | Register a prompt version |
| GET | `/prompts/active` | List active prompts |
| GET | `/health` | Health check |

---

## Configuration

Key `.env` variables:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...           # optional, for Gemini models

GENERATION_MODEL=gpt-4.1
GENERATION_PROVIDER=openai

CONTEXT_MODEL=gpt-4o-mini
CONTEXT_PROVIDER=openai

JUDGE_MODEL=gpt-4o-mini
JUDGE_PROVIDER=openai

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=capg_knowledge
QDRANT_HIERARCHY_COLLECTION=capg_hierarchy

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
