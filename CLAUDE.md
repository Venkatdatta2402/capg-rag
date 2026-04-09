# PG-CARAG — System Design Document

**Context-Aware Prompt Governance System for RAG-based AI Applications**
**Domain:** CBSE/NCERT Educational AI | **Version:** 5.0

---

## What This System Does

Most RAG systems treat prompts as static text and the context window as a dump of retrieved chunks. This system solves both problems:

**Problem 1 — Static prompts decay.** A prompt that works for Grade 5 Maths in one term may silently fail for Grade 10 Science in another. Without governance, nobody knows until outcomes degrade.

**Problem 2 — Unengineered context.** Sending a raw query ignores everything the system already knows about the learner — their grade, weak areas, what they understood five minutes ago.

**Solution:** Prompts are versioned, governed, self-improving assets. Context construction is an engineering discipline, not an afterthought.

---

## System Overview

Two parallel, connected loops:

```
LOOP A — Runtime (real-time, per interaction)
  Student query → pipeline → answer + MCQ quiz → judge evaluation → retry or done

LOOP B — Governance (offline, async)
  Feedback records → analysis → risk assessment → prompt candidate → canary test → promote/reject
```

---

## Loop A — Runtime Pipeline

### LangGraph State Graph

```
fetch_profile
    ↓
initial_retrieve         ← HybridRetriever, grade-scoped only (broad signal collection)
    ↓
transform_query          ← context LLM (gpt-4o-mini) + chunk keywords/concepts as signals
    ↓ ↘
retrieve                  select_prompt   ← run in parallel
    ↓                         ↓
rerank                        ↓
    ↓ ↘                      ↗
      → → → generate ← ← ←
               ↓
              END
```

### Node Responsibilities

**fetch_profile**
Loads `LearnerProfileDocument` (ES: `learner_profiles`) and `SessionState` (ES: `session_memory`).

**initial_retrieve**
Pass 1: broad hybrid retrieval (dense + BM25 RRF) scoped to grade only. Purpose is signal collection — chunks returned here are never used for generation, only their `keywords` and `concepts` fields are fed into `transform_query`.

**transform_query**
Query Transformation Agent (context LLM). Pools `keywords` and `concepts` from initial_retrieve chunks, selects relevant ones, rewrites the query to be retrieval-precise. Detects subject, topic, section hints. Outputs `EnrichedQuery`. Session context (covered topics, performance trend, recent topics) injected only when quiz was unattempted (`_judge_followup=False`).

**retrieve** *(parallel with select_prompt)*
Pass 2: hierarchical retrieval using the enriched query. Progressively narrows scope — unit → chapter → section → subsection — using embedded hierarchy node summaries in `capg_hierarchy`. Top-3 at every level, each level filters using `MatchAny` on the parent level's top-3 values. Final chunk retrieval: parallel dense search per top-3 deepest-scope + sparse BM25, fused via RRF.

**rerank**
Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks retrieved candidates. Average score drives quiz generation decision.

**select_prompt** *(parallel with retrieve)*
Single LLM call analyzes `LearnerProfileDocument` + `EnrichedQuery` → reasons all `ContextObject` signals. Reads `retry_mode`/`retry_count` from persisted `ContextObject` in session. Builds tag set → picks best prompt by tag overlap. Saves `ContextObject` to `session_memory`.

**generate**
Generation LLM (gpt-4.1) produces the answer. When retrieval quality is GOOD (`avg_rerank_score > 0.75`) and not `retry_mode`, calls `present_mcq` tool in the **same LLM pass** to output 3 MCQ questions. Correct answers stored server-side in `session_interactions`; client receives questions only.

---

## ContextObject

Output of `PromptSelector.select()` — NOT an input to it.

```python
class ContextObject:
    grade: str                      # from LearnerProfileDocument
    learning_styles: list[str]      # LLM-reasoned learnstyle: tags
    softskills_strong: list[str]    # LLM-reasoned softskill: tags (relevant strengths)
    softskills_weak: list[str]      # LLM-reasoned softskill: tags (relevant weaknesses)
    topic_strength: str             # "topic:strong" | "topic:weak"
    retry_mode: bool                # written by judge path ONLY
    retry_count: int                # written by judge path ONLY
```

Persisted in `session_memory` ES across turns. `retry_mode`/`retry_count` written by `session_store.set_retry_mode()` after quiz failure.

---

## Session Memory Access Rules

| Agent | What it reads |
|-------|--------------|
| `query_transform` | `summary_of_past` + `recent_interactions` — only when `_judge_followup=False` |
| `judge` | `summary_of_past` + `recent_interactions` — only on quiz FAILED |
| `select_prompt` | `context_object` only — always |

---

## Tag Namespaces

```
learnstyle:  visual, textual, example_driven, abstract_first, guided, exploratory,
             step_by_step, challenge_based, immediate_feedback, delayed_reflection, hint_sensitive

softskill:   decomposition, abstraction, pattern_mapping, working_memory, attention_control,
             process_discipline, error_detection, confidence_calibration, reflection

topic:       topic:strong | topic:weak   (LLM-estimated per query in select_prompt)

retry        (added to context_tags when retry_mode=True)
```

Storage convention: `LearnerProfileDocument` keys are bare (`"decomposition"`); `ContextObject` and prompt tags use prefixed keys (`"softskill:decomposition"`).

---

## Comprehension Testing — MCQ + Judge Flow

### MCQ Generation
- Triggered when: `avg_rerank_score > 0.75` AND `retry_mode=False`
- `RAGAgent` calls `present_mcq` tool → LLM outputs 3 questions with 4 options + `correct_answer`
- Full quiz stored in ES `session_interactions` (server-side); client gets questions without correct answers

### POST /quiz/submit

1. Load interaction from `session_interactions` by `interaction_id`
2. Grade each answer **deterministically**: `selected == correct_answer` (no LLM)
3. Wrong answers only: judge LLM generates `retrieval_feedback` (concept that was missing)
4. Update ES with `student_response` + `quiz.status = "submitted"`
5. Mirror score to `session_memory` via `session_store.update_quiz_result()`
6. Overall: PASSED if ≥ 2/3 correct

**On FAILED:**
- Judge reads session memory (`summary_of_past` + `recent_interactions`)
- Judge generates a follow-up question targeting the knowledge gap
- `session_store.set_retry_mode()` → `ContextObject.retry_mode = True`
- Full pipeline re-runs with judge question (`run_judge_followup` — skips session context in `transform_query`)
- Follow-up answer + quiz returned inline in the same `/quiz/submit` response

---

## Ingestion Pipeline

```
POST /ingest (PDF upload)
  → PDFParser         — extracts pages with section/subsection metadata via regex
  → Chunker           — structure-aware: subsection → section → paragraph breaks
                        chunk_size=450 words, overlap=67 words
  → ChunkEnricher     — single LLM call per 8-chunk batch
                        JSON array output: [{"keywords": [...], "concepts": [...]}, ...]
                        keywords: 5-8 per chunk, concepts: 2-4 per chunk
  → Indexer           — upserts enriched chunks into Qdrant capg_knowledge
  → HierarchyBuilder  — one node per unique (level, scope): unit/chapter/section/subsection
                        LLM generates 2-3 sentence summary per node (600-word context)
                        aggregates keywords + concepts from constituent chunks
  → HierarchyStore    — batch-embeds summaries → upserts into Qdrant capg_hierarchy
```

---

## Retrieval Architecture

### Qdrant Collections

| Collection | Contents | Used by |
|-----------|----------|---------|
| `capg_knowledge` | Enriched chunks with full NCERT hierarchy metadata | HybridRetriever, HierarchicalRetriever |
| `capg_hierarchy` | HierarchyNodes (one per level scope) with embedded summaries | HierarchyStore (traversal) |

### NCERT Metadata Schema (per chunk)
```
grade, subject
unit              e.g. "Unit II"
chapter_number    e.g. "6"
chapter_title     e.g. "Life Processes"
section_number    e.g. "6.1"
section_title     e.g. "Nutrition"
subsection_number e.g. "6.1.1"
subsection_title  e.g. "Autotrophic Nutrition"
keywords          list[str]   — LLM-extracted at ingestion
concepts          list[str]   — LLM-extracted at ingestion
```

### Hierarchical Traversal (HierarchyStore scoring)

At each level, top-3 nodes scored by:
```
score = 0.90 × vector_similarity
      + 0.05 × keyword_overlap      # |query_kw ∩ node_kw| / len(query_kw)
      + 0.03 × concept_overlap      # |query_ct ∩ node_ct| / len(query_ct)
      + 0.02 × title_match          # 1.0 if any query token in title, else 0.0
```
Weights are renormalised when `query_keywords` or `query_concepts` is empty.

### Traversal Flow
```
Level 1 — unit:        no parent filter                        → top-3 units
Level 2 — chapter:     unit IN [u1, u2, u3]                   → top-3 chapters
Level 3 — section:     chapter_title IN [c1, c2, c3]          → top-3 sections
Level 4 — subsection:  section_number IN [s1, s2, s3]         → top-3 subsections
Final:    parallel dense search per top-3 deepest scope + BM25 sparse → RRF → top_k chunks
```

### Cross-Encoder Reranking
`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair.

---

## LearnerProfileDocument

ES index: `learner_profiles`. One doc per learner. **Read-only during a session.**

```json
{
  "learner_id": "...",
  "grade": "Class 5",
  "softskills":       { "decomposition": { "score": 0.6, "count": 12 } },
  "learning_style":   { "example_driven": { "score": 0.7, "count": 15 } },
  "technical_skills": { "algebra": { "score": 0.5, "count": 20 } },
  "last_updated_session_id": "..."
}
```

**Weighted average update** (only at session end):
`new_score = (old_score × old_count + contribution) / (old_count + 1)`
Strong → +1.0, Weak → −1.0

---

## Session End

`POST /session/end` → fires two background tasks:

**SessionReviewAgent:**
1. Reads full interaction log from `InteractionStore` (ES `session_interactions`)
2. LLM reasons updated profile fields from quiz scores, wrong answers, topics covered
3. `LearnerProfileStore.update_from_review()` applies weighted average
4. Writes `FeedbackRecord` for governance pipeline
5. Deletes session from `session_memory`

**EvalAgent:**
1. Reads all interactions from `session_interactions` for the session
2. Per-interaction LLM evaluation: correctness, groundedness, answer_relevance, coherence, sufficiency, error_type
3. Writes `EvalResult` to ES `session_evaluations`

---

## ES Index Summary

| Index | Store class | Structure | Lifecycle |
|-------|-------------|-----------|-----------|
| `session_memory` | `SessionMemoryStore` | Recent 5 interactions + PastSummary + ContextObject | Deleted at session end |
| `session_interactions` | `InteractionStore` | Full interaction log with MCQ, correct answers, student responses | Persistent |
| `learner_profiles` | `LearnerProfileStore` | score/count per skill | Persistent |
| `session_evaluations` | `EvalStore` | Per-interaction eval scores (correctness, groundedness, etc.) | Persistent |

Both session stores follow the same pattern: in-memory dict as primary read source, ES upserted on every write.

---

## Teaching Style Catalog

| Style ID | Variant | Targets |
|----------|---------|---------|
| `analogy_first` | standard | Abstract concepts — leads with real-life analogy |
| `step_by_step` | standard | Process/mechanism gaps — numbered breakdown |
| `misconception_correct` | remedial | Wrong beliefs — opens by naming and correcting |
| `example_driven` | standard | Missing concrete anchor — NCERT example before theory |
| `standard_cbse` | standard | Definition/terminology gaps — textbook language |
| `advanced_connections` | advanced | Application failures — cross-chapter links |

---

## Models

| Role | Default | Purpose |
|------|---------|---------|
| Generation | gpt-4.1 | Answer + MCQ tool call |
| Context | gpt-4o-mini | Query transform, prompt selection, chunk enrichment |
| Judge | gpt-4o-mini | MCQ grading feedback + follow-up question generation |

Supports OpenAI and Gemini providers.

---

## Architecture Rules (Do Not Break)

- **Never write to `LearnerProfileDocument` mid-session** — only `SessionReviewAgent` at session end
- **ContextObject is OUTPUT of selector** — never build it before `select_prompt` runs
- **No QuizStore** — ES `session_interactions` is source of truth for correct answers
- **No Redis, no PostgreSQL** — ES handles all session and profile stores
- **MCQ grading is deterministic** — LLM called only on wrong answers for retrieval_feedback
- **SessionReviewAgent reads InteractionStore** — not SessionState
- **No `neutral` topic strength** — only `topic:strong` or `topic:weak`
- **No keyword store** — chunk keywords/concepts extracted at ingestion, used as signals at retrieval
- **Judge generates follow-up question inline** — no `/quiz/regenerate`, no prompt choice MCQ returned to client
- **`query_transform` skips session context on judge follow-up** — `_judge_followup=True` in pipeline state

---

## Key Files

| File | Purpose |
|------|---------|
| `src/orchestrator/pipeline.py` | LangGraph graph — two-pass retrieval, parallel select_prompt |
| `src/retrieval/hierarchical.py` | Top-3-at-every-level hierarchical retrieval with RRF fusion |
| `src/retrieval/hierarchy_store.py` | Qdrant capg_hierarchy — embed summaries, composite scoring |
| `src/ingestion/hierarchy_builder.py` | Builds HierarchyNodes from chunks with LLM summaries |
| `src/ingestion/chunk_enricher.py` | Batch LLM enrichment — keywords + concepts per chunk |
| `src/ingestion/chunker.py` | Structure-aware chunker (450 words, 67-word overlap) |
| `src/prompt_service/selector.py` | LLM context analysis + tag matching |
| `src/storage/session_memory.py` | ES `session_memory` store |
| `src/storage/interaction_store.py` | ES `session_interactions` store |
| `src/storage/learner_profile_store.py` | ES `learner_profiles` with weighted update |
| `src/storage/eval_store.py` | ES `session_evaluations` store |
| `src/agents/judge.py` | Deterministic grading + follow-up question generation |
| `src/agents/eval_agent.py` | Per-interaction quality scoring at session end |
| `src/agents/session_review.py` | End-of-session LLM review → updated learner profile |
| `src/models/hierarchy.py` | `HierarchyNode` model |
| `src/models/query.py` | `ContextObject`, `EnrichedQuery` definitions |
| `src/models/profile_document.py` | `LearnerProfileDocument` with `SkillEntry` |
| `src/tools/present_mcq.py` | Tool schema for MCQ generation |
| `api/routes/quiz.py` | MCQ grading + judge follow-up + retry_mode |
| `api/routes/session.py` | Fire-and-forget session end (review + eval) |
| `api/routes/ingest.py` | PDF ingestion — chunks + hierarchy nodes |

---

## Running

```bash
conda activate pg-carag
uvicorn api.main:app --reload
pytest tests/ -v
python scripts/benchmark.py
```

## Design Decision Rationale

**Why two-pass retrieval?**
Pass 1 (broad hybrid, grade-only) collects domain signals — chunk keywords and concepts — without knowing the subject. These signals drive query rewriting in `transform_query`, which then feeds the precise hierarchical pass 2. This eliminates the need for a separate keyword store.

**Why same LLM call for answer + MCQ?**
The model attends simultaneously to the prompt, retrieved chunks, and its own answer while generating questions. A separate call loses this context and produces weaker questions.

**Why MCQ instead of free-text quiz?**
Deterministic grading (no LLM needed for pass/fail). Reduces latency and cost — LLM only called on wrong answers for targeted feedback.

**Why judge generates follow-up question inline?**
Returning prompt choices to the client adds a round-trip and exposes internal prompt strategy. The judge generates a targeted question server-side and the pipeline re-runs immediately — the client sees a seamless follow-up response.

**Why a dedicated judge model?**
A model evaluating its own output has leniency bias. A separate lighter model (gpt-4o-mini) evaluating gpt-4.1 output is more adversarial.

**Why LLM reasons all ContextObject signals?**
Deterministic score thresholds miss query-specific relevance. LLM can reason "this query requires attention_control even if the score is borderline" by considering the query type and topic together.

**Why hierarchical retrieval with embedded summaries?**
NCERT books have deep structure. BM25/vector on raw chunk text is imprecise at coarse levels (unit, chapter). Embedding LLM-generated summaries per level gives a semantically clean signal for scope resolution before retrieving chunks.

**Why top-3 at every level with MatchAny?**
Top-1 scope resolution is brittle when a query sits near a boundary. Top-3 with MatchAny propagation (3 units → 3 chapters across all 3 units → ...) captures ambiguity without exponential branching.

**Why ES for all storage instead of Redis + PostgreSQL?**
ES already in the stack. Two indices (session_memory + session_interactions) with different structures and lifecycles cover all needs. Eliminates two dependencies without losing capability.
