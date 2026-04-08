# PG-CARAG — System Design Document

**Context-Aware Prompt Governance System for RAG-based AI Applications**
**Domain:** CBSE/NCERT Educational AI | **Version:** 3.0

---

## What This System Does

Most RAG systems treat prompts as static text and the context window as a dump of retrieved chunks. This system solves both problems:

**Problem 1 — Static prompts decay.** A prompt that works for Grade 5 Maths in one term may silently fail for Grade 10 Science in another. Without governance, nobody knows until outcomes degrade.

**Problem 2 — Unengineered context.** Sending a raw query ignores everything the system already knows about the learner — their grade, weak areas, what they understood five minutes ago.

**Solution:** Prompts are versioned, governed, self-improving assets. Context construction is an engineering discipline, not an afterthought. The system doesn't just answer questions — it learns which way of answering works best for each learner type and continuously improves without manual intervention.

---

## System Overview

Two parallel, connected loops:

```
LOOP A — Runtime (real-time, per interaction)
  Student query → pipeline → answer + quiz → judge evaluation → retry or done

LOOP B — Governance (offline, async)
  Feedback records → analysis → risk assessment → prompt candidate → canary test → promote/reject
```

---

## Loop A — Runtime Pipeline

### LangGraph State Graph

```
fetch_profile
    ↓
lookup_keywords          ← keyword store (grade+subject → curated NCERT terms)
    ↓
transform_query          ← context LLM (gpt-4o-mini)
    ↓
retrieve                 ← hierarchical retriever → Qdrant + BM25 (hybrid, RRF)
    ↓
build_context            ← deterministic (profile + session → context object)
    ↓
rerank                   ← cross-encoder (ms-marco-MiniLM)
    ↓
select_prompt            ← prompt selector (or override from quiz/regenerate)
    ↓
generate                 ← generation LLM (gpt-4.1) — answer + quiz in ONE call
    ↓
END
```

### Node Responsibilities

**fetch_profile**
Loads learner profile (grade, comprehension score, weak areas, learning style) and session state (retry count, concepts understood this session) from storage.

**lookup_keywords**
Fetches curated NCERT domain keywords for the learner's grade from the keyword store (keyed by `{grade}_{subject}`). Subject is not yet known at this stage, so a broad grade-level lookup is performed. The transform agent filters to subject-relevant terms.

**transform_query**
Query Transformation Agent (context LLM). Detects subject, topic, chapter hint, section hint. Selects relevant domain keywords + adds NCERT-specific terms. Rewrites query to be retrieval-friendly. Outputs a structured `EnrichedQuery`.

**retrieve**
Hierarchical retriever progressively narrows search scope:
- Always filters by grade + subject (mandatory)
- Narrows to chapter if topic identified
- Narrows to section/subsection if sub_topic identified
- Hybrid retrieval: Qdrant dense + BM25 sparse fused via Reciprocal Rank Fusion (RRF)

**build_context**
Deterministic (no LLM). Computes `comprehension_level` from profile score, assembles `ContextObject` used by prompt selector.

**rerank**
Cross-encoder reranks retrieved candidates. Returns top-K chunks with rerank scores. Average score used downstream to decide quiz generation.

**select_prompt**
Picks a versioned teaching-style prompt from the registry based on context object (comprehension level, retry mode). Canary router may route 5% traffic to a candidate prompt. If `_prompt_override_id` is set (from `/quiz/regenerate`), bypasses selector entirely.

**generate**
Generation LLM produces the answer. When `avg_rerank_score > 0.75 AND not retry_mode`, quiz generation is appended to the **same LLM call** via `QUIZ_SUFFIX`. The model simultaneously attends to the prompt template, retrieved chunks, and its own answer — producing tighter, misconception-targeted questions than any separate call.

Quiz output format (parsed from same response):
```
QUIZ_Q1: <question>  QUIZ_A1: <expected answer>
QUIZ_Q2: <question>  QUIZ_A2: <expected answer>
QUIZ_Q3: <question>  QUIZ_A3: <expected answer>
```

Expected answers are stored server-side (quiz store). Client receives only questions.

---

## Comprehension Testing — Quiz + Judge Flow

### Quiz Generation Conditions
- Retrieval quality GOOD (avg rerank score > 0.75)
- Not in retry mode

When either condition fails, quiz is skipped and `quiz_form.skipped = True` with a reason.

### POST /quiz/submit

1. Load expected answers from quiz store (keyed by `quiz_id = session_id`)
2. For each submitted answer, run `JudgeAgent.evaluate()`:
   - Dedicated judge model (gpt-4o-mini), separate from generation
   - Adversarial CoT: 5-step reasoning process
   - Outputs: VERDICT (UNDERSTOOD/NOT_UNDERSTOOD), RATIONALE, RETRIEVAL_FEEDBACK
   - `RETRIEVAL_FEEDBACK`: a retrieval query fragment describing what concept was absent from the explanation (e.g. "role of semipermeable membrane in osmosis")
3. Delete quiz from store (prevents replay)
4. Overall: PASSED if ≥ 2/3 questions UNDERSTOOD

**On FAILED:**
- Aggregate all non-empty `retrieval_feedback` into one `retrieval_hint`
- `PromptSelector.select_candidates_from_feedback()` scores all 6 teaching styles against the retrieval hint using an LLM
- Returns 3 best-fit prompt choices as MCQ (one per variant, diversity enforced)

### POST /quiz/regenerate

1. Learner picks a teaching style from the MCQ
2. `retrieval_hint` appended to original query: `"[Focus also on: {hint}]"`
3. Full pipeline re-runs with enriched query + chosen prompt (bypasses selector)
4. Returns fresh answer + new quiz form

---

## Teaching Style Catalog (Fixed, Seeded)

Six styles × 2 grades (5, 10) = 12 prompt versions. Selector scores all against judge feedback and picks top 3.

| Style ID | Variant Bucket | Targets |
|----------|---------------|---------|
| `analogy_first` | standard | Abstract concepts — leads with real-life analogy before formal definition |
| `step_by_step` | standard | Process/mechanism gaps — numbered breakdown, "why this happens" notes |
| `misconception_correct` | remedial | Wrong beliefs — opens by naming and correcting the misconception |
| `example_driven` | standard | Missing concrete anchor — NCERT example before theory |
| `standard_cbse` | standard | Definition/terminology gaps — textbook language, exam-aligned structure |
| `advanced_connections` | advanced | Application failures — cross-chapter links, embedded application questions |

---

## Keyword Store + Extraction

Keywords are extracted from NCERT PDFs using the context LLM and stored as:
```
key:   "{grade}_{subject}"     e.g. "10_Science"
value: [list of curated terms] e.g. ["osmosis", "xylem", "photosynthesis", ...]
```

The `lookup_keywords` pipeline node fetches these before query transformation. The transform agent uses them to select precise NCERT terms rather than guessing.

Keyword extraction script: `python scripts/extract_keywords.py --pdf <path> --grade 10 --subject Science`

---

## Retrieval Architecture

### Single Qdrant Collection
All grades and subjects in one collection (`capg_knowledge`). Grade + subject are mandatory filters on every search. Chapter/section/subsection progressively narrow scope (Arch B hierarchical approach).

### NCERT Metadata Schema (per chunk)
```
grade, subject
unit            e.g. "Unit II", "Part I"
chapter_number  e.g. "6"
chapter_title   e.g. "Life Processes"
section_number  e.g. "6.1"
section_title   e.g. "Nutrition"
subsection_number  e.g. "6.1.1"
subsection_title   e.g. "Autotrophic Nutrition"
```

Section and subsection metadata are auto-detected during PDF parsing via regex on decimal-numbered headings (e.g. "6.1 Nutrition", "6.1.1 Autotrophic Nutrition") and carried forward across page boundaries.

### Hybrid Retrieval
Dense (Qdrant cosine) + Sparse (BM25) fused via Reciprocal Rank Fusion. Ensures both semantic and keyword recall.

### Cross-Encoder Reranking
`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair. Average score used to gate quiz generation and report retrieval quality (GOOD / MARGINAL / POOR).

---

## Prompt Governance — Loop B

Offline loop triggered by accumulated feedback records. High-level flow:

```
gather_feedback → analyze → assess_risk → suggest_candidate
    → run_experiment → deploy_candidate OR reject_candidate
    → next_version (loop back or end)
```

Each feedback record includes:
- Judge verdict (UNDERSTOOD / NOT_UNDERSTOOD)
- Retrieval quality score + flag
- Prompt version + cohort (control / canary)
- Retry count, latency, architecture

**Canary deployment:** New prompt candidates are tested on 5% of traffic. The canary router splits based on `canary_traffic_percent` setting. Promotion only happens when experiment results confirm improvement.

---

## Models

| LLM Role | Default | Purpose |
|----------|---------|---------|
| Generation | gpt-4.1 | Answer + quiz generation |
| Context | gpt-4o-mini | Query transformation, prompt candidate scoring |
| Judge | gpt-4o-mini | Adversarial CoT answer evaluation |

All three roles are independently configurable via `.env`. Supports OpenAI and Gemini providers.

---

## Data Stores

| Store | Backend | Purpose |
|-------|---------|---------|
| Vector store | Qdrant | Chunk embeddings + metadata |
| Quiz store | In-memory (→ Redis) | Expected answers, deleted after evaluation |
| Keyword store | In-memory (→ Redis) | Grade+subject keyword pools |
| Feedback store | In-memory (→ PostgreSQL) | Interaction outcome records |
| Session memory | In-memory (→ Redis) | Per-session retry count, understood concepts |
| User profiles | In-memory (→ PostgreSQL) | Long-term learner data |
| Prompt registry | In-memory (→ PostgreSQL) | Versioned prompts + candidates |

---

## Key Design Decisions

**Why same LLM call for answer + quiz?**
The model attends simultaneously to the prompt, retrieved chunks, and its own answer while generating questions. This produces tighter, misconception-targeted questions. A separate call loses this context.

**Why a dedicated judge model?**
A model that evaluates its own output has leniency bias. A separate, lighter model (gpt-4o-mini) evaluating gpt-4.1 output is more adversarial and catches gaps the generation model would overlook.

**Why feedback-aware prompt selection instead of fixed variants?**
The judge's retrieval_feedback is semantic natural language describing what concept was missing. LLM-scored selection against teaching style descriptions + tags captures this semantic meaning better than rule-based variant mapping.

**Why hierarchical retrieval?**
NCERT books have deep structure (unit → chapter → section → subsection). Progressively narrowing the search scope before cross-encoder reranking means the reranker sees only relevant-scope candidates — higher precision and lower reranking latency than flat retrieval.

**Why keyword injection before query transformation?**
NCERT uses precise domain terminology (e.g. "stomata", "xylem", "osmotic pressure"). Without domain keywords, the transform agent may use colloquial synonyms that miss exact chunk matches. Pre-injecting curated terms ensures the rewritten query uses textbook vocabulary.
