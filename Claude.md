================================================================================

CONTEXT-AWARE PROMPT GOVERNANCE SYSTEM FOR RAG-BASED AI APPLICATIONS

END-TO-END AGENTIC FLOW DOCUMENT (FINAL)

================================================================================

VERSION : 2.0 (Final)

DOMAIN : Educational RAG (CBSE/NCERT) \-- Domain-Agnostic by Design

APPROACH : Context Engineering + Multi-Agent PromptOps Governance

SUGGESTED MODIFICATION:\

While the architecture is positioned as domain-agnostic at a high level,
internal design choices (such as retrieval strategies, ontology
structuring, chunking logic, and evaluation signals) naturally evolve to
suit specific domains. True domain agnosticism exists at the governance
layer, but execution layers should be intentionally domain-adaptive.

================================================================================

WHAT THIS SYSTEM DOES (AND WHY IT MATTERS)

================================================================================

Most RAG-based AI systems solve retrieval adequately but treat prompts
as

static, hand-crafted text that developers edit occasionally and
manually.

They also treat the context window as a simple container \-- query goes
in,

answer comes out \-- with no structured intelligence in how that context

is assembled.

This system solves both problems simultaneously.

PROBLEM 1 \-- STATIC PROMPTS

Prompts decay. A prompt that works for Class 3 Maths in April may fail

for Class 3 Science in August. Without a governance layer, nobody knows

until learner outcomes silently degrade.

PROBLEM 2 \-- UNENGINEERED CONTEXT

Sending a raw query to a RAG pipeline ignores everything the system
knows

about the learner \-- their grade, their weak areas, what they
understood

five minutes ago. This produces generic answers when learners need

personalised ones.

This system treats prompts as versioned, governed, self-improving
assets,

and treats context construction as an engineering discipline, not an

afterthought.

The result: a system that does not just answer questions \-- it learns

which way of answering works best for each learner type, continuously

improves its teaching quality without human intervention, and never

ships a prompt change that has not been risk-assessed, experimentally

validated, and canary-tested in production.

This is what Context Engineering + PromptOps looks like in production.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

KEY MARKET DIFFERENTIATORS

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

\[1\] CONTEXT ENGINEERING AT RUNTIME

Every LLM call receives a dynamically constructed context window

assembled from learner profile, session state, enriched query,

and a versioned prompt \-- not a static template.

\[2\] ADAPTIVE TEACHING LOOP

Real-time personalisation per learner. The system retries with a

narrower query and simpler prompt when understanding fails.

\[3\] CLOSED FEEDBACK CYCLE WITH RETRIEVAL QUALITY SIGNAL

Every interaction logs both the Judge verdict AND the retrieval

quality score. Analysis never blames the prompt for a retrieval

problem.

\[4\] RISK-GATED + CANARY-DEPLOYED PROMPT UPDATES

No prompt ships to 100% of users. New versions run on 5% traffic

first. Promotion to default only happens when data confirms the

improvement.

\[5\] MULTI-PROVIDER EXPERIMENTAL VALIDATION

Every prompt candidate is benchmarked across OpenAI and Gemini

before it is eligible for deployment.

\[6\] DOMAIN-AGNOSTIC ARCHITECTURE

Education is the demo domain. The same governance layer applies

to healthcare, legal, and enterprise analytics \-- only the

knowledge base and evaluation dataset change.

SUGGESTED MODIFICATION:\
\
While the governance layer remains domain-agnostic, the domain-specific
knowledge base itself introduces the need for domain-aware adaptations.

================================================================================

WHAT IS CONTEXT ENGINEERING (AND WHERE IT LIVES IN THIS SYSTEM)

================================================================================

Context Engineering is the practice of dynamically and precisely
constructing

the full input that an LLM receives \-- the complete context window \--
rather

than using a static prompt.

In a naive RAG system:

LLM input = \[static system prompt\] + \[raw user query\] + \[retrieved
chunks\]

In this system:

LLM input = \[dynamically selected versioned prompt\]

 + \[enriched and rephrased query\]

 + \[learner grade, comprehension level, weak areas\]

 + \[session progress: what was understood, what was not\]

 + \[top-K reranked retrieved chunks\]

 + \[retry context if applicable\]

Every element of that input is deliberately constructed by the Context &

Rephrase Agent before the LLM is ever called. This is context
engineering.

The prompt governance layer (Agents 4 through 7) operates on top of
this:

it governs what goes INTO the context construction over time, ensuring
the

versioned prompt component keeps improving based on real interaction
evidence.

The two together \-- context engineering at runtime + prompt governance

offline \-- are what give this system its self-improving character.

================================================================================

SYSTEM ARCHITECTURE OVERVIEW

================================================================================

The system runs two parallel, connected loops:

LOOP A \-- RUNTIME (per user interaction, real-time)

\-\-\-\-\-\--

User submits query

\--\> \[PARALLEL\] Context & Rephrase Agent enriches query + builds
context

Vector DB retrieval begins simultaneously (not after)

\--\> Prompt Service selects versioned prompt from context signals

\--\> RAG Agent generates grounded, grade-appropriate answer

\--\> User receives answer

\--\> Judge Agent verifies understanding with adversarial CoT evaluation

\--\> If UNDERSTOOD : log success, proceed

\--\> If NOT UNDERSTOOD : Adaptive Retry Loop (narrow query, remedial
prompt)

SUGGESTED MODIFICATION:\

Instead of positioning query transformation parallel to retrieval, treat
query transformation as a precision amplifier. Specifically,
keyword-enriched query transformation (derived from a structured keyword
pool based on grade, and subject) when combined with hierarchical
retrieval (chapter → section → paragraph) significantly improves
retrieval quality beyond naive parallel execution.

Because this approach already yields high-precision candidates,
re-ranking can be applied over a much smaller set of chunks compared to
naive retrieval. This not only improves final precision but also reduces
latency, since cross-encoders re-ranking is typically powered by
BERT-like models that perform pairwise relevance scoring per
query--chunk combination, with latency scaling roughly linearly with the
number of candidates.

LOOP B \-- OFFLINE IMPROVEMENT (async, event or schedule triggered)

\-\-\-\-\-\--

Accumulated feedback (with retrieval quality signals) from Loop A

\--\> Analysis Agent (pattern mining, separates prompt failures

from retrieval failures using quality signal)

\--\> Risk Agent (predicts regression, complexity, provider risk)

\--\> Experiment Agent (benchmarks candidates across providers +
datasets)

\--\> Suggestion Agent (generates improved, annotated prompt version)

\--\> Prompt Registry (published as candidate \-- canary deployment
begins)

\--\> Prompt Service (routes 5% traffic to new version)

\--\> Analysis Agent (compares 5% cohort vs 95% \-- promotes or reverts)

The two loops connect through:

 - Feedback Store : Loop A writes structured records, Loop B reads them

 - Prompt Registry : Loop B publishes new versions, Loop A selects from
them

This creates a self-improving flywheel. Prompts get better. Learner
outcomes

improve. Cost per successful explanation goes down. No developer
manually

edits a prompt after initial setup.

================================================================================

LOOP A \-- RUNTIME AGENTIC FLOW (STEP BY STEP)

================================================================================

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 1 : LEARNER SUBMITS A QUESTION

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

The learner interacts through a chat interface.

Example used throughout this document:

A Class 3 CBSE student types: \"what is M to cm?\"

The system receives:

 - query_text : \"what is M to cm?\"

 - user_id : (used to fetch learner profile)

 - session_id : (used to fetch current session state)

Nothing else is known yet. The Context & Rephrase Agent immediately
begins

building a full picture of who this learner is and what they need.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 2 : CONTEXT & REPHRASE AGENT + RETRIEVAL (PARALLELISED)

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

ARCHITECTURAL NOTE \-- PARALLELISATION (Latency Fix)

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

In a naive sequential design:

Context Agent finishes (2s) \--\> Retrieval runs (2s) \--\> Generation
(2s)

Total: 6+ seconds. Unacceptable for a real-time learning system.

In this design:

Context Agent and Retrieval run IN PARALLEL.

The Context Agent enriches the query and builds the context object.

Simultaneously, the Vector DB begins retrieval on the raw query.

When the Context Agent finishes, retrieval is refined or re-ranked

using the enriched query. The delay the user experiences is dominated

by the longer of the two tasks, not their sum.

Additionally:

The Context & Rephrase Agent uses a smaller, faster model

(e.g., Gemini Flash or GPT-4o-mini) for context enrichment.

The heavy, more capable model (e.g., GPT-4.1) is reserved for

the final generation step only.

This reduces cost and latency without sacrificing answer quality.

SUGGESTED MODIFICATION:

\- Use the agent to construct a keyword-enriched query using a curated
keyword pool specific to grade and subject\
- Perform hierarchical retrieval (chapter → section → paragraph) to
progressively narrow the search space\
- Ensure high-quality candidate chunks before ranking\
- Apply a cross-encoder re-ranker for final selection

Having retrieval on raw query results in lower precision and introduces
need for higher number of chunks to be sent to cross-encoder re-ranking.
This significantly increases the cost and latency as each query-chunk
passing through cross-encoder is an LLM call (2s)

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

WHAT THE CONTEXT & REPHRASE AGENT DOES:

This is the context engineering layer. It transforms a raw, ambiguous

user query into a precisely constructed context window that carries the

full state of the learner.

2a. FETCH LONG-TERM LEARNER PROFILE (from User Profile DB / PostgreSQL
JSONB)

 - Grade level : Class 3, CBSE

 - Comprehension score : 5 / 10 (medium-low)

 - Learning style : example-driven, prefers concrete objects

 - Historical weak areas: \[\"unit conversion\", \"reading scales\"\]

 - Mastered concepts : \[\"division\", \"multiplication\", \"basic
fractions\"\]

2b. FETCH SHORT-TERM SESSION MEMORY

 - Topics discussed this session: \[\"length introduction\"\]

 - Concepts confirmed understood : \[\"what is a metre\"\]

 - Active knowledge gaps : none yet for this topic

 - Retry count this session : 0

2c. PARSE INTENT FROM QUERY

 - Subject : Mathematics

 - Topic : Measurement \> Length \> Unit Conversion

 - Sub-topic : Metres to Centimetres

 - Type : Conceptual + Procedural

2d. REWRITE QUERY FOR RETRIEVAL

The raw query is short and ambiguous. The agent rewrites it to

maximise retrieval precision against the NCERT knowledge base.

Original : \"what is M to cm?\"

Rewritten : \"Explain conversion of metres to centimetres for a

CBSE Class 3 student. Include a step-by-step

explanation with a real object example such as a ruler

or height. Focus on: 1 metre equals 100 centimetres.\"

This enriched query is used for retrieval and generation.

The learner never sees it. This is the query engineering step.

2e. BUILD CONTEXT OBJECT FOR PROMPT SELECTION

The agent produces a structured context object sent to the Prompt
Service:

learner_grade : \"Class 3\"

comprehension_level : \"low-medium\"

learning_style : \"example-driven\"

weak_areas : \[\"unit conversion\"\]

session_understood : \[\"what is a metre\"\]

current_topic : \"metres to centimetres\"

query_type : \"conceptual_and_procedural\"

retry_mode : false

OUTPUTS (produced in parallel with retrieval):

 - Enriched query \--\> RAG Agent retrieval + generation

 - Context object \--\> Prompt Service

 - Session memory updated

SUGGESTED MODIFICATION:\

Clearly separate responsibilities:\
- 2d (Query Transformation Agent): Focused purely on retrieval
optimization through structured rephrasing and keyword injection\
- 2e (Context Object Builder): Responsible only for constructing the
context object used for prompt selection and downstream orchestration\
This separation prevents overloading a single agent with dual
responsibilities and improves modularity and parallelization and also in
adaptive retry loops.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

WHAT RETRIEVAL DOES IN PARALLEL:

While the Context Agent is running, initial retrieval begins on the raw

or partially-enriched query:

 - Dense retrieval : semantic vector search against NCERT/CBSE PDFs

(Vector DB: Qdrant or Weaviate for BM25 + vector mix)

 - Sparse retrieval : BM25 keyword search for subject-specific terms

(\"centimetre\", \"metre stick\", \"Class 3 measurement\")

When the Context Agent finishes, the enriched query is used to:

 - Re-run or refine retrieval with hierarchical narrowing:

Level 1: Chapter (\"Measurement\")

Level 2: Section (\"Length\")

Level 3: Passage (\"Converting m to cm\")

 - Apply cross-encoder reranking to re-score top candidates

 - Select top-K passages (e.g., top 3) for generation

SUGGESTED MODIFICATION:\

Enhanced Retrieval Flow Example:\
User Query: \'what is M to cm?\'\
→ Query Transformation: Adds keywords like \[\'metre\', \'centimetre\',
\'conversion\'\]\
→ Hierarchical Retrieval:\
Chapter: Measurement\
Section: Length\
Paragraph: Conversion rules\
→ Candidate chunks selected\
→ Cross-encoder re-ranking applied\
→ Top-K high-quality chunks passed to generation\
This ensures retrieval quality is high before generation begins.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 3 : PROMPT SERVICE \-- SELECT THE RIGHT VERSIONED PROMPT

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

The Prompt Service is not an agent. It is a governed registry and a

context-driven selector. It holds all prompt versions with metadata

and picks the best match for the current learner and situation.

It receives the context object and applies selection rules:

 - retry_mode = false + comprehension_level = \"low-medium\"

\--\> Select: prompt_v3_grade3_standard (current active version)

 - retry_mode = true

\--\> Select: prompt_v3_grade3_remedial

 - comprehension_level = \"high\"

\--\> Select: prompt_v3_grade3_advanced

CANARY DEPLOYMENT AWARENESS:

If a new candidate version (e.g., v4) is in canary testing, the Prompt

Service routes 5% of matching traffic to v4 and 95% to v3.

This split is transparent to the learner and all downstream agents.

The version tag on every feedback record tracks which cohort each

interaction belongs to.

OUTPUT:

 - Selected versioned prompt \--\> RAG Agent

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 4 : RAG AGENT \-- GENERATE GROUNDED ANSWER

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

The RAG Agent receives the enriched query, selected prompt, and top-K

retrieved passages. It has no awareness of the learner\'s profile \--

that context has already been encoded into the enriched query and
prompt.

4a. GENERATION

 - Combine: \[selected prompt\] + \[enriched query\] + \[retrieved
passages\]

 - This is the fully engineered context window.

 - Send to the heavy LLM (e.g., GPT-4.1).

 - The LLM generates an explanation that:

a. Uses grade-appropriate language (guided by the prompt)

b. Follows the tone and structure in the selected prompt

(warm teacher mode, step-by-step, example-first)

c. Grounds every claim in the retrieved NCERT content

Example output to the learner:

\"Great question! In India we measure length in metres (m) and

centimetres (cm). Here is the most important rule:

1 metre = 100 centimetres.

Think of your school ruler \-- it is 30 cm long. So 1 metre

is like 3 rulers placed end to end!

Let\'s try: if your room is 3 metres long, how many cm is that?

3 x 100 = 300 cm. You just multiply metres by 100.\"

4b. COMPUTE AND ATTACH RETRIEVAL QUALITY SCORE

Before passing the response forward, the RAG Agent scores the quality

of the retrieval step:

retrieval_quality_score : average cosine similarity of top-K chunks

to the enriched query (e.g., 0.91)

retrieval_quality_flag : GOOD (\> 0.75) / MARGINAL (0.5-0.75)

/ POOR (\< 0.5)

retrieved_sources : \[\"NCERT Class 3 Math Ch.6 pg.72\"\]

This score is attached to the response metadata and will later be

written to the Feedback Store alongside the Judge verdict.

PURPOSE: If a learner fails to understand, the Analysis Agent can

distinguish \"the prompt was wrong\" from \"the retrieved content was

irrelevant.\" Without this signal, prompt changes are made to fix

problems that actually live in the embedding or chunking layer.

4c. PACKAGE FULL RESPONSE METADATA

 - prompt_version : \"v3_grade3_standard\"

 - retrieved_sources : \[list\]

 - retrieval_quality_score: 0.91

 - retrieval_quality_flag : \"GOOD\"

 - generation_model : \"gpt-4.1\"

 - prompt_cohort : \"v3_control\" or \"v4_canary\"

 - timestamp : ISO timestamp

OUTPUT:

 - Generated answer \--\> shown to learner

 - Response metadata \--\> held for Feedback Service (written after
verdict)

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 5 : LEARNER READS THE ANSWER

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

The learner reads the explanation. The Judge Agent immediately generates

a comprehension check and presents it. The learner must respond before

the session can proceed.

SUGGESTED MODIFICATION:\

Instead of using a separate judge model for generating comprehension
questions, leverage the same RAG generation model. Since it already
contains full reasoning context and retrieved knowledge, it can generate
more relevant and aligned follow-up questions keeping the adversarial
judge prompt idea. This approach improves quality, coherence, and
latency simultaneously.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

STEP 6 : JUDGE AGENT \-- VERIFY UNDERSTANDING WITH ADVERSARIAL COT

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

ARCHITECTURAL NOTE \-- JUDGE BIAS FIX

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

If the same LLM that generated the answer also evaluates whether the

learner understood it, it is effectively grading its own homework.

Studies on LLM-as-judge show this produces leniency bias \-- the judge

is less likely to mark a response as FAILED when it closely resembles

the explanation it just gave.

This system addresses the bias in two ways:

FIX 1 \-- CHAIN OF THOUGHT BEFORE VERDICT

The Judge Agent is not allowed to output a verdict directly.

It must first reason through its evaluation in a structured CoT format:

Step 1: What was the core concept the learner needed to demonstrate?

Step 2: What did the learner actually say or answer?

Step 3: Does the response contain the key concept? What is missing?

Step 4: Is this a genuine understanding signal or a surface guess?

Step 5: Verdict \-- UNDERSTOOD or NOT UNDERSTOOD, and why.

This structured reasoning makes the evaluation process transparent

and auditable, and reduces lenient snap judgements.

FIX 2 \-- ADVERSARIAL JUDGE PROMPT

The Judge Agent operates under a strict adversarial system prompt:

\"You are a rigorous academic evaluator. Your job is to find gaps

in understanding, not to confirm that the learner did well.

Assume partial understanding is NOT sufficient. Only mark UNDERSTOOD

if the learner demonstrates the core concept without prompting.\"

This counteracts the leniency bias that arises when the judge model

has seen the explanation.

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

6a. GENERATE COMPREHENSION CHECK

The Judge Agent generates a targeted follow-up question based on

the content of the explanation just given.

Three check modes:

MODE A \-- Quiz (primary, most reliable)

\"If 1 metre = 100 cm, how many centimetres is 2 metres?\"

MODE B \-- Explanation Prompt (for conceptual topics)

\"Can you tell me in your own words: what does 1 metre equal?\"

MODE C \-- Confidence Check (lightweight, for simple confirmations)

\"Did you understand the ruler example?\"

The check is generated DYNAMICALLY from the content just taught.

It is not pulled from a static question bank \-- this prevents the

learner from gaming repeated questions across sessions.

Mode escalation rule:

If the learner has said \"yes\" to MODE C twice but then failed a

MODE A quiz, the system escalates ALL future checks to MODE A

for this learner in this session.

6b. EVALUATE THE LEARNER\'S RESPONSE WITH COT

The Judge runs its chain of thought before issuing a verdict:

\[CoT Step 1\] Core concept required:

Learner must demonstrate that 1 metre = 100 centimetres

and apply multiplication to a new value.

\[CoT Step 2\] What the learner said:

\"200 cm\" (in response to \"how many cm is 2 metres?\")

\[CoT Step 3\] Concept check:

Response contains correct numeric answer (200).

Implicit multiplication of 2 x 100 was applied correctly.

Unit label \"cm\" present.

\[CoT Step 4\] Surface guess check:

A surface guess would be \"100\" (just repeating the definition).

\"200\" requires actual application. Not a surface guess.

\[CoT Step 5\] Verdict: UNDERSTOOD.

Rationale: Learner applied the conversion correctly to a new value.

6c. ROUTING DECISION

\-- IF UNDERSTOOD \--

a. Mark concept LEARNED in Session Memory

b. Update User Profile DB \-- concept added to mastered list

c. Write full feedback record to Feedback Store:

{prompt_version, verdict: \"UNDERSTOOD\", topic, grade,

retrieval_quality_score, retrieval_quality_flag,

retry_count: 0, user_signal: \"quiz_correct\",

judge_cot_summary, prompt_cohort, timestamp, \...}

d. Confirm to learner: \"Excellent! You\'ve got it exactly right.\"

e. Session proceeds to next topic.

\-- IF NOT UNDERSTOOD \--

a. Mark concept as LEARNING GAP in Session Memory

b. Write full feedback record to Feedback Store:

{prompt_version, verdict: \"NOT_UNDERSTOOD\", topic, grade,

retrieval_quality_score, retrieval_quality_flag,

retry_count: 1, user_signal: \"quiz_incorrect\",

judge_cot_summary, prompt_cohort, timestamp, \...}

c. Signal Context Agent to activate ADAPTIVE RETRY LOOP.

\-- ADAPTIVE RETRY LOOP \--

Context Agent is called again with retry_mode = true:

 - Identifies the specific sub-concept that failed

(e.g., learner answered \"100\" \-- did not apply multiplication)

 - Narrows query to only that sub-concept:

\"Explain only what multiplying metres by 100 means, using

the example of counting 100 small steps to make one big step,

for a Class 3 student.\"

 - Sets retry_mode = true

SUGGESTED MODIFICATION:\

Retry triggering should be based on routing signals (UNDERSTOOD / NOT
UNDERSTOOD), intead of on retry_mode flags.\
- retry_mode and related properties in the context object are to be used
for prompt selection and improvement\
\
Additionally:\
- In adaptive retry loops, 2e (context object construction) becomes
unnecessary\
- 2d (query transformation) can run in parallel with prompt
selection/improvement workflows of prompt governance.\
- Allow prompt governance agents to have controlled edit access to a
shared context object\

Prompt Service selects the remedial prompt version:

 - Simpler vocabulary, single concept focus, analogy-based

RAG Agent regenerates a focused explanation.

Judge Agent issues a new comprehension check after the retry.

\-- IF MAX RETRIES REACHED (configurable, e.g., 3 attempts) \--

a. Flag concept as PERSISTENT WEAK AREA in User Profile DB

b. Switch to analogy-based or story-based explanation strategy

c. Inform learner: \"This is a tricky concept. Let\'s come back to

it. Try asking your teacher or a parent to walk through it.\"

d. Move the session forward \-- do not loop indefinitely.

OUTPUT:

 - Routing signal : UNDERSTOOD \| RETRY \| MAX_RETRY_ESCALATE

 - Feedback record : written to Feedback Store (with retrieval quality)

 - Memory updates : Session Memory + User Profile DB

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

RUNTIME LOOP \-- WHAT ONE COMPLETE INTERACTION PRODUCES

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

After one interaction, the system has:

 - Constructed a fully engineered context window for the LLM call

 - Personalised the query and prompt to this specific learner

 - Retrieved grounded content from NCERT knowledge base

 - Generated a grade-appropriate, example-driven explanation

 - Verified genuine understanding with adversarial CoT evaluation

 - Logged a structured feedback record that includes:

\* prompt_version, topic, grade, learner profile signals

\* retrieval_quality_score and flag (GOOD / MARGINAL / POOR)

\* judge verdict with CoT rationale

\* canary cohort tag (control or canary)

\* retry count, user signals

 - Updated both short-term session memory and long-term learner profile

Every one of these records accumulates in the Feedback Store.

When a threshold is crossed, the offline improvement loop activates.

================================================================================

LOOP B \-- OFFLINE PROMPT IMPROVEMENT FLOW (STEP BY STEP)

================================================================================

Runs asynchronously. Does NOT interrupt the user experience.

Activation triggers:

 - A new prompt version is submitted by a developer

 - Scheduled batch job (e.g., nightly)

 - Feedback volume threshold crossed (e.g., 500 new records since last
run)

 - Canary cohort has accumulated enough data for a comparison

================================================================================

TECHNOLOGY RECOMMENDATIONS

================================================================================

Component Recommended Technology Reason

\-\-\-\-\-\-\-\-- \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--
\-\-\-\-\--

Vector DB Qdrant or Weaviate Native BM25 + vector hybrid

search without extra tooling.

Avoids building a separate

sparse retrieval pipeline.

User Profile DB PostgreSQL with JSONB Flexible learner attribute

storage. New fields added

without schema migration.

Relational integrity for

user and session joins.

Session Memory Redis Sub-millisecond reads for

session state. TTL-based

auto-expiry at session end.

Context/Rephrase Model Gemini Flash / GPT-4o-mini Small, fast model for

context enrichment step.

Saves the heavy model call

for generation only.

Generation Model GPT-4.1 / Gemini Pro Full capability for the

final answer generation.

Experiment Framework Promptfoo or LangSmith Automates the evaluation

matrix across prompts x

questions x providers.

Built-in scoring pipelines,

comparison dashboards, and

regression alerts.

Orchestration LangGraph or custom Agent loop coordination,

lightweight orchestrator state handoff between

Context Agent, RAG Agent,

and Judge Agent.

================================================================================

IMPLEMENTATION STATUS

================================================================================

STATUS: Repository scaffolded with dual-architecture support (2026-04-07)

REPOSITORY STRUCTURE:

config/              Settings and architecture feature flags
src/models/          Shared Pydantic data models (query, learner, retrieval, response, feedback, prompt)
src/llm/             LLM provider abstraction (OpenAI, Gemini) with factory pattern
src/retrieval/       Vector store (Qdrant), BM25 sparse, hybrid fusion, hierarchical (Arch B), cross-encoder reranker
src/agents/arch_a/   Combined Context & Rephrase Agent, Separate Judge Agent
src/agents/arch_b/   Query Transform Agent, Context Object Builder, Unified Judge Agent
src/agents/shared/   RAG Generation Agent (shared)
src/orchestrator/    Pipeline orchestration (arch_a_pipeline, arch_b_pipeline, factory, retry)
src/prompt_service/  Prompt Registry, context-driven Selector, Canary Router
src/governance/      Loop B stubs (Analysis, Risk, Experiment, Suggestion agents + pipeline)
src/storage/         Database engine, User Profile store, Session Memory, Feedback Store
src/ingestion/       PDF parser, Chunker, Indexer
api/                 FastAPI app with chat, ingest, and prompt management endpoints
tests/               Test suite for agents, retrieval, orchestrator, API
scripts/             Benchmark (A vs B comparison) and seed prompts scripts

ARCHITECTURE SWITCHING:

Set CAPG_ARCHITECTURE=A or CAPG_ARCHITECTURE=B in .env to switch between:
- Architecture A (Parallel): Combined context+rephrase agent parallel with raw query retrieval
- Architecture B (Sequential Precision): Separate query transform → hierarchical retrieval

The API layer is architecture-agnostic. The orchestrator factory returns the correct pipeline.

GOVERNANCE (LOOP B) STATUS:

High-level stubs with clear interfaces. Each governance agent has:
- Input/output Pydantic models
- Stub implementation with logging
- Docstrings describing full intended behavior

Pipeline wiring: Feedback Store → Analysis → Risk → Experiment → Suggestion → Prompt Registry

NEXT STEPS:

1. Implement full governance agents (Loop B) with LLM-powered prompt improvement
2. Build evaluation dataset for benchmark script
3. Add PostgreSQL and Redis backends (currently in-memory for development)
4. Implement canary cohort comparison in the Analysis Agent
5. Add LangSmith integration for experiment tracking
6. Production deployment configuration (Docker, CI/CD)
