# Product Requirements Document (PRD)

## JKLU Compass --- Local University Knowledge Assistant

**Document status:** Draft v1.0\
**Product type:** Local-first AI knowledge assistant\
**Primary audience:** JK Lakshmipat University (JKLU) students and
project evaluators\
**Project context:** Deep Learning academic assignment\
**Owner:** Project team / student developer\
**Last updated:** 19 October 2026

------------------------------------------------------------------------

## 1. Executive Summary

JKLU Compass is a local-first AI assistant designed to answer questions
about JK Lakshmipat University using a curated collection of university
information. It combines a small pretrained open-weight language model,
Retrieval-Augmented Generation (RAG), a local vector index, source
citations, and an optional parameter-efficient fine-tuning (PEFT) stage.

The product is not intended to train a foundation model from scratch. It
will begin with an existing model such as Qwen and use RAG as the
primary mechanism for providing university-specific facts. Fine-tuning
with supervised fine-tuning (SFT) and LoRA/QLoRA will be an experimental
enhancement to improve response behavior, formatting, context use, and
handling of missing information.

The system should run its inference pipeline locally on a user's
computer after required models and dependencies have been downloaded.
Google Colab may be used for model fine-tuning; this training workflow
is separate from local inference.

The central principle is:

> **RAG supplies the university knowledge; fine-tuning, if evaluation
> supports it, improves how the model responds using that knowledge.**

The project will demonstrate a functioning application and an
evidence-based comparison of: 1. Base LLM without RAG. 2. Base LLM with
RAG. 3. Fine-tuned LLM without RAG. 4. Fine-tuned LLM with RAG.

The project succeeds when it can answer a defined set of university
questions with useful, grounded responses, verifiable source links,
measured retrieval and answer quality, and acceptable local performance.

------------------------------------------------------------------------

## 2. Problem Statement

University information is distributed across web pages and may be
difficult for students to locate, interpret, or compare. Conventional
search requires users to open multiple pages and manually assemble the
answer. A generic chatbot may not know current university-specific
details and can produce unsupported claims.

JKLU Compass will provide a conversational interface over a curated
university knowledge base. It will retrieve relevant source passages,
use a local LLM to compose an answer, and expose the source pages used.
When the retrieved information is insufficient, the assistant should
state that limitation rather than inventing an answer.

### 2.1 Problem to solve

Students need a convenient way to ask natural-language questions about
publicly available university information and receive: - A concise,
understandable response. - Relevant information drawn from university
sources. - Links to sources that can be checked. - A clear indication
when the knowledge base does not contain enough information.

### 2.2 Why a local-first design?

A local inference design can reduce dependence on paid inference APIs,
support offline use after setup, and keep ordinary queries on the user's
device. This privacy benefit applies only when the complete inference
path---including the LLM, embedding model, vector database, and
application---is configured locally. Initial downloads and optional
cloud fine-tuning still require network access.

------------------------------------------------------------------------

## 3. Product Vision and Goals

### 3.1 Vision

Build a practical, explainable, and locally runnable university
knowledge assistant that demonstrates how RAG and parameter-efficient
fine-tuning can complement one another.

### 3.2 Primary goals

1.  Turn the existing JKLU web crawl into a clean, traceable knowledge
    base.
2.  Retrieve relevant university passages for a user's question.
3.  Generate answers using a local open-weight LLM.
4.  Provide source citations linked to the original pages.
5.  Acknowledge when available evidence is insufficient.
6.  Measure retrieval quality, answer quality, latency, and resource
    use.
7.  Experimentally determine whether fine-tuning improves the RAG
    system.
8.  Deliver a reproducible codebase, setup guide, evaluation results,
    and academic report.

### 3.3 Non-goals

The first release will not: - Train a foundation model from random
initialization. - Guarantee that every answer is correct. - Replace
official university communication, student services, or academic
advising. - Access private student records, internal systems, or
authenticated university portals. - Perform transactions such as
submitting applications or changing student records. - Claim to provide
live information unless the source has been refreshed and its freshness
is clear. - Require fine-tuning for the first working release. - Depend
on a cloud LLM API for ordinary local inference. - Treat fine-tuning as
the primary storage mechanism for changing facts.

------------------------------------------------------------------------

## 4. Target Users and Use Cases

### 4.1 Primary users

**JKLU students** - Find university information. - Understand academic
programs and published policies. - Explore campus facilities and
publicly listed research information. - Locate relevant official
webpages.

**Prospective students** - Find published admissions information. -
Explore program descriptions. - Identify official pages for further
details.

**Project evaluator or developer** - Inspect retrieved chunks and
citations. - Run evaluation questions. - Compare base and fine-tuned
model configurations. - Reproduce the system setup and experiments.

### 4.2 Initial supported use cases

1.  Ask a question about publicly available JKLU information.
2.  Summarize a program, department, faculty profile, campus feature, or
    published policy from indexed sources.
3.  Compare two programs when the relevant information is available.
4.  Ask multi-part questions that require more than one retrieved
    passage.
5.  Find the source page supporting an answer.
6.  Ask about information not present in the dataset and receive an
    explicit limitation response.
7.  Evaluate system behavior using a fixed question set.

### 4.3 Out-of-scope use cases

-   Personalized academic, legal, medical, or financial decisions.
-   Official confirmation of eligibility or admission outcomes.
-   Claims about unpublished or private university information.
-   Retrieval of confidential student or employee data.
-   Autonomous actions on university systems.

------------------------------------------------------------------------

## 5. Existing Data and Assumptions

The project already has a scraped and processed JKLU dataset.

### 5.1 Known dataset baseline

-   Approximately 150 non-empty Markdown documents.
-   Approximately 3,970,302 characters.
-   Approximately 392,842 words.
-   Processed JSONL path: `data/processed/jklu_documents.jsonl`.
-   The recorded cleaning pass found no duplicates, empty records,
    processing errors, or missing URLs.
-   Categories include:
    -   `general`: 49
    -   `faculty`: 32
    -   `academics`: 30
    -   `campus`: 15
    -   `about_jklu`: 11
    -   `policies`: 6
    -   `admissions`: 3
    -   `placements`: 3
    -   `research`: 1

These figures describe the current dataset snapshot and should be
rechecked by the ingestion and validation scripts. A clean processing
report does not prove that all source content is current, complete,
authoritative, or free from near-duplicates.

### 5.2 Data requirements

Every indexed document should retain, where available: - Stable document
ID. - Title. - Original source URL. - Category. - Cleaned text. - Crawl
timestamp. - Source update date, if explicitly available. - Optional
content hash and ingestion version.

The crawl timestamp must not be presented as the source's publication or
last-updated date.

### 5.3 Data quality requirements

The ingestion pipeline must: - Remove repeated navigation and irrelevant
boilerplate without damaging meaningful content. - Normalize whitespace
and malformed text. - Preserve headings, lists, tables, and section
structure where useful. - Detect exact duplicates and report potential
near-duplicates. - Validate that source URLs are present and
well-formed. - Flag empty, unusually short, malformed, or potentially
stale records. - Keep a machine-readable data quality report. - Avoid
indexing private or unauthorized information.

The system should prioritize official university sources. If conflicting
source content is found, it should preserve source provenance and avoid
silently treating one statement as definitively correct.

------------------------------------------------------------------------

## 6. Product Scope and Release Plan

### Release 1 --- Local RAG baseline (required)

-   Clean and validate existing documents.
-   Chunk documents with source metadata.
-   Generate local embeddings.
-   Store and search chunks in ChromaDB.
-   Run a small local LLM.
-   Build a context-grounded prompt.
-   Generate answers with source links.
-   Handle insufficient evidence.
-   Provide a simple chat UI.
-   Establish a baseline evaluation.

### Release 2 --- Fine-tuning experiment (required for the full research comparison if resources allow)

-   Create a reviewed SFT dataset.
-   Split data into training, validation, and held-out test sets.
-   Fine-tune a compatible base model with LoRA or QLoRA in Google
    Colab.
-   Save adapter and training configuration.
-   Export or load the model in a format supported by the local
    inference stack.
-   Compare fine-tuned and base models using the same test set.

### Release 3 --- Retrieval improvements (optional, based on evaluation)

-   Hybrid keyword and vector retrieval.
-   Reranking.
-   Category filters.
-   Better chunking strategies.
-   Query rewriting, if a measured need exists.
-   Retrieval and response caching.
-   Expanded source freshness checks.

### Release 4 --- UI polish (optional)

-   Improved responsive interface.
-   Conversation history stored locally.
-   Source previews.
-   User feedback controls.
-   Evaluation/debug mode.

The team must not delay a working RAG baseline in order to build
optional features.

------------------------------------------------------------------------

## 7. Functional Requirements

Priority labels: **P0** = required for the core project; **P1** =
important enhancement; **P2** = optional.

### FR-1: Document ingestion (P0)

The system shall read processed university documents from JSONL.

**Acceptance criteria** - Each valid record is parsed into a normalized
internal document representation. - Invalid records are logged with a
useful error message. - Source URL and document ID are preserved. - An
ingestion summary reports records read, accepted, skipped, and rejected.

### FR-2: Cleaning and deduplication (P0)

The system shall clean text and identify duplicate or unusable records.

**Acceptance criteria** - Cleaning rules are documented and
repeatable. - Exact duplicates can be detected by normalized content
hash. - Potential near-duplicates are reported separately rather than
automatically deleted without review. - Headings and meaningful lists
remain usable.

### FR-3: Chunking (P0)

The system shall split documents into semantically useful chunks while
preserving source metadata.

**Initial experimental defaults** - Target size: approximately 400
tokens. - Overlap: approximately 60 tokens. - Retrieve: initially 5
chunks per question.

These are starting values, not fixed truths. Chunk sizes of
approximately 250, 400, and 600 tokens and multiple retrieval counts
should be evaluated.

**Acceptance criteria** - Each chunk has a stable ID, document ID,
source URL, title, category, and text. - Chunk boundaries avoid
splitting meaningful sections when practical. - Chunk counts and
token-length distributions are reported.

### FR-4: Embedding generation (P0)

The system shall use a local embedding model to encode document chunks
and user questions.

Candidate: `nomic-embed-text` through Ollama or a suitable English
sentence-transformers model.

**Acceptance criteria** - The same embedding model and compatible
preprocessing are used for indexing and query-time retrieval. - The
embedding model name and version are recorded in configuration. -
Changing the embedding model triggers a re-indexing workflow. - Ordinary
query embedding does not require a cloud API.

### FR-5: Vector index (P0)

The system shall store embeddings, chunk text, and metadata in ChromaDB.

**Acceptance criteria** - The index can be built from the processed
dataset. - The index persists between application runs. - Re-indexing is
repeatable and does not create uncontrolled duplicate entries. - A
document update can be reflected by re-indexing the affected content or
rebuilding the index.

### FR-6: Retrieval (P0)

The system shall retrieve relevant chunks for a user query.

**Acceptance criteria** - The top retrieved chunks and their metadata
can be inspected in development mode. - The number of retrieved chunks
is configurable. - Retrieval results are ranked by the selected search
method. - The system logs retrieval timing without storing unnecessary
personal query data. - The application can indicate when retrieval
confidence or relevance is insufficient, using a threshold validated
during evaluation rather than an arbitrary universal cutoff.

### FR-7: Local answer generation (P0)

The system shall use a locally running LLM to answer a question using
the question and retrieved context.

**Initial model candidate:** Qwen3 1.7B, subject to verifying model
availability, license, hardware behavior, and inference compatibility. A
compatible small Llama model may be used as a comparison candidate.

**Acceptance criteria** - The model can run locally through a supported
inference engine such as Ollama. - The prompt separates instructions,
retrieved context, and the user's question. - The system does not
require a cloud LLM API for ordinary inference. - Model name,
quantization, and generation settings are configurable. - The model's
license and redistribution conditions are checked before packaging.

### FR-8: Grounding and abstention (P0)

The assistant shall prioritize supplied evidence and acknowledge missing
or conflicting information.

**Acceptance criteria** - The prompt instructs the model not to invent
university facts. - When context is insufficient, the assistant gives a
clear limitation response. - Unsupported questions are included in
evaluation. - The application does not claim that a refusal or grounding
instruction guarantees correctness. - Important policy, admissions, and
placement claims should be checked against their source.

### FR-9: Source citations (P0)

The assistant shall provide source links based on retrieved document
metadata.

**Acceptance criteria** - Each displayed source link originates from the
indexed document metadata. - The UI shows a useful title and link where
available. - The application does not rely on the LLM to invent URLs. -
Citations are checked for relevance and support during evaluation. - If
no usable source was retrieved, the UI must not fabricate one.

### FR-10: Chat interface (P0)

The first UI may use Streamlit for implementation speed.

**Acceptance criteria** - A user can submit a question and see the
response. - The response and source links are displayed clearly. - A
loading indicator appears while retrieval and generation run. - Errors
are presented in understandable language. - The UI identifies the active
local model. - The UI makes clear that answers are AI-generated and
should be verified for consequential decisions.

### FR-11: Fine-tuning dataset preparation (P1)

The system shall support creating a curated SFT dataset containing
context-grounded examples.

Example categories: - Factual question answering using supplied
context. - Multi-passage answering. - Summarization. - Program
comparison. - Structured response formatting. - Handling missing or
ambiguous information.

**Acceptance criteria** - Training examples follow the selected model's
chat template. - Incorrect, unsupported, duplicated, or ambiguous
examples are filtered or reviewed. - Source provenance is retained for
audit. - Training, validation, and test sets are separated. - Related
examples from the same source page are grouped appropriately to reduce
leakage between splits. - Generated examples are not automatically
assumed to be correct.

### FR-12: PEFT fine-tuning (P1)

The project shall support an SFT experiment using LoRA or QLoRA, where
supported by the selected model and available GPU.

Candidate stack: Transformers, Datasets, TRL, PEFT, and compatible
quantization libraries. Training may be performed in Google Colab.

**Acceptance criteria** - Base model, tokenizer, chat template, training
configuration, and dataset version are recorded. - Training and
validation loss are logged. - Checkpoints or adapters are saved. - The
training run can be reproduced from documented settings to the extent
allowed by the environment. - The resulting adapter is evaluated on
held-out questions before acceptance. - The project does not claim that
fine-tuning improved performance without supporting measurements.

### FR-13: Model export and local deployment (P1)

The system shall provide a documented path to deploy the selected
fine-tuned model locally.

**Acceptance criteria** - The adapter is loaded with its compatible base
model or merged where supported. - Export and quantization use a format
compatible with the selected inference engine. - A clean local inference
test is completed. - Model quality is evaluated again after
quantization. - The deployment instructions identify required memory and
storage as measured, not assumed.

### FR-14: Evaluation runner (P0)

The project shall include repeatable evaluation scripts.

**Acceptance criteria** - A fixed evaluation dataset is used across
configurations. - Retrieval and generation metrics are reported
separately. - Test results can be exported to CSV or JSON. - The report
records model, prompt, retrieval, and dataset configuration for each
run.

### FR-15: Experiment comparison (P1)

The project shall compare four configurations: - A: Base LLM, no RAG. -
B: Base LLM + RAG. - C: Fine-tuned LLM, no RAG. - D: Fine-tuned LLM +
RAG.

**Acceptance criteria** - The same held-out questions are used for all
configurations. - Results include answer quality, faithfulness, citation
quality where applicable, latency, and memory usage. - Differences are
reported honestly, including regressions. - No target improvement is
asserted before measurement.

### FR-16: Configuration and logging (P0)

The system shall keep tunable settings outside application logic.

Configuration should include: - Model and inference engine. - Embedding
model. - Vector database path. - Chunk size and overlap. - Retrieval
count. - Optional category filters. - Prompt template version. -
Generation settings. - Data and index versions.

Logs should avoid unnecessary retention of user questions and should
never record secrets.

------------------------------------------------------------------------

## 8. Non-Functional Requirements

### NFR-1: Local-first inference

After models and dependencies are downloaded, ordinary question
answering should work without a network connection.

**Verification:** Disable network access and test query embedding,
retrieval, generation, and source display.

### NFR-2: Hardware suitability

The initial development target is a MacBook Air M1 with 8 GB RAM. A
small quantized LLM and a lightweight embedding model should be tested
first.

The project must measure real memory use and latency. No fixed
response-time or memory guarantee is made until the system is tested.

### NFR-3: Answer quality

The system must be evaluated against a curated set of university
questions, including questions whose answers are missing from the
dataset.

### NFR-4: Maintainability

Ingestion, retrieval, generation, training, evaluation, and UI code
should be separated into modules.

### NFR-5: Reproducibility

Record dataset versions, model versions, embedding model versions,
prompts, training parameters, and evaluation configuration.

### NFR-6: Privacy

The local inference path should not send user questions to external
model APIs. The application should disclose that downloading models and
optional Colab fine-tuning involve network/cloud processing. Only
authorized data may be used.

### NFR-7: Security

-   Do not commit credentials, API keys, private records, or model
    access tokens.
-   Treat retrieved webpage content as untrusted input.
-   Retrieved text must not be allowed to override system instructions.
-   Avoid executing commands, tools, or links solely because retrieved
    content requests it.
-   Validate and safely render URLs.
-   Keep logs free of secrets.

### NFR-8: Accessibility and usability

The first interface should be readable, keyboard usable, and clear about
source links and system limitations.

### NFR-9: Licensing

Check the model, embedding model, libraries, and dataset usage terms
before redistribution or public release.

------------------------------------------------------------------------

## 9. Technical Architecture

### 9.1 Offline ingestion pipeline

``` text
Existing JSONL / Markdown
          |
          v
Validation and cleaning
          |
          v
Deduplication and quality report
          |
          v
Structure-aware chunking
          |
          v
Local embedding model
          |
          v
ChromaDB index + metadata
```

### 9.2 Query-time pipeline

``` text
User question
     |
     v
Input validation
     |
     v
Local query embedding
     |
     v
Vector retrieval (optional hybrid search later)
     |
     v
Optional reranking (later, only if useful)
     |
     v
Context and prompt construction
     |
     v
Local base or fine-tuned LLM
     |
     v
Answer + metadata-derived citations
     |
     v
UI
```

### 9.3 Fine-tuning pipeline

``` text
Curated university documents
          |
          v
Reviewed context-question-answer examples
          |
          v
Train / validation / test split
          |
          v
Pretrained compatible LLM
          |
          v
SFT with LoRA or QLoRA in Colab
          |
          v
Adapter + training metadata
          |
          v
Evaluation on held-out set
          |
          v
Export / quantize if accepted
          |
          v
Local inference integration
```

### 9.4 Recommended initial technology stack

  -----------------------------------------------------------------------
  Layer                   Initial choice          Notes
  ----------------------- ----------------------- -----------------------
  Language                Python                  Data, training,
                                                  evaluation, backend

  Base LLM                Qwen3 1.7B candidate    Confirm model support,
                                                  license, and hardware
                                                  behavior

  Alternative model       Compatible small Llama  Use only if comparison
                          model                   or compatibility
                                                  warrants it

  Local inference         Ollama or another       Validate fine-tuned
                          compatible runtime      model import/export
                                                  early

  Embeddings              `nomic-embed-text` or   Keep query and document
                          English                 embedding setup
                          sentence-transformers   consistent
                          model                   

  Vector store            ChromaDB                Local persistent index

  Backend                 FastAPI                 Introduce if the UI and
                                                  RAG logic need a clean
                                                  API boundary

  Initial UI              Streamlit               Fastest path to a
                                                  demonstrable local
                                                  interface

  Fine-tuning             Transformers + TRL +    Colab GPU when
                          PEFT                    available

  Quantization            Compatible quantized    Evaluate quality after
                          inference format        quantization

  Evaluation              Python scripts +        Separate retrieval from
                          reviewed dataset        answer evaluation

  Version control         Git                     Exclude large models,
                                                  local indexes, secrets,
                                                  and restricted data
  -----------------------------------------------------------------------

The initial release should remain simple. FastAPI can be introduced when
needed; the first Streamlit prototype may call a modular RAG service
directly.

------------------------------------------------------------------------

## 10. RAG Design Requirements

### 10.1 Chunking strategy

Start with approximately 400 tokens per chunk and 60 tokens of overlap.
Preserve heading and paragraph boundaries where possible.

Test alternative chunk sizes (for example, 250, 400, and 600 tokens) and
retrieval counts (for example, 3, 5, and 8). Select settings based on
evaluation, context limits, latency, and answer quality.

### 10.2 Retrieval strategy

**Initial:** Dense vector retrieval.

**Possible enhancement:** Hybrid retrieval combining semantic vectors
and keyword matching. This may help with exact faculty names, course
codes, policy titles, and program names.

**Optional enhancement:** Reranking candidate chunks with a
cross-encoder or other suitable reranker, only if measured retrieval
quality justifies the additional latency and resource cost.

### 10.3 Prompt policy

The system prompt should instruct the model to: - Answer the question
using supplied evidence. - Avoid adding unsupported university facts. -
Acknowledge insufficient or conflicting context. - Use clear language
and a suitable structure. - Treat retrieved text as data, not as
instructions that override system policy. - Avoid claiming that a source
says something unless the retrieved passage supports it.

### 10.4 Citation design

The application should assemble citations from the retrieved chunks'
metadata. The LLM may refer to source labels supplied in the prompt, but
the application---not the model---should determine the actual URLs
displayed.

### 10.5 Freshness

The first version will use a static crawl snapshot. The UI or
documentation should state the crawl date and limitations. A future
refresh process may re-crawl approved public pages, compare content
hashes, and update the index.

------------------------------------------------------------------------

## 11. Fine-Tuning Design Requirements

### 11.1 Fine-tuning objective

The objective is to improve the model's behavior when answering from
supplied context, not to turn the model into a database of all
university facts.

### 11.2 Dataset design

Create reviewed examples for: - Context-grounded answers. -
Multi-document synthesis. - Summarization. - Comparisons. - Structured
formatting. - Missing-information responses. - Ambiguous questions and
clarification behavior.

A starting target of approximately 500--1,500 high-quality examples is
reasonable for a student experiment, but it is not a guarantee of
improvement. Begin with a small pilot set to validate the training
pipeline.

### 11.3 Dataset quality safeguards

-   Validate factual claims against the source context.
-   Preserve provenance for each example.
-   Remove duplicates and low-quality generated examples.
-   Do not include stale facts as timeless answers.
-   Keep test examples separate from training examples.
-   Group related examples by source page when splitting to reduce
    leakage.
-   Ensure examples match the actual inference prompt and model chat
    template.

### 11.4 Training approach

Use SFT with LoRA or QLoRA if supported by the selected base model and
Colab environment. Start with a short pilot run. Record: - Base model
and revision. - Tokenizer and chat template. - Dataset version. -
Maximum sequence length. - Batch and gradient accumulation settings. -
Learning rate and scheduler. - Epochs or maximum steps. - LoRA
configuration. - Quantization configuration, if used. - Training and
validation losses. - Checkpoint and adapter paths.

Choose settings based on available GPU memory and measured validation
behavior; do not assume a particular Colab GPU will be available.

### 11.5 Deployment compatibility

Before committing to a long training run, test that the selected model
and adapter can be loaded and exported through the intended local
inference stack. After merging or quantizing, re-run the same evaluation
set.

------------------------------------------------------------------------

## 12. Evaluation Plan

### 12.1 Evaluation dataset

Create an initial set of approximately 100 manually reviewed questions,
expanding it as time allows. Include: - Direct factual questions. -
Questions requiring multiple passages. - Summarization. - Program
comparisons. - Faculty and research questions. - Policy and admissions
questions. - Ambiguous questions. - Questions intentionally not
answerable from the corpus.

For each question, record: - Question ID. - Question text. - Expected
answer or key facts. - Relevant source document IDs and URLs. - Whether
the question is answerable from the indexed corpus. - Notes for scoring.

The evaluation dataset must not be used to generate fine-tuning
examples.

### 12.2 Retrieval metrics

**Hit Rate@k:** Fraction of questions for which at least one relevant
passage appears in the top k results.

**Recall@k:** Fraction of known relevant passages recovered in the top k
results.

**Mean Reciprocal Rank (MRR):** Rewards placing the first relevant
passage near the top of the ranked results.

Use human-verified relevance labels for a meaningful evaluation.

### 12.3 Generation metrics

Measure: - **Correctness:** Does the answer match the reference facts? -
**Faithfulness:** Are claims supported by retrieved context? -
**Citation accuracy:** Do displayed sources support the claims? -
**Completeness:** Does the response cover the key points required by the
question? - **Abstention quality:** Does the system avoid unsupported
answers when evidence is missing? - **Readability:** Is the response
understandable and appropriately structured?

Use manual scoring for a representative subset. Automated or model-based
evaluation may assist but should not be treated as unquestionable ground
truth.

### 12.4 Efficiency metrics

Record: - End-to-end response latency. - Retrieval latency. - Generation
latency. - Peak memory usage where measurable. - Model and index disk
size. - Tokens generated per second where the runtime exposes this
metric. - Whether the full query pipeline works with network access
disabled.

### 12.5 Required experiment matrix

  --------------------------------------------------------------------------
  ID           Model                     RAG       Fine-tuned Purpose
  ------------ ------------ ---------------- ---------------- --------------
  A            Base LLM                   No               No General
                                                              baseline

  B            Base LLM                  Yes               No Measure RAG
                                                              contribution

  C            Fine-tuned                 No              Yes Inspect
               LLM                                            behavior
                                                              change without
                                                              retrieval

  D            Fine-tuned                Yes              Yes Measure
               LLM                                            combined
                                                              system
  --------------------------------------------------------------------------

Use the same held-out questions and consistent scoring criteria across
experiments. Record prompt versions and configuration. Report
improvements and regressions honestly.

### 12.6 Proposed acceptance targets

These are project targets to validate, not claims about current
performance: - All displayed citations originate from indexed source
metadata. - The application can complete a query with the network
disabled after setup. - The evaluation runner produces reproducible
metrics for all implemented configurations. - At least 100 reviewed
evaluation questions are available for the final comparison, subject to
project time. - Every known evaluation case has source relevance labels
or a documented reason why it does not. - Fine-tuning is included in the
final architecture only if it is technically deployable and its measured
value is discussed, even if results are neutral or negative. - Retrieval
and answer-generation failures can be diagnosed separately.

Do not set a hard answer-accuracy target until the baseline is measured
and the scoring method is defined.

------------------------------------------------------------------------

## 13. User Experience Requirements

### 13.1 Main chat screen

The screen should include: - Product name: JKLU Compass. - A concise
description of the assistant's scope. - A question input and submit
action. - A loading state. - An answer area. - Source links and document
titles. - A clear message when information is insufficient. - Active
model and local/offline status where feasible.

### 13.2 Source inspection

Users should be able to open the original source page. A
developer/evaluation mode may also show the retrieved passage text and
relevance scores, but this diagnostic detail need not be exposed in the
default student interface.

### 13.3 Errors

The UI should handle: - Model runtime not running. - Model not
downloaded or not found. - Vector index missing or incompatible. - No
relevant results. - Invalid input. - Generation timeout or runtime
error. - Source URL unavailable.

Error messages should explain the next useful action without exposing
stack traces or secrets.

### 13.4 Product disclaimer

The interface should state that responses are AI-generated and may be
incomplete. For admissions, policies, deadlines, or other consequential
details, users should verify the official source.

------------------------------------------------------------------------

## 14. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  Scraped content is      Incorrect answer        Track crawl date, show
  outdated                                        sources, refresh
                                                  approved pages,
                                                  evaluate
                                                  freshness-sensitive
                                                  questions

  Retrieval returns       Unsupported or          Tune chunking and
  irrelevant chunks       incomplete answer       retrieval; inspect
                                                  failures; consider
                                                  hybrid search or
                                                  reranking

  LLM invents facts       Loss of trust           Context-grounded
                                                  prompts, abstention
                                                  examples, citations,
                                                  held-out evaluation

  Fine-tuning overfits    Poor generalization     Validation set,
                                                  source-grouped split,
                                                  checkpoint comparison,
                                                  early stopping

  Training examples       Model learns incorrect  Review and filter
  contain errors          behavior                examples; preserve
                                                  source provenance

  Model is too slow on 8  Poor user experience    Use a small quantized
  GB laptop                                       model, reduce context
                                                  length, measure memory
                                                  and latency

  Fine-tuned adapter      Deployment blocked      Test export
  cannot be exported                              compatibility before a
                                                  long run

  Quantization reduces    Lower answer quality    Re-evaluate after
  quality                                         quantization

  Data leakage between    Inflated evaluation     Split by source page or
  splits                  results                 related document group

  Website prompt          Unsafe or irrelevant    Treat retrieved text as
  injection in retrieved  behavior                untrusted data; do not
  content                                         allow it to override
                                                  system instructions

  Private data is         Privacy risk            Restrict scope to
  accidentally included                           authorized public data;
                                                  audit sources; exclude
                                                  secrets and personal
                                                  records

  Model or library        Redistribution          Review licenses and
  licensing issue         limitations             document constraints

  Colab runtime           Training delay          Use a small pilot, save
  availability varies                             checkpoints, record
                                                  configuration, plan for
                                                  variable GPU resources
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Privacy, Safety, and Data Governance

1.  Use only data the project is authorized to collect and process.
2.  Keep the first release limited to public university information.
3.  Do not ingest student records, credentials, private staff data, or
    authenticated portal content.
4.  Do not commit secrets or personal data to Git.
5.  Avoid retaining chat histories unless necessary and clearly
    disclosed.
6.  Make the local inference boundary explicit.
7.  Document that Colab fine-tuning processes training data in a cloud
    environment.
8.  Treat retrieved web content as untrusted input.
9.  Provide source links so users can independently verify claims.
10. Clearly identify the product as an unofficial academic project
    unless formal university authorization is obtained.

------------------------------------------------------------------------

## 16. Repository and File Structure

``` text
jklu-compass/
├── data/
│   ├── raw/
│   │   └── markdown/
│   ├── processed/
│   │   ├── jklu_documents.jsonl
│   │   └── chunks.jsonl
│   ├── finetuning/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── evaluation/
│       ├── questions.jsonl
│       └── reference_answers.jsonl
├── notebooks/
│   ├── data_exploration.ipynb
│   └── fine_tuning_colab.ipynb
├── src/
│   ├── ingestion/
│   │   ├── clean.py
│   │   ├── deduplicate.py
│   │   ├── chunk.py
│   │   └── build_index.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   ├── prompt.py
│   │   └── generation.py
│   ├── training/
│   │   ├── prepare_dataset.py
│   │   ├── train_lora.py
│   │   └── export_model.py
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py
│   │   └── evaluate_answers.py
│   └── api/
│       └── main.py
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── experiments/
│   └── results.csv
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Create modules incrementally. The first release does not need every file
to exist. Large model weights, generated indexes, secrets, and
restricted data should generally be excluded from version control.

------------------------------------------------------------------------

## 17. Suggested Milestones

### Milestone 1 --- Dataset validation

-   Inspect the existing JSONL dataset.
-   Validate metadata and source URLs.
-   Produce a data quality report.
-   Confirm a clean sample manually.

**Deliverable:** Validated dataset and documented data schema.

### Milestone 2 --- Chunking and indexing

-   Implement structure-aware chunking.
-   Generate local embeddings.
-   Create a persistent ChromaDB index.
-   Verify that representative queries retrieve relevant passages.

**Deliverable:** Reproducible local knowledge base.

### Milestone 3 --- Baseline RAG

-   Run the base local LLM.
-   Implement prompt construction.
-   Generate answers with source metadata.
-   Implement insufficient-context behavior.

**Deliverable:** Working local question-answering pipeline.

### Milestone 4 --- Basic UI and baseline evaluation

-   Add a simple chat interface.
-   Create and review an initial evaluation set.
-   Measure retrieval and answer quality.
-   Save baseline metrics.

**Deliverable:** Demonstrable baseline application and evaluation
report.

### Milestone 5 --- Fine-tuning data and pilot

-   Generate and review SFT examples.
-   Split data without source leakage.
-   Run a short LoRA/QLoRA experiment in Colab.
-   Check validation behavior.

**Deliverable:** Adapter, training logs, and documented configuration.

### Milestone 6 --- Local fine-tuned deployment

-   Validate adapter compatibility.
-   Export or load the fine-tuned model.
-   Quantize if needed.
-   Test offline inference.

**Deliverable:** Deployable fine-tuned model configuration.

### Milestone 7 --- Comparative evaluation

-   Run experiments A--D.
-   Compare quality, grounding, citations, latency, and memory.
-   Document limitations and unexpected results.

**Deliverable:** Results table, charts if useful, and conclusions
supported by evidence.

### Milestone 8 --- Final submission

-   Complete README and setup instructions.
-   Add architecture diagram and reproducibility details.
-   Prepare the report and viva explanation.
-   Demonstrate the end-to-end system.

**Deliverable:** Source code, runnable demo, model/data instructions,
evaluation results, and academic report.

------------------------------------------------------------------------

## 18. Definition of Done

The project is ready for final demonstration when:

-   [ ] Existing JKLU data is validated and its provenance is preserved.
-   [ ] Chunking and local embedding generation work reproducibly.
-   [ ] ChromaDB retrieves relevant source passages.
-   [ ] A small local LLM generates answers from retrieved context.
-   [ ] Source links are generated from document metadata, not invented
    by the LLM.
-   [ ] The system acknowledges insufficient evidence in tested cases.
-   [ ] A basic chat UI is usable.
-   [ ] The core inference pipeline works offline after setup.
-   [ ] A reviewed evaluation dataset and baseline results exist.
-   [ ] Fine-tuning data is separated from the held-out evaluation set.
-   [ ] The LoRA/QLoRA experiment is documented and evaluated if
    implemented.
-   [ ] The project compares base and RAG configurations, and fine-tuned
    configurations where available.
-   [ ] Model licenses, data scope, limitations, and setup steps are
    documented.
-   [ ] The final report describes measured results rather than assumed
    improvements.

------------------------------------------------------------------------

## 19. Final Product Recommendation

The project should be developed in this order:

1.  **Build a reliable RAG baseline.**
2.  **Measure its failures and performance.**
3.  **Fine-tune a compatible small model to improve context-grounded
    response behavior.**
4.  **Integrate and evaluate the fine-tuned model with the same RAG
    pipeline.**
5.  **Keep fine-tuning only as a justified improvement, and document
    results even if it does not help.**

For the initial implementation, Qwen3 1.7B is a candidate model, with a
local embedding model, ChromaDB, and a lightweight local interface.
Google Colab is the proposed environment for the optional/experimental
training stage, while local inference targets the MacBook Air M1 with 8
GB RAM.

The project's technical contribution is not simply the use of RAG and
fine-tuning together. It is the construction of a traceable local
university assistant and the measurement of what each component
contributes.

**Success means a useful, source-grounded, locally runnable assistant
with a reproducible evaluation---not merely a model that has been
fine-tuned.**
