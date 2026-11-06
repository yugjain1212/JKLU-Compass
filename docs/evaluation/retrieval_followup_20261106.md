# JKLU Compass retrieval and live RAG follow-up

**Date:** 6 November 2026  
**Scope:** Targeted retrieval and answer-grounding improvements. The official benchmark and Chroma collection were read only.

## Changes

- B.Tech programme-list intent now requires a programme, course, degree, major, or specialisation request. Semester exchange and mobility questions stay on the normal retrieval and generation path.
- Single-degree questions focus the context on that degree. Fee questions also prefer the requested fee fields and matching academic year, and mixed-degree tables are reduced to the requested degree row.
- Explicit annual degree-fee questions use deterministic rendering when one cited passage supplies the requested amount and year. A request for one fee returns only that field; a general annual-fee request can return both academic and technology fees. Yearless answers disclose the source year, and multi-year questions remain in generation.
- Retrieval gives a narrowly targeted priority to exact degree/year fee evidence, comprehensive lab lists, and hostel passages that state occupancy and AC options. The priority does not apply to a question about equipment in one named lab. Other queries keep the existing source-diversity selection behavior.
- The generation prompt now reinforces degree scope, exact-year support, concise answers, and omission of unasked scholarship or fee details.

## Retrieval evaluation

The same 24-question benchmark and top-6 setting were used before and after. `data/evaluation/benchmark.json` was not edited.

| Metric | Before this follow-up | After |
|---|---:|---:|
| Precision@6 | 0.1597 | 0.1806 |
| Recall@6 | 0.5625 | 0.6875 |
| Hit rate@6 | 0.6667 | 0.7917 |
| Mean reciprocal rank | 0.4451 | 0.6035 |

Four benchmark questions improved:

| Question | Change |
|---|---|
| BTECH005 — specialised laboratories | No relevant result in top 6 → relevant list at rank 1 |
| ADM001 — B.Tech 2026–27 fees | Relevant result at rank 5 → rank 1 |
| ADM002 — M.Tech 2026–27 academic fee | No relevant result in top 6 → relevant result at rank 1 |
| CAMP001 — hostel room types | No relevant result in top 6 → relevant result at rank 1 |

Academics, admissions, and campus category metrics improved. Faculty/research and placements were unchanged. General remains at zero because of the existing benchmark labels and candidate-generation gap; no category regressed. Average distinct sources in the top six increased slightly from 4.167 to 4.208 per query.

Five questions still have zero hits under the unchanged relevance labels: BTECH007, BTECH008, GEN001, GEN002, and GEN003. The first four have retrieved, answer-bearing alternatives documented in `proposed_benchmark_corrections_20261105.md`; they are label gaps rather than proof that the answer is unavailable. GEN003 remains a candidate-generation failure: its source wording is absent from both dense and BM25 candidate sets, including the earlier exploratory-depth run. The official labels were left unchanged pending review.

The final retrieval report is `data/evaluation/results/retrieval_evaluation_20261106_034918.json`.

## Live RAG checks

Live queries ran through the actual retriever, Chroma collection, and local Ollama model.

- “What is the annual B.Tech academic fee for 2026–27?” returned **₹2,84,000** with a citation to fee-table chunk `chunk_d8e5cf9706f0ddcd`.
- “What are the annual fees for B.Tech for academic year 2026–27?” returned **₹2,84,000 academic fee** and **₹10,000 technology fee**, citing the same year-specific table.
- The semester-study query stayed on exchange opportunities instead of entering the B.Tech programme-list route.
- The specialised-lab query returned the nine labs listed in the engineering brochure, with inline citations.
- The hostel-room query returned double/triple occupancy with AC/non-AC options, with an inline citation to the Hostel Life Policy.

The model did not include inline citation labels in every live response: the semester-study answer omitted them while the lab and hostel answers included them. The API still returned the source list. If inline citations are mandatory for every answer, citation compliance needs its own generation check before release. Live generation also took a noticeable wait on the local Ollama model; latency should be measured separately.

Chroma contained 5,716 records before and after live checks and evaluation. No ingestion, embedding, deletion, reset, or other collection write was performed.

## Validation

- Full test suite: **175 passed**.
- `ruff check .`: passed.
- `ruff format --check .`: passed (59 files).
- `git diff --check`: passed.
- Evaluation collection count: 5,716 before and after.

Retrieval evaluation does not by itself prove answer quality. The live checks above cover several high-impact cases; source freshness and the GEN003 candidate-generation gap remain open.
