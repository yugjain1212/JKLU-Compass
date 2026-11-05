# JKLU Compass retrieval diagnosis

**Date:** 5 November 2026  
**Scope:** Read-only diagnosis and retrieval-layer changes. The official benchmark and Chroma records were not changed.

## Summary

The measurable retrieval defect was source selection: the HTML one-chunk-per-URL rule discarded high-ranked chunks that contained distinct, needed information. The rule now permits a second HTML chunk only when it contributes query terms not present in the first, within the existing per-source limit. This fixed three benchmark misses without reducing any category metric.

The diagnostic also previously compared production retrieval at 18 candidates with independently calculated rankings over 60 candidates. It now invokes the shared production retriever at both depths and labels the results separately. For example, the expected B.Tech exchange chunk ranks 18 at production depth and 5 at exploratory depth. Those ranks are not interchangeable.

Aggregate results on the unchanged 24-question benchmark improved from Precision@6 0.1389, Recall@6 0.4583, Hit rate@6 0.5417, MRR 0.4201 to 0.1597, 0.5625, 0.6667, and 0.4451. Admissions and faculty improved; all other category metrics were unchanged. Retrieval still has unresolved misses, including a source-selection miss for the hostel-room question and a candidate-generation miss for the general-disciplines question.

## Confirmed findings and changes

| Finding | Evidence | Status |
|---|---|---|
| HTML source selection could hide useful same-page chunks. | ADM001’s expected fee chunk was dense rank 5, BM25 rank 2, RRF rank 2, but the HTML cap rejected it. The same pattern affected ADM004 and FAC001. | Fixed with a query-distinct second-chunk allowance. The existing URL canonicalization and one-chunk behavior for non-distinct HTML chunks remain covered by tests. |
| Diagnostic depth could produce misleading RRF ranks. | Production uses top_k 6 × multiplier 3 = 18 candidates per method; the older diagnostic used 60. BTECH007’s expected chunk is RRF rank 18 at production depth and rank 5 at exploratory depth. | Fixed by calling the same retriever with separate production-equivalent and exploratory depths. |
| Candidate records need defensive handling. | Duplicate IDs could otherwise contribute more than once, and a BM25 result with no corresponding record could fail during result construction. No duplicate or missing IDs were observed in the current collection. | Hardened and covered with regression tests; valid unique-record RRF scores and distance tie-breaking are preserved. |
| The hostel room-type answer is still filtered out. | CAMP001’s expected chunk is production RRF rank 4, but the final six contain four diverse results followed by two other chunks from the same hostel PDF. The expected chunk receives not_selected_top_k. | Unresolved. The current fill rule favors the rank-3 chunk over the rank-4 room-specific chunk. A more targeted relevance signal needs evaluation before changing selection again. |
| One academic answer is a near-miss in candidate ranking. | BTECH005’s expected lab-list chunk is BM25 rank 3, absent from Dense top 18, and RRF rank 7, just outside top six. The retrieved AI Lab chunk describes only one lab and cannot answer the full list. | Unresolved. Broader candidate recall or a source-aware list-selection rule needs a separate measured change. |
| A general question has a true candidate-generation gap. | GEN003’s expected chunk is absent from Dense and BM25 candidates even in the 60-candidate exploratory run. The source says “interdisciplinary programs in engineering, management and design,” which shares little wording with the query. | Unresolved. No evidence supports changing the embedding model or BM25 implementation based on this one case. |
| Some zero scores reflect incomplete relevance labels. | Retrieved chunks directly answer GEN001, GEN002, ADM002, BTECH007 and BTECH008, but are not among their current expected IDs. | Correction candidates are documented separately; the official benchmark remains unchanged. |

### Candidate and ranking behavior

RRF remains the existing sum of 1/(k + rank), with k=60. Candidate IDs are deduplicated before they contribute, missing BM25 records are skipped and reported, and ties continue to sort by RRF score then dense distance with stable insertion order for exact ties. The current database has 5,716 records and no observed duplicate IDs or missing BM25 records.

The retrieval code keeps the structured B.Tech programme path. The candidate-depth diagnostic does not calculate a second RRF implementation; it calls the shared retrieve method with candidate depth 18 or 60 and records its trace.

## Question-level root causes

These are all benchmark questions with Recall@6 below 1. A zero or partial benchmark score does not always mean the returned evidence is wrong; the proposed label report identifies verified alternatives.

| Question | Current evidence and cause |
|---|---|
| BTECH005 | Expected lab-list chunk: Dense absent, BM25 rank 3, production RRF rank 7, not selected. Returned AI Lab chunk is only partial evidence for a question asking for all specialized labs. |
| BTECH007 | Expected exchange chunk: Dense rank 10, BM25 absent from production top 18, RRF rank 18. A returned IET chunk at rank 4 directly describes B.Tech semester study at IITs, IIITs and other institutions; current labels do not credit it. |
| BTECH008 | Expected brochure chunk: BM25 rank 4, RRF rank 7, not selected. Returned brochure chunk 31dfc3ee9c7351c0 lists the five minors; this is a benchmark-label gap. |
| ADM001 | The first expected fee chunk is now selected at final rank 5, fixing the hit. A second same-page chunk remains blocked at the two-chunk HTML limit; another selected chunk from that fee page provides the exact B.Tech academic and technology amounts. |
| ADM002 | Expected chunk 612dc77b8cba2f47 is not selected, but retrieved chunk d8e5cf9706f0ddcd contains the 2026–27 fee table, including the M.Tech academic fee. The metric undercounts the returned answer. |
| ADM003 | One expected scholarship-table chunk is returned at rank 1. The second expected chunk is dense rank 7, absent from BM25 top 18, RRF rank 13 and excluded by the source limit; the retrieved table supports the answer, while the second chunk appears to continue the table. |
| PLAC002 | Expected chunk 2fb244a13f4c17da is RRF rank 9 and is not selected. The returned placement-support chunk directly describes resume/interview preparation, mentorship, industry insights and networking; the missing label is a verified alternative. |
| CAMP001 | Expected room-types chunk is RRF rank 4 but is not selected. Two other hostel-policy chunks occupy fill slots; the retrieved evidence does not provide the requested double/triple and AC/non-AC room types. |
| CAMP003 | The first expected common-room chunk is returned. The second is Dense rank 9, BM25 rank 11, RRF rank 7 and outside top six. Review whether the first chunk alone contains the complete facility list before changing labels. |
| FAC001 | The expected research chunk was previously blocked by the HTML source cap despite Dense and BM25 rank 2. It is now selected at final rank 5; the benchmark hit is fixed. |
| FAC002 | The first expected biography chunk is selected. The second is BM25 rank 18, RRF rank 29, and not selected by the same-page HTML rule; the selected biography already lists the academic degrees. |
| GEN001 | Expected chunk e057ef9bd1e44ce4 is BM25 rank 5 and production RRF rank 18. Returned chunk a5fe05267560aac1 states the 2011 founding and JK Group legacy, so the zero score is due to missing relevance labeling. |
| GEN002 | Expected prospectus chunk e057ef9bd1e44ce4 is BM25 rank 13 and production RRF rank 22; it contains the “A” grade statement but is not in the final six. Returned prospectus cover chunk 7711bb635b08bd46 also says “A” grade by NAAC. Add it as a proposed alternative label. |
| GEN003 | Expected chunk is absent from both Dense and BM25 at production and exploratory depth. This is the confirmed candidate-generation failure described above. |

## Evaluation

The two evaluations used the same benchmark, top_k=6, collection, embedding model and original relevance labels.

| Metric | Baseline | New |
|---|---:|---:|
| Precision@6 | 0.1389 | 0.1597 |
| Recall@6 | 0.4583 | 0.5625 |
| Hit rate@6 | 0.5417 | 0.6667 |
| MRR | 0.4201 | 0.4451 |

| Category | Precision@6 baseline → new | Recall@6 baseline → new | Hit rate baseline → new | MRR baseline → new |
|---|---:|---:|---:|---:|
| Academics | 0.2083 → 0.2083 | 0.6250 → 0.6250 | 0.6250 → 0.6250 | 0.3229 → 0.3229 |
| Admissions | 0.0417 → 0.1250 | 0.1250 → 0.5000 | 0.2500 → 0.7500 | 0.2500 → 0.3500 |
| Campus | 0.2083 → 0.2083 | 0.6250 → 0.6250 | 0.7500 → 0.7500 | 0.7500 → 0.7500 |
| Faculty & Research | 0.1111 → 0.1667 | 0.5000 → 0.8333 | 0.6667 → 1.0000 | 0.5000 → 0.5667 |
| General | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 0.0000 → 0.0000 |
| Placements | 0.1667 → 0.1667 | 0.7500 → 0.7500 | 1.0000 → 1.0000 | 1.0000 → 1.0000 |

Only three questions improved on the original labels; the other 21 question-level metrics were unchanged:

| Question | Precision@6 | Recall@6 | Hit@6 | MRR |
|---|---:|---:|---:|---:|
| ADM001 | 0.0000 → 0.1667 | 0.0000 → 0.5000 | 0 → 1 | 0.0000 → 0.2000 |
| ADM004 | 0.0000 → 0.1667 | 0.0000 → 1.0000 | 0 → 1 | 0.0000 → 0.2000 |
| FAC001 | 0.0000 → 0.1667 | 0.0000 → 1.0000 | 0 → 1 | 0.0000 → 0.2000 |

The original baseline is data/evaluation/results/retrieval_evaluation_20261105_163743.json. The final evaluation is data/evaluation/results/retrieval_evaluation_20261105_172916.json. The final all-question candidate trace is data/evaluation/diagnostics/retrieval_diagnostic_20261105_172946.json. These JSON outputs are ignored by Git under the existing data/evaluation ignore rule but remain in the workspace.

## Metadata and source freshness

The read-only metadata inventory found:

- 5,716 records with source, file_path, title, source_url, chunk_index, file_name and source_type keys.
- 2,544 title values were empty, null or the literal string “undefined.”
- 5,481 records had heading_level, heading_path and section_title; 235 lacked those structural fields.
- All 5,716 records lacked a category key, so metadata-based category inspection reports them as uncategorized.

The Engineering brochure metadata has a confirmed version mismatch: its title says “Feb2025 V8” while its source URL points to “March2025 V9.” The university’s brochure page lists an Institute of Engineering & Technology brochure, and the linked PDF itself is labeled March 2025 V9. The live B.Tech page currently presents 2026–27 admissions and a current labs/specialisations list, which differs in version and content from the stored 2025–26 brochure. This establishes a freshness risk, not that every stored fact is wrong. [JKLU brochures](https://jklu.edu.in/admissions/brochures), [Engineering brochure PDF](https://jklu.edu.in/uploads/course/pdf/Engineering-JKLU-New-Brochure-March2025-V9%20%282%29.pdf), [current B.Tech page](https://applications.jklu.edu.in/btech/).

Other historical sources in the collection include 2021–22 yearly reports, 2022–23 and 2023–24 academic calendars, 2024–25 calendars and handbooks, 2025–26 calendars, and older programme syllabi. Their presence is confirmed; their current validity was not independently established. A future source refresh needs an approved document/version list and a separate safe ingestion plan.

## Tests and database integrity

- Before this change: 157 tests passed; Ruff passed.
- After this change: `ruff check .` passed, `ruff format --check .` passed, and `pytest -q` passed with 160 tests in 6.60 seconds.
- Three regression tests were added: query-distinct second HTML chunk selection, missing BM25 record handling, and duplicate candidate ID handling. Existing URL-tracking canonicalization and HTML one-chunk tests remain intact.
- Collection: jklu_compass at data/chroma.
- Initial count: 5,716. Final count: 5,716.
- No Chroma mutation or ingestion call occurred. Records and embeddings were read only.

Answer generation was not evaluated. Retrieval metrics do not establish that the local LLM produces correct answers.

## Files changed and reports created

Code and regression coverage:

- `src/jklu_compass/retrieval/retriever.py` — shared trace instrumentation, defensive candidate handling, and query-distinct same-page selection.
- `scripts/diagnose_retrieval_candidates.py` — production-depth and exploratory-depth reports through the shared retriever, with full candidate text, metadata and selection reasons.
- `tests/test_retriever.py` — three regression tests listed above.

Review documents:

- `docs/evaluation/retrieval_root_cause_report_20261105.md`
- `docs/evaluation/proposed_benchmark_corrections_20261105.md`

Generated local JSON evidence, ignored by Git under the existing data/evaluation rule:

- Evaluations: `retrieval_evaluation_20261105_170310.json`, `retrieval_evaluation_20261105_172431.json`, and final `retrieval_evaluation_20261105_172916.json`.
- Diagnostics: `retrieval_diagnostic_20261105_171650.json`, `retrieval_diagnostic_20261105_172257.json`, and final `retrieval_diagnostic_20261105_172946.json`.
