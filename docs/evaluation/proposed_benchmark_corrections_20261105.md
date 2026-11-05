# Proposed JKLU Compass benchmark corrections

**Status:** Proposal only. `data/evaluation/benchmark.json` was not modified.  
**Evidence:** Stored chunk text and metadata from the production-equivalent diagnostic, plus the source URL where source context matters.

## Recommended additions

| Question | Candidate chunk | Classification | Proposed change and reason |
|---|---|---|---|
| JKLU-GEN-001 | `chunk_a5fe05267560aac1` | Fully relevant | Add. It says JKLU was founded in 2011 by the 140-year-old JK Group and continues its legacy, directly answering both parts. |
| JKLU-GEN-002 | `chunk_7711bb635b08bd46` | Fully relevant | Add. The prospectus cover explicitly says “Accredited ‘A’ Grade by NAAC” and identifies the 2025–26 prospectus. |
| JKLU-ADM-001 | `chunk_ae590d7f4807bbee` | Fully relevant | Add. The JKLU fee-structure page chunk gives the B.Tech annual academic fee of ₹2,84,000 and technology fee of ₹10,000. The year heading is in a neighboring chunk on the same page, so retain source URL context when judging it. |
| JKLU-ADM-002 | `chunk_d8e5cf9706f0ddcd` | Fully relevant | Add. The 2026–27 fee table includes the M.Tech annual academic fee of ₹1,80,000. It is already returned at rank 1, while the current label points to another chunk from the same page. |
| JKLU-BTECH-007 | `chunk_f958ac954f8e5b10` | Fully relevant | Add. The IET page says B.Tech includes semester study opportunities at prestigious IITs, IIITs and other leading institutions. It was returned at rank 4. |
| JKLU-BTECH-008 | `chunk_31dfc3ee9c7351c0` | Fully relevant | Add. The March 2025 Engineering brochure chunk lists minors including Design, Mathematics, Entrepreneurship, Banking and Fintech, and Robotics. It was returned at rank 1. |
| JKLU-PLAC-002 | `chunk_2fb244a13f4c17da` | Fully relevant | Add. It describes preparation through industry linkages, continuous skill-building and personalized mentorship, matching the question. It is from the same placements page as the returned support chunk. |

## Reviewed candidates not recommended for addition

| Question | Candidate chunk | Classification | Reason |
|---|---|---|---|
| JKLU-GEN-002 | `chunk_8c6a1b85c065ec12` | Related but insufficient for the source-constrained question | It states that JKLU has an “A” NAAC grade, but comes from a milestones page rather than the prospectus asked about. Prefer the prospectus chunk above. |
| JKLU-ADM-004 | `chunk_7c479f5714efef8a` | Related but insufficient | It contains scholarship table ranges but does not show the requested above-95% row. Adding it would credit incomplete evidence. |
| JKLU-BTECH-005 | `chunk_8cd6a6247982ede3` | Related but insufficient | It describes the Artificial Intelligence Lab only; the question asks for the full specialized-lab list. |

## Applying this proposal

No questions, answers, or labels should be changed until the proposal is reviewed and approved. The reported before/after evaluation uses the original labels. If these changes are approved, preserve that run and report corrected-label metrics separately; do not compare them as if they used the same benchmark.
