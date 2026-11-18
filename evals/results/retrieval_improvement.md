### What was improved, and how (retrieval)

**Problem found by the baseline:** equal-weight RRF of vector + full-text (hybrid) scored *below* vector
alone (nDCG@10 0.668 vs 0.687): the body full-text retriever is noisy on long natural-language queries
and dragged the fusion down.

**Change:** (1) a third retriever, full-text on the *title only* (new generated `title_tsv` column + GIN
index, migration 0004); (2) weighted RRF with weights vector : body-fts : title-fts = 1 : 0.5 : 0.5.

**Protocol:** variants were compared on a DEV split (odd query ids) and checked on a HOLDOUT split
(even ids) to limit overfitting to the golden set. All variants tried:


| Variant | DEV nDCG@10 | DEV MRR | HOLDOUT nDCG@10 | HOLDOUT MRR | ALL nDCG@10 |
|---|---|---|---|---|---|
| vector only | 0.708 | 0.845 | 0.665 | 0.860 | 0.687 |
| RRF vector+fts (baseline hybrid) | 0.669 | 0.867 | 0.666 | 0.841 | 0.668 |
| weighted RRF 1 : 0.5 (vec : fts) | 0.732 | 0.932 | 0.701 | 0.869 | 0.717 |
| weighted RRF 1 : 0.25 (vec : fts) | 0.741 | 0.955 | 0.689 | 0.844 | 0.716 |
| vector + title-fts | 0.701 | 0.848 | 0.702 | 0.785 | 0.701 |
| vector + title-fts (0.5) | 0.735 | 0.886 | 0.721 | 0.804 | 0.728 |
| vector + fts + title-fts (1 : 0.5 : 0.5) | 0.744 | 0.907 | 0.718 | 0.807 | 0.731 |
| vector + fts + title-fts (1 : 0.25 : 0.5) | 0.746 | 0.882 | 0.715 | 0.813 | 0.731 |

Result on the full set: hybrid nDCG@10 **0.668 → 0.731**, recall@10 **0.630 → 0.712**. On top of that,
the LLM rerank of the fused top 50 lifts nDCG@10 to **0.799** and MRR to **0.913** (same snapshot,
same run), at a much higher latency (an LLM call; cached afterwards).

Caveats: tuned and checked on the same 43 queries (split, but small); title-only relevance rubrics favour
title-matching signals; rerank latency here includes free-tier rate-limit waits.
