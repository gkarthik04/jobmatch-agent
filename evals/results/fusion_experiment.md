### Fusion experiment (all variants tried)

Chosen on DEV (odd query ids) only; HOLDOUT (even ids) is the unbiased check.

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
