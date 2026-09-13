# Data dictionary

## Shared definitions

Expert identifiers run from `Expert_1` to `Expert_37` and are consistent across the two long-format CSV files. They are taken from the workbook's numerical expert index. No identifying key is supplied.

| Code | Criterion |
| --- | --- |
| C1 | Risk controllability |
| C2 | Risk mitigation capability |
| C3 | Financial capacity |
| C4 | Legal and contractual authority |
| C5 | Information availability |
| C6 | Impact absorption capacity |
| C7 | Risk transfer efficiency |

All seven allocation scores are benefit-oriented: larger means greater stated capability. Alternatives are A1 Client, A2 Contractor, A3 Consultant and A4 Shared. R1 is payment delay; R2 design change; R3 unforeseen site conditions; R4 material price escalation; R5 force majeure; R6 regulatory change.

## ahp_comparisons.csv

777 rows, one per expert and unique criterion pair. `expert_id` identifies the response; `criterion_left` and `criterion_right` identify the pair; `comparison_ratio` is the recorded importance ratio of the left criterion to the right. Reverse ratios are reciprocals and diagonal entries are one. Reverse and diagonal entries are derived, not additional observations.

## r1_scores.csv

1,036 rows. `expert_id`, `risk_id`, `alternative_id`, `alternative`, `criterion` and `score` identify each original scoring item. Scores are integers from 1 to 9. No R2–R6 individual scores are present.

## Computational JSON and CSV files

`analysis/expert_inputs.json` stores AHP as a 37 × 7 × 7 array and R1_scores as a 37 × 4 × 7 array. Expert, alternative and criterion orders follow the definitions above. `analysis/aggregate_scores.csv` has 24 scenario-alternative rows, seven aggregate scores and the corresponding direct TOPSIS target. R1 aggregates are medians; R2–R6 rows are reconstructed inputs. `analysis/all_predictions.csv` provides targets and predictions for ANFIS, the 20-row non-adaptive FIS and MLR references, and the matched 16-row initial FIS. Predictions are unbounded numerical regression outputs, not probabilities.
