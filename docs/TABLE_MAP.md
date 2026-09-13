# Manuscript table and output map

Table numbers refer to Revised_Manuscript.docx supplied with the revision, not the expanded comparison copy.

| Manuscript item | Reproducible source |
| --- | --- |
| Table 4 group comparison matrix | Geometric mean of `expert_inputs.json` AHP arrays in `audit_analysis.py`; also regenerated `part1_results.json` |
| Tables 5 and 5a weights and individual quartiles | `audit_results.json`: weights and experts |
| Table 6 R1 medians | Median of `expert_inputs.json`: R1_scores |
| Table 7 R1 TOPSIS distances and coefficients | `audit_results.json`: topsis[0] |
| Table 8 six scenario coefficients | `audit_results.json`: topsis |
| Table 9 one-at-a-time sensitivity | `audit_results.json`: oat |
| Tables 9a and 9b score dispersion and expert robustness | `audit_results.json`: experts |
| Table 10 ANFIS folds | `part2_results.json`: folds |
| Table 11 reference-model comparison | `part2_results.json`: summary |
| Table 12 matched initial FIS ablation | `audit_results.json`: matched_ablation and matched_initial_summary |
| Appendix B matrices and distances | `aggregate_inputs.json` risk_matrices and `audit_results.json` topsis |
| Appendix C predictions and targets | `part2_results.json`: preds and y; `all_predictions.csv` |
| Figures 1–5 | `make_figures.py` |

The scripts retain full precision; manuscript values are rounded. Training/validation/test splits, initialisation seeds and selected update counts are included in the computation. The archive's original_reference directory explains earlier discrepancies and is not a second authoritative analysis.
