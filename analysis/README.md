# Supplementary data and code

This package accompanies the revised manuscript. Run `python audit_analysis.py` from any directory after installing requirements.txt. Python and NumPy versions used in this revision are stored in audit_results.json. The script writes results beside itself. It reconstructs all revised numerical results, model comparisons, the matched ablation, and expert dispersion and robustness results. The manuscript rounds results for display; calculations retain full precision.

## Data provenance and scope

37_Experts_Data.xlsx is the unmodified workbook designated by the author as the actual expert response dataset on 12 September 2026. Its original worksheet names are Synthetic_AHP_Raw and Synthetic_TOPSIS_R1_Raw. The author explicitly stated that these records are not synthetic. This confirmation is the basis for treating them as responses; original questionnaires, survey timestamps and independent provenance verification are not included. Naming and numerical consistency alone cannot authenticate survey origin. The original worksheet labels are preserved for transparency.

The file contains 37 expert identifiers, 777 AHP comparisons and 1,036 R1 allocation scores. expert_inputs.json is an exact machine-readable transcription: reverse AHP comparisons are reciprocal entries. aggregate_inputs.json contains the manuscript's old displayed AHP matrix and six score matrices; the active analysis replaces its AHP matrix with the geometric mean of the designated individual responses and replaces R1 with their median scores. R2–R6 remain reconstructed numerical inputs, not observed individual expert scores. No other workbook from the source directory is used.

## Files and results

- audit_analysis.py: full revised aggregate calculation, AHP/TOPSIS, arbitrary weight sensitivity, expert quartiles, individual rankings, Kendall W, paired expert bootstrap, leave-one-expert-out analysis, counterexample and matched FIS ablation.
- reproduce_anfis.py: retained two-rule finite-difference hybrid-learning implementation with unambiguous completed-update counter (zero means initial model). Invoked by the audit script.
- audit_results.json and part2_results.json: full-precision revised numerical results, folds, updates, convergence series and model metrics.
- part1_results.json and r2r6.json: regenerated interface inputs to the retained learning implementation. Use audit_analysis.py first to regenerate these consistently.
- aggregate_scores.csv: all revised 24 input rows and their exact targets.
- all_predictions.csv: all 24 held-out targets and predictions for ANFIS, the 20-row FIS and MLR references, and the matched 16-row initial FIS.
- make_figures.py: recreates all five manuscript figures from the revised numerical results. Run after audit_analysis.py.
- verify_workbook.py: optionally verifies every extracted AHP comparison and R1 score against the unmodified source workbook.
- original_reference/: original computational files and original reported results retained for audit. Their main sensitivity script used earlier R2–R6 matrices; they are not the revised pipeline. The original epoch counter is ambiguous and the original ANFIS/FIS comparison uses different fitting data. These original files are included to explain discrepancies, not as the authoritative computation.

## Interpretation

Seed 7 governs model initialisations, 42 the uniform weight perturbations, and 20260912 the 10,000 paired expert bootstrap draws. Bootstrap records pair each expert's entire AHP matrix with the same expert's R1 scoring matrix. All 37 responses are retained; no imputation or consistency-based exclusion occurs. The bootstrap describes the empirical panel and is not external validation. R2–R6 have no individual-response analysis. AHP weights and scenario definitions are fixed context in the learning exercise, so it is not end-to-end validation of a newly elicited case. Consequent ridge shrinkage does not regularise the premises. Finite-difference training and its selected candidate may vary with numerical environment; full-precision outputs and versions are included.

The source workbook and extracted raw inputs are the data available for these analyses. No independent project outcomes, dispute records, or new panels are included. The conditional mathematical counterexample is constructed deliberately as an illustration and is not an expert observation.
