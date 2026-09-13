# AHP TOPSIS risk allocation with an exploratory ANFIS evaluation

Computational companion to **AHP TOPSIS Decision Support for Contractual Risk Allocation with an Exploratory ANFIS Evaluation**, by Sulaksha Wimalasena and Tharushika Prabhathi.

This repository reproduces the manuscript's numerical analysis, including the available expert-response dispersion analysis, six TOPSIS matrices, two-rule ANFIS comparison, matched non-adaptive FIS ablation, sensitivity analyses and five figures.

## What the evidence supports

Direct TOPSIS is the operational calculation once the complete decision matrix is available. A single alternative's seven scores do not generally determine its TOPSIS coefficient. The ANFIS experiment does not demonstrate an adaptive advantage. The results describe this numerical exercise and the stability of one expert panel, not validated contractual outcomes or generalisation to new projects.

## Data scope

The author-designated response workbook contains 37 experts' AHP comparisons and allocation scores for payment delay (R1): 777 pairwise comparisons and 1,036 allocation scores. R2–R6 are reconstructed numerical inputs; they are not individual expert observations. The original workbook and worksheet names are preserved. See [data provenance](docs/DATA_PROVENANCE.md) and [data dictionary](docs/DATA_DICTIONARY.md).

## Reproduce the analysis

Python 3.12 and NumPy 2.3.5 were used for the reference computation (exact recorded Python version: 3.12.14).

```text
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```text
python -m pip install -r requirements.txt
python reproduce.py
```

The runner checks the source workbook against the extracted inputs, reruns the calculations in a temporary directory, compares the numerical results with the stored reference outputs and writes fresh results and figures to `reproduced/`. It does not overwrite the archived inputs or reference results. Run `python reproduce.py --no-figures` to check numbers only.

Finite-difference premise updates can be sensitive to numerical environments. A failed comparison is reported rather than silently replacing the archived results. The reference outputs and environment remain available for inspection.

## Contents

| Path | Purpose |
| --- | --- |
| `analysis/37_Experts_Data.xlsx` | Unmodified author-designated response workbook |
| `analysis/expert_inputs.json` | Exact computational transcription of AHP and R1 responses |
| `data/ahp_comparisons.csv` | 777 observed pairwise comparisons in long format |
| `data/r1_scores.csv` | 1,036 R1 scores in long format |
| `analysis/aggregate_inputs.json` | Legacy displayed matrix plus six scenario matrices; the active script replaces AHP and R1 from individual records |
| `analysis/audit_analysis.py` | Authoritative revised pipeline |
| `analysis/reproduce_anfis.py` | Two-rule ANFIS and reference-model calculation |
| `analysis/audit_results.json` | Full-precision TOPSIS, dispersion, sensitivity and ablation outputs |
| `analysis/part2_results.json` | Model predictions, fold metrics and learning curves |
| `analysis/all_predictions.csv` | All 24 held-out predictions and targets |
| `analysis/original_reference/` | Superseded scripts and results retained for discrepancy auditing; not the active pipeline |
| `docs/TABLE_MAP.md` | Links manuscript tables to computation outputs |
| `CITATION.cff` | Machine-readable citation metadata |
| `SHA256SUMS.txt` | File-integrity manifest |

## Citation and archive

GitHub repository: [https://github.com/Oblivion-afk-pyth/ahp-topsis-anfis-risk-allocation](https://github.com/Oblivion-afk-pyth/ahp-topsis-anfis-risk-allocation). Version prepared: **1.0.0**. Zenodo archiving is pending; no DOI has been assigned. Cite the immutable Zenodo version DOI once the first GitHub release has been archived.

## Reuse

Reuse licences are awaiting the authors' selection. No open licence is granted by this draft package. Licence files and citation metadata must be finalised before the Zenodo archive release.
