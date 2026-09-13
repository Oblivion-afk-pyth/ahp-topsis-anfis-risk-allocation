# GitHub and Zenodo release procedure

1. Finalise the authors' reuse licences. Use a public GitHub repository named `ahp-topsis-anfis-risk-allocation` under the chosen author or organisation account. Upload this folder's contents, not only a ZIP file.
2. Run `python reproduce.py` and retain the successful report. Commit the tested repository contents. Exclude the generated reproduced folder and local virtual environment.
3. Sign in to Zenodo and link the intended GitHub account. Review the scope of GitHub permissions. In Zenodo's GitHub settings, sync repositories and enable this repository before making its first release.
4. Create GitHub tag and release `v1.0.0` with the tested commit and the release notes below. Zenodo archives new releases after integration is enabled. A normal repository URL or an unpublished draft is not an archived release.
5. Wait for Zenodo processing and open its resulting record. Verify the authors, version, licences, files and public access. Copy the DOI for this specific version. Do not substitute an invented DOI or the DOI of an unrelated article.
6. Add the verified GitHub URL and Zenodo version DOI to the manuscript Data Availability section and Reviewer 3 response. Add the DOI to the README and CITATION.cff on the main branch. This documentation update need not create another release. If computational files change, release a new version and cite that version's DOI.

## Proposed release notes

Reproducible numerical companion to the revised AHP–TOPSIS risk-allocation manuscript with exploratory ANFIS evaluation. Includes the author-designated 37-expert AHP and R1 workbook, machine-readable inputs, five reconstructed scenarios, complete reference outputs, expert dispersion and resampling analyses, matched non-adaptive FIS ablation, all model predictions, figures and a reproduction runner. No independent contractual outcome validation is claimed.

## Official instructions

- Enable integration: https://help.zenodo.org/docs/github/enable-repository/
- Archive a release: https://help.zenodo.org/docs/github/archive-software/github-upload/
- Citation metadata: https://help.zenodo.org/docs/github/describe-software/citation-file/

These instructions were checked on 13 September 2026. This local preparation has not created a public repository or DOI.
