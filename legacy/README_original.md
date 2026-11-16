# ICALTER 2026 – School non-enrollment in Peru (ENAHO 2017–2025)

Run order (Python 3, pandas, pyreadstat, scikit-learn, xgboost, shap, statsmodels, matplotlib, pyarrow). Seed: 2026.
1. scripts/01_inspect.py <enaho_raw>        – checks that the variables exist in every year
2. scripts/02_labels.py <enaho_raw>         – variable and value labels (verification)
3. scripts/03_build.py <enaho_raw> base_6_17.parquet  – merges modules 300, 200, 100, 612 and sumaria
4. scripts/04_features.py base_6_17.parquet analytic.parquet – population 12–17, April–December, predictors
5. scripts/04b_harmonize_internet.py         – internet break from 2023 (p1144 vs p114b1)
6. scripts/05_models.py                      – E1–E5 (CV, temporal transfer, retraining, subgroups, SHAP)
7. scripts/06_sensitivity_prev.py            – E6 ablation and design-based prevalence
8. scripts/08_mode_month.py                  – calendar artefact by month and 2020 results by interview mode
9. scripts/07_figures.py                     – figures (run after 08)
<enaho_raw> = folder with one subfolder per year (2017…2025) containing the .dta files extracted from INEI's zips
(modules 01, 02, 03, 18, 34; codes 603, 634, 687, 737, 759, 784, 906, 966, 1031).
Scripts 04b, 08 and the build step use the path /home/claude/enaho_raw: change it to yours.
manuscript/main.tex (English, for submission) and manuscript/main_es.tex (Spanish) compile with pdflatex (IEEEtran class).
