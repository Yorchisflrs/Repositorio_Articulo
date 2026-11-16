# School non-enrollment in Peru before and after COVID-19

Code, result files and manuscript source for the paper *Predicting School Non-Enrollment in Peru Before and
After COVID-19: Temporal Robustness and Subgroup Calibration of Explainable Machine Learning*
(Edgar Jeferson Cusihuaman Garate, Universidad Nacional del Altiplano, Puno).

The study uses nine annual rounds (2017-2025) of the Peruvian National Household Survey (ENAHO) to ask three
questions about models of adolescent school non-enrollment: whether a model trained before the pandemic keeps
its discrimination and calibration afterwards, whether its errors fall unevenly across groups, and what the
survey outcome actually measures.

## Main results

All figures are survey-weighted, with 95% intervals from a stratified cluster bootstrap (1,000 replicates).
Every number below is in `results/` and in `results/tables/manuscript_numbers.csv`.

* 84,502 adolescents aged 12-17 without completed secondary school, interviewed April-December; 3,607 not
  enrolled. January and February interviews are excluded because the school year starts in March.
* XGBoost tuned by nested cross-validation on 2017-2019: ROC-AUC 0.930 in the training period, 0.837 in 2020
  and 0.888 in 2025 without refitting. A model cross-validated inside 2020 reaches 0.844, so the 2020 drop
  reflects a harder year rather than a stale model.
* Calibration failed during 2020-2021 (risk under-predicted, calibration-in-the-large 0.41 and 0.43) and
  recovered without updating from 2022. Re-estimating the intercept on the previous year over-corrected 2022.
* After the pandemic, overall calibration hides under-prediction for rural and Indigenous-language
  adolescents and over-prediction for urban ones (rural-urban gap +0.49 [0.32, 0.68]).
* About two in three non-enrolled adolescents were already out of school the year before. The model reaches
  ROC-AUC 0.964 for that persistent group, 0.790 for incident non-enrollment and 0.815 for non-enrollment one
  year ahead in the survey panel. It is a tool for describing the out-of-school population, not an early
  warning system.

## Repository layout

```
configs/config.json         years, seed, grids, bootstrap replicates
src/                        package: data access, features, models, validation, calibration, SHAP, metrics
scripts/01-10_*.py          the pipeline, run in order (see docs/reproducibility.md)
tests/                      unit tests, regression tests on the data and manuscript consistency checks
notebooks/                  results_overview.ipynb: reads result files only
data/                       README_DATA.md and DATA_MANIFEST.csv (no microdata in the repository)
results/metrics/            every metric with its interval, one CSV or JSON per experiment
results/tables/             data validation tables, paper tables as CSV, manuscript number registry
results/figures/            paper figures (PDF and 600 dpi PNG)
paper/                      manuscript.tex, references.bib, generated tables and numbers, tools
docs/                       methodology, data dictionary, harmonization, reproducibility
legacy/                     the original scripts, tables and manuscript, unchanged
```

## Experiments

| ID | Question | Script | Main output |
|---|---|---|---|
| E1 | In-period performance (nested grouped CV) | `04`, `05` | `E1_E2_performance.csv` |
| E2 | Frozen model applied to 2020-2025; same-year reference; shift diagnostics | `05` | `E1_E2_performance.csv`, `E2_*.csv`, `shift_diagnostics.csv` |
| E3 | Recalibration and retraining on 2025; rolling origin | `06` | `E3_*.csv` |
| E4 | Subgroup reliability in three periods; group-specific intercepts | `07` | `E4_*.csv` |
| E5 | SHAP by family, stability over refits, permutation importance | `08` | `E5_*.csv` |
| E6 | Ablation; persistent vs incident outcome; prospective panel outcome | `09`, `09b` | `E6_*.csv`, `leakage_*.csv`, `panel_*.csv` |

## Quick start

```bash
pip install -r requirements.txt
# put the ENAHO files in data/raw/<year>/ (data/README_DATA.md), then
for s in 01_prepare_data 02_harmonize_data 03_validate_data 04_train_models 05_temporal_validation \
         06_calibration 07_subgroup_analysis 08_shap_analysis 09_ablation 09b_panel_prospective 10_generate_outputs; do
  python scripts/$s.py
done
python -m pytest -q tests
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

Without the microdata, `python scripts/10_generate_outputs.py` and the tests that read result files still run,
because the result files are part of the repository.

## Traceability

The manuscript contains no hand-typed statistics. `scripts/10_generate_outputs.py` writes every number to
`paper/generated/numbers.tex`, and the manuscript prints them with `\V{key}`. The file
`results/tables/manuscript_numbers.csv` lists each key with its value, the result file and the row it comes
from. `tests/test_manuscript.py` fails if the text contains a decimal number typed by hand or uses an
undefined key; `tests/test_claims.py` checks the qualitative statements of the paper against the result files.

## Data

ENAHO microdata are public and free at <https://proyectos.inei.gob.pe/microdatos/>. They are not included
here. `data/DATA_MANIFEST.csv` lists the 45 files used, with their official URL and SHA-256 checksums.

## Status

The manuscript is complete except for the author's contact e-mail, which is marked `EMAIL-PENDIENTE`. The
license has not been chosen yet (`LICENSE_PENDING.md`). See `CHANGELOG.md` for the differences from the
original project kept in `legacy/`.

## Citation

See `CITATION.cff`.
