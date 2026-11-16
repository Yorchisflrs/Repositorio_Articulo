# Reproducibility

Everything reported in the manuscript is produced by the ten numbered scripts in `scripts/`, run in order.
No figure in the text is typed by hand: `scripts/10_generate_outputs.py` writes the tables, the figures and a
macro file (`paper/generated/numbers.tex`) from the CSV and JSON files in `results/metrics/` and
`results/tables/`, and the manuscript reads its numbers from that file.

## Environment

The reported results were obtained with Python 3.13.18 on Linux x86-64 (Ubuntu 24.04, 2 CPU cores, 8 GB of
memory) and the package versions pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`environment.yml` creates the same environment with conda.

## Data

Download the 45 ENAHO archives listed in `data/DATA_MANIFEST.csv` and unzip them as described in
`data/README_DATA.md`. `python scripts/01_prepare_data.py` stops with a clear message if a file is missing.
To check that the files are the ones used here, compare their SHA-256 with the manifest.

## Running the pipeline

```bash
python scripts/01_prepare_data.py        # read the five modules, join them, keep persons aged 10-18
python scripts/02_harmonize_data.py      # harmonise variables, build the analytic sample and the flow table
python scripts/03_validate_data.py       # data validation tables (labels, month artefact, design, overlap)
python scripts/04_train_models.py        # nested grouped CV and frozen models (E1)
python scripts/05_temporal_validation.py # forward-in-time validation, in-year reference, shift diagnostics (E2)
python scripts/06_calibration.py         # recalibration, retraining, rolling origin (E3)
python scripts/07_subgroup_analysis.py   # subgroup reliability (E4)
python scripts/08_shap_analysis.py       # SHAP by family, stability, permutation importance (E5)
python scripts/09_ablation.py            # ablation and outcome-type analysis (E6)
python scripts/09b_panel_prospective.py  # prospective outcome on the panel subsample (E6)
python scripts/10_generate_outputs.py    # tables, figures and manuscript numbers
python -m pytest -q tests                # unit tests and consistency checks
```

Approximate run times on the machine described above: steps 1-3 about 6 minutes, step 4 about 15 minutes,
steps 5 and 6 about 10 minutes each, step 7 about 45 minutes, steps 8-9b about 30 minutes in total, step 10
under a minute. Steps 5-9b only need the files written by step 4 and can be run in parallel if memory allows
(each uses 1-1.5 GB).

## Determinism

One seed (`configs/config.json`, `seed = 2026`) drives every random step. Each script derives its own
generators from that seed plus a fixed offset, so running one script again, or changing one, does not alter
the random numbers used by the others. With the pinned versions the outputs are identical between runs on
the same machine. Across machines, gradient boosting with the histogram method can differ in the last
decimals; the conclusions do not depend on them.

Bootstrap replicate weights are generated once per evaluation set and reused for every model and metric, so
differences between models are paired by construction.

## What each result file contains

`06_REPRODUCIBILIDAD/EXPERIMENT_MANIFEST.csv` in the delivery folder (and the table in the README) maps every
experiment to its script, its inputs and its output files. `results/tables/manuscript_numbers.csv` maps every
number printed in the manuscript to the file and the row it comes from.

## Building the manuscript

```bash
cd paper
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
pdfinfo manuscript.pdf | grep Pages     # must not exceed 7
```

`tests/test_manuscript.py` fails if the manuscript uses a number that is not in the generated macro file, if
the abstract is longer than 250 words, or if the compiled PDF has more than seven pages.

## Known limits of reproducibility

INEI occasionally republishes microdata files. If a checksum differs from the manifest, small differences in
counts are possible. The files used here cannot be redistributed with this repository; they are public and
free of charge at the INEI portal.
