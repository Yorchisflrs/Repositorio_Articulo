# Data

This repository contains **no ENAHO microdata**. The microdata are public and are downloaded from the
National Institute of Statistics and Informatics of Peru (INEI).

## Source

Encuesta Nacional de Hogares sobre Condiciones de Vida y Pobreza (ENAHO), annual files 2017-2025, Stata format.
Portal: <https://proyectos.inei.gob.pe/microdatos/>.
Direct download pattern: `https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/<code>-Modulo<NN>.zip`

| Year | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| INEI survey code | 603 | 634 | 687 | 737 | 759 | 784 | 906 | 966 | 1031 |

Five modules are needed for each year (45 zips, about 626 MB):

| INEI module | File used | Content |
|---|---|---|
| 01 | `enaho01-<year>-100.dta` | Dwelling and household (phone, internet, housing, interview mode in 2020) |
| 02 | `enaho01-<year>-200.dta` | Household roster (age, sex, kinship, membership, person panel code) |
| 03 | `enaho01a-<year>-300.dta` | Education (attainment, enrollment, mother tongue, person weight) |
| 18 | `enaho01-<year>-612.dta` | Household durables (computer) |
| 34 | `sumaria-<year>.dta` | Household summary (poverty, expenditure, size) |

`DATA_MANIFEST.csv` lists the 45 zips with their URL, size and SHA-256, and the SHA-256 of the Stata file
used from each one. The checksums describe the files the author downloaded from the INEI portal and that were
used for this study in November 2026. INEI occasionally republishes files, so a
different checksum means the file changed after that date.

## Layout expected by the scripts

```
data/raw/2017/enaho01-2017-100.dta
data/raw/2017/enaho01-2017-200.dta
data/raw/2017/enaho01a-2017-300.dta
data/raw/2017/enaho01-2017-612.dta
data/raw/2017/sumaria-2017.dta
...
data/raw/2025/...
```

Unzip every archive into the folder of its year (sub-folders inside the zip can be flattened). Another
location can be used by setting the environment variable `ENAHO_RAW`.

## Files written to `data/processed/` (not versioned)

| File | Written by | Content |
|---|---|---|
| `persons_10_18.parquet` | `01_prepare_data.py` | Roster persons aged 10-18 with the variables of the five modules |
| `harmonized_10_18.parquet` | `02_harmonize_data.py` | Same rows with harmonised variables |
| `analytic.parquet` | `02_harmonize_data.py` | Analytic population (84,502 adolescents) |
| `predictions.parquet` | `04_train_models.py` | Out-of-fold and frozen-model predictions |
| `models/*.pkl` | `04_train_models.py` | Frozen models |
| `panel_pairs.parquet` | `09b_panel_prospective.py` | Linked person-pairs (t, t+1) |

## Terms of use

ENAHO microdata are published by INEI for public use. Cite INEI as the source. Check INEI's current terms
before redistributing any derived person-level file (see `LICENSE_PENDING.md`).
