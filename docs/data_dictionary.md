# Data dictionary

Every analysis variable, its ENAHO source, the transformation applied in `src/preprocessing/features.py`
and the check that covers it. Value labels for each year are in `results/tables/value_labels_by_year.csv`
(read from the Stata files) and observed code counts in `results/tables/raw_code_counts_by_year.csv`.

## Outcome and population

| Variable | Source (module: item) | Original coding | Transformation | Check |
|---|---|---|---|---|
| `y` (not enrolled this year) | 300: P306 "Este año, ¿está matriculado en algún centro o programa de educación básica o superior?" | 1 = sí, 2 = no (all years) | `y = 1` if P306 = 2 | Labels identical 2017-2025; only codes 1 and 2 observed; never missing in the analytic population |
| Age filter | 200: P208A (completed years) | integer | 12 <= age <= 17 | `test_no_duplicate_persons_and_valid_ranges` |
| Household member | 200: P204 | 1 = sí, 2 = no | keep P204 = 1 | flow table |
| Not absent | 200: P205 (absent 30 days or more) | 1 = sí, 2 = no | keep P205 != 1 | flow table |
| Secondary not completed | 300: P301A | 1 none ... 5 incomplete secondary, 6 complete secondary, 7-11 post-secondary, 12 special basic education | keep P301A < 6 | `results/tables/exclusion_detail.csv` |
| Interview month | 200: MES | 1-12 | keep 4-12 | `results/tables/prevalence_by_month.csv` |
| Enrolled last year (leakage analysis only) | 300: P303 | 1 = sí, 2 = no | used as is | Missing for every 2020 telephone interview (item not asked) |
| Person panel code (panel analysis only) | 200: P201P | 17-character code | exact match between years t and t+1 | unique within year; sex equal and age +0..2 in 99.5-100% of matches |

## Predictors of the main model (26 columns)

| Variable | Source | Original coding | Transformation | Notes |
|---|---|---|---|---|
| `age` | 200: P208A | years | integer 12-17 | |
| `female` | 200: P207 | 1 = hombre, 2 = mujer | 1 if P207 = 2 | |
| `indig_tongue` | 300: P300A | 1 quechua, 2 aimara, 3 otra lengua nativa, 4 castellano, 6 portugués, 7 otra extranjera, 8 no escucha/no habla, 9 lengua de señas (from 2018), 10-15 Amazonian languages (from 2020) | 1 if code in {1, 2, 3, 10, ..., 15} | Before 2020 Amazonian languages fall under code 3 |
| `school_years` | 300: P301A, P301B (year), P301C (grade) | see above | 0 if P301A <= 2; P301C if incomplete primary; 6 if complete primary; 6 + P301B if incomplete secondary | Primary grade is stored in P301C and secondary year in P301B (verified in the data) |
| `overage` (age-grade distortion) | derived | | `age - 6 - school_years`, clipped to [-2, 12] | Deterministic function of `age` and `school_years` |
| `child_of_head` | 200: P203 | 1 jefe, 2 cónyuge, 3 hijo(a), ..., 11 hermano(a) (from 2018) | 1 if P203 = 3 | Code 11 added in 2018 does not affect code 3 |
| `hh_mobile` | 100: P1142 | 0 = pase, 1 = sí | 1 if P1142 = 1 | |
| `hh_computer` | 612: P612N = 7, P612 | P612: 1 = sí, 2 = no | 1 if the household owns item 7 | Label "computadora" (2017), "computadora/laptop" (2018 on); tablet is a separate item (28) from 2022 and is not counted; 1.4% missing |
| `poor`, `extreme_poor` | sumaria: POBREZA | 1 pobre extremo, 2 pobre no extremo, 3 no pobre | `poor` = 1 or 2; `extreme_poor` = 1 | Official monetary poverty |
| `exp_quintile` | sumaria: GASHOG2D / MIEPERHO | soles per year | weighted quintile inside the analytic sample of each year | Within-year ranks avoid inflation drift |
| `head_educ_years` | 300: P301A of the head (P203 = 1) | level | level mapped to years (0, 0, 3, 6, 8, 11, 13, 14, 14, 16, 18) | Approximation from level only; 0.04% missing |
| `head_female`, `head_age` | 200: P207, P208A of the head | | | |
| `hh_size` | sumaria: MIEPERHO | persons | as is | |
| `n_kids` | 200 | | members (P204 = 1) younger than 18 | |
| `rural` | 200: ESTRATO | 1-8 (size of population centre) | 1 if ESTRATO >= 6 | 6-8 = rural strata |
| `inadequate_housing` | 100: NBI1 | 0/1 | as is | |
| `dom_1` ... `dom_8` | 200: DOMINIO | 1 Costa Norte, 2 Costa Centro, 3 Costa Sur, 4 Sierra Norte, 5 Sierra Centro, 6 Sierra Sur, 7 Selva, 8 Lima Metropolitana | indicator columns | |

## Variables used only in sensitivity analyses

| Variable | Source | Transformation | Notes |
|---|---|---|---|
| `hh_internet_raw` | 100: P1144 | 1 if P1144 = 1 | Not comparable from 2023 (see `harmonization.md`) |
| `hh_internet_fixed` | 100: P1144 (to 2022), P114B1 (from 2023) | P1144 up to 2022; P114B1 = 1 from 2023 | P114B1 is blank when the household has no internet; coded 0 |
| `telephone` | 100: TIPOENTREVISTA (2020 only) | 1 if TIPOENTREVISTA = 2 | 1 = presencial, 2 = telefónico |

## Design variables

| Variable | Source | Use |
|---|---|---|
| `factora07` | 300: FACTORA07 | Person expansion factor of the education module; weights every estimate |
| `estrato` | 200: ESTRATO | Stratum in variance estimation (INEI specification) |
| `conglome` | all modules | Primary sampling unit; grouping unit in cross-validation |
| `dept` | first two digits of UBIGEO | Alternative strata (department x ESTRATO) in a sensitivity analysis |
