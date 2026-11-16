# Harmonisation of ENAHO 2017-2025

What changes between years in the variables used, how each change is handled, and the evidence. All
statements below were checked against the value labels stored in the Stata files of the nine years
(`results/tables/value_labels_by_year.csv`), the PDF dictionaries of 2017, 2020 and 2025, and the household
questionnaires of 2022 and 2023.

## Merge keys

Households are identified by `CONGLOME + VIVIENDA + HOGAR` and persons by those three plus `CODPERSO`.
`results/logs/join_audit.csv` reports, for every year: zero duplicated keys in modules 100, 200, 300 and
sumaria; zero education records without a roster record; 100% of roster persons aged 10-18 matched to their
household (module 100) and to sumaria; 97.4-98.2% matched to an education record (the rest are roster
persons outside the education module, such as non-members); 96.8-99.8% of persons with the durables module
available. No merge duplicates rows (asserted in `01_prepare_data.py`).

## Changes found and decisions

| Variable | Change | Years | Decision |
|---|---|---|---|
| P306 (enrollment this year) | None: 1 = sí, 2 = no | 2017-2025 | Used as is |
| P300A (mother tongue) | Code 9 "lengua de señas peruanas" added in 2018; codes 10-15 (ashaninka, awajún, shipibo-konibo, shawi, matsigenka, achuar) added in 2020, previously inside code 3 "otra lengua nativa" | 2018, 2020 | One indicator for codes 1, 2, 3, 10-15, so the definition is constant |
| P301A/B/C (attainment) | Wording of code 11 only ("postgrado" to "maestría/doctorado") | 2018 | No effect on ages 12-17 |
| P203 (kinship) | Code 11 "hermano(a)" added; code 3 reworded "hijo(a)/hijastro(a)" | 2018 | `child_of_head` uses code 3 only; its share is stable across 2017-2018 |
| P612N item 7 | "computadora" becomes "computadora/laptop"; "tablet" added as item 28 | 2018, 2022 | Item 7 only; tablet excluded |
| P1142 (mobile phone) | Label only ("celular" to "teléfono celular") | 2019 | Used as is |
| P1144 (household internet) | Label "internet" becomes "conexión a internet" in 2019. In 2023 the questionnaire item becomes "conexión a internet (fijo/móvil)" and items P114B1-B3 (fixed, post-paid mobile, prepaid mobile) are added | 2019, 2023 | **Excluded from the main model** (see below) |
| TIPOENTREVISTA | Published only in 2020 (1 = presencial, 2 = telefónico) | 2020 | Used to split 2020 results by interview mode |
| P303 (enrolled last year) | Coding constant, but the item was not asked in the 2020 telephone interviews | 2020 | Leakage analysis uses 2017-2019 and 2021-2025 |
| POBREZA, NBI1, ESTRATO, DOMINIO | None | 2017-2025 | Used as is |

## The household-internet item

Among adolescents of the analytic population, the share living in a household with internet according to
P1144 moves from 25.0% (2017) to 57.1% (2022) and then to 95.8% (2023), 97.2% (2024) and 97.1% (2025)
(`results/tables/internet_item_by_year.csv`). A jump of 38.8 points in one year coincides with the
questionnaire change, not with a plausible change in connectivity: the fixed-connection item P114B1 gives
28.7% in 2023, 29.1% in 2024 and 30.7% in 2025, and about three in four connected households in 2023-2025
report a mobile connection only.

No single series is consistent over the nine years. P1144 before 2023 does not say whether mobile data
plans were counted, so it is neither "any connection" nor "fixed connection" with certainty. For that
reason:

* the main model **does not use** household internet;
* the ablation reports two sensitivity variants, P1144 as published and a "fixed connection" series
  (P1144 to 2022, P114B1 from 2023), and both change the ROC-AUC by less than 0.005 in every year;
* P114B1 is blank by skip pattern when the household reports no internet, and is coded 0 (all blank cases
  have P1144 = 0).

## The interview-month restriction

P306 asks about enrollment "this year", and the Peruvian school year starts in March. Before the month
filter, the weighted share answering "no" is 81.3% in January, 51.6% in February and 13.0% in March, and
between 3.3% and 4.5% in every month from April to December (`results/tables/prevalence_by_month.csv`).
January-February interviews measure the vacation and March the enrollment period, so the analytic
population keeps April-December interviews (25,463 adolescents excluded).

## Automatic promotion in 2020-2021

Among adolescents enrolled the previous year, the share reporting "desaprobado" for that year (P305) falls
from about 2% in 2017-2019 to 0.2-0.4% in 2021-2023 and returns to 1.3-1.5% in 2024-2025 (unweighted shares). Age-grade
distortion therefore accumulates more slowly for the cohorts schooled during the pandemic, which changes
what the same value of the predictor means after 2021. This is documented as a source of concept drift;
no correction is applied.
