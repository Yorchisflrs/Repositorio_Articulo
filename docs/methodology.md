# Methodology

This note states what is estimated, how, and under which assumptions. File names refer to this repository.

## 1. Target, population and scope

* **Outcome.** `y = 1` when the adolescent answers "no" to ENAHO item P306 ("This year, are you enrolled in a
  basic or higher education program?"). The outcome is **current non-enrollment**, measured in the same
  interview as the predictors. It is not future dropout.
* **Population.** Household members aged 12-17, not absent from the dwelling for 30 days or more, who have
  not completed secondary school and were interviewed between April and December (2017-2025). The flow of
  exclusions is in `results/tables/sample_flow.csv`.
* **Three targets are kept apart.**
  1. *Status* (main analysis): not enrolled this year.
  2. *Incident non-enrollment*: not enrolled this year among those enrolled last year (item P303).
  3. *Prospective*: not enrolled in year t+1 among those enrolled in year t, for persons re-interviewed in
     the ENAHO panel component (linked by the person panel code P201P).
  Targets 2 and 3 are validity checks for the claim that the model identifies risk and not only a spell of
  non-enrollment that already started.

## 2. Survey design

ENAHO is a probability, stratified, multistage, area sample, independent in each department. Persons carry
the expansion factor `FACTORA07` (education module). Following INEI's variance specification, the primary
sampling unit is the cluster `CONGLOME` and the stratum is `ESTRATO`; department x `ESTRATO` strata are used
as a sensitivity analysis. Design facts by year (PSUs, single-PSU strata, weight variation) are in
`results/tables/design_structure.csv`.

All performance metrics are **population estimates**: every sum is weighted by the expansion factor
(`src/evaluation/metrics.py`).

## 3. Uncertainty: Rao-Wu rescaled bootstrap

Intervals come from the rescaled bootstrap of Rao and Wu (`src/validation/bootstrap.py`). In each of 1,000
replicates and each stratum h with n_h PSUs, n_h - 1 PSUs are drawn with replacement and the weight of every
person in PSU j is multiplied by (n_h / (n_h - 1)) x r_hj, r_hj being the number of draws of PSU j. Metrics
are recomputed with each replicate weight vector; 95% intervals are the 2.5th and 97.5th percentiles.

* Whole PSUs are resampled inside strata, so clustering and stratification are respected. A bootstrap of
  persons would ignore both.
* Strata with a single PSU are merged into the smallest other stratum before resampling (only relevant for
  the department x `ESTRATO` variant; the number merged is reported).
* When several years are pooled, a cluster code revisited in different years is one PSU, so dependence
  created by the panel component stays inside the resampling unit.
* Comparisons between models, strategies or groups are **paired**: both quantities are computed with the
  same replicate weights and the interval of the difference is reported.
* The models are held fixed, so intervals reflect test-sample uncertainty, not the variability of training.
  Training variability is examined separately for the explanations (refits on cluster-bootstrap samples).
* First-stage sampling fractions are treated as negligible.

For weighted proportions (prevalence), linearised standard errors with correct domain handling are also
reported and agree with the bootstrap (`results/tables/prevalence_by_year.csv`).

## 4. Metrics

With w_i the expansion factor, p_i the prediction and l_i = logit(p_i):

| Metric | Definition | Reading |
|---|---|---|
| ROC-AUC | weighted probability that a non-enrolled adolescent has a higher score than an enrolled one (ties count one half) | discrimination |
| PR-AUC | average precision, sum over thresholds of (R_k - R_{k-1}) P_k | discrimination under a rare outcome; its chance level is the prevalence |
| Brier score | sum w_i (p_i - y_i)^2 / sum w_i | overall accuracy of the probabilities |
| Calibration-in-the-large (CITL) | intercept a of logit P(y=1) = a + l_i (slope fixed at 1) | a > 0: risk is under-predicted; a < 0: over-predicted; 0 is ideal |
| Calibration slope | b in logit P(y=1) = a' + b l_i | b < 1: predictions too extreme; b > 1: too timid; 1 is ideal |
| O/E | observed prevalence / mean prediction | same direction as CITL |
| Recall, precision, specificity, F1 | from the weighted confusion matrix at a fixed threshold | operating point |

The sign conventions are covered by unit tests (`tests/test_metrics.py`). The flagging threshold is the
weighted 90th percentile of **out-of-fold** risk in 2017-2019, fixed before looking at test years. The
original project used in-sample predictions for this threshold, which are over-fitted for tree models and
give a higher cut-off.

## 5. Models and tuning

L2 logistic regression (standardised inputs), random forest and XGBoost; 26 predictors, all measured in the
same interview (`docs/data_dictionary.md`). Hyper-parameters are selected by **nested** grouped
cross-validation on 2017-2019 (outer 5 folds, inner 3 folds, survey-weighted PR-AUC), with folds formed on
the cluster code pooled across years so that dwellings revisited in another year never straddle folds. The
final configuration is selected by grouped 5-fold CV on the whole training period. Test years are never
used for tuning. Grids are in `configs/config.json`; choices per fold are in
`results/metrics/hyperparameters.json`.

**Survey weights in training.** Each model is trained with and without the expansion factors (rescaled to
mean 1). Unweighted training estimates risk in the sample, where rural clusters are over-represented;
weighted training targets the population risk at the cost of a larger variance. Both are evaluated with the
same weighted metrics and compared with paired intervals (`results/metrics/E2_paired_differences.csv`).

## 6. Experiments

| ID | Question | Design | Script |
|---|---|---|---|
| E1 | In-period performance | Nested grouped CV, 2017-2019 | `04`, `05` |
| E2 | Does a frozen model hold after the shock? | Train 2017-2019, test each year 2020-2025, no refitting. An in-year grouped CV gives the performance a fresh model reaches in the same year | `05` |
| E3 | Is updating worth it? | (A) fit on 2022-2024, test 2025: intercept, intercept + slope, retraining. (B) rolling origin: recalibration on year t-1 or retraining on earlier years, test year t | `06` |
| E4 | Is reliability equal across groups? | Area, poverty, mother tongue, sex and intersections with at least 50 events; three periods; paired gaps; measured effect of group-specific intercepts | `07` |
| E5 | How does the model use the predictors? | TreeSHAP by period, year and group; family-level attributions; stability over 30 cluster-bootstrap refits; permutation importance | `08` |
| E6 | What does performance depend on? | Refits without predictor families; single-variable benchmark | `09` |
| L1 | Is the status target contaminated by past non-enrollment? | Persistent vs incident non-enrollment (P303); incident-only models | `09` |
| L2 | Does the score anticipate next-year non-enrollment? | Panel pairs (t, t+1) linked by P201P | `09b` |

## 7. Shift diagnostics

Prior shift is the ratio of prevalences. Covariate shift is summarised by the out-of-fold ROC-AUC of a
classifier that separates the training period from each test year (0.5 means indistinguishable) and by the
population stability index of each predictor. Concept shift cannot be observed directly; it is bounded by
comparing the frozen model with the in-year reference of E2 and with the retrained models of E3.

## 8. Explanations

SHAP values are computed with TreeSHAP on the log-odds scale. Age-grade distortion equals
age - 6 - years of schooling, so credit among these three predictors is not identified; attributions are
therefore also reported for predictor **families** (sum of SHAP values within the family, then mean
absolute value). Importance describes how the model uses a predictor and carries no causal meaning.

## 9. Known limits

* The main target is contemporaneous. About two in three non-enrolled adolescents (68% unweighted, 63-73% weighted by year) were already out of school the
  previous year; results for incident and prospective targets are reported next to the main ones.
* The cross-sectional files carry no longitudinal weights; panel results describe the panel subsample.
* Predictors describe the household and the adolescent; there are no school-side variables.
* 2020 combines the lockdown with a change to telephone interviewing; the two cannot be separated.
* The household-internet item is not comparable from 2023 and is excluded from the main model.
