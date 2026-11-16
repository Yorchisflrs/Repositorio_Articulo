# Changelog

## 2.0.0 - 2026-11-16

Full reconstruction of the analysis. The original project is kept unchanged in `legacy/`.

### Scope of the claims
- The outcome is now described as current non-enrollment (a prevalent condition), not as dropout or early
  warning. Item P303 shows that about two in three non-enrolled adolescents were already out of school the
  year before.
- New outcome-validity analysis: persistent and incident non-enrollment evaluated separately, a model
  trained on the incident outcome, and a prospective outcome one year ahead built with the person panel code.

### Statistical design
- Rao-Wu rescaled bootstrap with 1,000 replicates, clusters resampled within strata, for every metric
  (the original resampled clusters ignoring strata, 200 replicates, discrimination metrics only).
- All comparisons between models, strategies and groups are paired on the same replicate weights.
- Nested grouped cross-validation (5 outer, 3 inner folds); groups are cluster codes pooled across years.
- Flagging threshold taken from out-of-fold predictions (the original used in-sample predictions).
- Each model trained with and without survey weights and compared.
- Same-year reference model to separate a harder year from a stale model.
- Updating experiment with a common fitting set (2022-2024) and a rolling-origin design.
- Subgroup audit with intervals, paired gaps, a minimum of 50 events per group and three periods.
- SHAP by predictor family with stability over 30 cluster-bootstrap refits and permutation importance.
- Ablation with explicitly defined predictor sets and a single-variable benchmark.
- Two-sided bootstrap p-values are undefined (not 2.0) when two metrics are identical by construction.

### Data
- Pipeline starts from the household roster; join audit and sample-flow table written to disk.
- Expenditure quintiles computed with weights within year and a stable ordering (3 of 84,502 values differ
  from the original).
- Validation tables: value labels by year, month artefact, internet item, interview mode, design structure,
  overlap of test years with the training period through the panel.

### Engineering
- Code organised as a package (`src/`) with ten numbered scripts, one configuration file, relative paths,
  pinned requirements and unit tests.
- One seed with per-script offsets; no shared global random state.
- Tables, figures and every number of the manuscript are generated from result files
  (`scripts/10_generate_outputs.py`, `paper/generated/numbers.tex`, `results/tables/manuscript_numbers.csv`).

### Manuscript
- Rewritten around the new results; random forest reported next to the other two models with the same
  metrics; references moved to BibTeX and limited to verified entries.

## 1.1.0 - 2026-11 (previous review)
- Three rounding errors fixed in the text; reference list completed and two mis-attributed citations
  corrected; Spanish version aligned with the English one.

## 1.0.0 - original project
- Nine scripts, seven result tables, five figures and the first manuscript (kept in `legacy/`).
