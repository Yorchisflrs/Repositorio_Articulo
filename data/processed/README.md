# data/processed

Files written here by the pipeline (person-level, derived from ENAHO microdata) are not versioned and are not
part of the delivery: `persons_10_18.parquet`, `harmonized_10_18.parquet`, `analytic.parquet`,
`predictions.parquet`, `panel_pairs.parquet`, `shap_frozen_xgb.npz` and the fitted models in `models/`.
Run `scripts/01_prepare_data.py` to `scripts/09b_panel_prospective.py` to create them (see
`../README_DATA.md`).
