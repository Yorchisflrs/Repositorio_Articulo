"""Regression tests on the processed ENAHO files (skipped when the microdata have not been processed)."""
import pandas as pd

from src.config import FEATURES, PERSON_KEYS, TRAIN_YEARS
from tests.conftest import needs_data, processed


@needs_data
def test_analytic_population_size():
    a = pd.read_parquet(processed("analytic.parquet"))
    assert len(a) == 84502 and int(a.y.sum()) == 3607
    assert a.groupby("year").size().to_dict()[2017] == 9892 and a[a.year.isin(TRAIN_YEARS)].y.sum() == 1382


@needs_data
def test_no_duplicate_persons_and_valid_ranges():
    a = pd.read_parquet(processed("analytic.parquet"))
    assert not a.duplicated(["year"] + PERSON_KEYS).any()
    assert a.age.between(12, 17).all() and a.mes.between(4, 12).all() and (a.factora07 > 0).all()
    assert a[FEATURES].isna().mean().max() < 0.05
    assert ((a.overage == (a.age - 6 - a.school_years).clip(-2, 12))).all()


@needs_data
def test_matches_the_original_pipeline_outputs():
    """Design-based prevalence must equal the table shipped with the original project (legacy/original_tables)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    old = pd.read_csv(root / "legacy" / "original_tables" / "E0_prevalence_design_based.csv")
    new = pd.read_csv(root / "results" / "tables" / "prevalence_by_year.csv")
    o = old[old.area == "National"].set_index("year").prev; n = new[new.area == "National"].set_index("year").prev
    assert (o - n).abs().max() < 1e-12
