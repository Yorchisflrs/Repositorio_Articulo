"""Shared fixtures. Tests that need the processed ENAHO files are skipped when those files are absent."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def toy():
    rng = np.random.default_rng(7); n = 6000
    p = np.round(rng.beta(1, 14, n), 3)                      # rounded on purpose: creates ties
    y = (rng.random(n) < np.clip(p * 1.4, 0, 1)).astype(int)
    w = rng.gamma(2.0, 150.0, n)
    strata = rng.integers(0, 6, n); psu = strata * 1000 + rng.integers(0, 60, n)
    return dict(y=y, p=p, w=w, strata=strata, psu=psu)


def processed(name: str) -> Path:
    return ROOT / "data" / "processed" / name


needs_data = pytest.mark.skipif(not processed("analytic.parquet").exists(), reason="processed ENAHO files not available")
