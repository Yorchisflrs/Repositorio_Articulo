"""Readers for the official ENAHO Stata files (INEI).

The annual zips of modules 01, 02, 03, 18 and 34 must be unpacked into ``<raw>/<year>/``.
Column names change case between years, so every reader lower-cases them.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyreadstat

# file pattern of each module inside <raw>/<year>/
MODULES = {
    "100": "enaho01-*-100.dta",       # dwelling and household
    "200": "enaho01-*-200.dta",       # household roster
    "300": "enaho01a-*-300.dta",      # education
    "612": "enaho01-*-612.dta",       # household durables
    "sumaria": "sumaria-[0-9][0-9][0-9][0-9].dta",
}
# INEI download codes (https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/<code>-Modulo<NN>.zip)
INEI_CODES = {2017: 603, 2018: 634, 2019: 687, 2020: 737, 2021: 759, 2022: 784, 2023: 906, 2024: 966, 2025: 1031}
INEI_MODULES = {"100": "01", "200": "02", "300": "03", "612": "18", "sumaria": "34"}


def module_path(raw: Path, year: int, module: str) -> Path:
    hits = sorted(glob.glob(str(Path(raw) / str(year) / MODULES[module])))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one file for module {module} in {year}, found {hits}")
    return Path(hits[0])


def read_module(raw: Path, year: int, module: str, cols: list[str]) -> pd.DataFrame:
    """Read the requested columns (those that exist that year), lower-casing names."""
    f = module_path(raw, year, module)
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    real = {c.lower(): c for c in meta.column_names}
    df, _ = pyreadstat.read_dta(str(f), usecols=[real[c] for c in cols if c in real])
    df.columns = [c.lower() for c in df.columns]
    return df


def module_meta(raw: Path, year: int, module: str):
    """Variable labels and value labels of a module (lower-cased names)."""
    _, meta = pyreadstat.read_dta(str(module_path(raw, year, module)), metadataonly=True)
    labels = {c.lower(): l for c, l in zip(meta.column_names, meta.column_labels)}
    values = {k.lower(): v for k, v in meta.variable_value_labels.items()}
    return labels, values, meta.number_rows
