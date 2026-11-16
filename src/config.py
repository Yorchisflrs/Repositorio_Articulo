"""Central configuration: paths, years, seeds and variable lists shared by every script."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))

RAW_DIR = Path(os.environ.get("ENAHO_RAW", ROOT / CFG["paths"]["raw"]))
PROC_DIR = ROOT / CFG["paths"]["processed"]
RES_DIR = ROOT / "results"
TAB_DIR = RES_DIR / "tables"
FIG_DIR = RES_DIR / "figures"
MET_DIR = RES_DIR / "metrics"
LOG_DIR = RES_DIR / "logs"
MODEL_DIR = PROC_DIR / "models"
for _d in (PROC_DIR, TAB_DIR, FIG_DIR, MET_DIR, LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED: int = CFG["seed"]
YEARS = list(range(CFG["years"]["first"], CFG["years"]["last"] + 1))
TRAIN_YEARS = CFG["years"]["train"]
TEST_YEARS = CFG["years"]["test"]
N_BOOT: int = CFG["bootstrap"]["replicates"]

HH_KEYS = ["conglome", "vivienda", "hogar"]
PERSON_KEYS = HH_KEYS + ["codperso"]

# Predictor families (used by the ablation and leakage analyses)
FAM_INDIVIDUAL = ["age", "female", "indig_tongue", "child_of_head"]
FAM_TRAJECTORY = ["school_years", "overage"]
FAM_DIGITAL = ["hh_mobile", "hh_computer"]
FAM_HOUSEHOLD = ["poor", "extreme_poor", "exp_quintile", "head_educ_years", "head_female", "head_age",
                 "hh_size", "n_kids", "inadequate_housing"]
FAM_AREA = ["rural"] + [f"dom_{k}" for k in range(1, 9)]
FEATURES = (["age", "female", "indig_tongue", "school_years", "overage", "child_of_head"] + FAM_DIGITAL
            + FAM_HOUSEHOLD[:-1] + ["rural", "inadequate_housing"] + [f"dom_{k}" for k in range(1, 9)])
assert len(FEATURES) == 26 and len(set(FEATURES)) == 26

FEATURE_LABELS = {
    "age": "Age", "female": "Female", "indig_tongue": "Indigenous mother tongue", "school_years": "Years of schooling",
    "overage": "Age-grade distortion", "child_of_head": "Child of household head", "hh_mobile": "Mobile phone at home",
    "hh_computer": "Computer at home", "poor": "Poor household", "extreme_poor": "Extremely poor household",
    "exp_quintile": "Expenditure quintile", "head_educ_years": "Head education (yrs)", "head_female": "Female head",
    "head_age": "Head age", "hh_size": "Household size", "n_kids": "Children in household", "rural": "Rural area",
    "inadequate_housing": "Inadequate housing", "dom_1": "Domain: North coast", "dom_2": "Domain: Central coast",
    "dom_3": "Domain: South coast", "dom_4": "Domain: North highlands", "dom_5": "Domain: Central highlands",
    "dom_6": "Domain: South highlands", "dom_7": "Domain: Rainforest", "dom_8": "Domain: Metropolitan Lima",
}
