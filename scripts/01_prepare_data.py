"""Step 1 - Read the official ENAHO modules and build one person-level file per year (ages 10-18).

Inputs : <raw>/<year>/ Stata files of modules 100, 200, 300, 612 and sumaria (INEI, 2017-2025)
Outputs: data/processed/persons_10_18.parquet
         results/logs/join_audit.csv      (rows, key duplicates and match rates of every merge)
The roster (module 200) is the base table, so persons without an education record stay visible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import HH_KEYS, LOG_DIR, PERSON_KEYS, PROC_DIR, RAW_DIR, YEARS  # noqa: E402
from src.data.enaho import read_module  # noqa: E402

C200 = PERSON_KEYS + ["mes", "ubigeo", "dominio", "estrato", "p201p", "p203", "p204", "p205", "p206", "p207", "p208a"]
C300 = PERSON_KEYS + ["p300a", "p301a", "p301b", "p301c", "p303", "p304a", "p305", "p306", "p307", "p308a", "p313",
                      "factora07", "factor07"]
C100 = HH_KEYS + ["result", "panel", "tipenc", "nconglome", "p1141", "p1142", "p1144", "p114b1", "p114b2", "p114b3",
                  "nbi1", "tipoentrevista"]
CSUM = HH_KEYS + ["mieperho", "pobreza", "gashog2d", "linea", "factor07"]

frames, audit = [], []
for y in YEARS:
    ros = read_module(RAW_DIR, y, "200", C200)
    edu = read_module(RAW_DIR, y, "300", C300).rename(columns={"factor07": "factor07_300"})
    hog = read_module(RAW_DIR, y, "100", C100)
    eq = read_module(RAW_DIR, y, "612", HH_KEYS + ["p612n", "p612"])
    suma = read_module(RAW_DIR, y, "sumaria", CSUM)
    pc = eq[eq.p612n == 7].groupby(HH_KEYS, as_index=False).p612.min().rename(columns={"p612": "computer"})
    # household head: sex and age from the roster, education level from module 300
    head = (ros[ros.p203 == 1][PERSON_KEYS + ["p207", "p208a"]]
            .merge(edu[PERSON_KEYS + ["p301a"]], on=PERSON_KEYS, how="left")
            .rename(columns={"p207": "head_sex", "p208a": "head_age", "p301a": "head_educ"}))
    n_heads_dup = int(head.duplicated(HH_KEYS).sum())
    head = head.drop_duplicates(HH_KEYS)[HH_KEYS + ["head_sex", "head_age", "head_educ"]]
    kids = (ros[(ros.p208a < 18) & (ros.p204 == 1)].groupby(HH_KEYS).size().rename("n_kids").reset_index())

    base = ros[ros.p208a.between(10, 18)].copy()
    n0 = len(base)
    d = base.merge(edu, on=PERSON_KEYS, how="left", indicator="m300")
    d = d.merge(hog, on=HH_KEYS, how="left", indicator="m100")
    d = d.merge(pc, on=HH_KEYS, how="left").merge(suma, on=HH_KEYS, how="left", indicator="msum")
    d = d.merge(head, on=HH_KEYS, how="left").merge(kids, on=HH_KEYS, how="left")
    assert len(d) == n0, "a merge duplicated rows"
    d["year"] = y
    audit.append(dict(
        year=y, rows_200=len(ros), rows_300=len(edu), rows_100=len(hog), rows_612=len(eq), rows_sumaria=len(suma),
        dup_keys_200=int(ros.duplicated(PERSON_KEYS).sum()), dup_keys_300=int(edu.duplicated(PERSON_KEYS).sum()),
        dup_keys_100=int(hog.duplicated(HH_KEYS).sum()), dup_keys_sumaria=int(suma.duplicated(HH_KEYS).sum()),
        dup_keys_612_item7=int(eq[eq.p612n == 7].duplicated(HH_KEYS).sum()), households_with_two_heads=n_heads_dup,
        persons_10_18=n0, match_300=float((d.m300 == "both").mean()), match_100=float((d.m100 == "both").mean()),
        match_sumaria=float((d.msum == "both").mean()), match_612=float(d.computer.notna().mean()),
        match_head=float(d.head_age.notna().mean()), rows_300_not_in_200=int((~edu.set_index(PERSON_KEYS).index.isin(ros.set_index(PERSON_KEYS).index)).sum()),
    ))
    for c in ("m300", "m100", "msum"):
        d[c] = (d[c] == "both").astype("int8")
    frames.append(d)
    print(y, n0, f"match300={audit[-1]['match_300']:.4f} match100={audit[-1]['match_100']:.4f} sumaria={audit[-1]['match_sumaria']:.4f}")

df = pd.concat(frames, ignore_index=True)
for c in ("mes", "dominio", "estrato"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ("conglome", "vivienda", "hogar", "codperso", "ubigeo", "p201p", "nconglome"):
    if c in df:
        df[c] = df[c].astype(str).str.strip()
df.to_parquet(PROC_DIR / "persons_10_18.parquet")
pd.DataFrame(audit).to_csv(LOG_DIR / "join_audit.csv", index=False)
print("saved", df.shape)
