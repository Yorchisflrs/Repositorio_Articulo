"""Consistency between the manuscript and the result files.

These tests need no microdata: they read paper/manuscript.tex, paper/generated/ and results/.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
NUMBERS = PAPER / "generated" / "numbers.tex"
needs_outputs = pytest.mark.skipif(not NUMBERS.exists(), reason="run scripts/10_generate_outputs.py first")


def body_text() -> str:
    """Prose of the manuscript: comments, tables, figures, captions, labels and citations removed."""
    t = "\n".join(l for l in TEX.splitlines() if not l.lstrip().startswith("%"))
    t = t.split("\\begin{document}")[1].split("\\bibliographystyle")[0].split("\\section*{Acknowledgment}")[0]
    t = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", t)
    return t


@needs_outputs
def test_every_number_macro_is_defined():
    defined = set(re.findall(r"\\setval\{([^}]+)\}", NUMBERS.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\V\{([^}]+)\}", TEX))
    assert used, "the manuscript should read its numbers from the generated macro file"
    assert not used - defined, f"numbers used but not generated: {sorted(used - defined)}"


@needs_outputs
def test_number_registry_points_to_existing_files():
    reg = pd.read_csv(ROOT / "results" / "tables" / "manuscript_numbers.csv")
    assert reg.key.is_unique
    for src in reg.source.unique():
        assert (ROOT / src).exists(), f"source file missing: {src}"


def test_no_hand_typed_statistics_in_the_prose():
    """Decimals such as 0.930 or 4.51 must come from macros. Whole numbers that describe the design
    (years, ages, counts of folds or replicates) are allowed."""
    prose = re.sub(r"\\V\{[^}]+\}", " ", body_text())
    prose = re.sub(r"\$[^$]*\$", " ", prose)
    found = re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", prose)
    assert not found, f"decimal numbers typed by hand: {found}"
    # whole numbers allowed in the prose: years, ages, design constants and facts about cited work
    allowed = {str(y) for y in range(2017, 2026)} | {"12", "16", "17", "26", "30", "50", "90", "1,000", "110", "36"}
    ints = set(re.findall(r"(?<![\w.{-])\d[\d,]*\d|(?<![\w.{-])\d(?![\w}.])", prose))
    ints = {i.rstrip(",") for i in ints if len(i.rstrip(",")) > 1}
    assert ints <= allowed, f"whole numbers typed by hand: {sorted(ints - allowed)}"


def test_abstract_has_at_most_250_words():
    abstract = TEX.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    if NUMBERS.exists():
        values = dict(re.findall(r"\\setval\{([^}]+)\}\{(.*)\}\s*$", NUMBERS.read_text(encoding="utf-8"), flags=re.M))
        abstract = re.sub(r"\\V\{([^}]+)\}", lambda m: values.get(m.group(1), "0"), abstract)
    abstract = re.sub(r"\\[a-zA-Z]+|[{}$]", " ", abstract).replace("--", "-")
    words = [w for w in abstract.split() if re.search(r"[A-Za-z0-9]", w)]
    assert 100 <= len(words) <= 250, f"abstract has {len(words)} words"


def test_tables_are_generated_not_typed():
    """Every tabular in the manuscript must come from paper/generated/."""
    assert "\\begin{tabular}" not in TEX
    for name in re.findall(r"\\input\{generated/(table_[a-z_]+)\}", TEX):
        assert (PAPER / "generated" / f"{name}.tex").exists() or not NUMBERS.exists(), name


def test_no_unsupported_novelty_claims():
    low = body_text().lower()
    for phrase in ("first study", "novel framework", "for the first time", "state-of-the-art", "unprecedented", "early warning system that"):
        assert phrase not in low, phrase


def test_every_citation_has_a_bib_entry():
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", TEX) for k in group.split(",")}
    assert cited and not cited - keys, sorted(cited - keys)


def test_page_limit():
    pdf = PAPER / "manuscript.pdf"
    if not pdf.exists() or shutil.which("pdfinfo") is None:
        pytest.skip("compile the manuscript and install poppler to check the page count")
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", out).group(1))
    assert pages <= 7, f"the manuscript has {pages} pages; the limit is 7"


@pytest.mark.xfail(reason="the author still has to supply the contact e-mail (EMAIL-PENDIENTE)", strict=False)
def test_author_email_has_been_supplied():
    assert "EMAIL-PENDIENTE" not in TEX
