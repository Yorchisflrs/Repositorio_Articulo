"""Write a single-file version of the manuscript.

manuscript.tex reads its numbers from generated/numbers.tex, its tables from generated/table_*.tex and its
references from references.bib. Submission systems and the Word converter are easier to feed with one
self-contained file, so this script inlines the three: every \\V{key} is replaced by its value, every
\\input{generated/...} by the file, and \\bibliography{references} by the thebibliography block that
BibTeX wrote to manuscript.bbl. The output compiles to the same PDF.

usage: python tools/flatten_tex.py [manuscript.tex] [build/manuscript_flat.tex]   (run from paper/)
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "manuscript.tex")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "build/manuscript_flat.tex")
base = src.parent
tex = src.read_text(encoding="utf-8")

values = dict(re.findall(r"\\setval\{([^}]+)\}\{(.*)\}\s*$", (base / "generated" / "numbers.tex").read_text(encoding="utf-8"), flags=re.M))
used = set()


def value(m: re.Match) -> str:
    key = m.group(1)
    if key not in values:
        raise KeyError(f"manuscript number not defined: {key}")
    used.add(key)
    return values[key]


# drop the macro machinery (no longer needed once the values are inlined)
tex = re.sub(r"% Every number in the text.*?\\input\{generated/numbers\}\n", "", tex, flags=re.S)
tex = re.sub(r"\\input\{(generated/[^}]+)\}",
             lambda m: "\n".join(l for l in (base / (m.group(1) + ".tex")).read_text(encoding="utf-8").splitlines() if not l.startswith("%")),
             tex)
tex = re.sub(r"\\V\{([^}]+)\}", value, tex)
email = re.search(r"\\newcommand\{\\authoremail\}\{([^}]*)\}", tex)
if email:
    tex = tex.replace(email.group(0) + "\n", "").replace("\\authoremail", email.group(1))
    tex = re.sub(r"% Author e-mail:.*\n", "", tex)
bbl = (base / (src.stem + ".bbl")).read_text(encoding="utf-8")
tex = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}", lambda m: bbl.strip(), tex)
tex = re.sub(r"\n{3,}", "\n\n", tex)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tex, encoding="utf-8")
print(f"{src} -> {out} ({len(used)} numbers inlined)")
