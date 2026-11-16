"""Convert the flattened manuscript (build/manuscript_flat.tex) into a JSON structure for the Word builder.

Only the LaTeX constructs used in the manuscript are supported; anything else raises an error, so a new
construct cannot slip silently into the Word version.

usage: python tools/tex2json.py build/manuscript_flat.tex build/manuscript.json   (run from paper/)
"""
import json
import re
import sys
import unicodedata

src, out = sys.argv[1], sys.argv[2]
tex = open(src, encoding="utf-8").read()
tex = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
body = tex.split("\\begin{document}")[1].split("\\end{document}")[0]


def braced(s: str, i: int):
    """s[i] == '{'; return (content, index after the closing brace)."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


# ---------- numbering for \cite and \ref
bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", body)
cite_no = {k: i + 1 for i, k in enumerate(bib_keys)}
roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
ref_no = {}
t_i = f_i = s_i = ss_i = 0
for m in re.finditer(r"\\begin\{(table|figure)\*?\}|\\section\{|\\subsection\{|\\label\{([^}]+)\}", body):
    if m.group(1) == "table":
        t_i += 1; cur = roman[t_i - 1]
    elif m.group(1) == "figure":
        f_i += 1; cur = str(f_i)
    elif m.group(0).startswith("\\section"):
        s_i += 1; ss_i = 0; cur = roman[s_i - 1]
    elif m.group(0).startswith("\\subsection"):
        ss_i += 1; cur = f"{roman[s_i - 1]}-{chr(64 + ss_i)}"
    else:
        ref_no[m.group(2)] = cur

GREEK = {"\\Delta": "\u0394", "\\times": "\u00d7", "\\le": "\u2264", "\\ge": "\u2265", "\\approx": "\u2248"}
ACCENT = {"'": "\u0301", "~": "\u0303", '"': "\u0308", "`": "\u0300", "^": "\u0302"}


def plain(s: str) -> str:
    """LaTeX inline text (no formatting commands left) -> unicode."""
    s = s.replace("{\\o}", "\u00f8").replace("\\o ", "\u00f8")
    s = re.sub(r"\\(['~\"`^])\{?\\i\}?", lambda m: unicodedata.normalize("NFC", "i" + ACCENT[m.group(1)]), s)
    s = re.sub(r"\{?\\(['~\"`^])\{?([A-Za-z])\}?\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENT[m.group(1)]), s)
    s = s.replace("$-$", "\u2212").replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace("`", "\u2018").replace("'", "\u2019")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\,", "\u2009").replace("~", "\u00a0").replace("\\ ", " ")
    s = s.replace("{,}", ",").replace("\\quad", "\u2003").replace("\\_", "_").replace("\\#", "#")
    s = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + "], [".join(str(cite_no[k.strip()]) for k in m.group(1).split(",")) + "]", s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: ref_no[m.group(1)], s)
    s = re.sub(r"\\url\{([^}]+)\}", r"\1", s)
    while re.search(r"\{([^{}\\]*)\}", s):                  # BibTeX case-protection braces
        s = re.sub(r"\{([^{}\\]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s


def math(s: str) -> list:
    """Very small math subset: letters italic, sub/superscripts, a few symbols."""
    s = s.strip()
    if s == "-":
        return [{"t": "\u2212"}]
    m = re.fullmatch(r"\^\{\\mathrm\{([a-z])\}\}", s)
    if m:
        return [{"t": m.group(1), "sup": True}]
    out = []
    i = 0
    while i < len(s):
        hit = next((k for k in GREEK if s.startswith(k, i)), None)
        if hit:
            out.append({"t": GREEK[hit]}); i += len(hit)
        elif s[i] == "_":
            sub, i = (braced(s, i + 1) if s[i + 1] == "{" else (s[i + 1], i + 2))
            out.append({"t": sub, "sub": True, "i": sub.isalpha()})
        elif s[i].isalpha():
            out.append({"t": s[i], "i": True}); i += 1
        elif s[i] == "-":
            out.append({"t": "\u2212"}); i += 1
        elif s[i] in "0123456789+=<>()., |":
            out.append({"t": s[i]}); i += 1
        else:
            raise ValueError(f"unsupported math: {s!r}")
    return out


def runs(s: str) -> list:
    """LaTeX inline text -> list of runs {t, i, sup, sub}."""
    s = s.strip()
    out, i, buf = [], 0, ""

    def flush():
        nonlocal buf
        if buf:
            out.append({"t": plain(buf)}); buf = ""

    while i < len(s):
        if s.startswith("\\textit{", i) or s.startswith("\\emph{", i):
            flush(); k = s.index("{", i); inner, i = braced(s, k)
            out.extend({**r, "i": True} for r in runs(inner))
        elif s.startswith("$-$", i):
            buf += "$-$"; i += 3
        elif s[i] == "$":
            flush(); j = s.index("$", i + 1); out.extend(math(s[i + 1:j])); i = j + 1
        else:
            buf += s[i]; i += 1
    flush()
    left = [r["t"] for r in out if "\\" in r["t"] or "$" in r["t"] or "{" in r["t"]]
    if left:
        raise ValueError(f"unconverted LaTeX: {left}")
    # keep spaces between runs; drop empty runs
    return [r for r in out if r["t"] != ""]


doc = {"blocks": []}
B = doc["blocks"]
# ---------- front matter
doc["title"] = plain(braced(body, body.index("\\title{") + 6)[0])
author, _ = braced(body, body.index("\\author{") + 7)
doc["author_name"] = plain(braced(author, author.index("\\IEEEauthorblockN{") + 17)[0])
aff = braced(author, author.index("\\IEEEauthorblockA{") + 17)[0]
doc["affiliation"] = [runs(l) for l in re.split(r"\\\\", aff) if l.strip()]
doc["abstract"] = runs(body.split("\\begin{abstract}")[1].split("\\end{abstract}")[0])
doc["keywords"] = runs(body.split("\\begin{IEEEkeywords}")[1].split("\\end{IEEEkeywords}")[0])
doc["lang"] = "en"

main = body.split("\\end{IEEEkeywords}")[1].split("\\begin{thebibliography}")[0]
bib = body.split("\\begin{thebibliography}")[1].split("\\end{thebibliography}")[0]

pattern = re.compile(r"\\(section\*?|subsection)\{|\\begin\{(table|figure)(\*?)\}(\[[^\]]*\])?")


def flush_text(chunk: str) -> None:
    for para in re.split(r"\n\s*\n", chunk):
        if para.strip():
            B.append({"type": "p", "runs": runs(para)})


def parse_table(t: str, wide: bool) -> dict:
    cap = runs(braced(t, t.index("\\caption{") + 8)[0])
    spec, k = braced(t, t.index("\\begin{tabular}") + 15)
    tab = t[k:t.index("\\end{tabular}")]
    cols = re.findall(r"([lcr])", re.sub(r"@\{[^}]*\}|>\{[^}]*\}", "", spec))
    rows, rule = [], "top"
    for line in tab.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("\\toprule"):
            rule = "top"; continue
        if line.startswith("\\midrule"):
            rule = "mid"; continue
        if line.startswith("\\addlinespace"):
            rule = "space"; continue
        if line.startswith("\\bottomrule"):
            rows[-1]["rule_after"] = "bottom"; continue
        if line.endswith("\\\\"):
            line = line[:-2]
        cells = []
        for c in line.split("&"):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{(?:@\{\})?(\w)(?:@\{\})?\}\{(.*)\}$", c)
            if m:
                cells.append({"runs": runs(m.group(3)) if m.group(3) else [], "span": int(m.group(1)), "align": m.group(2)})
            else:
                cells.append({"runs": runs(c) if c else [], "span": 1})
        rows.append({"cells": cells, "rule_before": rule})
        rule = None
    note = re.search(r"\\parbox\{[^}]*\}\{\\scriptsize (.*)\}\s*$", t.strip(), flags=re.S)
    lab = re.search(r"\\label\{([^}]+)\}", t).group(1)
    return {"type": "table", "num": ref_no[lab], "caption": cap, "cols": [{"align": a} for a in cols], "rows": rows,
            "wide": wide, "note": runs(note.group(1)) if note else None}


def parse_figure(t: str, wide: bool) -> dict:
    w = re.search(r"\\includegraphics\[width=([\d.]*)\\(?:columnwidth|textwidth)\]\{([^}]+)\}", t)
    lab = re.search(r"\\label\{([^}]+)\}", t).group(1)
    return {"type": "figure", "num": ref_no[lab], "file": re.sub(r"(\.pdf)?$", ".png", w.group(2), count=1), "width": float(w.group(1) or 1),
            "caption": runs(braced(t, t.index("\\caption{") + 8)[0]), "wide": wide}


pos = 0
sec_no = sub_no = 0
for m in pattern.finditer(main):
    if m.start() < pos:
        continue
    flush_text(main[pos:m.start()])
    if m.group(1):
        title, pos = braced(main, m.end() - 1)
        lab = re.match(r"\s*\\label\{[^}]+\}", main[pos:])
        if lab:
            pos += lab.end()
        if m.group(1) == "section":
            sec_no += 1; sub_no = 0
            B.append({"type": "h1", "text": plain(title), "num": roman[sec_no - 1]})
        elif m.group(1) == "section*":
            B.append({"type": "h1", "text": plain(title), "num": None})
        else:
            B.append({"type": "h2", "text": plain(title), "num": chr(ord("A") + sub_no)}); sub_no += 1
    else:
        env = m.group(2) + m.group(3); end = main.index("\\end{" + env + "}", m.end())
        t = main[m.end():end]; pos = end + len("\\end{" + env + "}")
        B.append(parse_table(t, bool(m.group(3))) if m.group(2) == "table" else parse_figure(t, bool(m.group(3))))
flush_text(main[pos:])

# ---------- references (IEEEtran.bst output)
doc["references"] = []
for k, txt in re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", bib, flags=re.S):
    txt = re.sub(r"\\BIBentry(ALT|STD)interwordspacing", "", txt)
    txt = txt.replace("\\newblock", " ").replace("\\penalty0 ", "")
    txt = re.sub(r"\\bibinfo\{[^}]*\}\{([^}]*)\}", r"\1", txt)
    doc["references"].append({"n": cite_no[k], "runs": runs(txt)})

json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(src, "->", out, "| blocks", len(B), "| refs", len(doc["references"]),
      "| tables", sum(b["type"] == "table" for b in B), "| figures", sum(b["type"] == "figure" for b in B))
