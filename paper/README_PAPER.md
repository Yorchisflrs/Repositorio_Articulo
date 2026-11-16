# Manuscript

`manuscript.tex` is the source of the paper (IEEE conference template, `IEEEtran` class). It does not contain
typed statistics or tables: it reads

* `generated/numbers.tex`: one `\setval{key}{value}` per number, written by `scripts/10_generate_outputs.py`;
  the text prints them with `\V{key}`, and an undefined key stops compilation;
* `generated/table_*.tex`: the five tables, written by the same script;
* `../results/figures/figure_01.pdf` and `figure_02.pdf`: the figures, written by the same script;
* `references.bib`: the bibliography, formatted by `IEEEtran.bst`.

## Build

```bash
python ../scripts/10_generate_outputs.py    # only if results changed
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
pdfinfo manuscript.pdf | grep Pages         # 7 pages, limit 7
```

`tools/flatten_tex.py` writes `build/manuscript_flat.tex`, a single self-contained file with the numbers,
tables and bibliography inlined, which compiles to the same PDF and is what `03_CODIGO_LATEX/main.tex` in the
delivery contains. `tools/tex2json.py` and `tools/build_docx.js` turn that file into the Word version
(`node tools/build_docx.js build/manuscript.json ../results/figures ARTICULO_FINAL.docx`, needs the npm
package `docx`).

## Checks

`../tests/test_manuscript.py`: every `\V{}` key exists; no decimal number and no unexpected whole number is
typed in the prose; the abstract has at most 250 words (it has 219); every citation has a BibTeX entry; no
unsupported novelty phrases; the compiled PDF has at most seven pages. `../tests/test_claims.py` checks the
qualitative statements of the text against the result files.

## Pending before submission

* **Author e-mail.** The author block shows `EMAIL-PENDIENTE`. Replace it in the `\authoremail` definition at
  the top of `manuscript.tex`. The manuscript is not ready for submission until this is done
  (`test_author_email_has_been_supplied` is marked as an expected failure until then).
* **Page limit of the venue.** The ICALTER call consulted earlier in this project mentioned six pages; this
  version follows the seven-page limit set for this revision and has exactly seven pages including
  references. If the venue enforces six, the shortest path is to move Fig. 2 and Table V to a supplement.
* **ORCID and funding statements**, if the venue requires them, must come from the author.

## References: verification status

Checked against the publisher record, DOI resolver or an indexed catalogue during this project.

| Key | Status |
|---|---|
| rodriguez2023, adelman2018, jacha2024, jimenez2024, gallego2025, moreno2012, vancalster2019, baker2022, lakkaraju2015, sansone2019, romero2020, khatun2025, alcazar2009, anaya2021, breiman2001, chen2016, lumley2004, saito2015, lundberg2020, raowu1988, raowuyue1992, davis2017, steyerberg2004, macnell2023, deho2024, kleinberg2017, aas2021, christodoulou2019 | Verified (authors, title, venue, volume, pages, DOI) |
| bowers2013 | DOI and content verified; volume and pages not confirmed online |
| lundberg2017 | Title, venue and year verified; page range not confirmed online |
| cavero2011 | Working paper number and date verified; only the abstract could be read, and the paper cites it only for what the abstract states |
| inei | Portal URL of the INEI microdata service |

What each source is cited for was checked against the source in the earlier review of this project
(`legacy/` and the delivery's audit folder): two citations of the original manuscript attributed claims the
sources do not make and were rewritten.
