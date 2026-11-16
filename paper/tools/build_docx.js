// Build an IEEE-conference-style Word document from the JSON written by tex2json.py.
// usage: node tools/build_docx.js build/manuscript.json <figures_dir> ARTICULO_FINAL.docx   (run from paper/)
// Requires the npm package "docx".
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, SectionType, TabStopType, VerticalAlign, LineRuleType,
} = require('docx');

const [, , inJson, figDir, outDocx] = process.argv;
const d = JSON.parse(fs.readFileSync(inJson, 'utf8'));
const L = { abstract: 'Abstract', keywords: 'Index Terms', table: 'TABLE', fig: 'Fig.', refs: 'References' };

const FONT = 'Times New Roman';
const MARGIN = { top: 1080, bottom: 1440, left: 900, right: 900 };
const PAGE = { size: { width: 12240, height: 15840 }, margin: MARGIN };
const TEXT_W = 12240 - MARGIN.left - MARGIN.right;   // 10440 DXA
const GAP = 360;
const COL_W = (TEXT_W - GAP) / 2;                    // 5040 DXA = 3.5 in
const pt = (x) => Math.round(x * 2);                 // half-points

function mkRuns(runs, base = {}) {
  return runs.map((r) => {
    const o = { text: r.t, font: FONT, size: base.size || pt(10), italics: !!(r.i || base.italics), bold: !!base.bold };
    if (r.sup) o.superScript = true;
    if (r.sub) o.subScript = true;
    if (base.smallCaps) o.smallCaps = true;
    if (r.t.includes('EMAIL-PENDIENTE')) o.highlight = 'yellow';
    return new TextRun(o);
  });
}
const P = (opts) => new Paragraph(opts);
const text = (runs) => runs.map((x) => x.t).join('');

function bodyPara(runs) {
  return P({ children: mkRuns(runs), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 288 },
    spacing: { before: 0, after: 0, line: 240, lineRule: LineRuleType.AUTO }, widowControl: true });
}
function h1(b) {
  const label = b.num ? `${b.num}. ${b.text}` : b.text;
  return P({ children: [new TextRun({ text: label, font: FONT, size: pt(10), smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true });
}
function h2(b) {
  return P({ children: [new TextRun({ text: `${b.num}. ${b.text}`, font: FONT, size: pt(10), italics: true })],
    alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 }, keepNext: true });
}

function figure(b) {
  const buf = fs.readFileSync(path.join(figDir, b.file));
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20);          // PNG size from the IHDR chunk
  const full = (b.wide ? TEXT_W : COL_W) / 1440 * 96;                   // available width in px at 96 px/in
  const wpx = Math.round(full * b.width), hpx = Math.round(wpx * ph / pw);
  return [
    P({ children: [new ImageRun({ type: 'png', data: buf, transformation: { width: wpx, height: hpx },
      altText: { title: `${L.fig} ${b.num}`, description: text(b.caption), name: b.file } })],
      alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 }, keepNext: true, keepLines: true }),
    P({ children: [new TextRun({ text: `${L.fig} ${b.num}. `, font: FONT, size: pt(8) }), ...mkRuns(b.caption, { size: pt(8) })],
      alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 120 } }),
  ];
}

function table(b) {
  const W = b.wide ? TEXT_W : COL_W;
  const nCol = b.cols.length;
  const fontSize = pt(b.wide ? 8 : 7.5);
  const em = fontSize / 2 * 20;   // font size in DXA
  // approximate Times New Roman advance widths, enough to size columns to their content
  const textW = (t) => [...t].reduce((a, ch) => a + em * (/[A-ZΔ]/.test(ch) ? 0.72 : /[0-9a-z−+]/.test(ch) ? 0.5
    : /\s/.test(ch) ? 0.25 : /[‑\-.,:;()\[\]\/]/.test(ch) ? 0.34 : ch === ' ' ? 1.0 : 0.56), 0);
  const need = Array(nCol).fill(0);
  b.rows.forEach((r) => {
    let k = 0;
    r.cells.forEach((c) => { if (c.span === 1) need[k] = Math.max(need[k], Math.ceil(textW(text(c.runs))) + 100); k += c.span; });
  });
  const sum = need.reduce((a, x) => a + x, 0);
  const widths = sum <= W ? need.map((w) => w + Math.floor((W - sum) / nCol)) : need.map((w) => Math.floor(w * W / sum));
  widths[nCol - 1] += W - widths.reduce((a, x) => a + x, 0);

  const NONE = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
  const THICK = { style: BorderStyle.SINGLE, size: 8, color: '000000' };
  const THIN = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
  const alignOf = (a) => (a === 'c' ? AlignmentType.CENTER : a === 'r' ? AlignmentType.RIGHT : AlignmentType.LEFT);

  const rows = b.rows.map((r, ri) => {
    let col = 0;
    const isLast = ri === b.rows.length - 1;
    const cells = r.cells.map((c) => {
      const w = widths.slice(col, col + c.span).reduce((a, x) => a + x, 0);
      const align = c.span > 1 ? alignOf(c.align || 'c') : alignOf(b.cols[col].align);
      col += c.span;
      const top = r.rule_before === 'top' ? THICK : r.rule_before === 'mid' ? THIN : NONE;
      const bottom = isLast || r.rule_after === 'bottom' ? THICK : NONE;
      return new TableCell({
        width: { size: w, type: WidthType.DXA }, columnSpan: c.span > 1 ? c.span : undefined,
        borders: { top, bottom, left: NONE, right: NONE },
        margins: { top: r.rule_before === 'space' ? 45 : 10, bottom: 15, left: 30, right: 30 }, verticalAlign: VerticalAlign.CENTER,
        children: [P({ children: c.runs.length ? mkRuns(c.runs.map((x) => ({ ...x, t: x.t.replace(/-/g, '‑').replace(/ /g, ' ') })), { size: fontSize })
          : [new TextRun({ text: '', size: fontSize, font: FONT })], alignment: align, spacing: { before: 0, after: 0 }, keepNext: !isLast })],
      });
    });
    return new TableRow({ children: cells, cantSplit: true });
  });

  const out = [
    P({ children: [new TextRun({ text: `${L.table} ${b.num}`, font: FONT, size: pt(8) })],
      alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true }),
    P({ children: mkRuns(b.caption, { size: pt(8), smallCaps: true }), alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 }, keepNext: true }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, rows }),
  ];
  if (b.note) out.push(P({ children: mkRuns(b.note, { size: pt(7) }), alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 0 } }));
  out.push(P({ children: [], spacing: { before: 0, after: 120 } }));
  return out;
}

// ---------- sections: title block (one column), then two-column text; a wide table or figure gets its own one-column section
const sections = [];
const one = (children) => sections.push({ properties: { type: sections.length ? SectionType.CONTINUOUS : undefined, page: PAGE }, children });
const two = (children) => sections.push({ properties: { type: SectionType.CONTINUOUS, page: PAGE, column: { count: 2, space: GAP, equalWidth: true } }, children });

one([
  P({ children: [new TextRun({ text: d.title, font: FONT, size: pt(24) })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 } }),
  P({ children: [new TextRun({ text: d.author_name, font: FONT, size: pt(11) })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 } }),
  ...d.affiliation.map((l, i) => P({ children: mkRuns(l, { size: pt(10) }), alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: i === d.affiliation.length - 1 ? 360 : 0 } })),
]);

let cur = [
  P({ children: [new TextRun({ text: `${L.abstract}—`, font: FONT, size: pt(9), bold: true, italics: true }), ...mkRuns(d.abstract, { size: pt(9), bold: true })],
    alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 288 }, spacing: { before: 0, after: 120 } }),
  P({ children: [new TextRun({ text: `${L.keywords}—`, font: FONT, size: pt(9), bold: true, italics: true }), ...mkRuns(d.keywords, { size: pt(9), bold: true })],
    alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 288 }, spacing: { before: 0, after: 120 } }),
];
// Word has no floats: a full-width table or figure breaks the two-column flow where it stands. When two
// full-width items are only a few paragraphs apart, they are placed together so the page is not left half empty.
const blocks = d.blocks.slice();
for (let i = 0; i < blocks.length; i++) {
  if (!blocks[i].wide) continue;
  const j = blocks.findIndex((x, k) => k > i + 1 && k <= i + 5 && x.wide);
  if (j > 0) blocks.splice(i + 1, 0, ...blocks.splice(j, 1));
}
// LaTeX puts a full-width float at the top of the next page; Word breaks the columns where the item is.
// Moving each group of full-width items DEFER body paragraphs later imitates the LaTeX placement.
const DEFER = Number(process.env.DOCX_WIDE_DEFER ?? 2);
if (DEFER > 0) {
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (!blocks[i].wide || (i > 0 && blocks[i - 1].wide)) continue;
    let n = 1; while (blocks[i + n] && blocks[i + n].wide) n++;
    const group = blocks.splice(i, n);
    let k = i, seen = 0;
    while (k < blocks.length && seen < DEFER) { if (blocks[k].type === 'p') seen++; k++; }
    blocks.splice(k, 0, ...group);
  }
}
for (const b of blocks) {
  if (b.wide) {
    if (cur.length) { two(cur); cur = []; }
    one(b.type === 'table' ? table(b) : figure(b));
  } else if (b.type === 'p') cur.push(bodyPara(b.runs));
  else if (b.type === 'h1') cur.push(h1(b));
  else if (b.type === 'h2') cur.push(h2(b));
  else if (b.type === 'figure') cur.push(...figure(b));
  else if (b.type === 'table') cur.push(...table(b));
}
cur.push(P({ children: [new TextRun({ text: L.refs, font: FONT, size: pt(10), smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true }));
for (const r of d.references) {
  cur.push(P({ children: [new TextRun({ text: `[${r.n}]\t`, font: FONT, size: pt(8) }), ...mkRuns(r.runs, { size: pt(8) })],
    alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 }],
    spacing: { before: 0, after: 0 } }));
}
two(cur);
// a final continuous break balances the two columns of the last page
one([P({ children: [], spacing: { before: 0, after: 0 } })]);

const doc = new Document({
  creator: d.author_name, title: d.title,
  styles: { default: { document: { run: { font: FONT, size: pt(10) } } } },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outDocx, buf); console.log('written', outDocx, buf.length); });
