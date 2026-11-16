const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, Footer, PageNumber, TabStopType, LevelFormat, VerticalAlign, HeadingLevel,
} = require("docx");

const ZH = process.env.LANG_ZH === "1";
const spec = JSON.parse(fs.readFileSync(process.env.SPEC || "/home/claude/paper/spec.json", "utf8"));
const FIG = "/home/claude/paper/figs";
const FONT = ZH ? { ascii: "Palatino Linotype", hAnsi: "Palatino Linotype", eastAsia: "SimSun", cs: "Palatino Linotype" } : "Palatino Linotype";
const L = ZH ? { fig: "图", tab: "表", art: "研究论文", abs: "摘要：", kw: "关键词：", abbr: "缩略语", abbrIntro: "本文使用的缩略语如下：", refs: "参考文献", of: " / ", sep: "：" }
             : { fig: "Figure", tab: "Table", art: "Article", abs: "Abstract:", kw: "Keywords:", abbr: "Abbreviations", abbrIntro: "The following abbreviations are used in this manuscript:", refs: "References", of: " of ", sep: ":" };
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins

// ---------- inline markup: **bold**, *italic*, _{sub}, ^{sup} ----------
function strip(s) { return s.replace(/[_^]\{([^{}]*)\}/g, "$1").replace(/\*/g, ""); }
function runs(text, base = {}) {
  const out = []; let i = 0, buf = "", bold = false, ital = false;
  const flush = () => { if (buf) { out.push(new TextRun({ text: buf, bold: bold || base.bold, italics: ital || base.italics, font: FONT, size: base.size })); buf = ""; } };
  while (i < text.length) {
    if (text.startsWith("**", i)) { flush(); bold = !bold; i += 2; continue; }
    if (text[i] === "*") { flush(); ital = !ital; i += 1; continue; }
    if ((text[i] === "_" || text[i] === "^") && text[i + 1] === "{") {
      flush(); const sub = text[i] === "_"; let depth = 0, j = i + 1;
      for (; j < text.length; j++) { if (text[j] === "{") depth++; else if (text[j] === "}") { depth--; if (depth === 0) break; } }
      const inner = strip(text.slice(i + 2, j));
      out.push(new TextRun({ text: inner, subScript: sub, superScript: !sub, bold: bold || base.bold, italics: ital || base.italics, font: FONT, size: base.size }));
      i = j + 1; continue;
    }
    buf += text[i]; i++;
  }
  flush(); return out;
}
const P = (text, opt = {}) => new Paragraph({
  children: runs(text, { size: opt.size || 20, bold: opt.bold, italics: opt.italics }),
  alignment: opt.align || AlignmentType.JUSTIFIED,
  spacing: { before: opt.before ?? 0, after: opt.after ?? 80, line: opt.line ?? 260 },
  indent: opt.indent, keepNext: opt.keepNext, numbering: opt.numbering,
});

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function figure(f, cap, num) {
  const file = path.join(FIG, f); const { w, h } = pngSize(file);
  const width = 600, height = Math.round(width * h / w); // px at 96 dpi ≈ 15.9 cm
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: `Figure ${num}`, description: strip(cap), name: f } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: 240 },
      children: [new TextRun({ text: `${L.fig} ${num}. `, bold: true, font: FONT, size: 18 }), ...runs(cap, { size: 18 })] }),
  ];
}

const thick = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(t) {
  const sum = t.w.reduce((a, b) => a + b, 0);
  const widths = t.w.map(x => Math.floor(x * CONTENT_W / sum)); widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const size = t.small ? 15 : 16;
  const cell = (txt, ci, ri, isHead, isLast, isGroup) => new TableCell({
    width: { size: widths[ci], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    verticalAlign: VerticalAlign.CENTER,
    shading: isGroup ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined,
    borders: { top: ri === 0 ? thick : none, bottom: isHead ? thin : (isLast ? thick : none), left: none, right: none },
    children: [new Paragraph({ alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { before: 0, after: 0, line: 220 },
      children: runs(txt, { size, bold: isHead, italics: isGroup }) })],
  });
  const rows = [new TableRow({ tableHeader: true, children: t.head.map((h, ci) => cell(h, ci, 0, true, false, false)) })];
  t.rows.forEach((r, ri) => {
    const isGroup = r.slice(1).every(c => c === "") && r[0] !== "";
    const isLast = ri === t.rows.length - 1;
    rows.push(new TableRow({ cantSplit: true, children: r.map((c, ci) => cell(c, ci, ri + 1, false, isLast, isGroup)) }));
  });
  const capParts = t.cap.match(/^((?:Table|表) ?\d+\.)\s*(.*)$/);
  const out = [
    new Paragraph({ spacing: { before: 160, after: 80, line: 240 }, keepNext: true, alignment: AlignmentType.JUSTIFIED,
      children: [new TextRun({ text: capParts[1] + " ", bold: true, font: FONT, size: 18 }), ...runs(capParts[2], { size: 18 })] }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows }),
  ];
  if (t.note) out.push(P(t.note, { size: 16, after: 120 }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

const children = [];
children.push(P(L.art, { italics: true, size: 20, align: AlignmentType.LEFT, after: 120 }));
children.push(new Paragraph({ spacing: { after: 200, line: ZH ? 460 : 300 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: spec.title, bold: true, font: FONT, size: 36 })] }));
children.push(P(`**${spec.author}**`, { align: AlignmentType.LEFT, after: 120 }));
children.push(P(spec.affil, { size: 16, align: AlignmentType.LEFT, after: 240 }));
children.push(P(`**${L.abs}** ${spec.abstract}`, { size: 18, after: 120, line: 240 }));
children.push(P(`**${L.kw}** ${spec.keywords}`, { size: 18, after: 240, line: 240 }));

let fig = 0, firstPara = true;
for (const b of spec.blocks) {
  if (b.t === "h1") children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 }, keepNext: true,
    children: [new TextRun({ text: b.x, bold: true, font: FONT, size: 20 })] }));
  else if (b.t === "h2") children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 }, keepNext: true,
    children: [new TextRun({ text: b.x, italics: !ZH, bold: ZH, font: FONT, size: 20 })] }));
  else if (b.t === "p") children.push(P(b.x, { indent: (/^[a-z]/.test(b.x) || b.noindent) ? undefined : { firstLine: ZH ? 400 : 425 } }));
  else if (b.t === "li") children.push(P(b.x, { numbering: { reference: "bul", level: 0 } }));
  else if (b.t === "eq") children.push(new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(CONTENT_W / 2) }, { type: TabStopType.RIGHT, position: CONTENT_W }],
    spacing: { before: 80, after: 120 }, children: [new TextRun({ text: "\t", font: FONT }), ...runs(b.x, { size: 20, italics: false }), new TextRun({ text: `\t(${b.n})`, font: FONT, size: 20 })] }));
  else if (b.t === "fig") { fig += 1; children.push(...figure(b.f, b.cap, fig)); }
  else if (b.t === "tab") children.push(...table(b));
}
// back matter
children.push(new Paragraph({ spacing: { before: 240 }, children: [] }));
for (const [k, v] of spec.back) children.push(P(`**${k}${L.sep}** ${v}`, { size: 18, line: 240, after: 100 }));
children.push(P(`**${L.abbr}**`, { size: 18, before: 120, after: 60 }));
children.push(P(L.abbrIntro, { size: 18, after: 60 }));
for (const [a, d] of spec.abbr) children.push(new Paragraph({ tabStops: [{ type: TabStopType.LEFT, position: 1400 }], spacing: { after: 0, line: 240 },
  children: [new TextRun({ text: a, font: FONT, size: 18 }), new TextRun({ text: "\t" + d, font: FONT, size: 18 })] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: L.refs, bold: true, font: FONT, size: 20 })] }));
spec.refs.forEach((r) => children.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, spacing: { after: 40, line: 230 }, alignment: AlignmentType.LEFT,
  children: runs(r, { size: 17 }) })));

const doc = new Document({
  creator: spec.author, title: spec.title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 20, bold: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 20, italics: true }, paragraph: { outlineLevel: 1 } }] },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 425, hanging: 283 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 454, hanging: 454 } }, run: { font: FONT, size: 17 } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1134, bottom: 1134, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 }), new TextRun({ text: L.of, font: FONT, size: 16 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2] || "/home/claude/paper/manuscript.docx", buf); console.log("written"); });
