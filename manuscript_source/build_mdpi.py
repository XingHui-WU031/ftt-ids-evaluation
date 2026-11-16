"""Build the manuscript inside the official MDPI Sensors Word template.
usage: python build_mdpi.py <template.dot|.docx> <spec.json> <figs_dir> <out.docx>"""
import copy, io, json, re, sys, zipfile
import docx
from docx.shared import Cm, Pt, Emu
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

tpl, spec_path, FIG, out = sys.argv[1:5]
spec = json.load(open(spec_path, encoding="utf8"))

# --- open the template as a document (.dot is an OOXML template: switch its content type)
zi = zipfile.ZipFile(tpl); buf = io.BytesIO(); zo = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
for it in zi.infolist():
    d = zi.read(it.filename)
    if it.filename == "[Content_Types].xml":
        d = d.replace(b"wordprocessingml.template.main+xml", b"wordprocessingml.document.main+xml")
    zo.writestr(it, d)
zo.close(); buf.seek(0)
doc = docx.Document(buf)
body = doc.element.body
paras = list(doc.paragraphs)
IND = 2608          # left indent of the MDPI text column (twips)
FULL = 10465        # full text width (twips)
COL = FULL - IND

def runs(par, text, size=None, bold=False, italic=False):
    i, b, it, bufs = 0, False, False, ""
    def flush():
        nonlocal bufs
        if bufs:
            r = par.add_run(bufs); r.bold = True if (b or bold) else None; r.italic = True if (it or italic) else None
            if size: r.font.size = Pt(size)
            bufs = ""
    while i < len(text):
        if text.startswith("**", i): flush(); b = not b; i += 2; continue
        if text[i] == "*": flush(); it = not it; i += 1; continue
        if text[i] in "_^" and i + 1 < len(text) and text[i + 1] == "{":
            flush(); sub = text[i] == "_"; depth = 0; j = i + 1
            while j < len(text):
                if text[j] == "{": depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0: break
                j += 1
            inner = re.sub(r"[_^]\{([^{}]*)\}", r"\1", text[i + 2:j]).replace("*", "")
            r = par.add_run(inner); r.font.subscript = True if sub else None; r.font.superscript = None if sub else True
            r.bold = True if (b or bold) else None; r.italic = True if (it or italic) else None
            if size: r.font.size = Pt(size)
            i = j + 1; continue
        bufs += text[i]; i += 1
    flush()

def set_text(par, text, **kw):
    for r in list(par.runs): r._r.getparent().remove(r._r)
    runs(par, text, **kw)

def set_indent(par, left):
    pPr = par._p.get_or_add_pPr(); ind = pPr.find(qn("w:ind"))
    if ind is None: ind = OxmlElement("w:ind"); pPr.append(ind)
    ind.set(qn("w:left"), str(left)); ind.set(qn("w:firstLine"), "0")

# ---------------- front matter (reuse the template paragraphs)
set_text(paras[0], "Article")
set_text(paras[1], spec["title"])
set_text(paras[2], spec["author"])
aff, _, mail = spec["affil"].partition(";")
set_text(paras[3], aff.strip())
body.remove(paras[4]._p)
set_text(paras[5], "Correspondence:" + mail)
for k in range(6, 14): body.remove(paras[k]._p)          # optional Highlights block
set_text(paras[14], "**Abstract:** " + spec["abstract"])
body.remove(paras[15]._p)
set_text(paras[16], "**Keywords:** " + spec["keywords"])
line = paras[17]._p
end = [p for p in paras if p.style.name == "MDPI_6.3_notes"][-1]._p       # publisher's note stays last
kids = list(body); a, z = kids.index(line), kids.index(end)
for el in kids[a + 1:z]: body.remove(el)

def P(style, text="", **kw):
    par = doc.add_paragraph(style=style); end.addprevious(par._p)
    if text: runs(par, text, **kw)
    return par

def border(el, tag, sz):
    b = OxmlElement("w:" + tag); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), str(sz)); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto"); el.append(b)

def make_table(ncols, widths, wide):
    t = doc.add_table(rows=0, cols=ncols); end.addprevious(t._tbl)
    pr = t._tbl.tblPr
    for ch in list(pr): pr.remove(ch)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), str(sum(widths))); w.set(qn("w:type"), "dxa"); pr.append(w)
    if wide:
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); pr.append(jc)
    else:
        ind = OxmlElement("w:tblInd"); ind.set(qn("w:w"), str(IND)); ind.set(qn("w:type"), "dxa"); pr.append(ind)
    bs = OxmlElement("w:tblBorders"); border(bs, "top", 8); border(bs, "bottom", 8); pr.append(bs)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); pr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        m = OxmlElement("w:" + side); m.set(qn("w:w"), "40"); m.set(qn("w:type"), "dxa"); mar.append(m)
    pr.append(mar)
    grid = t._tbl.tblGrid
    for g, wd in zip(grid.findall(qn("w:gridCol")), widths): g.set(qn("w:w"), str(wd))
    return t

def add_row(t, cells, widths, size, head=False, group=False):
    row = t.add_row()
    trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
    if head:
        th = OxmlElement("w:tblHeader"); trPr.append(th)
    for c, txt, wd in zip(row.cells, cells, widths):
        tcPr = c._tc.get_or_add_tcPr()
        for ch in list(tcPr): tcPr.remove(ch)
        tw = OxmlElement("w:tcW"); tw.set(qn("w:w"), str(wd)); tw.set(qn("w:type"), "dxa"); tcPr.append(tw)
        if head:
            tb = OxmlElement("w:tcBorders"); border(tb, "bottom", 4); tcPr.append(tb)
        va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)
        par = c.paragraphs[0]; par.style = doc.styles["MDPI_4.2_table_body"]
        runs(par, txt, size=size, bold=head, italic=group)

fig_n = 0
NARROW_FIG = {"fig1_framework.png", "fig2_architecture.png", "fig_sttl_shift.png"}
for b in spec["blocks"]:
    t = b["t"]
    if t == "h1": P("MDPI_2.1_heading1", b["x"])
    elif t == "h2": P("MDPI_2.2_heading2", b["x"])
    elif t == "p": P("MDPI_3.2_text_no_indent" if re.match(r"[a-z]", b["x"]) else "MDPI_3.1_text", b["x"])
    elif t == "li": P("MDPI_3.8_bullet", b["x"])
    elif t == "eq":
        tb = doc.add_table(rows=1, cols=2); end.addprevious(tb._tbl)
        pr = tb._tbl.tblPr
        for ch in list(pr): pr.remove(ch)
        w = OxmlElement("w:tblW"); w.set(qn("w:w"), str(COL)); w.set(qn("w:type"), "dxa"); pr.append(w)
        ind = OxmlElement("w:tblInd"); ind.set(qn("w:w"), str(IND)); ind.set(qn("w:type"), "dxa"); pr.append(ind)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); pr.append(lay)
        ws = [COL - 500, 500]
        for g, wd in zip(tb._tbl.tblGrid.findall(qn("w:gridCol")), ws): g.set(qn("w:w"), str(wd))
        for c, wd, sty, txt in zip(tb.rows[0].cells, ws, ("MDPI_3.9_equation", "MDPI_3.a_equation_number"), (b["x"], f"({b['n']})")):
            tcPr = c._tc.get_or_add_tcPr()
            for ch in list(tcPr): tcPr.remove(ch)
            tw = OxmlElement("w:tcW"); tw.set(qn("w:w"), str(wd)); tw.set(qn("w:type"), "dxa"); tcPr.append(tw)
            va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)
            c.paragraphs[0].style = doc.styles[sty]; runs(c.paragraphs[0], txt)
            if sty == "MDPI_3.9_equation": set_indent(c.paragraphs[0], 0)
    elif t == "fig":
        fig_n += 1; narrow = b["f"] in NARROW_FIG
        par = P("MDPI_5.2_figure")
        if narrow: set_indent(par, IND)
        par.add_run().add_picture(f"{FIG}/{b['f']}", width=Cm(13.6 if narrow else 17.6))
        cap = P("MDPI_5.1_figure_caption", f"**Figure {fig_n}.** " + b["cap"])
        if not narrow: set_indent(cap, 0)
    elif t == "tab":
        m = re.match(r"^(Table ?\d+\.)\s*(.*)$", b["cap"]); n = len(b["head"])
        wide = not (n <= 4 and m.group(1) in ("Table 4.", "Table 14.", "Table 15."))
        total = FULL if wide else COL
        s = sum(b["w"]); widths = [int(x * total / s) for x in b["w"]]; widths[-1] += total - sum(widths)
        cap = P("MDPI_4.1_table_caption", f"**{m.group(1)}** " + m.group(2))
        if wide: set_indent(cap, 0)
        size = 7.5 if n >= 8 else (8 if n >= 6 else 9)
        tb = make_table(n, widths, wide)
        add_row(tb, b["head"], widths, size, head=True)
        for r in b["rows"]:
            add_row(tb, r, widths, size, group=(r[0] != "" and all(c == "" for c in r[1:])))
        sp = P("MDPI_3.1_text"); sp.paragraph_format.space_after = Pt(6)

# ---------------- back matter
back = list(spec["back"])
back.insert(2, ["Institutional Review Board Statement", "Not applicable."])
back.insert(3, ["Informed Consent Statement", "Not applicable."])
for k, v in back: P("MDPI_6.2_back_matter", f"**{k}:** {v}")
P("MDPI_2.1_heading1", "Abbreviations")
P("MDPI_3.2_text_no_indent", "The following abbreviations are used in this manuscript:")
tb = make_table(2, [1300, COL - 1300], False)
tb._tbl.tblPr.remove(tb._tbl.tblPr.find(qn("w:tblBorders")))
for a_, d_ in spec["abbr"]:
    row = tb.add_row()
    for c, txt, wd in zip(row.cells, (a_, d_), (1300, COL - 1300)):
        tcPr = c._tc.get_or_add_tcPr()
        for ch in list(tcPr): tcPr.remove(ch)
        tw = OxmlElement("w:tcW"); tw.set(qn("w:w"), str(wd)); tw.set(qn("w:type"), "dxa"); tcPr.append(tw)
        c.paragraphs[0].style = doc.styles["MDPI_4.2_table_body"]; c.paragraphs[0].alignment = 0; runs(c.paragraphs[0], txt, size=9)
P("MDPI_2.1_heading1", "References")
for r in spec["refs"]:
    par = P("MDPI_8.1_references", r)
    pPr = par._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr"); il = OxmlElement("w:ilvl"); il.set(qn("w:val"), "0"); ni = OxmlElement("w:numId"); ni.set(qn("w:val"), "4")
    numPr.append(il); numPr.append(ni); pPr.append(numPr)
    ind = OxmlElement("w:ind"); ind.set(qn("w:left"), "425"); ind.set(qn("w:hanging"), "425"); pPr.append(ind)
# the template's section carries a right-to-left flag that makes some word processors (WPS, LibreOffice) mirror tables
for sp in body.iter(qn("w:sectPr")):
    for bd in sp.findall(qn("w:bidi")): sp.remove(bd)
doc.save(out); print("written", out)
