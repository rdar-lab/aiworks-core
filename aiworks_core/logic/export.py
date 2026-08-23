"""Export logic for session reports.

Provides helpers to convert a markdown report into:
  - Word document (.docx) via python-docx
  - PDF document (.pdf) via markdown → HTML → weasyprint
"""

import io
import re

import markdown
import weasyprint
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

__all__ = ["generate_docx", "generate_pdf"]

# RTL Unicode ranges (Arabic, Hebrew, Syriac, Thaana, etc.)
_RTL_CODEPOINTS = [
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),    # Syriac
    (0x0750, 0x077F),   # Arabic Supplement
    (0x07C0, 0x07FF),   # Nkoo
    (0x0800, 0x083F),   # Samaritan
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0x10800, 0x108FF), # Cypriot
    (0x10A60, 0x10AFF), # Old South Arabian
    (0x10B80, 0x10BFF), # Psalter Pahlavi
    (0x10C00, 0x10C4F), # Old Turkic
    (0x10F00, 0x10F2F), # Old Hungarian
    (0x1E800, 0x1E8FF), # Mandaic
    (0x1EE00, 0x1EEFF), # Arabic Mathematical Alphabetic Symbols
    (0x1F800, 0x1F8FF), # Modifier Tone Letters
    (0x20670, 0x2069F), # Invisible overflow
]


def _is_rtl_codepoint(cp: int) -> bool:
    for start, end in _RTL_CODEPOINTS:
        if start <= cp <= end:
            return True
    return False


def is_rtl_text(text: str) -> bool:
    """Return True if text is predominantly RTL (Arabic, Hebrew, etc.)."""
    rtl_count = 0
    non_rtl_alpha_count = 0
    for char in text:
        cp = ord(char)
        if _is_rtl_codepoint(cp):
            rtl_count += 1
        elif char.isalpha():
            non_rtl_alpha_count += 1
    return rtl_count > non_rtl_alpha_count


def _set_document_direction(doc: Document, rtl: bool) -> None:
    """Set document-level text direction to RTL if requested."""
    if not rtl:
        return
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for section in doc.sections:
        bidi = OxmlElement("w:bidi")
        bidi.set(qn("w:val"), "1")
        section._sectPr.insert_element_before(bidi, "w:pgMar")


# ---------------------------------------------------------------------------
# DOCX generation
# ---------------------------------------------------------------------------


def generate_docx(title: str, markdown_content: str) -> io.BytesIO:
    """Return a BytesIO buffer containing a .docx document.

    The document contains the session title as the document heading followed
    by the markdown content rendered as Word paragraphs with basic formatting
    (headings H1-H6, bold, italic, bullet lists, numbered lists, code blocks,
    horizontal rules, tables).
    """
    doc = Document()

    # Set document margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.25)

    # Set default font for the document
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    # Configure heading styles
    for i in range(1, 7):
        try:
            heading_style = doc.styles[f"Heading {i}"]
            heading_style.font.name = "Arial"
            heading_style.font.bold = True
            if i == 1:
                heading_style.font.size = Pt(18)
            elif i == 2:
                heading_style.font.size = Pt(14)
            else:
                heading_style.font.size = Pt(12)
        except KeyError:
            pass

    doc.add_heading(title, level=0)

    rtl = is_rtl_text(markdown_content)
    _set_document_direction(doc, rtl)

    _render_markdown_to_docx(doc, markdown_content, rtl=rtl)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# Inline-formatting patterns
_BOLD_ITALIC_RE = re.compile(r"\*\*\*(.*?)\*\*\*|___(.*?)___")
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*|__(.*?)__")
_ITALIC_RE = re.compile(r"\*(.*?)\*|_(.*?)_")
_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_ANCHOR_TAG_RE = re.compile(r"<a\s+name=\"([^\"]*)\"\s*>(?:(?!</a>).)*(?:</a>)?", re.DOTALL)
_ANCHOR_SELF_CLOSING_RE = re.compile(r"<a\s+name=\"([^\"]*)\"\s*/>")


def _set_paragraph_spacing(para, before: Pt = Pt(6), after: Pt = Pt(6)) -> None:
    """Set consistent spacing for paragraphs."""
    para.paragraph_format.space_before = before
    para.paragraph_format.space_after = after


def _rtl_run_element():
    """Return an <w:rtl> element to mark a run as right-to-left."""
    from docx.oxml import OxmlElement
    rtl = OxmlElement("w:rtl")
    return rtl


def _add_formatted_run(para, text: str, rtl: bool = False) -> None:
    """Add *text* to *para*, splitting on inline bold/italic/code markers and anchor tags."""
    tokens = _tokenize_inline(text)
    for segment, bold, italic, code in tokens:
        if not segment:
            continue
        segment = _strip_anchor_tags(segment)
        run = para.add_run(segment)
        run.bold = bold
        run.italic = italic
        if code:
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        if rtl:
            run._r.get_or_add_rPr().append(_rtl_run_element())


def _strip_anchor_tags(text: str) -> str:
    """Remove anchor tags while preserving inner content."""
    # Remove self-closing anchor tags like <a name="..."/>
    text = _ANCHOR_SELF_CLOSING_RE.sub("", text)
    # Remove opening/closing anchor pairs while keeping content between them
    # e.g., <a name="...">content</a> -> content
    text = _ANCHOR_TAG_RE.sub(lambda m: _extract_anchor_text(m.group(0)), text)
    return text


def _extract_anchor_text(anchor_tag: str) -> str:
    """Extract inner text from an anchor tag, stripping the tags themselves."""
    # Strip <a ...> at start and </a> at end, return what's between
    inner = re.sub(r"<a\s+[^>]*>", "", anchor_tag)
    inner = re.sub(r"</a>", "", inner)
    return inner


def _tokenize_inline(text: str):
    """Return a list of (segment, bold, italic, code) tuples."""
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        # Code span
        m = _CODE_INLINE_RE.search(text, i)
        bi = _BOLD_ITALIC_RE.search(text, i)
        b = _BOLD_RE.search(text, i)
        it = _ITALIC_RE.search(text, i)

        # Find the earliest match
        candidates = [(m, "code"), (bi, "bold_italic"), (b, "bold"), (it, "italic")]
        candidates = [(x, t) for x, t in candidates if x is not None]
        if not candidates:
            tokens.append((text[i:], False, False, False))
            break

        earliest = min(candidates, key=lambda x: x[0].start())
        match, kind = earliest

        # Plain text before the match
        if match.start() > i:
            tokens.append((text[i : match.start()], False, False, False))

        groups = match.groups()
        if len(groups) >= 2:
            inner = groups[0] if groups[0] is not None else (groups[1] or "")
        else:
            inner = groups[0] or ""

        if kind == "code":
            tokens.append((inner, False, False, True))
        elif kind == "bold_italic":
            tokens.append((inner, True, True, False))
        elif kind == "bold":
            tokens.append((inner, True, False, False))
        elif kind == "italic":
            tokens.append((inner, False, True, False))

        i = match.end()

    return tokens


def _render_markdown_to_docx(doc, markdown_content: str, rtl: bool = False) -> None:  # noqa: C901
    """Walk through markdown lines and add corresponding Word elements."""
    lines = markdown_content.splitlines()
    in_code_block = False
    code_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Fenced code block ---
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                _add_code_block(doc, code_text)
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        # --- Table ---
        if _is_markdown_table_line(stripped):
            table_rows = []
            while i < len(lines):
                stripped_line = lines[i].strip()
                if _is_separator_row(stripped_line):
                    i += 1
                    continue
                if not _is_markdown_table_line(stripped_line):
                    break
                table_rows.append(stripped_line)
                i += 1
            _add_markdown_table(doc, table_rows, rtl)
            continue

        # --- Headings ---
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            text = _strip_anchor_tags(text)
            para = doc.add_heading(text, level=level)
            # Add extra space before/after headings
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            i += 1
            continue

        # --- Horizontal rule ---
        if re.match(r"^(-{3,}|_{3,}|\*{3,})$", stripped):
            _add_horizontal_rule(doc)
            i += 1
            continue

        # --- Bullet list item ---
        bullet_match = re.match(r"^[*\-+]\s+(.*)", stripped)
        if bullet_match:
            para = doc.add_paragraph(style="List Bullet")
            _add_formatted_run(para, bullet_match.group(1), rtl)
            i += 1
            continue

        # --- Numbered list item ---
        ordered_match = re.match(r"^\d+\.\s+(.*)", stripped)
        if ordered_match:
            para = doc.add_paragraph(style="List Number")
            _add_formatted_run(para, ordered_match.group(1), rtl)
            i += 1
            continue

        # --- Blockquote ---
        if stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(36)
            _set_paragraph_spacing(para, Pt(3), Pt(3))
            _add_formatted_run(para, text, rtl)
            i += 1
            continue

        # --- Empty line ---
        if not stripped:
            i += 1
            continue

        # --- Normal paragraph ---
        para = doc.add_paragraph()
        _set_paragraph_spacing(para, Pt(4), Pt(4))
        _add_formatted_run(para, stripped, rtl)
        i += 1


def _is_separator_row(line: str) -> bool:
    """Return True if line is a markdown table separator row (|---|---|...)."""
    stripped = line.strip()
    return bool(re.match(r"^\|?[\s\-:|]+\|?$", stripped))


def _is_markdown_table_line(line: str) -> bool:
    """Return True if line looks like a markdown table row (not a separator)."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    if _is_separator_row(stripped):
        return False
    return "|" in stripped


def _parse_markdown_table_row(line: str) -> list[str]:
    """Parse a markdown table row into cell values."""
    stripped = line.strip().strip("|")
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells


def _add_markdown_table(doc, rows: list[str], rtl: bool = False) -> None:
    """Add a markdown table to the Word document."""
    if len(rows) < 2:
        return

    header_row = _parse_markdown_table_row(rows[0])
    num_cols = len(header_row)

    data_rows = []
    for row in rows[1:]:
        if not _is_separator_row(row):
            data_rows.append(row)

    table = doc.add_table(rows=1, cols=num_cols)
    table.style = "Table Grid"

    def _set_cell_text(cell, text):
        cell.text = ""
        para = cell.paragraphs[0]
        para.paragraph_format.space_before = Pt(3)
        para.paragraph_format.space_after = Pt(3)
        _add_formatted_run(para, text, rtl)

    def _apply_header_style(cell):
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.name = "Arial"
                run.font.size = Pt(10)

    def _apply_data_style(cell):
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(10)

    hdr_cells = table.rows[0].cells
    for idx, cell_text in enumerate(header_row):
        if idx < len(hdr_cells):
            _set_cell_text(hdr_cells[idx], cell_text)
            _apply_header_style(hdr_cells[idx])

    for row in data_rows:
        cells = _parse_markdown_table_row(row)
        row_cells = table.add_row().cells
        for idx, cell_text in enumerate(cells):
            if idx < len(row_cells):
                _set_cell_text(row_cells[idx], cell_text)
                _apply_data_style(row_cells[idx])


def _add_code_block(doc, code_text: str) -> None:
    """Add a monospace code block paragraph."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    run = para.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    # Light gray background shading for code blocks
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), "F5F5F5")
    para._p.get_or_add_pPr().append(shading)


def _add_horizontal_rule(doc) -> None:
    """Add a horizontal rule paragraph with a bottom border."""
    para = doc.add_paragraph()
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    pBdr.append(bottom)
    pPr.append(pBdr)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html{dir_attr}>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 20px 20px;
    color: #1e293b;
    font-size: 12pt;
    line-height: 1.6;
  }}
  h1 {{
    color: #0f172a;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 6px;
    font-size: 22pt;
    margin-top: 0;
  }}
  h2 {{ color: #1e3a5f; font-size: 16pt; margin-top: 24px; }}
  h3 {{ color: #1e3a5f; font-size: 13pt; margin-top: 18px; }}
  h4, h5, h6 {{ color: #334155; font-size: 11pt; margin-top: 14px; }}
  p {{ margin: 8px 0; }}
  ul, ol {{ margin: 8px 0; padding-right: 24px; }}
  li {{ margin: 4px 0; }}
  code {{
    background: #f1f5f9;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: "Courier New", monospace;
    font-size: 10pt;
  }}
  pre {{
    background: #f1f5f9;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: "Courier New", monospace;
    font-size: 10pt;
    overflow: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }}
  blockquote {{
    border-right: 4px solid #3b82f6;
    padding-right: 12px;
    color: #475569;
    margin-right: 0;
  }}
  hr {{ border: none; border-top: 1px solid #cbd5e1; margin: 16px 0; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
    text-align: right;
  }}
  th {{ background: #e2e8f0; font-weight: bold; }}
  strong {{ font-weight: bold; }}
  em {{ font-style: italic; }}
  a {{ font-weight: inherit; }}
  a[name] {{ font-weight: inherit; }}
  .title-block {{
    margin-bottom: 28px;
  }}
  .title-label {{
    font-size: 9pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 4px;
  }}
</style>
</head>
<body>
  <div class="title-block">
    <p class="title-label">Research Report</p>
    <h1>{title}</h1>
  </div>
  {body}
</body>
</html>"""

_RTL_CSS = "\n  body { direction: rtl; }\n"


def generate_pdf(title: str, markdown_content: str) -> bytes:
    """Return PDF bytes for the given markdown content.

    Uses the *markdown* library to convert to HTML, then *weasyprint* to
    render the HTML as a PDF document.
    """
    rtl = is_rtl_text(markdown_content)
    html_body = markdown.markdown(
        markdown_content,
        extensions=["fenced_code", "tables", "nl2br", "md_in_html"],
    )
    dir_attr = ' dir="rtl"' if rtl else ""
    extra_css = _RTL_CSS if rtl else ""
    full_html = _HTML_TEMPLATE.format(
        title=_escape_html(title),
        body=html_body,
        dir_attr=dir_attr,
    )
    if extra_css:
        full_html = full_html.replace("</style>", f"{extra_css}</style>")
    return weasyprint.HTML(string=full_html).write_pdf()


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for plain-text values embedded in HTML."""
    return (\
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
