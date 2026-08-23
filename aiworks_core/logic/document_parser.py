"""
Document parsing utilities for extracting text content from various file formats.

This module provides a standalone document parser that can extract text from:
- PDF files (.pdf)
- Word documents (.docx, .doc, .rtf)
- Excel spreadsheets (.xlsx, .xls, .ods)
- PowerPoint presentations (.pptx, .ppt, .odp)
- Text files (.txt, .md, .csv)
- OpenDocument formats (.odt, .ods, .odp)

Usage:
    from document_parser import parse_document

    with open('document.pdf', 'rb') as f:
        text = parse_document(f, 'document.pdf')
"""

import io
import logging
import os
import tempfile
from typing import BinaryIO

import docx
import olefile
import openpyxl
import pypdf
import xlrd
from bidi.algorithm import get_display
from docx.oxml.ns import qn
from odf import opendocument, table, text as odf_text
from pptx import Presentation
from pyantiword.antiword_wrapper import extract_text_with_antiword
from striprtf.striprtf import rtf_to_text

logger = logging.getLogger(__name__)

# Maximum file size: 50MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Supported file extensions
ALLOWED_UPLOAD_EXTENSIONS = {
    # PDF
    '.pdf',
    # Word
    '.docx', '.doc', '.rtf',
    # Excel
    '.xlsx', '.xls', '.ods',
    # PowerPoint
    '.pptx', '.ppt', '.odp',
    # Text
    '.txt', '.md', '.csv', '.json',
    # OpenDocument Text
    '.odt',
}

def can_extract_text(filename: str) -> bool:
    """Return True if the file type supports text extraction."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_UPLOAD_EXTENSIONS


def extract_text_from_binary(binary: bytes, filename: str) -> str:
    """Extract text content from binary data.

    Args:
        binary: Raw file bytes
        filename: Original filename (for extension detection)

    Returns:
        Extracted text, or empty string on failure
    """
    try:
        return parse_document(io.BytesIO(binary), filename, validate_size=False)
    except Exception:
        logger.exception("extract_text_from_binary | filename=%s", filename)
        return ""


def validate_file_size(file_content: bytes, filename: str) -> None:
    """
    Validate that the file size is within the allowed limit.

    Args:
        file_content: The file content as bytes
        filename: The name of the file (for error messages)

    Raises:
        ValueError: If the file size exceeds MAX_FILE_SIZE_BYTES
    """
    size = len(file_content)
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(
            f"File '{filename}' is too large ({size_mb:.1f}MB). "
            f"Maximum allowed size is {max_mb:.0f}MB."
        )


def _parse_pdf(file_obj: BinaryIO) -> str:
    try:
        reader = pypdf.PdfReader(file_obj)
        text = '\n'.join(
            page.extract_text(extraction_mode='layout') or '' for page in reader.pages
        )
        if text:
            text = get_display(text)

        return str(text)
    except Exception as e:
        logger.warning(f"Failed to parse PDF: {e}")
        return ""


def _parse_docx(file_obj: BinaryIO) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_obj)
        texts = [para.text for para in doc.paragraphs]
        for doc_table in doc.tables:
            # python-docx returns the same _Cell object multiple times for
            # merged cells (once per grid position they span), so dedup by
            # object identity to avoid repeated text.
            seen_cells: set[int] = set()
            for row in doc_table.rows:
                for cell in row.cells:
                    if id(cell) not in seen_cells:
                        seen_cells.add(id(cell))
                        texts.append(cell.text)
        # Extract text from text boxes (w:txbxContent), which python-docx
        # does not include in doc.paragraphs or doc.tables.  Documents with
        # complex/presentation-style layouts often store all their content
        # exclusively inside text boxes.
        for txbx in doc.element.body.iter(qn('w:txbxContent')):
            for para in txbx.iter(qn('w:p')):
                text = ''.join(
                    node.text for node in para.iter(qn('w:t')) if node.text
                )
                if text:
                    texts.append(text)
        return '\n'.join(texts)
    except Exception as e:
        logger.warning(f"Failed to parse DOCX: {e}")
        return ""


def _parse_rtf(file_obj: BinaryIO) -> str:
    """Extract text from an RTF file."""
    try:
        content = file_obj.read()
        text = rtf_to_text(content.decode('utf-8', errors='ignore'))
        return text
    except Exception as e:
        logger.warning(f"Failed to parse RTF: {e}")
        return ""


def _parse_doc(file_obj: BinaryIO) -> str:
    """Extract text from a legacy DOC file."""
    content = file_obj.read()
    try:
        # pyantiword's wrapper accepts a file path, so persist bytes temporarily.
        with tempfile.NamedTemporaryFile(suffix='.doc') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            text = extract_text_with_antiword(tmp_file.name)
        return text.decode('utf-8', errors='replace') if isinstance(text, bytes) else text
    except Exception as e:
        logger.warning(f"Failed to parse DOC with pyantiword (trying olefile): {e}")
    try:
        ole = olefile.OleFileIO(io.BytesIO(content))
        if ole.exists('WordDocument'):
            stream = ole.openstream('WordDocument')
            raw = stream.read()
            text = raw.decode('latin-1', errors='ignore')
            return ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
        ole.close()
    except Exception as e2:
        logger.warning(f"DOC parsing failed completely: {e2}")
    return ""


def _parse_excel(file_obj: BinaryIO, filename: str) -> str:
    """Extract text from Excel files (.xlsx, .xls, .ods)."""
    try:
        # For .xlsx files, use openpyxl
        if filename.lower().endswith('.xlsx'):
            wb = openpyxl.load_workbook(file_obj, data_only=True)
            texts = []
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                texts.append(f"Sheet: {sheet_name}")
                for row in sheet.iter_rows(values_only=True):
                    row_text = '\t'.join(str(cell) if cell is not None else '' for cell in row)
                    if row_text.strip():
                        texts.append(row_text)
            return '\n'.join(texts)
        # For .xls files, use xlrd
        elif filename.lower().endswith('.xls'):
            wb = xlrd.open_workbook(file_contents=file_obj.read())
            texts = []
            for sheet in wb.sheets():
                texts.append(f"Sheet: {sheet.name}")
                for row_idx in range(sheet.nrows):
                    row = sheet.row(row_idx)
                    row_text = '\t'.join(str(cell.value) for cell in row)
                    if row_text.strip():
                        texts.append(row_text)
            return '\n'.join(texts)
        # For .ods files, use odfpy
        elif filename.lower().endswith('.ods'):
            doc = opendocument.load(file_obj)
            texts = []
            for sheet in doc.spreadsheet.getElementsByType(table.Table):
                sheet_name = sheet.getAttribute('name')
                texts.append(f"Sheet: {sheet_name}")
                for row in sheet.getElementsByType(table.TableRow):
                    cells = row.getElementsByType(table.TableCell)
                    row_values = []
                    for cell in cells:
                        # Get text content from cell
                        cell_text = []
                        for p in cell.getElementsByType(odf_text.P):
                            cell_text.append(str(p))
                        row_values.append(''.join(cell_text))
                    row_text = '\t'.join(row_values)
                    if row_text.strip():
                        texts.append(row_text)
            return '\n'.join(texts)
    except Exception as e:
        logger.warning(f"Failed to parse Excel file: {e}")
    return ""


def _parse_powerpoint(file_obj: BinaryIO, filename: str) -> str:
    """Extract text from PowerPoint files (.pptx, .ppt, .odp)."""
    try:
        # For .pptx files, use python-pptx
        if filename.lower().endswith('.pptx'):
            prs = Presentation(file_obj)
            texts = []
            for slide_num, slide in enumerate(prs.slides, 1):
                texts.append(f"Slide {slide_num}:")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        texts.append(shape.text)
            return '\n'.join(texts)
        elif filename.lower().endswith('.ppt'):
            try:
                ole = olefile.OleFileIO(file_obj)
                if not ole.exists('PowerPoint Document'):
                    return ""
                stream = ole.openstream('PowerPoint Document')
                data = stream.read()
                texts = []
                i = 0
                while i + 8 <= len(data):
                    rec_type = int.from_bytes(data[i + 2:i + 4], 'little')
                    rec_len = int.from_bytes(data[i + 4:i + 8], 'little')
                    body = data[i + 8:i + 8 + rec_len]
                    if rec_type == 0x0FA0:  # TextCharsAtom (UTF-16LE)
                        texts.append(body.decode('utf-16-le', errors='ignore'))
                    elif rec_type == 0x0FA8:  # TextBytesAtom (Latin-1)
                        texts.append(body.decode('latin-1', errors='ignore'))
                    i += 8 + rec_len
                ole.close()
                return '\n'.join(t for t in texts if t.strip())
            except Exception as e:
                logger.warning(f"Failed to parse legacy PPT: {e}")
            return ""
        # For .odp files, use odfpy
        elif filename.lower().endswith('.odp'):
            doc = opendocument.load(file_obj)
            texts = []
            # Extract all text paragraphs
            for para in doc.getElementsByType(odf_text.P):
                texts.append(str(para))
            return '\n'.join(texts)
    except Exception as e:
        logger.warning(f"Failed to parse PowerPoint file: {e}")
    return ""


def _parse_odt(file_obj: BinaryIO) -> str:
    """Extract text from OpenDocument Text files (.odt)."""
    try:
        doc = opendocument.load(file_obj)
        texts = []
        for para in doc.getElementsByType(odf_text.P):
            texts.append(str(para))
        return '\n'.join(texts)
    except Exception as e:
        logger.warning(f"Failed to parse ODT: {e}")
        return ""


def _parse_text(file_obj: BinaryIO) -> str:
    """Extract text from plain text files."""
    content = file_obj.read()
    return content.decode('utf-8', errors='replace')


def parse_document(file_obj: BinaryIO, filename: str, validate_size: bool = True) -> str:
    """
    Extract text content from an uploaded document.

    This function handles various document formats and returns the extracted text.
    The file object will be seeked back to the beginning after parsing.

    Args:
        file_obj: A file-like object opened in binary mode
        filename: The name of the file (used to determine file type)
        validate_size: Whether to validate file size (default: True)

    Returns:
        The extracted text content, or an empty string if parsing fails

    Raises:
        ValueError: If validate_size is True and file size exceeds limit
    """
    # Read file content
    content = file_obj.read()

    # Validate file size if requested
    if validate_size:
        validate_file_size(content, filename)

    # Create a new BytesIO object for parsing
    content_io = io.BytesIO(content)

    # Determine file type and parse accordingly
    name_lower = filename.lower()

    try:
        parsed_content = None

        if name_lower.endswith('.pdf'):
            parsed_content =  _parse_pdf(content_io)
        elif name_lower.endswith('.docx'):
            parsed_content =  _parse_docx(content_io)
        elif name_lower.endswith('.rtf'):
            parsed_content =  _parse_rtf(content_io)
        elif name_lower.endswith('.doc'):
            parsed_content =  _parse_doc(content_io)
        elif name_lower.endswith(('.xlsx', '.xls', '.ods')):
            parsed_content =  _parse_excel(content_io, filename)
        elif name_lower.endswith(('.pptx', '.ppt', '.odp')):
            parsed_content =  _parse_powerpoint(content_io, filename)
        elif name_lower.endswith('.odt'):
            parsed_content =  _parse_odt(content_io)

        # Fallback to parse as text
        if not parsed_content:
            parsed_content = _parse_text(io.BytesIO(content))

        return parsed_content
    finally:
        # Ensure file_obj is at the beginning for any subsequent reads
        file_obj.seek(0)
