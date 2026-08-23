"""
Tests for document parsing utilities.
"""

import io
import os
from unittest.mock import MagicMock, patch

from django.test import TestCase

from ..logic.document_parser import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    _parse_doc,
    parse_document,
    validate_file_size,
)


class DocumentParserTests(TestCase):
    """Test document parsing functionality."""

    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

    def _get_fixture_path(self, filename):
        """Get the full path to a fixture file."""
        return os.path.join(self.FIXTURES_DIR, filename)

    def _read_fixture(self, filename):
        """Read a fixture file and return a BytesIO object."""
        path = self._get_fixture_path(filename)
        with open(path, 'rb') as f:
            content = f.read()
        return io.BytesIO(content)

    def test_allowed_extensions(self):
        """Test that all expected file extensions are allowed."""
        expected_extensions = {
            '.pdf',
            '.docx', '.doc', '.rtf',
            '.xlsx', '.xls', '.ods',
            '.pptx', '.ppt', '.odp',
            '.txt', '.md', '.csv', '.json',
            '.odt',
        }
        self.assertEqual(ALLOWED_UPLOAD_EXTENSIONS, expected_extensions)

    # noinspection PyMethodMayBeStatic
    def test_validate_file_size_pass(self):
        """Test that small files pass validation."""
        small_file = b"Small content"
        # Should not raise
        validate_file_size(small_file, "small.txt")

    def test_validate_file_size_fail(self):
        """Test that large files fail validation."""
        # Create a file larger than MAX_FILE_SIZE_BYTES
        large_file = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with self.assertRaises(ValueError) as cm:
            validate_file_size(large_file, "large.txt")
        self.assertIn("too large", str(cm.exception))

    def test_parse_text_file(self):
        """Test parsing plain text files."""
        file_obj = self._read_fixture('sample.txt')
        text = parse_document(file_obj, 'sample.txt', validate_size=False)
        self.assertIn("sample text file", text)
        self.assertIn("multiple lines", text)

    def test_parse_markdown_file(self):
        """Test parsing markdown files."""
        file_obj = self._read_fixture('sample.md')
        text = parse_document(file_obj, 'sample.md', validate_size=False)
        self.assertIn("Sample Markdown", text)
        self.assertIn("Features", text)

    def test_parse_csv_file(self):
        """Test parsing CSV files."""
        file_obj = self._read_fixture('sample.csv')
        text = parse_document(file_obj, 'sample.csv', validate_size=False)
        self.assertIn("Name", text)
        self.assertIn("Alice", text)
        self.assertIn("Bob", text)

    def test_parse_rtf_file(self):
        """Test parsing RTF files."""
        file_obj = self._read_fixture('sample.rtf')
        text = parse_document(file_obj, 'sample.rtf', validate_size=False)
        # RTF parsing extracts plain text
        self.assertIn("sample RTF file", text)

    def test_parse_docx_file(self):
        """Test parsing DOCX files."""
        # Skip if fixture doesn't exist yet
        path = self._get_fixture_path('sample.docx')
        if not os.path.exists(path):
            self.skipTest("DOCX fixture not available")

        file_obj = self._read_fixture('sample.docx')
        text = parse_document(file_obj, 'sample.docx', validate_size=False)
        self.assertIn("sample DOCX file", text)

    def test_parse_pdf_file(self):
        """Test parsing PDF files."""
        # Skip if fixture doesn't exist yet
        path = self._get_fixture_path('sample.pdf')
        if not os.path.exists(path):
            self.skipTest("PDF fixture not available")

        file_obj = self._read_fixture('sample.pdf')
        text = parse_document(file_obj, 'sample.pdf', validate_size=False)
        # PDF may have extractable text
        self.assertIsInstance(text, str)

    def test_parse_xlsx_file(self):
        """Test parsing XLSX files."""
        # Skip if fixture doesn't exist yet
        path = self._get_fixture_path('sample.xlsx')
        if not os.path.exists(path):
            self.skipTest("XLSX fixture not available")

        file_obj = self._read_fixture('sample.xlsx')
        text = parse_document(file_obj, 'sample.xlsx', validate_size=False)
        self.assertIn("Sheet", text)

    def test_parse_pptx_file(self):
        """Test parsing PPTX files."""
        # Skip if fixture doesn't exist yet
        path = self._get_fixture_path('sample.pptx')
        if not os.path.exists(path):
            self.skipTest("PPTX fixture not available")

        file_obj = self._read_fixture('sample.pptx')
        text = parse_document(file_obj, 'sample.pptx', validate_size=False)
        self.assertIn("Slide", text)

    def test_file_object_seeked_after_parse(self):
        """Test that file object is at the beginning after parsing."""
        file_obj = self._read_fixture('sample.txt')
        parse_document(file_obj, 'sample.txt', validate_size=False)
        # File should be seeked back to 0
        self.assertEqual(file_obj.tell(), 0)

    def test_parse_unknown_extension(self):
        """Test parsing file with unknown extension as text."""
        content = b"Unknown file content"
        file_obj = io.BytesIO(content)
        text = parse_document(file_obj, 'unknown.xyz', validate_size=False)
        self.assertEqual(text, "Unknown file content")

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        file_obj = io.BytesIO(b"")
        text = parse_document(file_obj, 'empty.txt', validate_size=False)
        self.assertEqual(text, "")

    def test_parse_with_validation(self):
        """Test that parse_document validates file size by default."""
        large_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        file_obj = io.BytesIO(large_content)
        with self.assertRaises(ValueError):
            parse_document(file_obj, 'large.txt', validate_size=True)

    @patch('aiworks_core.logic.document_parser.extract_text_with_antiword', return_value='antiword text')
    def test_parse_doc_uses_pyantiword_wrapper(self, extract_text_mock):
        """Test that .doc parsing uses pyantiword wrapper function."""
        text = _parse_doc(io.BytesIO(b'legacy-doc-content'))
        self.assertEqual(text, 'antiword text')
        extract_text_mock.assert_called_once()
        called_path = extract_text_mock.call_args.args[0]
        self.assertTrue(called_path.endswith('.doc'))

    @patch('aiworks_core.logic.document_parser.extract_text_with_antiword', side_effect=RuntimeError('antiword failed'))
    @patch('aiworks_core.logic.document_parser.olefile.OleFileIO')
    def test_parse_doc_falls_back_to_olefile(self, ole_cls_mock, _extract_text_mock):
        """Test that .doc parser falls back to olefile when pyantiword fails."""
        ole_mock = MagicMock()
        ole_mock.exists.return_value = True
        ole_stream_mock = MagicMock()
        ole_stream_mock.read.return_value = b'hello\x00\nworld\t'
        ole_mock.openstream.return_value = ole_stream_mock
        ole_cls_mock.return_value = ole_mock

        text = _parse_doc(io.BytesIO(b'legacy-doc-content'))

        self.assertEqual(text, 'hello\nworld\t')
        ole_mock.exists.assert_called_once_with('WordDocument')
        ole_mock.openstream.assert_called_once_with('WordDocument')

    @patch('aiworks_core.logic.document_parser.extract_text_with_antiword', side_effect=RuntimeError('antiword failed'))
    @patch('aiworks_core.logic.document_parser.olefile.OleFileIO', side_effect=RuntimeError('ole failed'))
    def test_parse_doc_returns_empty_when_all_methods_fail(self, _ole_cls_mock, _extract_text_mock):
        """Test that .doc parser returns empty text when all parse strategies fail."""
        text = _parse_doc(io.BytesIO(b'legacy-doc-content'))
        self.assertEqual(text, '')
