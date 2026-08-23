"""
Tests for filename sanitization utilities: make_safe_filename and make_content_disposition.
"""

from django.test import TestCase

from ..logic.logic_utils import make_safe_filename
from ..views.views_utils import make_content_disposition


class MakeSafeFilenameTests(TestCase):
    """Tests for logic_utils.make_safe_filename."""

    def test_pure_ascii_passes_through_unchanged(self):
        self.assertEqual(make_safe_filename("report.txt"), "report.txt")
        self.assertEqual(make_safe_filename("my-document_v1.pdf"), "my-document_v1.pdf")

    def test_alphanumeric_dots_underscores_hyphens_preserved(self):
        self.assertEqual(make_safe_filename("file-name_2.0.tar.gz"), "file-name_2.0.tar.gz")

    def test_null_byte_replaced(self):
        self.assertEqual(make_safe_filename("file\x00name.txt"), "file_name.txt")

    def test_forward_slash_replaced(self):
        self.assertEqual(make_safe_filename("path/to/file.txt"), "path_to_file.txt")

    def test_backslash_replaced(self):
        self.assertEqual(make_safe_filename("path\\to\\file.txt"), "path_to_file.txt")

    def test_colon_replaced(self):
        self.assertEqual(make_safe_filename("file:name.txt"), "file_name.txt")

    def test_asterisk_replaced(self):
        self.assertEqual(make_safe_filename("file*name.txt"), "file_name.txt")

    def test_question_mark_replaced(self):
        self.assertEqual(make_safe_filename("file?name.txt"), "file_name.txt")

    def test_double_quote_replaced(self):
        self.assertEqual(make_safe_filename('file"name.txt'), "file_name.txt")

    def test_less_than_replaced(self):
        self.assertEqual(make_safe_filename("file<name.txt"), "file_name.txt")

    def test_greater_than_replaced(self):
        self.assertEqual(make_safe_filename("file>name.txt"), "file_name.txt")

    def test_pipe_replaced(self):
        self.assertEqual(make_safe_filename("file|name.txt"), "file_name.txt")

    def test_all_dangerous_chars_replaced(self):
        self.assertEqual(
            make_safe_filename("a/b\\c:d*e?f\"g<h>i|f.txt"),
            "a_b_c_d_e_f_g_h_i_f.txt",
        )

    def test_unicode_chinese_pass_through(self):
        self.assertEqual(make_safe_filename("报告.txt"), "报告.txt")
        self.assertEqual(make_safe_filename("日本語ファイル.pdf"), "日本語ファイル.pdf")

    def test_unicode_arabic_pass_through(self):
        self.assertEqual(make_safe_filename("تقرير.pdf"), "تقرير.pdf")

    def test_unicode_hebrew_pass_through(self):
        self.assertEqual(make_safe_filename("דוח.pdf"), "דוח.pdf")

    def test_mixed_ascii_unicode_preserves_unicode(self):
        self.assertEqual(make_safe_filename("report_日本語.txt"), "report_日本語.txt")

    def test_mixed_with_dangerous_chars(self):
        self.assertEqual(make_safe_filename("path/to/日本語.txt"), "path_to_日本語.txt")

    def test_empty_string_returns_empty(self):
        self.assertEqual(make_safe_filename(""), "")

    def test_path_parts_multiple_slashes(self):
        self.assertEqual(make_safe_filename("a/b/c/file.txt"), "a_b_c_file.txt")


class MakeContentDispositionTests(TestCase):
    """Tests for views_utils.make_content_disposition."""

    def test_pure_ascii_returns_simple_header(self):
        result = make_content_disposition("report.txt")
        self.assertEqual(result, 'attachment; filename="report.txt"')

    def test_ascii_filename_with_dangerous_chars_sanitized(self):
        result = make_content_disposition("file:name.txt")
        self.assertEqual(result, 'attachment; filename="file_name.txt"')

    def test_non_ascii_returns_rfc598_encoding(self):
        result = make_content_disposition("报告.txt")
        self.assertIn('filename*=UTF-8\'\'', result)
        self.assertIn("报告", result)
        self.assertEqual(result.count(";"), 2)

    def test_unicode_with_dangerous_chars_uses_rfc598(self):
        result = make_content_disposition("path/to/日本語.txt")
        self.assertIn('filename*=UTF-8\'\'', result)
        self.assertIn("path_to", result)

    def test_unicode_with_only_safe_chars_returns_rfc598(self):
        result = make_content_disposition("דוח")
        self.assertIn('filename*=UTF-8\'\'', result)
        self.assertIn("דוח", result)

    def test_simple_zip_extension(self):
        result = make_content_disposition("package-files.zip")
        self.assertIn(".zip", result)

    def test_simple_pdf_extension(self):
        result = make_content_disposition("report.pdf")
        self.assertIn(".pdf", result)

    def test_simple_docx_extension(self):
        result = make_content_disposition("document.docx")
        self.assertIn(".docx", result)

    def test_hebrew_non_ascii_returns_rfc598(self):
        result = make_content_disposition("דוח.pdf")
        self.assertIn('filename*=UTF-8\'\'', result)

    def test_mixed_language_non_ascii_returns_rfc598(self):
        result = make_content_disposition("rapport_日本語_final.pdf")
        self.assertIn('filename*=UTF-8\'\'', result)