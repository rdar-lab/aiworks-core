"""
Tests for deepagent_validation_tools.py — VFS-aware validation tools.

Verifies:
- Each validator reads content from VFS via backend.read() and delegates to original validator
- FileNotFoundError when backend cannot find the file
- RuntimeError when backend is not set
- Validation errors are returned as {"is_valid": False, "report": ...}
- get_deepagent_validation_tools returns all 6 tools
"""

from unittest.mock import MagicMock, patch
from ..logic.llm import init_agent_backend
from django.test import TestCase

from ..logic.deepagent_validation_tools import (
    get_deepagent_validation_tools,
    plan_json_repair,
    validate_css,
    validate_html,
    validate_js,
    validate_json,
    validate_xml,
    validate_yaml,
)


class ReadFileFromVFSTests(TestCase):
    """Tests for _read_file_from_vfs via the validation tool call chain."""

    def setUp(self):
        init_agent_backend(None)

    def tearDown(self):
        init_agent_backend(None)

    def _make_backend(self, files: dict):
        backend = MagicMock()
        def read_side_effect(path):
            for p, data in files.items():
                if p == path or p == path.lstrip("/") or "/" + p.lstrip("/") == path:
                    result = MagicMock()
                    result.error = None
                    result.file_data = data
                    return result
            result = MagicMock()
            result.error = f"File not found: {path}"
            result.file_data = None
            return result
        backend.read.side_effect = read_side_effect
        return backend

    def _set_backend(self, backend):
        init_agent_backend(backend)

    def test_validate_json_success(self):
        backend = self._make_backend({"/config.json": {"content": '{"key": "value"}', "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value='{"key": "value"}'):
            result = validate_json.run("/config.json")
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_validate_json_file_not_found(self):
        backend = self._make_backend({})
        self._set_backend(backend)
        result = validate_json.run("/nonexistent.json")
        self.assertFalse(result["is_valid"])
        self.assertIn("FileNotFoundError", result["report"])

    def test_validate_json_backend_not_set(self):
        init_agent_backend(None)
        result = validate_json.run("/config.json")
        self.assertFalse(result["is_valid"])
        self.assertIn("Agent backend not set", result["report"])

    def test_validate_html_success(self):
        backend = self._make_backend({"/page.html": {"content": "<!DOCTYPE html><html><body>Hello</body></html>", "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value="<!DOCTYPE html><html><body>Hello</body></html>"):
            result = validate_html.run("/page.html")
        self.assertTrue(result["is_valid"])

    def test_validate_html_file_not_found(self):
        backend = self._make_backend({})
        self._set_backend(backend)
        result = validate_html.run("/nonexistent.html")
        self.assertFalse(result["is_valid"])

    def test_validate_xml_success(self):
        backend = self._make_backend({"/data.xml": {"content": "<root><item>value</item></root>", "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value="<root><item>value</item></root>"):
            result = validate_xml.run("/data.xml")
        self.assertTrue(result["is_valid"])

    def test_validate_yaml_success(self):
        backend = self._make_backend({"/config.yaml": {"content": "key: value\nnum: 42", "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value="key: value\nnum: 42"):
            result = validate_yaml.run("/config.yaml")
        self.assertTrue(result["is_valid"])

    def test_validate_css_success(self):
        backend = self._make_backend({"/style.css": {"content": "body { color: red; }", "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        result = validate_css.run("/style.css")
        self.assertTrue(result["is_valid"])

    def test_validate_js_success(self):
        backend = self._make_backend({"/script.js": {"content": "const x = 42;", "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value="const x = 42;"):
            result = validate_js.run("/script.js")
        self.assertTrue(result["is_valid"])

    def test_validate_json_invalid_content(self):
        backend = self._make_backend({"/bad.json": {"content": '{"key": "value",}', "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value='{"key": "value",}'):
            result = validate_json.run("/bad.json")
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])


class GetDeepAgentValidationToolsTests(TestCase):
    """Tests for get_deepagent_validation_tools."""

    def test_returns_eight_tools(self):
        tools = get_deepagent_validation_tools()
        self.assertEqual(len(tools), 8)

    def test_tools_are_langchain_tools(self):
        tools = get_deepagent_validation_tools()
        for tool in tools:
            self.assertTrue(hasattr(tool, "name"))
            self.assertTrue(callable(tool.run))

    def test_tool_names(self):
        tools = get_deepagent_validation_tools()
        names = {t.name for t in tools}
        expected = {"validate_json", "validate_html", "validate_xml", "validate_yaml", "validate_css", "validate_js", "validate_json_with_schema", "plan_json_repair"}
        self.assertEqual(names, expected)

    def test_tools_in_expected_order(self):
        tools = get_deepagent_validation_tools()
        self.assertEqual(tools[0].name, "validate_json")
        self.assertEqual(tools[1].name, "validate_html")
        self.assertEqual(tools[2].name, "validate_xml")
        self.assertEqual(tools[3].name, "validate_yaml")
        self.assertEqual(tools[4].name, "validate_css")
        self.assertEqual(tools[5].name, "validate_js")
        self.assertEqual(tools[6].name, "validate_json_with_schema")
        self.assertEqual(tools[7].name, "plan_json_repair")


class DeepAgentPlanJSONRepairTests(TestCase):
    """Tests for plan_json_repair tool in deepagent context."""

    def setUp(self):
        init_agent_backend(None)

    def tearDown(self):
        init_agent_backend(None)

    def _make_backend(self, files: dict):
        backend = MagicMock()
        def read_side_effect(path):
            for p, data in files.items():
                if p == path or p == path.lstrip("/") or "/" + p.lstrip("/") == path:
                    result = MagicMock()
                    result.error = None
                    result.file_data = data
                    return result
            result = MagicMock()
            result.error = f"File not found: {path}"
            result.file_data = None
            return result
        backend.read.side_effect = read_side_effect
        return backend

    def _set_backend(self, backend):
        init_agent_backend(backend)

    def _load_fixture(self, filename: str) -> str:
        import os
        path = os.path.join(os.path.dirname(__file__), "fixtures", "json_repair", filename)
        with open(path, "r") as f:
            return f.read()

    def test_plan_json_repair_success(self):
        content = self._load_fixture("trailing_comma.json")
        backend = self._make_backend({"/presentation.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/presentation.json")
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "trailing_comma")
        self.assertTrue(result["repair_plan"])
        self.assertNotIn("write_file", result["repair_plan"])
        self.assertNotIn("validate_json", result["repair_plan"])

    def test_plan_json_repair_file_not_found(self):
        backend = self._make_backend({})
        self._set_backend(backend)
        result = plan_json_repair.run("/nonexistent.json")
        self.assertFalse(result["can_repair"])
        self.assertIn("FileNotFoundError", result["error"])

    def test_plan_json_repair_backend_not_set(self):
        init_agent_backend(None)
        result = plan_json_repair.run("/presentation.json")
        self.assertFalse(result["can_repair"])
        self.assertIn("Agent backend not set", result["error"])

    def test_plan_json_repair_valid_json(self):
        content = self._load_fixture("valid_simple.json")
        backend = self._make_backend({"/config.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/config.json")
        self.assertTrue(result["can_repair"])
        self.assertEqual(result["issues"], [])

    def test_plan_json_repair_mixed_corruptions(self):
        content = self._load_fixture("mixed_corruptions.json")
        backend = self._make_backend({"/posts.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/posts.json")
        self.assertTrue(result["can_repair"])
        self.assertGreaterEqual(len(result["issues"]), 1)

    def test_plan_json_repair_single_quoted_strings(self):
        content = self._load_fixture("single_quoted_strings.json")
        backend = self._make_backend({"/data.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/data.json")
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "single_quoted_strings")

    def test_plan_json_repair_unclosed_bracket(self):
        content = self._load_fixture("unclosed_object.json")
        backend = self._make_backend({"/brand.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/brand.json")
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "mid_structure_corruption")

    def test_plan_json_repair_post_repair_preview_has_line_numbers(self):
        content = self._load_fixture("trailing_comma.json")
        backend = self._make_backend({"/presentation.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/presentation.json")
        post = result["issues"][0]["post_repair_preview"]
        self.assertIsNotNone(post)
        self.assertIn(":", post)

    def test_plan_json_repair_pre_repair_preview_has_arrow_marker(self):
        content = self._load_fixture("trailing_comma.json")
        backend = self._make_backend({"/presentation.json": {"content": content, "encoding": "utf-8", "created_at": "", "modified_at": ""}})
        self._set_backend(backend)
        with patch("deepagents.backends.utils.file_data_to_string", return_value=content):
            result = plan_json_repair.run("/presentation.json")
        pre = result["issues"][0]["pre_repair_preview"]
        self.assertIn("→", pre)