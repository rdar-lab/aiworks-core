"""
Tests for validation_tools.py — validate_json, validate_html, validate_xml,
validate_yaml, validate_css, validate_js, and get_validation_tools.
"""

from unittest.mock import patch

from django.test import TestCase

from ..logic.json_repair import plan_json_repair
from ..logic.validation_tools import (
    get_validation_tools,
    validate_html,
    validate_js,
    validate_json,
    validate_xml,
    validate_yaml,
)


class ValidateJSONTests(TestCase):
    """Tests for validate_json."""

    def test_valid_json_dict(self):
        result = validate_json.run('{"key": "value", "num": 42}')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_json_array(self):
        result = validate_json.run('[1, 2, "three"]')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_json_nested(self):
        payload = '{"a": {"b": [1, 2, 3]}, "c": null}'
        result = validate_json.run(payload)
        self.assertTrue(result["is_valid"])

    def test_invalid_trailing_comma(self):
        result = validate_json.run('{"key": "value",}')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_missing_quotes(self):
        result = validate_json.run('{key: "value"}')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_unmatched_brace(self):
        result = validate_json.run('{"key": "value"')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_bad_data_type(self):
        result = validate_json.run('{"key": undefined}')
        self.assertFalse(result["is_valid"])

    def test_empty_string(self):
        result = validate_json.run('')
        self.assertFalse(result["is_valid"])

    def test_non_json_string(self):
        result = validate_json.run("not json at all")
        self.assertFalse(result["is_valid"])

    def test_runtime_error_malformed_json(self):
        result = validate_json.run("{}garbage")
        self.assertFalse(result["is_valid"])


class ValidateHTMLTests(TestCase):
    """Tests for validate_html."""

    def test_valid_html_minimal(self):
        result = validate_html.run('<!DOCTYPE html><html><body>Hello</body></html>')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_html_fragment_without_doctype(self):
        result = validate_html.run('<div><p>Paragraph</p></div>')
        self.assertFalse(result["is_valid"])
        self.assertIn("expected-doctype-but-got-start-tag", result["report"])

    def test_valid_html_self_closing_without_doctype(self):
        result = validate_html.run('<img src="x.png" alt="x"/>')
        self.assertFalse(result["is_valid"])
        self.assertIn("expected-doctype-but-got-start-tag", result["report"])

    def test_invalid_unclosed_tag(self):
        result = validate_html.run('<div><p>Unclosed')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_mismatched_tags(self):
        result = validate_html.run('<div></span>content</div>')
        self.assertFalse(result["is_valid"])

    def test_invalid_bad_attribute(self):
        result = validate_html.run('<img src="x" alt=x>')
        self.assertFalse(result["is_valid"])

    def test_invalid_raw_text(self):
        result = validate_html.run('<script>')
        self.assertFalse(result["is_valid"])

    def test_parser_failure_runtime_error(self):
        with patch("html5lib.HTMLParser.parse", side_effect=RuntimeError("boom")):
            result = validate_html.run('<div>test</div>')
        self.assertFalse(result["is_valid"])
        self.assertIn("CRITICAL PARSER FAILURE", result["report"])


class ValidateXMLTests(TestCase):
    """Tests for validate_xml."""

    def test_valid_xml_simple(self):
        result = validate_xml.run('<root><item>value</item></root>')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_xml_with_attributes(self):
        result = validate_xml.run('<root attr="val"><child id="1"/></root>')
        self.assertTrue(result["is_valid"])

    def test_valid_xml_declaration(self):
        result = validate_xml.run('<?xml version="1.0" encoding="UTF-8"?><root/>')
        self.assertTrue(result["is_valid"])

    def test_invalid_unclosed_tag(self):
        result = validate_xml.run('<root><item>value</root>')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_bad_attribute_quoting(self):
        result = validate_xml.run('<root attr="val><child/></root>')
        self.assertFalse(result["is_valid"])

    def test_invalid_missing_root(self):
        result = validate_xml.run('<item>value</item><item>value2</item>')
        self.assertFalse(result["is_valid"])

    def test_invalid_mismatched_closing_tag(self):
        result = validate_xml.run('<div></span>content</div>')
        self.assertFalse(result["is_valid"])

    def test_empty_string(self):
        result = validate_xml.run('')
        self.assertFalse(result["is_valid"])

    def test_runtime_error(self):
        with patch("xml.etree.ElementTree.fromstring", side_effect=RuntimeError("engine failure")):
            result = validate_xml.run('<root/>')
        self.assertFalse(result["is_valid"])
        self.assertIn("CRITICAL RUNTIME ERROR", result["report"])


class ValidateYAMLTests(TestCase):
    """Tests for validate_yaml."""

    def test_valid_yaml_simple(self):
        result = validate_yaml.run('key: value\nnum: 42')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_yaml_nested(self):
        yaml_str = """
parent:
  child: value
  list:
    - item1
    - item2
"""
        result = validate_yaml.run(yaml_str)
        self.assertTrue(result["is_valid"])

    def test_valid_yaml_list(self):
        result = validate_yaml.run('- item1\n- item2\n- item3')
        self.assertTrue(result["is_valid"])

    def test_valid_yaml_quoted_strings(self):
        result = validate_yaml.run('key: "quoted"\nmulti: \'single\'')
        self.assertTrue(result["is_valid"])

    def test_invalid_bad_indentation(self):
        result = validate_yaml.run('key: value\n  bad_indent: x')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_tab_usage(self):
        result = validate_yaml.run('key:\tvalue')
        self.assertFalse(result["is_valid"])

    def test_invalid_unclosed_quotes(self):
        result = validate_yaml.run('key: "unclosed')
        self.assertFalse(result["is_valid"])

    def test_invalid_directive_error(self):
        result = validate_yaml.run('%YAML 1.1\nkey: value')
        self.assertFalse(result["is_valid"])


class ValidateJSTests(TestCase):
    """Tests for validate_js."""

    def test_valid_js_simple(self):
        result = validate_js.run('const x = 42;')
        self.assertTrue(result["is_valid"])
        self.assertIn("SUCCESS", result["report"])

    def test_valid_js_function(self):
        result = validate_js.run('function foo() { return true; }')
        self.assertTrue(result["is_valid"])

    def test_valid_js_arrow_function(self):
        result = validate_js.run('const add = (a, b) => a + b;')
        self.assertTrue(result["is_valid"])

    def test_valid_js_async_function(self):
        result = validate_js.run('async function test() { await fetch("/api"); }')
        self.assertTrue(result["is_valid"])

    def test_valid_js_class(self):
        result = validate_js.run('class Foo { constructor() { this.x = 1; } }')
        self.assertTrue(result["is_valid"])

    def test_valid_js_esmodule_import(self):
        result = validate_js.run('import { foo } from "bar";')
        self.assertTrue(result["is_valid"])
        self.assertIn("Module", result["report"])

    def test_valid_js_esmodule_export(self):
        result = validate_js.run('export const x = 42;')
        self.assertTrue(result["is_valid"])
        self.assertIn("Module", result["report"])

    def test_invalid_unclosed_bracket(self):
        result = validate_js.run('const arr = [1, 2;')
        self.assertFalse(result["is_valid"])
        self.assertIn("VALIDATION FAILED", result["report"])

    def test_invalid_syntax_error(self):
        result = validate_js.run('function f( { return 1; }')
        self.assertFalse(result["is_valid"])

    def test_runtime_error(self):
        with patch("esprima.parseScript", side_effect=RuntimeError("engine failure")):
            result = validate_js.run('const x = 42;')
        self.assertFalse(result["is_valid"])
        self.assertIn("CRITICAL RUNTIME ERROR", result["report"])


class GetValidationToolsTests(TestCase):
    """Tests for get_validation_tools."""

    def test_returns_six_tools(self):
        tools = get_validation_tools()
        self.assertEqual(len(tools), 6)

    def test_tools_are_langchain_tools(self):
        tools = get_validation_tools()
        for tool in tools:
            self.assertTrue(hasattr(tool, "name"))
            self.assertTrue(callable(tool.run))

    def test_tool_names(self):
        tools = get_validation_tools()
        names = {t.name for t in tools}
        expected = {"validate_xml", "validate_json", "validate_html", "validate_yaml", "validate_css", "validate_js"}
        self.assertEqual(names, expected)

    def test_tools_in_expected_order(self):
        tools = get_validation_tools()
        self.assertEqual(tools[0].name, "validate_xml")
        self.assertEqual(tools[1].name, "validate_json")
        self.assertEqual(tools[2].name, "validate_html")
        self.assertEqual(tools[3].name, "validate_yaml")
        self.assertEqual(tools[4].name, "validate_css")
        self.assertEqual(tools[5].name, "validate_js")


class PlanJSONRepairTests(TestCase):
    """Tests for plan_json_repair."""

    def _load_fixture(self, filename: str) -> str:
        import os
        path = os.path.join(os.path.dirname(__file__), "fixtures", "json_repair", filename)
        with open(path, "r") as f:
            return f.read()

    def test_valid_json_returns_no_issues(self):
        content = self._load_fixture("valid_simple.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(result["repair_plan"], "")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["error"], "")

    def test_trailing_comma(self):
        content = self._load_fixture("trailing_comma.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "trailing_comma")
        self.assertIn("Line 4", result["issues"][0]["location"])
        self.assertIn("trailing comma", result["issues"][0]["description"].lower())
        self.assertTrue(result["issues"][0]["pre_repair_preview"])
        self.assertTrue(result["issues"][0]["post_repair_preview"])
        self.assertIn("1.", result["repair_plan"])
        self.assertNotIn("write_file", result["repair_plan"])
        self.assertNotIn("validate_json", result["repair_plan"])

    def test_unclosed_object(self):
        content = self._load_fixture("unclosed_object.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "mid_structure_corruption")
        self.assertIn("Line 7", result["issues"][0]["location"])
        self.assertTrue(result["issues"][0]["pre_repair_preview"])

    def test_unclosed_array(self):
        content = self._load_fixture("unclosed_array.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "mid_structure_corruption")
        self.assertIn("Line 3", result["issues"][0]["location"])
        self.assertTrue(result["issues"][0]["pre_repair_preview"])

    def test_overclosed_bracket(self):
        content = self._load_fixture("overclosed_object.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "overclosed_bracket")
        self.assertIn("Line 4", result["issues"][0]["location"])
        self.assertTrue(result["issues"][0]["post_repair_preview"])

    def test_single_quoted_strings(self):
        content = self._load_fixture("single_quoted_strings.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "single_quoted_strings")
        self.assertTrue(result["issues"][0]["post_repair_preview"])

    def test_js_single_line_comment(self):
        content = self._load_fixture("js_single_line_comment.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "js_comment")
        self.assertTrue(result["issues"][0]["post_repair_preview"])

    def test_js_multiline_comment(self):
        content = self._load_fixture("js_multiline_comment.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "js_comment")
        self.assertTrue(result["issues"][0]["post_repair_preview"])

    def test_trailing_garbage(self):
        content = self._load_fixture("trailing_garbage.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "overclosed_bracket")
        self.assertTrue(result["issues"][0]["post_repair_preview"])
        self.assertFalse(result["issues"][0]["post_repair_preview"].endswith("END"))

    def test_mid_structure_missing_comma(self):
        content = self._load_fixture("mid_structure_missing_comma.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "mid_structure_corruption")
        self.assertIn("Line 5", result["issues"][0]["location"])

    def test_mid_structure_extra_bracket(self):
        content = self._load_fixture("mid_structure_extra_bracket.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["type"], "mid_structure_corruption")

    def test_mixed_corruptions(self):
        content = self._load_fixture("mixed_corruptions.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertGreaterEqual(len(result["issues"]), 1)
        issue_types = {issue["type"] for issue in result["issues"]}
        self.assertTrue(any(t in issue_types for t in ["single_quoted_strings", "trailing_comma", "mid_structure_corruption"]))

    def test_presentation_corrupt(self):
        content = self._load_fixture("presentation_corrupt.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertGreaterEqual(len(result["issues"]), 1)
        self.assertTrue(result["repair_plan"])

    def test_posts_corrupt(self):
        content = self._load_fixture("posts_corrupt.json")
        result = plan_json_repair(content)
        self.assertTrue(result["can_repair"])
        self.assertGreaterEqual(len(result["issues"]), 1)

    def test_repair_plan_has_numbered_steps(self):
        content = self._load_fixture("trailing_comma.json")
        result = plan_json_repair(content)
        self.assertTrue(result["repair_plan"])
        lines = result["repair_plan"].split("\n")
        for line in lines:
            self.assertTrue(line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip())

    def test_pre_repair_preview_has_line_numbers(self):
        content = self._load_fixture("trailing_comma.json")
        result = plan_json_repair(content)
        preview = result["issues"][0]["pre_repair_preview"]
        self.assertIn(":", preview)
        self.assertIn("Line", preview or "1:")

    def test_post_repair_preview_has_line_numbers(self):
        content = self._load_fixture("trailing_comma.json")
        result = plan_json_repair(content)
        post = result["issues"][0]["post_repair_preview"]
        self.assertIsNotNone(post)
        self.assertIn(":", post)
        self.assertIn("Line", post or "1:")

    def test_pre_and_post_preview_show_line_markers(self):
        content = self._load_fixture("trailing_comma.json")
        result = plan_json_repair(content)
        pre = result["issues"][0]["pre_repair_preview"]
        post = result["issues"][0]["post_repair_preview"]
        self.assertIn("→", pre)
        self.assertIsNotNone(post)