"""Tests for schema_validation_utils.py"""

import json
import tempfile
import os
from jsonschema import validators, ValidationError
from ..logic.schema_validation_utils import (
    validate_json_with_schema_file,
    _format_instance_path,
    _validate_against_schema,
    _resolve_ref,
    _strip_dict_to_schema,
    force_to_schema,
)


def _validate_with_schema(data, schema):
    """Validate data against schema using jsonschema.validators.validate. Returns (is_valid, error)."""
    try:
        validators.validate(data, schema)
        return True, None
    except ValidationError as e:
        return False, e.message


class ValidateSchemaTests:
    def test_valid_content_passes_schema_validation(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }
        content = {"name": "Alice", "age": 30}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(json.dumps(content), schema_path)
            assert is_valid is True
            assert error == ""
        finally:
            os.unlink(schema_path)

    def test_invalid_content_fails_schema_validation(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }
        content = {"name": "Alice", "age": "not_a_number"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(json.dumps(content), schema_path)
            assert is_valid is False
            assert "age" in error
        finally:
            os.unlink(schema_path)

    def test_nested_path_extraction(self):
        schema = {
            "type": "object",
            "required": ["slides"],
            "properties": {
                "slides": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "elements"],
                        "properties": {
                            "type": {"type": "string"},
                            "elements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["type", "font_size"],
                                    "properties": {
                                        "type": {"type": "string"},
                                        "font_size": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        content = {
            "slides": [
                {
                    "type": "content",
                    "elements": [
                        {"type": "text", "font_size": "large"},
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(json.dumps(content), schema_path)
            assert is_valid is False
            assert "slides[0].elements[0].font_size" in error
        finally:
            os.unlink(schema_path)

    def test_malformed_json_returns_false(self):
        schema = {"type": "object"}
        malformed_json = "{ this is not valid json }"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(malformed_json, schema_path)
            assert is_valid is False
            assert "Malformed JSON" in error
        finally:
            os.unlink(schema_path)

    def test_schema_file_not_found_returns_false(self):
        is_valid, error = validate_json_with_schema_file('{"name": "test"}', "/nonexistent/path/schema.json")
        assert is_valid is False
        assert "Schema file not found" in error

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }
        content = {"name": "Alice"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(json.dumps(content), schema_path)
            assert is_valid is False
        finally:
            os.unlink(schema_path)

    def test_invalid_type_wrong_type_produces_error(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "number"},
            },
        }
        content = {"count": "not a number"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name

        try:
            is_valid, error = validate_json_with_schema_file(json.dumps(content), schema_path)
            assert is_valid is False
        finally:
            os.unlink(schema_path)


class FormatInstancePathTests:
    def test_empty_path_returns_empty_string(self):
        result = _format_instance_path([])
        assert result == ""

    def test_integer_indices_use_bracket_notation(self):
        result = _format_instance_path(["slides", 0, "elements", 2, "font_size"])
        assert "slides[0].elements[2].font_size" == result

    def test_first_element_is_string(self):
        result = _format_instance_path(["slides", 0])
        assert result == "slides[0]"


# =============================================================================
# Tests for _validate_against_schema (discriminator validation)
# =============================================================================

class ValidateAgainstSchemaDiscriminatorTests:
    """Tests for the discriminator-based schema validation used in oneOf branch selection."""

    def test_const_type_matches(self):
        """Data.type matches schema.properties.type.const -> valid."""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "table"}, "position": {"type": "object"}},
        }
        data = {"type": "table", "position": {"x": 0, "y": 0}}
        assert _validate_against_schema(data, schema) is True

    def test_const_type_mismatch_rejected(self):
        """Data.type does NOT match schema.properties.type.const -> invalid."""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "text"}, "position": {"type": "object"}},
        }
        data = {"type": "table", "position": {"x": 0, "y": 0}}
        assert _validate_against_schema(data, schema) is False

    def test_enum_type_in_list_valid(self):
        """Data.type is in schema.properties.type.enum -> valid."""
        schema = {
            "type": "object",
            "properties": {"type": {"enum": ["text", "rect", "image", "table"]}},
        }
        data = {"type": "table"}
        assert _validate_against_schema(data, schema) is True

    def test_enum_type_not_in_list_invalid(self):
        """Data.type is NOT in schema.properties.type.enum -> invalid."""
        schema = {
            "type": "object",
            "properties": {"type": {"enum": ["text", "rect", "image"]}},
        }
        data = {"type": "table"}
        assert _validate_against_schema(data, schema) is False

    def test_no_type_constraint_always_valid(self):
        """Schema without type constraint always passes."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"name": "Alice", "extra": "ignored"}
        assert _validate_against_schema(data, schema) is True

    def test_missing_type_in_data_when_const_expected_invalid(self):
        """Data missing type field when schema expects const -> invalid."""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "text"}},
        }
        data = {"position": {"x": 0}}
        assert _validate_against_schema(data, schema) is False

    def test_rect_element_type_discriminator(self):
        """Rect element with type=rect passes rect discriminator check."""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "rect"}, "position": {"type": "object"}},
        }
        data = {"type": "rect", "position": {"x": 1, "y": 2, "width": 10, "height": 5}, "fill": "#FF0000"}
        assert _validate_against_schema(data, schema) is True

    def test_image_element_type_discriminator(self):
        """Image element with type=image passes image discriminator check."""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "image"}, "position": {"type": "object"}},
        }
        data = {"type": "image", "position": {"x": 0, "y": 0}, "image_file_name": "logo.png"}
        assert _validate_against_schema(data, schema) is True


# =============================================================================
# Tests for _resolve_ref
# =============================================================================

class ResolveRefTests:
    """Tests for $ref resolution within JSON schemas."""

    def test_resolves_simple_ref(self):
        """$ref pointing to a definition is resolved."""
        schema = {
            "definitions": {"person": {"type": "object", "properties": {"name": {"type": "string"}}}},
            "properties": {"person": {"$ref": "#/definitions/person"}},
        }
        ref_schema = {"$ref": "#/definitions/person"}
        resolved = _resolve_ref(ref_schema, schema)
        assert resolved == {"type": "object", "properties": {"name": {"type": "string"}}}

    def test_non_ref_returns_original(self):
        """Schema without $ref is returned unchanged."""
        schema = {"type": "string"}
        assert _resolve_ref(schema, {}) == schema

    def test_ref_without_hash_prefix_returns_original(self):
        """$ref not starting with #/ is returned unchanged."""
        ref = {"$ref": "http://example.com/schema.json"}
        assert _resolve_ref(ref, {}) == ref

    def test_resolves_nested_ref(self):
        """$ref pointing to nested definition is resolved."""
        schema = {
            "definitions": {
                "address": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        }
        ref = {"$ref": "#/definitions/address"}
        resolved = _resolve_ref(ref, schema)
        assert resolved["properties"]["city"]["type"] == "string"

    def test_missing_definition_returns_original(self):
        """$ref pointing to non-existent definition returns original."""
        ref = {"$ref": "#/definitions/nonexistent"}
        result = _resolve_ref(ref, {"definitions": {}})
        assert result == ref


# =============================================================================
# Tests for _strip_dict_to_schema
# =============================================================================

class StripDictToSchemaTests:
    """Tests for stripping dicts to schema-defined properties."""

    def test_keeps_properties_in_schema(self):
        """Properties defined in schema are kept."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        result = _strip_dict_to_schema(data, schema)
        assert result == {"name": "Alice", "age": 30}
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_removes_properties_not_in_schema(self):
        """Properties NOT in schema are removed."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"name": "Alice", "extra": "value"}
        result = _strip_dict_to_schema(data, schema)
        assert result == {"name": "Alice"}
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_uses_resolved_ref_for_properties(self):
        """Resolved $ref is used to determine which properties to keep."""
        schema = {
            "definitions": {"person": {"type": "object", "properties": {"name": {"type": "string"}}}},
            "properties": {"person": {"$ref": "#/definitions/person"}},
        }
        ref_schema = {"$ref": "#/definitions/person"}
        data = {"name": "Alice", "extra": "value"}
        result = _strip_dict_to_schema(data, ref_schema, schema)
        assert result == {"name": "Alice"}
        is_valid, error = _validate_with_schema(result, {"type": "object", "properties": {"name": {"type": "string"}}})
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_empty_dict_for_non_dict_input(self):
        """Non-dict input returns empty dict."""
        assert _strip_dict_to_schema("string", {"type": "object"}) == {}
        assert _strip_dict_to_schema(None, {"type": "object"}) == {}
        assert _strip_dict_to_schema(123, {"type": "object"}) == {}


# =============================================================================
# Tests for force_to_schema - core coercion behavior
# =============================================================================

class ForceToSchemaBasicsTests:
    """Basic tests for force_to_schema function."""

    def test_passthrough_when_no_schema_constraints(self):
        """Data passes through unchanged when schema has no type/properties."""
        schema = {"type": "string"}
        data = "hello"
        result, removed = force_to_schema(data, schema, "root", {})
        assert result == "hello"
        assert removed == []

    def test_null_schema_returns_data(self):
        """Null schema returns data unchanged."""
        result, removed = force_to_schema({"key": "value"}, None, "root", {})
        assert result == {"key": "value"}

    def test_strips_unknown_properties(self):
        """Unknown properties are stripped from dicts."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        data = {"name": "Alice", "extra": "value"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result == {"name": "Alice"}
        assert removed == ["root.extra"]
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_returns_none_when_required_property_missing(self):
        """Dict missing required property returns None (should be removed)."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        data = {"age": 30}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_array_items_coerced_individually(self):
        """Array items are coerced individually - valid items pass through."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        data = ["a", "b", "c"]
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result == ["a", "b", "c"]
        assert removed == []
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_array_invalid_items_removed(self):
        """Invalid array items are removed from result."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        data = ["valid", 123, "also_valid"]
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result == ["valid", "also_valid"]
        assert "root[1]" in removed
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_non_array_data_with_array_schema_returns_none(self):
        """Non-array data with array schema returns None."""
        schema = {"type": "array", "items": {"type": "string"}}
        data = "not an array"
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None


# =============================================================================
# Tests for force_to_schema - primitive type validation
# =============================================================================

class ForceToSchemaPrimitivesTests:
    """Tests for primitive type validation."""

    def test_string_type_validated(self):
        """String type constraint validates correctly."""
        schema = {"type": "string"}
        result, _ = force_to_schema("hello", schema, "root", {}, {})
        assert result == "hello"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"
        assert force_to_schema(123, schema, "root", {}, {})[0] is None

    def test_number_type_validated(self):
        """Number type constraint validates correctly."""
        schema = {"type": "number"}
        result, _ = force_to_schema(123.5, schema, "root", {}, {})
        assert result == 123.5
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"
        assert force_to_schema("string", schema, "root", {}, {})[0] is None

    def test_integer_type_validated(self):
        """Integer type constraint validates correctly."""
        schema = {"type": "integer"}
        result, _ = force_to_schema(123, schema, "root", {}, {})
        assert result == 123
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"
        assert force_to_schema(123.5, schema, "root", {}, {})[0] is None

    def test_boolean_type_validated(self):
        """Boolean type constraint validates correctly."""
        schema = {"type": "boolean"}
        result, _ = force_to_schema(True, schema, "root", {}, {})
        assert result is True
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"
        assert force_to_schema("string", schema, "root", {}, {})[0] is None


# =============================================================================
# Integration tests - full schema coercion with ppt_schema-like structure
# =============================================================================

class ForceToSchemaIntegrationTests:
    """Integration tests mimicking ppt_schema.json structure."""

    def test_ppt_element_oneOf_all_types_preserved(self):
        """All element types (text, rect, image, table) preserve their properties."""
        ppt_schema_like = {
            "type": "object",
            "additionalProperties": False,
            "definitions": {
                "text_element": {
                    "type": "object",
                    "properties": {"type": {"const": "text"}, "position": {}, "text": {}},
                },
                "rect_element": {
                    "type": "object",
                    "properties": {"type": {"const": "rect"}, "position": {}, "fill": {}},
                },
                "image_element": {
                    "type": "object",
                    "properties": {"type": {"const": "image"}, "position": {}, "image_file_name": {}},
                },
                "table_element": {
                    "type": "object",
                    "properties": {"type": {"const": "table"}, "position": {}, "columns": {}, "rows": {}},
                },
            },
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"$ref": "#/definitions/text_element"},
                            {"$ref": "#/definitions/rect_element"},
                            {"$ref": "#/definitions/image_element"},
                            {"$ref": "#/definitions/table_element"},
                        ]
                    }
                }
            },
        }

        elements = [
            {"type": "text", "position": {"x": 0, "y": 0}, "text": "Hello"},
            {"type": "rect", "position": {"x": 0, "y": 0}, "fill": "#FF0000"},
            {"type": "image", "position": {"x": 0, "y": 0}, "image_file_name": "logo.png"},
            {
                "type": "table",
                "position": {"x": 0, "y": 0},
                "columns": [{"header": "Col1"}],
                "rows": [["Val1"]],
            },
        ]
        data = {"elements": elements}

        result, _ = force_to_schema(data, ppt_schema_like, "root", {}, ppt_schema_like)

        assert result["elements"][0]["type"] == "text"
        assert "text" in result["elements"][0]
        assert result["elements"][1]["type"] == "rect"
        assert "fill" in result["elements"][1]
        assert result["elements"][2]["type"] == "image"
        assert "image_file_name" in result["elements"][2]
        assert result["elements"][3]["type"] == "table"
        assert "columns" in result["elements"][3]
        assert "rows" in result["elements"][3]
        is_valid, error = _validate_with_schema(result, ppt_schema_like)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_mixed_slide_with_all_element_types(self):
        """Slide with mixed element types preserves each correctly."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "definitions": {
                "text_elem": {"type": "object", "properties": {"type": {"const": "text"}, "position": {}, "text": {}}},
                "rect_elem": {"type": "object", "properties": {"type": {"const": "rect"}, "position": {}, "fill": {}}},
                "table_elem": {"type": "object", "properties": {"type": {"const": "table"}, "position": {}, "columns": {}, "rows": {}}},
            },
            "properties": {
                "slides": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "elements": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"$ref": "#/definitions/text_elem"},
                                        {"$ref": "#/definitions/rect_elem"},
                                        {"$ref": "#/definitions/table_elem"},
                                    ]
                                }
                            }
                        }
                    }
                }
            },
        }

        data = {
            "slides": [
                {
                    "type": "content",
                    "elements": [
                        {"type": "rect", "position": {"x": 0, "y": 0}, "fill": "#000"},
                        {"type": "table", "position": {"x": 1, "y": 1}, "columns": [{"header": "A"}], "rows": [["B"]]},
                    ],
                }
            ]
        }

        result, removed = force_to_schema(data, schema, "root", {}, schema)

        assert len(result["slides"]) == 1
        assert len(result["slides"][0]["elements"]) == 2
        assert result["slides"][0]["elements"][0]["type"] == "rect"
        assert result["slides"][0]["elements"][1]["type"] == "table"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"


# =============================================================================
# Tests for anyOf branch handling
# =============================================================================

class ForceToSchemaAnyOfTests:
    """Tests for anyOf branch selection in force_to_schema."""

    def test_anyOf_first_branch_matches(self):
        """Data matching first anyOf branch is accepted."""
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"type": {"const": "text"}}, "additionalProperties": False},
                {"type": "object", "properties": {"type": {"const": "rect"}}, "additionalProperties": False},
            ]
        }
        data = {"type": "text", "text": "Hello"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is not None
        assert result["type"] == "text"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_anyOf_second_branch_matches(self):
        """Data matching second anyOf branch is accepted (first branch fails)."""
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"type": {"const": "text"}, "text": {"type": "string"}}, "additionalProperties": False},
                {"type": "object", "properties": {"type": {"const": "rect"}}, "additionalProperties": False},
            ]
        }
        data = {"type": "rect", "fill": "#FF0000"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is not None
        assert result["type"] == "rect"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_anyOf_no_branch_matches_returns_none(self):
        """Data matching no anyOf branch returns None."""
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"type": {"const": "text"}}, "additionalProperties": False},
                {"type": "object", "properties": {"type": {"const": "rect"}}, "additionalProperties": False},
            ]
        }
        data = {"type": "polygon", "points": []}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_anyOf_with_properties_and_additional_props(self):
        """anyOf with properties and additionalProperties mixed schema."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"properties": {"type": {"const": "text"}, "slot": {}}},
                            {"properties": {"type": {"const": "image"}, "image_data": {}}},
                        ]
                    }
                }
            }
        }
        data = {"elements": [{"type": "text", "slot": "body"}, {"type": "image", "image_data": "abc123"}]}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert len(result["elements"]) == 2
        assert result["elements"][0]["type"] == "text"
        assert result["elements"][1]["type"] == "image"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_anyOf_empty_returns_none(self):
        """Empty anyOf returns None."""
        schema = {"anyOf": []}
        data = {"type": "anything"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None


# =============================================================================
# Tests for allOf branch handling
# =============================================================================

class ForceToSchemaAllOfTests:
    """Tests for allOf branch handling in force_to_schema."""

    def test_allOf_all_branches_pass(self):
        """allOf where all branches pass returns coerced data."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": True},
                {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "additionalProperties": True},
            ]
        }
        data = {"name": "Alice", "age": 30}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_allOf_one_branch_fails_returns_none(self):
        """allOf where one branch fails returns None (missing required field)."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": True},
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["missing"], "additionalProperties": True},
            ]
        }
        data = {"name": "Alice"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_allOf_with_ref_branches(self):
        """allOf with $ref in branches resolves and validates correctly."""
        schema = {
            "definitions": {
                "has_name": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": True},
                "has_name_and_age": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "additionalProperties": True},
            },
            "allOf": [
                {"$ref": "#/definitions/has_name"},
                {"$ref": "#/definitions/has_name_and_age"},
            ]
        }
        data = {"name": "Bob", "age": 25}
        result, removed = force_to_schema(data, schema, "root", {}, schema)
        assert result is not None
        assert result["name"] == "Bob"
        assert result["age"] == 25

    def test_allOf_one_ref_branch_fails(self):
        """allOf with $ref branch that fails returns None (missing required field)."""
        schema = {
            "definitions": {
                "has_name": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": True},
                "has_name_and_age": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "required": ["age"], "additionalProperties": True},
            },
            "allOf": [
                {"$ref": "#/definitions/has_name"},
                {"$ref": "#/definitions/has_name_and_age"},
            ]
        }
        data = {"name": "Bob"}
        result, removed = force_to_schema(data, schema, "root", {}, schema)
        assert result is None

    def test_allOf_nested_oneOf_inside_branch(self):
        """allOf with oneOf inside a branch handles discriminator correctly."""
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": True,
                },
                {
                    "type": "object",
                    "properties": {
                        "elem": {
                            "oneOf": [
                                {"properties": {"type": {"const": "text"}, "text": {}}},
                                {"properties": {"type": {"const": "rect"}, "fill": {}}},
                            ]
                        }
                    },
                    "additionalProperties": True,
                },
            ]
        }
        data = {"name": "Alice", "elem": {"type": "rect", "fill": "#000000"}}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is not None
        assert result["elem"]["type"] == "rect"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_allOf_discriminator_failure(self):
        """allOf where discriminator check fails returns None."""
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"type": {"const": "text"}},
                },
                {
                    "type": "object",
                    "properties": {"position": {}},
                },
            ]
        }
        data = {"type": "image", "position": {"x": 0}}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None


# =============================================================================
# Tests for additionalProperties as dict schema
# =============================================================================

class ForceToSchemaAdditionalPropertiesDictTests:
    """Tests for additionalProperties used as a schema dict."""

    def test_additional_props_dict_accepts_valid_unknown(self):
        """Unknown props that match the additional schema are accepted."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "number"},
        }
        data = {"name": "Alice", "score": 100, "height": 175}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result["name"] == "Alice"
        assert result["score"] == 100
        assert result["height"] == 175
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_additional_props_dict_rejects_invalid_unknown(self):
        """Unknown props that don't match the additional schema are removed."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "number"},
        }
        data = {"name": "Alice", "score": 100, "city": "NYC"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result["name"] == "Alice"
        assert result["score"] == 100
        assert "city" not in result
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_additional_props_dict_nested_coerces(self):
        """Nested objects with additionalProperties dict are coerced recursively."""
        schema = {
            "type": "object",
            "properties": {"name": {}},
            "additionalProperties": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
            },
        }
        data = {"name": "Alice", "meta": {"value": 42, "extra": "ignored"}}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result["name"] == "Alice"
        assert result["meta"]["value"] == 42

    def test_additional_props_dict_in_array_items(self):
        """Array items with additionalProperties dict coerce unknown props."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tag": {"type": "string"}},
                "additionalProperties": {"type": "boolean"},
            },
        }
        data = [{"tag": "a", "flag": True}, {"tag": "b", "flag": False, "extra": "string"}]
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result[0]["tag"] == "a"
        assert result[0]["flag"] is True
        assert result[1]["tag"] == "b"
        assert result[1]["flag"] is False
        assert "extra" not in result[1]
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"


# =============================================================================
# Tests for edge cases and regression prevention
# =============================================================================

class ForceToSchemaEdgeCasesTests:
    """Edge cases and regression prevention tests."""

    def test_empty_oneOf_returns_none(self):
        """Empty oneOf returns None (element removed)."""
        schema = {"oneOf": []}
        data = {"type": "anything"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_element_without_matching_oneOf_returns_none(self):
        """Element not matching any oneOf branch returns None."""
        schema = {
            "oneOf": [
                {"properties": {"type": {"const": "text"}}},
                {"properties": {"type": {"const": "rect"}}},
            ]
        }
        data = {"type": "polygon"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_boolean_schema_true_always_passes(self):
        """Boolean schema True always passes."""
        schema = True
        data = {"anything": "allowed"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result == data
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_boolean_schema_false_always_fails(self):
        """Boolean schema False always fails."""
        schema = False
        data = {"anything": "disallowed"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result is None

    def test_strip_unknown_props_preserves_known_props(self):
        """Unknown properties are stripped but known props are preserved."""
        schema = {
            "type": "object",
            "properties": {"name": {}, "age": {}},
            "additionalProperties": False,
        }
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result == {"name": "Alice", "age": 30}
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"

    def test_strip_unknown_props_with_additional_props_dict(self):
        """additionalProperties as dict allows unknown props matching that schema."""
        schema = {
            "type": "object",
            "properties": {"name": {}},
            "additionalProperties": {"type": "string"},
        }
        data = {"name": "Alice", "city": "NYC"}
        result, removed = force_to_schema(data, schema, "root", {}, {})
        assert result["name"] == "Alice"
        assert result["city"] == "NYC"
        is_valid, error = _validate_with_schema(result, schema)
        assert is_valid, f"Result should pass schema validation: {error}"