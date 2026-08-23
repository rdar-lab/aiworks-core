"""Schema validation utilities for structured JSON output from LLM agents."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from django.conf import settings
from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)


def _get_lib_schema_path(filename: str) -> Path:
    """Return the absolute path to a schema file in aiworks_core/resources/schema/."""
    return Path(__file__).resolve().parent.parent / "resources" / "schema" / filename


def validate_json_with_schema_file(content: str, schema_file_path: str) -> tuple[bool, str]:
    """Validate JSON content against a JSON schema file.

    Returns (is_valid, error_message).
    error_message is a human-readable dot-notation path for recovery prompts.

    If *schema_file_path* is not absolute, it is first resolved relative to
    ``settings.BASE_DIR``.  If the file is not found there, it is resolved
    relative to ``aiworks_core/resources/schema/`` (handy in tests that bundle
    schemas alongside the library source).
    """
    if not os.path.isabs(schema_file_path):
        resolved = Path(settings.BASE_DIR) / schema_file_path
        if not resolved.exists():
            # Fall back to library-bundled schema (e.g. aiworks_core/resources/schema/)
            lib_schema_dir = Path(__file__).resolve().parent.parent / "resources" / "schema"
            resolved = lib_schema_dir / Path(schema_file_path).name
        schema_file_path = str(resolved)

    try:
        with open(schema_file_path) as f:
            schema = f.read()
    except FileNotFoundError:
        logger.error("validate_json_with_schema_file | schema file not found: %s", schema_file_path)
        return False, f"Schema file not found: {schema_file_path}"

    return validate_json_schema(content, schema)


def validate_json_schema(content: str, schema: str) -> tuple[bool, str]:
    """Validate JSON content against a JSON schema.

    Returns (is_valid, error_message).
    error_message is a human-readable dot-notation path for recovery prompts.
    """
    try:
        parsed_schema = json.loads(schema)
    except json.JSONDecodeError as e:
        logger.warning("validate_json_schema | malformed Schema JSON: %s", e)
        return False, f"Malformed schema JSON: {e}"

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("validate_json_schema | malformed JSON: %s", e)
        return False, f"Malformed JSON: {e}"

    try:
        validate(instance=parsed, schema=parsed_schema)
        return True, ""
    except ValidationError as e:
        try:
            path = _format_instance_path(e.path)
            return False, f"{path} {e.message}"
        except Exception as exp:
            logger.warning(f"Was unable to format specific error, returning the original, due to {exp}")
            return False, repr(e)


def _format_instance_path(path: Optional[object]) -> str:
    """Format a ValidationError instance_path deque as dot-notation string."""
    if not path:
        return ""
    parts = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            if parts:
                parts.append(f".{p}")
            else:
                parts.append(p)
    return "".join(parts)


def _resolve_ref(schema: dict, root_schema: Optional[dict] = None) -> dict:
    """Resolve $ref within a schema to the actual definition, following entire chain.

    Handles #/path/to/value refs by walking the root_schema.
    Keeps resolving until a schema without $ref is found.
    """
    if not isinstance(schema, dict) or "$ref" not in schema:
        return schema

    ref = schema["$ref"]
    if not ref.startswith("#/"):
        return schema

    # Start from root_schema (the full document), not from the local schema
    parts = ref[2:].split("/")
    current = root_schema if root_schema else schema
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return schema

    if not isinstance(current, dict):
        return schema

    # If the resolved schema also has $ref, keep resolving
    if "$ref" in current:
        return _resolve_ref(current, root_schema)

    return current


def _validate_against_schema(data: Any, schema: dict, root_schema: Optional[dict] = None) -> bool:
    """Check if data matches schema discriminator constraints.

    This is a lightweight validation for oneOf branch selection.
    Checks type discriminator (const or enum) if present.
    Full $ref-based validation would require more complex registry setup.
    """
    type_constraint = schema.get('properties', {}).get('type', {})
    if isinstance(type_constraint, dict):
        if 'const' in type_constraint:
            if data.get('type') != type_constraint['const']:
                return False
        elif 'enum' in type_constraint:
            if data.get('type') not in type_constraint['enum']:
                return False
    return True


def _strip_dict_to_schema(data: dict, schema: dict, root_schema: Optional[dict] = None) -> dict:
    """Strip a dict to only include properties defined in schema (one-level, no validation).

    For use in oneOf branch trials - strip to shape without validation.
    Resolves $ref in schema before stripping.
    """
    if not isinstance(data, dict):
        return {}

    # Resolve $ref if present
    resolved = schema
    if "$ref" in schema and root_schema:
        resolved = _resolve_ref(schema, root_schema)

    result = {}
    props = resolved.get("properties", {})
    additional_props = resolved.get("additionalProperties")

    for key in data:
        if key in props:
            result[key] = data[key]
        elif additional_props is True:
            result[key] = data[key]
    return result


def force_to_schema(
        data: Any, schema: Any, path: str = "", _resolved_cache: Optional[dict] = None,
        _root_schema: Optional[dict] = None
) -> Tuple[Any | None, list[str]]:
    """Force data to conform to schema. Returns (coerced_data, removed_paths).

    - oneOf: first branch that validates wins; none pass -> remove (path returned)
    - allOf: all branches must pass (as-is); any fail -> remove (path returned)
    - dict: strip unknown/invalid properties; check required; remove if missing
    - array: strip invalid items; removed items tracked with paths
    - $ref: resolve and recurse (with cycle detection)
    - removed_paths only contains element-level removals (dicts/array items),
      not individual property removals

    Args:
        data: The data to coerce
        schema: The JSON schema to validate against (dict, list, str, bool, or None)
        path: Current JSON path for tracking removals (internal use)
        _resolved_cache: Internal cache to prevent infinite recursion on $ref cycles
        _root_schema: The root schema for resolving $ref (defaults to schema if not provided)

    Returns:
        Tuple of (coerced_data, removed_paths).
        If coerced_data is None, the element should be removed.
        removed_paths is a list of paths (e.g., slides[0].elements[3]) that were removed.
    """
    # Initialize per-call cache for cycle detection
    if _resolved_cache is None:
        _resolved_cache = {}

    # Root schema for $ref resolution
    if _root_schema is None:
        _root_schema = schema if isinstance(schema, dict) else {}

    # Handle None / non-dict schema
    if schema is None:
        return data, []

    # Handle boolean schema before dict check - False always rejects
    if isinstance(schema, bool):
        if schema:
            return data, []
        else:
            return (None, [path]) if path else (None, [])

    if not isinstance(schema, dict):
        return data, []

    # Handle $ref - resolve with cycle detection
    if "$ref" in schema:
        # Cache by (schema_id, path) to detect cycles at same path level
        cache_key = (id(schema), path)
        if cache_key in _resolved_cache:
            # Already resolved this schema at this path - avoid cycle
            return data, []
        _resolved_cache[cache_key] = True

        resolved = _resolve_ref(schema, _root_schema)
        return force_to_schema(data, resolved, path, _resolved_cache, _root_schema)

    # Handle oneOf | anyOf (known issue oneOf mandate that exactly one branch fits, we don't check that)
    if "oneOf" in schema or "anyOf" in schema:
        for branch in schema.get("oneOf") or schema.get("anyOf") or []:
            # Resolve branch ref for recursion
            resolved_branch = branch
            if "$ref" in branch:
                resolved_branch = _resolve_ref(branch, _root_schema)
            # Try recursing with resolved branch directly
            # Let recursion handle validation and coercion
            coerced, sub_removed = force_to_schema(data, resolved_branch, path, _resolved_cache, _root_schema)
            if coerced is not None:
                # Verify the coercion actually validates against the schema.
                # This is needed because object-branch coercion strips unknown properties
                # but doesn't validate discriminator constraints (e.g., type=table vs type=text).
                if _validate_against_schema(coerced, resolved_branch, _root_schema):
                    return coerced, sub_removed
            # Branch failed or invalid, try next
        # No branch passed - remove element
        return (None, [path]) if path else (None, [])

    # Handle allOf
    if "allOf" in schema:
        coerced = data
        removed = []
        for branch in schema["allOf"]:
            resolved_branch = branch
            if "$ref" in branch:
                resolved_branch = _resolve_ref(branch, _root_schema)

            sub_coerced, sub_removed = force_to_schema(coerced, resolved_branch, path, _resolved_cache, _root_schema)
            if not sub_coerced or not _validate_against_schema(sub_coerced, branch):
                # This branch failed - remove element
                return (None, [path]) if path else (None, [])
            else:
                coerced = sub_coerced
                removed.extend(sub_removed)
        # All branches passed
        return coerced, removed

    # Handle array type
    if schema.get("type") == "array" and "items" in schema:
        if not isinstance(data, list):
            return (None, [path]) if path else (None, [])
        result = []
        removed = []
        items_schema = schema["items"]
        for i, item in enumerate(data):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            coerced, sub_removed = force_to_schema(item, items_schema, item_path, _resolved_cache, _root_schema)
            if coerced is not None:
                result.append(coerced)
                removed.extend(sub_removed)
            else:
                removed.append(item_path)
        return result, removed

    # Handle primitive type validation
    if "type" in schema and not any(k in schema for k in ("properties", "oneOf", "allOf", "anyOf")):
        expected_type = schema["type"]
        if expected_type == "string" and not isinstance(data, str):
            return (None, [path]) if path else (None, [])
        if expected_type == "number" and not isinstance(data, (int, float)):
            return (None, [path]) if path else (None, [])
        if expected_type == "integer" and not isinstance(data, int):
            return (None, [path]) if path else (None, [])
        if expected_type == "boolean" and not isinstance(data, bool):
            return (None, [path]) if path else (None, [])
        if expected_type == "null" and data is not None:
            return (None, [path]) if path else (None, [])
        # type matches or no specific type constraint - return as-is
        return data, []

    # Handle object type
    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(data, dict):
            return (None, [path]) if path else (None, [])
        result = {}
        removed_paths = []
        props = schema.get("properties", {})
        required_fields = set(schema.get("required", []))
        additional_props = schema.get("additionalProperties")

        for key, value in data.items():
            if key in props:
                prop_schema = props[key]
                coerced, sub_removed = force_to_schema(value, prop_schema, f"{path}.{key}" if path else key,
                                                       _resolved_cache, _root_schema)
                if coerced is not None:
                    result[key] = coerced
                    removed_paths.extend(sub_removed)
            elif isinstance(additional_props, bool) and additional_props:
                result[key] = value
            elif isinstance(additional_props, dict):
                coerced, sub_removed = force_to_schema(value, additional_props, f"{path}.{key}" if path else key,
                                                       _resolved_cache, _root_schema)
                if coerced is not None:
                    result[key] = value
                    removed_paths.extend(sub_removed)
            else:
                removed_paths.append(f"{path}.{key}" if path else key)

        # Check required fields
        for req in required_fields:
            if req not in result:
                return (None, [path]) if path else (None, [])

        return result, removed_paths

    # Default: return data as-is
    return data, []
