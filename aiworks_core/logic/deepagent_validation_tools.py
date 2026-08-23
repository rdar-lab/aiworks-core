"""Deep-agent VFS-aware validation tools.

Wraps validation_tools.py validators to accept file_path instead of content.
The LLM passes a path; this module reads the file from the deep-agent VFS
(OverridingStateBackend) via threading.local() and delegates to the original validator.
"""

import logging

from deepagents.backends.utils import file_data_to_string
from langchain_core.tools import tool
from . import schema_validation_utils
from .json_repair import plan_json_repair as orig_plan_json_repair
from .llm import get_agent_backend
from .validation_tools import (
    validate_json as orig_validate_json,
    validate_html as orig_validate_html,
    validate_xml as orig_validate_xml,
    validate_yaml as orig_validate_yaml,
    validate_css as orig_validate_css,
    validate_js as orig_validate_js,
)

logger = logging.getLogger(__name__)


def _read_file_from_vfs(file_path: str) -> str:
    backend = get_agent_backend()
    if backend is None:
        raise RuntimeError(
            "Agent backend not set. Cannot validate files outside agent context."
        )
    normalized = "/" + file_path.lstrip("/")
    result = backend.read(normalized)
    if result.error:
        raise FileNotFoundError(f"Cannot read {normalized}: {result.error}")
    if result.file_data is None:
        raise FileNotFoundError(f"No file data for {normalized}")
    return file_data_to_string(result.file_data)


def _safe_read_and_validate(file_path: str, validator):
    """Utility: read file from VFS and call the validator, handling all error cases."""
    try:
        content = _read_file_from_vfs(file_path)
    except Exception as e:
        return {"is_valid": False, "report": f"Was unable to read the file: {repr(e)}"}
    try:
        return validator.run(content)
    except Exception as e:
        return {"is_valid": False, "report": f"Validation error: {repr(e)}"}


@tool
def validate_json(file_path: str) -> dict:
    """
    Validates a JSON file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the JSON file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_json)


@tool
def validate_json_with_schema(file_path: str, schema_path: str) -> dict:
    """
    Validates a JSON file by reading its content from the agent's virtual filesystem and validating against a JSON schema in the agent VFS.

    Args:
        file_path: Absolute path to the JSON file in the agent's VFS.
        schema_path: Absolute path to the JSON schema file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    try:
        content = _read_file_from_vfs(file_path)
    except Exception as e:
        return {"is_valid": False, "report": f"Was unable to read the JSON file: {repr(e)}"}

    try:
        schema = _read_file_from_vfs(schema_path)
    except Exception as e:
        return {"is_valid": False, "report": f"Was unable to read the SCHEMA file: {repr(e)}"}

    try:
        is_valid, validation_error = schema_validation_utils.validate_json_schema(content, schema)
        if is_valid:
            return {
                "is_valid": True,
                "report": "✓ SUCCESS: The JSON document passes the schema validation"
            }
        else:
            return {
                "is_valid": False,
                "report": validation_error
            }
    except Exception as e:
        return {"is_valid": False, "report": f"Validation error: {repr(e)}"}


@tool
def validate_html(file_path: str) -> dict:
    """
    Validates an HTML file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the HTML file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_html)


@tool
def validate_xml(file_path: str) -> dict:
    """
    Validates an XML file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the XML file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_xml)


@tool
def validate_yaml(file_path: str) -> dict:
    """
    Validates a YAML file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the YAML file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_yaml)


@tool
def validate_css(file_path: str) -> dict:
    """
    Validates a CSS file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the CSS file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_css)


@tool
def validate_js(file_path: str) -> dict:
    """
    Validates a JavaScript file by reading its content from the agent's virtual filesystem.

    Args:
        file_path: Absolute path to the JavaScript file in the agent's VFS.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    return _safe_read_and_validate(file_path, orig_validate_js)


@tool
def plan_json_repair(file_path: str) -> dict:
    """
    Analyzes a corrupted JSON file from the agent's virtual filesystem and returns
    a structured repair plan with pre/post line snippets for each issue.

    The agent should use the fix instructions to repair the JSON using write_file,
    then re-validate with validate_json.

    Args:
        file_path: Absolute path to the JSON file in the agent's VFS.

    Returns:
        A dictionary containing:
            - can_repair: bool (always True)
            - repair_plan: str (numbered issue-specific steps)
            - issues: list of issue dicts with type, location, description,
                      fix, pre_repair_preview, post_repair_preview
            - error: str (empty string on success)
    """
    try:
        content = _read_file_from_vfs(file_path)
    except Exception as e:
        return {
            "can_repair": False,
            "repair_plan": "",
            "issues": [],
            "error": f"Was unable to read the file: {repr(e)}"
        }
    try:
        return orig_plan_json_repair(content)
    except Exception as e:
        return {
            "can_repair": False,
            "repair_plan": "",
            "issues": [],
            "error": f"Repair planning error: {repr(e)}"
        }


def get_deepagent_validation_tools():
    return [
        validate_json,
        validate_html,
        validate_xml,
        validate_yaml,
        validate_css,
        validate_js,
        validate_json_with_schema,
        plan_json_repair,
    ]