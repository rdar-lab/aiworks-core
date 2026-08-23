import json
import xml.etree.ElementTree as ET

import esprima
import html5lib
import tinycss2
import yaml
from langchain_core.tools import tool


@tool
def validate_json(json_content: str) -> dict:
    """
    Validates a JSON content string for strict syntax and structural validity.
    Catches trailing commas, missing quotes, unmatched braces, and bad data types.

    Args:
        json_content: The raw JSON string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    try:
        json.loads(json_content)
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The JSON document is structurally valid and well-formed."
        }
    except json.JSONDecodeError as e:
        report_text = (
            f"✗ VALIDATION FAILED: JSON syntax error detected.\n"
            f"- [Line {e.lineno}, Col {e.colno}]: {e.msg}"
        )
        return {
            "is_valid": False,
            "report": report_text
        }
    except Exception as e:
        return {
            "is_valid": False,
            "report": f"CRITICAL RUNTIME ERROR: Unexpected parsing engine failure.\nDetails: {str(e)}"
        }


@tool
def validate_html(html_content: str) -> dict:
    """
    Validates HTML content string for structural syntax errors using the HTML5 specification.

    Args:
        html_content: The raw HTML string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    parser = html5lib.HTMLParser()
    try:
        parser.parse(html_content)
    except Exception as e:
        return {
            "is_valid": False,
            "report": f"CRITICAL PARSER FAILURE: Could not execute parse engine.\nDetails: {str(e)}"
        }

    if not parser.errors:
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The HTML structure is valid according to the HTML5 specification."
        }

    error_lines = []
    for pos, error_code, datadict in parser.errors:
        # Crucial Fix: html5lib occasionally passes pos as None for structural file anomalies
        if pos and isinstance(pos, tuple):
            line, col = pos
        else:
            line, col = 1, 1

        detail = f"- [Line {line}, Col {col}]: {error_code}"
        if datadict:
            context = ", ".join(f"{k}: {v}" for k, v in datadict.items())
            detail += f" ({context})"
        error_lines.append(detail)

    report_text = "✗ VALIDATION FAILED:\n" + "\n".join(error_lines)
    return {
        "is_valid": False,
        "report": report_text
    }


@tool
def validate_xml(xml_content: str) -> dict:
    """
    Validates an XML content string for strict well-formedness syntax rules.
    Catches unclosed tags, attribute quoting errors, casing mismatches, and root violations.

    Args:
        xml_content: The raw XML string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    try:
        ET.fromstring(xml_content)
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The XML document is well-formed and syntactically valid."
        }
    except ET.ParseError as e:
        line, col = e.position if e.position else ("unknown", "unknown")
        report_text = (
            f"✗ VALIDATION FAILED: XML syntax error detected.\n"
            f"- [Line {line}, Col {col}]: {str(e)}"
        )
        return {
            "is_valid": False,
            "report": report_text
        }
    except Exception as e:
        return {
            "is_valid": False,
            "report": f"CRITICAL RUNTIME ERROR: Unexpected parsing engine failure.\nDetails: {str(e)}"
        }


@tool
def validate_yaml(yaml_content: str) -> dict:
    """
    Validates a YAML content string for structural correctness and syntax validity.
    Catches layout errors like bad indentation, missing colons, tab usage, or unclosed quotes.

    Args:
        yaml_content: The raw YAML string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    try:
        yaml.safe_load(yaml_content)
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The YAML document is structurally valid and well-formed."
        }
    except yaml.YAMLError as e:
        report_lines = ["✗ VALIDATION FAILED: YAML syntax or indentation error detected."]

        if isinstance(e, yaml.error.MarkedYAMLError) and e.problem_mark is not None:
            mark = e.problem_mark
            line = mark.line + 1
            col = mark.column + 1

            error_msg = e.problem if e.problem else "Syntax anomaly"
            report_lines.append(f"- [Line {line}, Col {col}]: {error_msg}")

            if e.context is not None:
                report_lines.append(f"  Context: {e.context}")
        else:
            report_lines.append(f"- Details: {str(e)}")

        return {
            "is_valid": False,
            "report": "\n".join(report_lines)
        }


@tool
def validate_css(css_content: str) -> dict:
    """
    Validates a CSS style content string for syntax errors, unclosed braces,
    mismatched tokens, and broken structural layout configurations.

    Args:
        css_content: The raw CSS string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    error_lines = []

    rules = tinycss2.parse_stylesheet(
        css_content,
        skip_comments=True,
        skip_whitespace=True
    )

    for rule in rules:
        if rule.type == 'error':
            # Crucial Fix: Fallback for missing/None properties on top-level error components
            line = getattr(rule, 'source_line', 1) or 1
            col = getattr(rule, 'source_column', 1) or 1
            msg = getattr(rule, 'message', 'Malformed rule declaration')
            error_lines.append(f"- [Line {line}, Col {col}]: Structural Rule Error ({msg})")

    for rule in rules:
        if hasattr(rule, 'content') and rule.content:
            for token in rule.content:
                if token.type == 'error':
                    # Crucial Fix: Fallback for missing/None internal parsing token attributes
                    line = getattr(token, 'source_line', 1) or 1
                    col = getattr(token, 'source_column', 1) or 1
                    msg = getattr(token, 'message', 'Invalid declaration sequence token')
                    error_lines.append(f"- [Line {line}, Col {col}]: Declaration Syntax Error ({msg})")

    if not error_lines:
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The CSS stylesheet structure is well-formed and valid."
        }

    unique_errors = sorted(list(set(error_lines)))
    report_text = "✗ VALIDATION FAILED: CSS parsing anomalies detected.\n" + "\n".join(unique_errors)
    return {
        "is_valid": False,
        "report": report_text
    }


@tool
def validate_js(js_content: str) -> dict:
    """
    Validates a JavaScript source code string for strict syntax correctness.
    Supports modern ECMAScript standards (up to ECMA 2025).

    Args:
        js_content: The raw JavaScript string to validate.

    Returns:
        A dictionary containing the boolean 'is_valid' flag and a detailed 'report'.
    """
    try:
        esprima.parseScript(js_content)
        return {
            "is_valid": True,
            "report": "✓ SUCCESS: The JavaScript code is structurally valid with clean syntax."
        }
    except esprima.Error as e:
        if "Unexpected token" in str(e) or "Cannot use import statement outside a module" in str(e):
            try:
                esprima.parseModule(js_content)
                return {
                    "is_valid": True,
                    "report": "✓ SUCCESS: The ES6 Module code is structurally valid with clean syntax."
                }
            except esprima.Error as module_error:
                e = module_error

        line = getattr(e, 'lineNumber', 1)
        col = getattr(e, 'column', 1)
        error_description = getattr(e, 'description', str(e).split(':', 1)[-1].strip())

        return {
            "is_valid": False,
            "report": f"✗ VALIDATION FAILED: JavaScript syntax error detected.\n- [Line {line}, Col {col}]: {error_description}"
        }
    except Exception as e:
        return {
            "is_valid": False,
            "report": f"CRITICAL RUNTIME ERROR: Unexpected parsing engine failure.\nDetails: {str(e)}"
        }


def get_validation_tools():
    return [
        validate_xml,
        validate_json,
        validate_html,
        validate_yaml,
        validate_css,
        validate_js
    ]