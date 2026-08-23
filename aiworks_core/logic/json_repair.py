import json
import re


def plan_json_repair(json_content: str) -> dict:
    """
    Analyzes corrupted JSON and returns a structured repair plan with
    pre/post repair line snippets (±5 lines from each issue location).

    Args:
        json_content: The raw JSON string to analyze.

    Returns:
        A dictionary containing:
            - can_repair: bool (always True — repairable or best-effort)
            - repair_plan: str (numbered issue-specific steps, no generic follow-ups)
            - issues: list of issue dicts, each with type, location, description,
                      fix, pre_repair_preview, post_repair_preview
            - error: str (set when repair is impossible, empty string otherwise)
    """
    lines = json_content.split('\n')
    total_lines = len(lines)

    try:
        json.loads(json_content)
        return {
            "can_repair": True,
            "repair_plan": "",
            "issues": [],
            "error": ""
        }
    except json.JSONDecodeError:
        pass

    issues = []
    fix_steps = []

    error_info = _classify_json_error(json_content)
    error_type = error_info["type"]
    error_lineno = error_info["lineno"]
    error_colno = error_info["colno"]

    issue_location = f"Line {error_lineno}, Col {error_colno}"

    pre_preview = _build_line_preview(lines, error_lineno, total_lines)

    if error_type == "trailing_comma":
        fixed = _fix_trailing_comma(json_content, error_lineno, error_colno)
        post_preview = _build_preview_from_fixed(fixed, lines, error_lineno, total_lines) if fixed else None
        issues.append({
            "type": "trailing_comma",
            "location": issue_location,
            "description": "Trailing comma before closing bracket",
            "fix": "Remove the trailing comma before the closing bracket",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Line {error_lineno}: Remove trailing comma")

    elif error_type == "unclosed_bracket":
        bracket_char = error_info.get("bracket_char", "")
        fixed = json_content.rstrip() + ("}" if bracket_char == "{" else "]")
        post_preview = _build_preview_from_fixed(fixed, lines, total_lines, total_lines)
        issues.append({
            "type": "unclosed_bracket",
            "location": issue_location,
            "description": f"Unclosed '{bracket_char}' — missing closing bracket",
            "fix": f"Add closing {'}' if bracket_char == '{' else ']'} at the end",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Line {total_lines}: Add missing closing bracket")

    elif error_type == "overclosed_bracket":
        fixed = _fix_overclosed_bracket(json_content, error_lineno, error_colno)
        post_preview = _build_preview_from_fixed(fixed, lines, error_lineno, total_lines) if fixed else None
        issues.append({
            "type": "overclosed_bracket",
            "location": issue_location,
            "description": "Extra closing bracket",
            "fix": "Remove the extra closing bracket",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Line {error_lineno}: Remove extra closing bracket")

    elif error_type == "single_quoted_strings":
        locations = error_info.get("quote_locations", [])
        fixed = _fix_single_quoted_strings(json_content)
        post_preview = _build_preview_from_fixed(fixed, lines, locations[0] if locations else error_lineno, total_lines) if fixed else None
        locations_str = ", ".join(f"Line {loc}" for loc in locations) if locations else issue_location
        issues.append({
            "type": "single_quoted_strings",
            "location": locations_str,
            "description": "Single-quoted string delimiters (JSON requires double quotes)",
            "fix": "Replace all single-quote string delimiters with double quotes",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Lines {locations_str}: Replace single-quote delimiters with double quotes")

    elif error_type == "js_comment":
        locations = error_info.get("comment_locations", [])
        fixed = _fix_js_comments(json_content)
        post_preview = _build_preview_from_fixed(fixed, lines, locations[0] if locations else error_lineno, total_lines) if fixed else None
        locations_str = ", ".join(f"Line {loc}" for loc in locations) if locations else issue_location
        issues.append({
            "type": "js_comment",
            "location": locations_str,
            "description": "JavaScript-style comment inside JSON",
            "fix": "Remove the JavaScript-style comment",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Lines {locations_str}: Remove JavaScript-style comments")

    elif error_type == "trailing_garbage":
        fixed = _fix_trailing_garbage(json_content)
        post_preview = _build_preview_from_fixed(fixed, lines, error_lineno, total_lines) if fixed else None
        issues.append({
            "type": "trailing_garbage",
            "location": issue_location,
            "description": "Content after valid JSON closing bracket",
            "fix": "Truncate the content to the valid JSON boundary",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Line {error_lineno}: Truncate content after valid JSON")

    elif error_type == "mid_structure_corruption":
        fixed = _fix_mid_structure(json_content, error_lineno, error_colno)
        post_preview = _build_preview_from_fixed(fixed, lines, error_lineno, total_lines) if fixed else None
        issues.append({
            "type": "mid_structure_corruption",
            "location": issue_location,
            "description": "Corruption in middle of JSON structure — fix may be ambiguous",
            "fix": "Insert the missing comma or bracket, or regenerate from scratch using full context",
            "pre_repair_preview": pre_preview,
            "post_repair_preview": post_preview
        })
        fix_steps.append(f"Line {error_lineno}: Insert missing delimiter or regenerate JSON")

    repair_plan = "\n".join(f"{i+1}. {step}" for i, step in enumerate(fix_steps)) if fix_steps else ""

    return {
        "can_repair": True,
        "repair_plan": repair_plan,
        "issues": issues,
        "error": ""
    }


def _classify_json_error(content: str) -> dict:
    error_info = {"type": "mid_structure_corruption", "lineno": 1, "colno": 1, "msg": "", "pos": 0}
    try:
        json.loads(content)
        return {"type": "valid", "lineno": 0, "colno": 0, "msg": ""}
    except json.JSONDecodeError as e:
        lineno = e.lineno if e.lineno else 1
        colno = e.colno if e.colno else 1
        msg = e.msg
        err_pos = e.pos if e.pos else 0

        error_info = {"type": "mid_structure_corruption", "lineno": lineno, "colno": colno, "msg": msg, "pos": err_pos}

    msg_lower = msg.lower()
    lines = content.split('\n')
    line_at_error = lines[lineno - 1] if 0 < lineno <= len(lines) else (lines[-1] if lines else "")

    after_error = line_at_error[colno - 1:] if colno <= len(line_at_error) else ""
    stripped_after = after_error.lstrip()
    is_only_close_bracket = stripped_after in ('}', ']')
    char_before_close = line_at_error[colno - 2] if colno > 1 else ''

    error_type = "mid_structure_corruption"
    if "extra data" in msg_lower or "too many" in msg_lower:
        error_type = "overclosed_bracket"
    elif is_only_close_bracket and char_before_close == ',':
        error_type = "trailing_comma"
    elif is_only_close_bracket and char_before_close != ',':
        error_type = "unclosed_bracket"
        bracket_char = '{' if '{' in line_at_error else ('[' if '[' in line_at_error else "")
        error_info["bracket_char"] = bracket_char
    elif "'" in content and (msg_lower == "expecting property name enclosed in double quotes" or "property name" in msg_lower):
        error_type = "single_quoted_strings"
        matches = [(m.start(), content[:m.start()].count('\n') + 1) for m in re.finditer(r":\s*'[^']*'", content)]
        error_info["quote_locations"] = [line_no for _, line_no in matches]
    elif "//" in content or "/*" in content:
        comment_pattern = r"//.*?(?=\n)|/\*[\s\S]*?\*/"
        matches = [(m.start(), content[:m.start()].count('\n') + 1) for m in re.finditer(comment_pattern, content, re.DOTALL)]
        if matches:
            error_type = "js_comment"
            error_info["comment_locations"] = [line_no for _, line_no in matches]
    else:
        remaining = content[err_pos:err_pos+20] if err_pos else ""
        if remaining and not remaining.strip().startswith((',', '}', ']')):
            if re.search(r":\s*'[^']*'", line_at_error):
                error_type = "single_quoted_strings"
                matches = [(m.start(), content[:m.start()].count('\n') + 1) for m in re.finditer(r":\s*'[^']*'", content)]
                error_info["quote_locations"] = [line_no for _, line_no in matches]

    error_info["type"] = error_type
    return error_info


def _build_line_preview(lines: list, error_lineno: int, total_lines: int) -> str:
    start = max(0, error_lineno - 6)
    end = min(total_lines, error_lineno + 5)
    preview_lines = []
    for i in range(start, end):
        line_num = i + 1
        marker = " →" if line_num == error_lineno else ""
        preview_lines.append(f"{line_num}: {lines[i]}{marker}")
    return "Lines {}-{}:\n{}".format(start + 1, end, "\n".join(preview_lines))


def _build_preview_from_fixed(fixed: str, original_lines: list, error_lineno: int, total_lines: int) -> str:
    if not fixed:
        return None
    fixed_lines = fixed.split('\n')
    fixed_total = len(fixed_lines)
    start = max(0, error_lineno - 6)
    end = min(fixed_total, error_lineno + 5)
    preview_lines = []
    for i in range(start, end):
        line_num = i + 1
        marker = " →" if line_num == error_lineno else ""
        preview_lines.append(f"{line_num}: {fixed_lines[i]}{marker}")
    return "Lines {}-{}:\n{}".format(start + 1, end, "\n".join(preview_lines))


def _fix_trailing_comma(content: str, lineno: int, colno: int) -> str:
    lines = content.split('\n')
    if 0 < lineno <= len(lines):
        line = lines[lineno - 1]
        line = re.sub(r',(\s*[}\]])', r'\1', line)
        lines[lineno - 1] = line
        return '\n'.join(lines)
    return None


def _fix_overclosed_bracket(content: str, lineno: int, colno: int) -> str:
    lines = content.split('\n')
    if 0 < lineno <= len(lines):
        line = lines[lineno - 1]
        line_stripped = line.lstrip()
        if line_stripped.startswith('}') or line_stripped.startswith(']'):
            removed = 0
            for i, ch in enumerate(line):
                if ch in ('}', ']') and removed < 2:
                    removed += 1
            line = line[removed:].lstrip()
            if not line:
                lines[lineno - 1] = ''
            else:
                lines[lineno - 1] = line
        return '\n'.join(lines)
    return None


def _fix_single_quoted_strings(content: str) -> str:
    result = []
    i = 0
    while i < len(content):
        if content[i] == "'":
            j = i + 1
            while j < len(content) and content[j] != "'":
                j += 1
            if j < len(content):
                result.append('"')
                result.append(content[i+1:j])
                result.append('"')
                i = j + 1
            else:
                result.append(content[i])
                i += 1
        else:
            result.append(content[i])
            i += 1
    return ''.join(result)


def _fix_js_comments(content: str) -> str:
    content = re.sub(r'//.*?($|\n)', r'\1', content)
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    return content


def _fix_trailing_garbage(content: str) -> str:
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass
    last_brace = max(content.rfind('}'), content.rfind(']'))
    if last_brace != -1:
        return content[:last_brace+1]
    return content[:200]


def _fix_mid_structure(content: str, lineno: int, colno: int) -> str:
    lines = content.split('\n')
    if 0 < lineno <= len(lines):
        line = lines[lineno - 1]
        match = re.search(r'(\w+)"\s*$', line)
        if match:
            line = line + ','
            lines[lineno - 1] = line
            return '\n'.join(lines)
    return None