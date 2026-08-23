import json
from typing import Any

from ..utils import async_to_sync
from django.http import Http404
from django.shortcuts import render
from django.utils.html import format_html, escape
from django.utils.safestring import mark_safe
from ..logic.mcp_tools import build_mcp_tools_from_servers
from ..models import MCPServer


def enumerate_mcp_servers(admin, request, queryset):
    results = []
    for server in queryset:
        try:
            tools = build_mcp_tools_from_servers([server])
            results.append((server.name, server.url, tools))
        except Exception as e:
            results.append((server.name, server.url, f"Error: {e}"))

    if not results:
        admin.message_user(request, "No MCP servers to enumerate.")
        return

    rows = []
    for name, url, tools_or_err in results:
        formatted_name = f'{name} ({url})'
        render_mcp_tools_server(formatted_name, tools_or_err, rows)
    admin.message_user(
        request,
        format_html(
            "<table style='border-collapse:collapse;width:100%'>{}</table>",
            mark_safe("".join(rows)),
        ),
    )


def render_mcp_tools_server(name, tools_or_err, rows: list[Any]):
    rows.append(format_html("<tr><th colspan='2'>=== {} ===</th></tr>", name))
    if isinstance(tools_or_err, str):
        rows.append(format_html("<tr><td colspan='2'>{}</td></tr>", tools_or_err))
    elif not tools_or_err:
        rows.append("<tr><td colspan='2'>(no tools or server unreachable)</td></tr>")
    else:
        for tool in tools_or_err:
            desc = getattr(tool, "description", "") or ""
            desc = desc.replace("\n", " ").replace("\r", "")
            rows.append(format_html("<tr><td style='white-space:nowrap'>{}</td><td>{}</td></tr>", tool.name, desc))


def execute_tool_inner(tools, tool_name, run_args, execute_flag):
    run_result = None
    error = None
    tool = None
    args_schema = None
    tool_description = None

    if tools and tool_name:
        try:
            tool = next(t for t in tools if t.name == tool_name)

            if hasattr(tool, "args_schema") and tool.args_schema:
                if isinstance(tool.args_schema, dict):
                    args_schema = json.dumps(tool.args_schema, indent=2)
                else:
                    args_schema = json.dumps(tool.args_schema.model_json_schema(), indent=2)
            else:
                args_schema = "{}"

            if hasattr(tool, "description") and tool.description:
                tool_description = tool.description
            else:
                tool_description = "No description"

            if execute_flag:
                if run_args:
                    tool_args_parsed = json.loads(run_args)
                else:
                    tool_args_parsed = {}
                run_result = async_to_sync(tool.ainvoke)(tool_args_parsed)
                run_result = _extract_run_result(run_result)


        except Exception as e:
            error = str(e)

    return {
        "tools": tools,
        "selected_tool": tool,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "run_args": run_args,
        "run_result": run_result,
        "error": error,
        "args_schema": args_schema,
    }


def _extract_run_result(run_result) -> Any:
    if run_result and isinstance(run_result, dict) and "content" in run_result:
        return _extract_run_result(run_result["content"])

    if run_result and isinstance(run_result, list) and len(run_result) == 1 and isinstance(run_result[0], dict) and \
            run_result[0].get("type") == "text" and run_result[0].get("text"):
        data = run_result[0].get("text")
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
        return _extract_run_result(data)

    if not isinstance(run_result, str):
        run_result = json.dumps(run_result, indent=2, default=str)

    run_result = escape(run_result)
    run_result = run_result.replace(" ", "&nbsp;")
    run_result = run_result.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    run_result = run_result.replace("\n", "<br/>")
    return run_result


def execute_tool_view(request, pk):
    try:
        server = MCPServer.objects.get(pk=pk)
    except MCPServer.DoesNotExist:
        raise Http404

    tools = build_mcp_tools_from_servers([server])
    tool_name = request.POST.get("tool_name") or request.GET.get("tool_name")
    run_args = request.POST.get("run_args", "{}")
    execute_flag = request.POST.get("execute_flag", "false") == 'true'
    result_params = execute_tool_inner(tools, tool_name, run_args, execute_flag)

    return render(
        request,
        "admin/aiworks_core/mcp_execute_tool.html",
        {
            "server": server,
            "title": f"Execute Tool — {server.name}",
            **result_params
        },
    )
