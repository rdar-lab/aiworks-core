import json

from ..utils import async_to_sync
from django.http import Http404
from django.shortcuts import render
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from .admin_mcp_helper import render_mcp_tools_server, execute_tool_inner
from ..logic.mcp_tunnel import MCPTunnelManager, build_tunnel_tools
from ..models import (
    Tunnel,
)


def execute_tool_view(request, pk):
    try:
        tunnel = Tunnel.objects.get(tunnel_id=pk)
    except Tunnel.DoesNotExist:
        raise Http404

    server_id = request.POST.get("server_id") or request.GET.get("server_id")
    tool_name = request.POST.get("tool_name") or request.GET.get("tool_name")
    run_args = request.POST.get("run_args", "{}")
    execute_flag = request.POST.get("execute_flag", "false") == 'true'
    servers = []
    tools = []

    try:
        servers = async_to_sync(MCPTunnelManager.pull_tools)(tunnel.tunnel_id, [])
        if server_id:
            tools = build_tunnel_tools([tunnel.tunnel_id], {tunnel.tunnel_id: [server_id]})
        result_params = execute_tool_inner(tools, tool_name, run_args, execute_flag)
    except Exception as e:
        result_params = {
            "tools": None,
            "selected_tool": None,
            "tool_name": tool_name,
            "tool_description": None,
            "run_args": run_args,
            "run_result": None,
            "args_schema": None,
            "error": str(e)
        }

    servers_json = json.dumps(servers or [])
    return render(
        request,
        "admin/aiworks_core/tunnel_execute_tool.html",
        {
            "tunnel": tunnel,
            "servers": servers,
            "servers_json": servers_json,
            "server_id": server_id,
            **result_params,
            "title": f"Execute Tool — Tunnel {tunnel.tunnel_id}",
        },
    )


def enumerate_tunnel_servers(admin, request, queryset):
    results = []
    for tunnel in queryset:
        try:
            servers = async_to_sync(MCPTunnelManager.pull_tools)(tunnel.tunnel_id, [])
            updated_servers = [
                {
                    "name": srv.get("name"),
                    "id": srv.get("id"),
                    "tools": build_tunnel_tools([tunnel.tunnel_id], {tunnel.tunnel_id: [srv.get("id")]})
                }
                for srv in servers
            ]
            results.append((tunnel.tunnel_id, updated_servers))
        except Exception as e:
            results.append((tunnel.tunnel_id, f"Error: {e}"))

    if not results:
        admin.message_user(request, "No tunnels to enumerate.")
        return

    rows = []
    for tid, servers_or_err in results:
        rows.append(format_html("<tr><th colspan='2'>=== Tunnel: {} ===</th></tr>", tid))
        if isinstance(servers_or_err, str):
            rows.append(format_html("<tr><td colspan='2'>{}</td></tr>", servers_or_err))
        elif not servers_or_err:
            rows.append(format_html("<tr><td colspan='2'>(no servers or tunnel disconnected)</td></tr>"))
        else:
            for srv in servers_or_err:
                server_name = srv.get("name", "?")
                server_id = srv.get("id", "?")
                formatted_server_name = f"{server_name} (id: {server_id})"
                tools = srv.get("tools", [])
                render_mcp_tools_server(formatted_server_name, tools, rows)

    admin.message_user(
        request,
        format_html(
            "<table style='border-collapse:collapse;width:100%'>{}</table>",
            mark_safe("".join(rows)),
        ),
    )
