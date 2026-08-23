"""Converts registered MCPServer records into LangChain tools for the agent."""
import asyncio
import base64
import copy
import json
import logging
from typing import Callable, List, Optional

import httpx
from ..utils import async_to_sync, sync_to_async, raise_for_status
from django.db import transaction
from fastmcp import FastMCP
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import create_model
from .logic_utils import validate_ssrf_safe_url
from .oauth import resolve_oauth_tokens
from ..models import MCPServer

logger = logging.getLogger(__name__)

_OAUTH_SERVER_ID_HEADER_NAME = "Oauth-Server-ID"

class RefreshableTokenAuth(httpx.Auth):
    """
    httpx Auth handler that reads OAuth tokens from the DB on every request
    and automatically refreshes on 401 with select_for_update locking.

    Used for OpenAPI MCP servers with auth_type='oauth'. Ensures concurrent
    requests on different workers don't produce lost updates during token refresh.
    """

    requires_response_body = True

    def __init__(self, server_id: int):
        self.server_id = server_id

    def auth_flow(self, request: httpx.Request):
        raise NotImplementedError("RefreshableTokenAuth - auth_flow should not be used directly")

    def _read_token_sync(self):
        server = MCPServer.objects.get(pk=self.server_id)
        return server.token

    async def _read_token_async(self):
        return await sync_to_async(self._read_token_sync)()

    def _refresh_token_sync(self):
        with transaction.atomic():
            server = MCPServer.objects.select_for_update().get(pk=self.server_id)
            if not server.refresh_token:
                logger.error("No refresh token for server %s", self.server_id)
                return None

            try:
                access_token, refresh_token = resolve_oauth_tokens(
                    url=server.url,
                    refresh_token=server.refresh_token,
                    oauth_metadata=server.oauth_metadata,
                    client_secret=getattr(
                        server.predefined_server, 'client_secret', ''
                    ) or '',
                )
            except ValueError as exc:
                logger.warning("OAuth refresh rejected for server %s: %s", self.server_id, exc)
                return None

            server.token = access_token
            server.refresh_token = refresh_token
            server.save(update_fields=["token", "refresh_token"])
        return server.token

    async def _refresh_token_async(self):
        return await sync_to_async(self._refresh_token_sync)()


    async def async_auth_flow(self, request: httpx.Request):
        logger.debug("Injecting OAuth token to request for server %s", self.server_id)
        token = await self._read_token_async()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if response.status_code == 401:
            logger.info("OAuth token rejected for server %s, attempting refresh", self.server_id)
            token = await self._refresh_token_async()

            if token:
                retry_request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    content=request.content,
                )
                retry_request.headers["Authorization"] = f"Bearer {token}"
                yield retry_request

    def sync_auth_flow(self, request: httpx.Request):
        token = self._read_token_sync()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if response.status_code == 401:
            logger.info("OAuth token rejected for server %s, attempting refresh", self.server_id)
            token = self._refresh_token_sync()

            if token:
                retry_request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    content=request.content,
                )
                retry_request.headers["Authorization"] = f"Bearer {token}"
                yield retry_request


async def _load_mcp_tools_async(
        url: str,
        headers: Optional[dict] = None,
) -> List[BaseTool]:
    auth_server_id: str | None = headers.get(_OAUTH_SERVER_ID_HEADER_NAME) if headers else None
    if auth_server_id:
        headers = copy.copy(headers)
        del headers[_OAUTH_SERVER_ID_HEADER_NAME]

    client = MultiServerMCPClient({
        "server": {
            "url": url,
            "transport": "streamable_http",
            "headers": headers,
            "auth": RefreshableTokenAuth(int(auth_server_id)) if auth_server_id else None
        },
    })
    return await asyncio.wait_for(client.get_tools(), timeout=30)


async def _validate_mcp_connection_async(url: str, headers: Optional[dict] = None) -> None:
    await _load_mcp_tools_async(url, headers)


def _json_schema_to_python_type(schema: dict) -> type:
    """Convert JSON Schema type to Python type for pydantic model creation."""
    typ = schema.get("type", "string")
    if typ == "integer":
        return int
    elif typ == "number":
        return float
    elif typ == "boolean":
        return bool
    elif typ == "array":
        return list
    elif typ == "object":
        return dict
    return str


def _build_openapi_langchain_tool(
        openapi_spec: dict,
        url: str,
        headers: dict,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict,
) -> Optional[StructuredTool]:
    """Build a LangChain StructuredTool that recreates httpx.AsyncClient on each call.

    Each .invoke() creates a fresh AsyncClient in the current event loop so no
    client is ever shared across loops. Avoids the "Event loop is closed" error
    that occurs when an AsyncClient created in one async loop is used from another.
    """
    props = tool_parameters.get("properties", {})
    if not props:
        return None

    model_fields = {}
    for prop_name, prop_schema in props.items():
        model_fields[prop_name] = (_json_schema_to_python_type(prop_schema), None)

    tool_input = create_model(
        f"{tool_name.title().replace('_', '')}Input",
        **model_fields,
    )

    auth_server_id: str | None = headers.get(_OAUTH_SERVER_ID_HEADER_NAME)
    if auth_server_id:
        headers = copy.copy(headers)
        del headers[_OAUTH_SERVER_ID_HEADER_NAME]

    def invoke_sync(**kwargs):
        client = httpx.AsyncClient(base_url=url, headers=headers,
                                   auth=RefreshableTokenAuth(int(auth_server_id)) if auth_server_id else None)
        mcp = FastMCP.from_openapi(openapi_spec=openapi_spec, client=client, name="OpenAPIServer")

        async def find_and_run():
            for ot in await mcp.list_tools():
                if ot.name == tool_name:
                    result = await ot.run(kwargs)
                    return result.text if hasattr(result, "text") else str(result)
            raise ValueError(f"Tool '{tool_name}' not found in spec")

        return async_to_sync(find_and_run)()

    return StructuredTool(
        name=tool_name,
        description=tool_description or "",
        args_schema=tool_input,
        func=invoke_sync,
    )


async def _load_openapi_tools_async(
        openapi_spec: dict,
        url: str,
        headers: dict,
        name: str,
) -> List[BaseTool]:
    """Load MCP tools from an OpenAPI spec using FastMCP, in-process.

    Each OpenAPITool from FastMCP.list_tools() is wrapped as a LangChain
    StructuredTool that recreates httpx.AsyncClient on each invocation.
    """

    effective_headers = headers
    auth_server_id: str | None = effective_headers.get(_OAUTH_SERVER_ID_HEADER_NAME)
    if auth_server_id:
        effective_headers = copy.copy(effective_headers)
        del effective_headers[_OAUTH_SERVER_ID_HEADER_NAME]

    client = httpx.AsyncClient(base_url=url, headers=effective_headers,
                               auth=RefreshableTokenAuth(int(auth_server_id)) if auth_server_id else None)
    mcp = FastMCP.from_openapi(openapi_spec=openapi_spec, client=client, name=name)
    openapi_tools = await mcp.list_tools()

    langchain_tools: List[BaseTool] = []
    for ot in openapi_tools:
        langchain_tool = _build_openapi_langchain_tool(
            openapi_spec,
            url,
            headers,
            ot.name,
            ot.description or "",
            ot.parameters or {},
        )
        if langchain_tool is not None:
            langchain_tools.append(langchain_tool)

    return langchain_tools


def validate_mcp_connection(url: str, headers: Optional[dict] = None) -> None:
    """
    Synchronously validate that the MCP server at ``url`` is reachable and accepts
    the provided ``headers``. Raises ``ValueError`` with a descriptive message on
    failure.

    Also performs SSRF safety checks (loopback, private IP, link-local, metadata
    endpoints) before attempting any outbound connection.
    """
    validate_ssrf_safe_url(url)
    try:
        async_to_sync(_validate_mcp_connection_async)(url, headers)
    except Exception as exc:
        logger.exception("Failed to load tools from MCP server: %s", exc)
        raise ValueError(f"Could not connect to MCP server at '{url}': {exc}") from exc


def validate_rest_api_connection(
        url: str,
        headers: Optional[dict],
        openapi_spec_url: str,
        openapi_spec: str,
) -> None:
    """
    Validate that the OpenAPI/REST server at ``url`` is reachable with the provided
    ``headers`` and a valid OpenAPI spec. Uses FastMCP.from_openapi() in-process to
    call get_tools() as the validation mechanism.

    Raises ``ValueError`` with a descriptive message on failure.
    """
    validate_ssrf_safe_url(url)
    if not openapi_spec_url and not openapi_spec:
        raise ValueError(
            "openapi_spec_url or openapi_spec is required for server_type='openapi'"
        )
    spec = _load_openapi_spec(openapi_spec_url, openapi_spec, headers or {})
    try:
        async_to_sync(_load_openapi_tools_async)(spec, url, headers or {}, "validate_connection")
    except Exception as exc:
        logger.exception("Failed to validate REST API at '%s': %s", url, exc)
        raise ValueError(f"Could not validate REST API at '{url}': {exc}") from exc


def build_mcp_tools(mcp_server_configs: List[dict]) -> List[BaseTool]:
    """
    Given a list of dicts with keys ``name``, ``url``, ``headers``,
    ``auth_type``, ``username``, ``password``, ``token``, ``refresh_token``,
    return a flat list of LangChain ``BaseTool`` instances.
    """
    tools: List[BaseTool] = []
    for cfg in mcp_server_configs:
        try:
            url = cfg["url"]
            validate_ssrf_safe_url(url)
            server_tools = async_to_sync(_load_mcp_tools_async)(
                url,
                cfg.get("headers", {}),
            )
            tools.extend(server_tools)
            logger.info("Loaded %d tools from MCP server '%s'", len(server_tools), cfg["url"])
        except Exception as exc:
            logger.exception("Failed to load tools from MCP server '%s': %s", cfg["url"], exc)
    return tools


def _load_openapi_spec(
        openapi_spec_url: str,
        openapi_spec: str,
        auth_headers: dict,
) -> dict:
    """Load and parse OpenAPI spec from URL or inline content."""
    if openapi_spec:
        try:
            return json.loads(openapi_spec)
        except json.JSONDecodeError as exc:
            raise ValueError(f"openapi_spec is not valid JSON: {exc}") from exc
    elif openapi_spec_url:
        validate_ssrf_safe_url(openapi_spec_url)
        response = httpx.get(openapi_spec_url, headers=auth_headers, timeout=30)
        raise_for_status(response)
        return response.json()
    else:
        raise ValueError("openapi_spec_url or openapi_spec is required for server_type=openapi")


def build_mcp_tools_from_servers(servers: List) -> List[BaseTool]:
    """
    Load MCP tools from a list of MCPServer objects (or dict-like objects with
    the same field interface).

    server_type="http"  → MultiServerMCPClient (existing flow)
    server_type="openapi" → FastMCP.from_openapi() called in-process

    Per-server error isolation: one failing server doesn't break others.
    Auth headers resolved internally via resolve_auth_headers().

    Returns flat list of BaseTool instances from all servers.
    """
    tools: List[BaseTool] = []

    for server in servers:
        server_type = getattr(server, "server_type", "http")
        url = getattr(server, "url", "")

        try:
            if server_type == "openapi":
                openapi_spec_url = getattr(server, "openapi_spec_url", "") or ""
                openapi_spec = getattr(server, "openapi_spec", "") or ""

                url, headers = _ensure_connection(server, validate_server_connection_openapi)

                spec = _load_openapi_spec(openapi_spec_url, openapi_spec, headers)
                server_tools = async_to_sync(_load_openapi_tools_async)(
                    spec, url, headers, getattr(server, "name", "OpenAPI Server")
                )
                tools.extend(server_tools)
                logger.info(
                    "Loaded %d tools from OpenAPI server '%s'",
                    len(server_tools), getattr(server, "name", url)
                )
            else:
                # server_type == "http" (default)
                url, headers = _ensure_connection(server, validate_server_connection)
                server_tools = build_mcp_tools([{"url": url, "headers": headers}])
                tools.extend(server_tools)
        except Exception as exc:
            logger.exception(
                "Failed to load tools from MCP server '%s' (type=%s): %s",
                getattr(server, "name", url), server_type, exc
            )

    return tools


def validate_server_connection(mcp_server: MCPServer):
    headers = mcp_server.headers
    auth_type = mcp_server.auth_type
    username = mcp_server.username
    password = mcp_server.password
    token = mcp_server.token
    url = mcp_server.url
    server_id = mcp_server.pk

    effective_headers = resolve_auth_headers(
        headers, auth_type, username, password, token, server_id
    )
    validate_mcp_connection(url, effective_headers)
    return url, effective_headers


def validate_server_connection_openapi(mcp_server: MCPServer):
    headers = mcp_server.headers
    auth_type = mcp_server.auth_type
    username = mcp_server.username
    password = mcp_server.password
    token = mcp_server.token
    url = mcp_server.url
    openapi_spec_url = mcp_server.openapi_spec_url or ""
    openapi_spec = mcp_server.openapi_spec or ""
    server_id = mcp_server.pk

    effective_headers = resolve_auth_headers(
        headers, auth_type, username, password, token, server_id
    )
    validate_rest_api_connection(url, effective_headers, openapi_spec_url, openapi_spec)
    return url, effective_headers


def _ensure_connection(mcp_server: MCPServer, validate_func: Callable) -> tuple[str, dict]:
    """
    Call validate_func(mcp_server) and retry with oauth_refresh on failure.
    validate_func is validate_server_connection (HTTP) or
    validate_server_connection_openapi (OpenAPI).
    """
    try:
        return validate_func(mcp_server)
    except Exception:
        if mcp_server.auth_type == "oauth":
            mcp_server = oauth_refresh(mcp_server)
            return validate_func(mcp_server)
        else:
            raise


def oauth_refresh(server: MCPServer):
    """
    Refresh an expired or expiring OAuth access token using the stored
    refresh token. The token endpoint is discovered on demand from the
    server URL so it does not need to be stored.

    Request body:
        {
            "mcp_server_id": 123,
            "client_id":      "..."    // optional — defaults to "aiworks-core"
        }

    Response:
        {
            "access_token":  "...",
            "token_type":    "bearer",
            "expires_in":    3600,       // optional
            "refresh_token": "..."        // optional
        }
    """
    if not server.refresh_token:
        raise ValueError(
            {"error": "MCP server has no stored refresh token"}
        )

    try:
        access_token, refresh_token = resolve_oauth_tokens(
            url=server.url,
            refresh_token=server.refresh_token,
            oauth_metadata=server.oauth_metadata,
            client_secret=getattr(server.predefined_server, 'client_secret', '') or '',
        )
    except ValueError as exc:
        logger.info("OAuth token refresh failed for server %s: %s", server.url, exc)
        raise ValueError(
            {"error": "Token refresh was rejected by the authorization server."}
        ) from exc

    with transaction.atomic():
        # select_for_update ensures concurrent refresh calls on the same
        # server don't produce a lost update when the server rotates tokens.
        locked = MCPServer.objects.select_for_update().get(
            pk=server.pk
        )
        locked.token = access_token
        locked.refresh_token = refresh_token
        locked.save(update_fields=["token", "refresh_token"])

    # Update the in-memory fields as well
    server.token = access_token
    server.refresh_token = refresh_token

    return server


def resolve_auth_headers(
        headers: Optional[dict],
        auth_type: str,
        username: str = "",
        password: str = "",
        token: str = "",
        server_id: int | None = None
) -> dict:
    """
    Return the effective HTTP headers dict for the given authentication type.

    For ``none`` and ``basic``/``bearer`` modes this is purely synchronous
    (no network calls).  For ``oauth``, the stored ``oauth_metadata`` is used
    to obtain the token endpoint; if not available, the server's
    authorization-server metadata is discovered on demand and the stored
    refresh token is exchanged for a fresh access token before each request.

    The caller's ``headers`` dict is merged with the computed
    ``Authorization`` value (auth takes precedence).
    """
    effective = dict(headers or {})

    if auth_type == "basic" and username:
        credentials = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode()
        effective["Authorization"] = f"Basic {credentials}"
    elif (auth_type == "bearer" or auth_type == "oauth") and token:
        effective["Authorization"] = f"Bearer {token}"

    if auth_type == "oauth" and server_id:
        effective[_OAUTH_SERVER_ID_HEADER_NAME] = str(server_id)

    return effective
