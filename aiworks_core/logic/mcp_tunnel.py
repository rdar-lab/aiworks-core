import asyncio
import logging
import secrets
import threading
import time
from typing import Any

from ..utils import async_to_sync, sync_to_async
from channels.layers import get_channel_layer
from django.db import transaction
from django.utils import timezone
from langchain_core.tools import StructuredTool, BaseTool
from ..models import Tunnel

logger = logging.getLogger(__name__)

ALIVE_BEAT_TIMEOUT = 60

_relay_sync: dict[str, tuple[Any, threading.Event]] = {}


def tunnel_group_name(tunnel_id: str) -> str:
    return f"tunnel_{tunnel_id}"


def call_group_name(call_id: str) -> str:
    return f"call__{call_id}"


async def listen_for_call_result(call_id, timeout):
    channel_layer = get_channel_layer()
    if not channel_layer:
        raise RuntimeError("Channel layer not available")

    # 1. Create a unique, temporary channel name for this specific listener
    # 'new_channel' is provided by the Redis backing to ensure uniqueness.
    channel_name = await channel_layer.new_channel()
    group_name = call_group_name(call_id)

    # 2. Join the group so we receive messages sent to it
    await channel_layer.group_add(group_name, channel_name)

    try:
        # 3. Wait for the message (Non-blocking to the rest of the app)
        # This will "pause" here until a message hits this specific channel_name or timeout
        # timeout will raise  asyncio.TimeoutError
        event = await asyncio.wait_for(
            channel_layer.receive(channel_name),
            timeout=timeout
        )

        # 4. Handle the message (Logic from your original consumer)
        msg = event.get("call_response")
        if call_id in _relay_sync:
            _, sync_event = _relay_sync[call_id]
            _relay_sync[call_id] = (msg, sync_event)
            sync_event.set()
        else:
            logger.warning("Received call response for unknown call_id %s: %s", call_id, msg)

    finally:
        # 5. Always clean up and leave the group
        await channel_layer.group_discard(group_name, channel_name)


class MCPTunnelManager:
    @classmethod
    def connect_sync(
            cls,
            tunnel_api_key: str,
            tunnel_id: str | None = None,
    ) -> dict[str, str]:
        with transaction.atomic():
            existing = None
            if tunnel_id:
                try:
                    existing = Tunnel.objects.select_for_update(
                        skip_locked=True
                    ).get(tunnel_id=tunnel_id)
                except Tunnel.DoesNotExist:
                    logger.warning(f"Tunnel {tunnel_id} not found - will register a new tunnel")
                    existing = None

            if existing:
                if existing.tunnel_api_key != tunnel_api_key:
                    raise ValueError("Tunnel API key mismatch")
            else:
                if tunnel_id is None:
                    tunnel_id = f"{secrets.token_urlsafe(16)}"
                Tunnel.objects.create(
                    tunnel_id=tunnel_id,
                    user=None,
                    tunnel_api_key=tunnel_api_key,
                    alive_beat=timezone.now(),
                )

        assert tunnel_id is not None

        return {"tunnel_id": tunnel_id}

    @classmethod
    async def connect(cls, *args, **kwargs) -> dict[str, str]:
        return await sync_to_async(cls.connect_sync)(*args, **kwargs)

    @classmethod
    def disconnect_sync(cls, tunnel_id: str) -> None:
        # No-op: the Tunnel DB record is kept intentionally to allow the desktop CLI
        # to reconnect with the same tunnel_id + tunnel_api_key. Disconnection is detected
        # via alive_beat staleness (is_connected checks alive_beat age, not record absence).
        pass

    @classmethod
    async def disconnect(cls, tunnel_id: str) -> None:
        # No-op: see disconnect_sync — DB record is kept by design.
        pass

    @classmethod
    def delete_tunnel_sync(cls, tunnel_id: str) -> None:
        try:
            with transaction.atomic():
                Tunnel.objects.filter(tunnel_id=tunnel_id).delete()
        except Exception as exp:
            logger.warning(f"Was unable to delete tunnel. due to {exp}")

    @classmethod
    async def refresh_alive_beat(cls, tunnel_id: str) -> None:
        try:
            await sync_to_async(
                lambda: Tunnel.objects.filter(tunnel_id=tunnel_id).update(alive_beat=timezone.now())
            )()
        except Exception:
            pass

    @classmethod
    def is_connected(cls, tunnel_id: str) -> bool:
        try:
            if not Tunnel.objects.filter(tunnel_id=tunnel_id).exists():
                return False
            tunnel = Tunnel.objects.get(tunnel_id=tunnel_id)
            return (timezone.now() - tunnel.alive_beat).total_seconds() < ALIVE_BEAT_TIMEOUT
        except Exception as exp:
            logger.warning("Error checking tunnel connection for %s. err= %s", tunnel_id, str(exp),
                           exc_info=True)
            return False

    @classmethod
    async def pull_tools(
            cls, tunnel_id: str, server_ids: list[str], timeout: int = 30
    ) -> list[dict[str, Any]]:
        channel_layer = get_channel_layer()
        if not channel_layer:
            raise RuntimeError("Channel layer not available")

        call_id = f"pull_{secrets.token_urlsafe(8)}"
        event = threading.Event()
        _relay_sync[call_id] = (None, event)

        asyncio.create_task(listen_for_call_result(call_id, timeout))

        await channel_layer.group_send(
            tunnel_group_name(tunnel_id),
            {
                "type": "tunnel_pull_tools",
                "call_id": call_id,
                "server_ids": server_ids,
            }
        )
        logger.debug("→ tunnel [%s] pull_tools servers=%s", tunnel_id, server_ids)

        loop = asyncio.get_running_loop()
        wait_done = await loop.run_in_executor(None, event.wait, timeout)
        if not wait_done:
            _relay_sync.pop(call_id, None)
            raise TimeoutError(f"pull_tools timed out after {timeout}s")
        # Re-read from dict after event is set — handler writes (msg, event) in place.
        result = _relay_sync.pop(call_id, (None, None))[0]
        return result.get("servers", []) if result else []

    @classmethod
    async def push_call(
            cls,
            tunnel_id: str,
            server_id: str,
            call_id: str,
            tool: str,
            args: dict[str, Any],
            timeout: int = 120,
    ) -> Any:
        _sent_at = time.monotonic()
        channel_layer = get_channel_layer()
        if not channel_layer:
            raise RuntimeError("Channel layer not available")

        event = threading.Event()
        _relay_sync[call_id] = (None, event)
        logger.debug("  [tunnel %s] registered sync relay for %s", tunnel_id, call_id)

        asyncio.create_task(listen_for_call_result(call_id, timeout))

        await channel_layer.group_send(
            tunnel_group_name(tunnel_id),
            {
                "type": "tunnel_call",
                "call_id": call_id,
                "server_id": server_id,
                "tool": tool,
                "args": args,
            }
        )
        logger.debug("→ tunnel [%s] call %s.%s", tunnel_id, server_id, tool)

        loop = asyncio.get_running_loop()
        wait_done = await loop.run_in_executor(None, event.wait, timeout)
        if not wait_done:
            _relay_sync.pop(call_id, None)
            logger.warning("push_call timed out for %s after %.1fs", call_id, time.monotonic() - _sent_at)
            raise TimeoutError(f"push_call timed out after {timeout}s")
        # Re-read from dict after event is set — handler writes (msg, event) in place.
        result = _relay_sync.pop(call_id, (None, None))[0]
        logger.debug("← tunnel [%s] push_call raw result for %s: %s", tunnel_id, call_id, result)
        logger.debug("← tunnel [%s] push_call result for %s after %.1fs", tunnel_id, call_id,
                     time.monotonic() - _sent_at)
        if not result:
            return None
        if result.get("error") is not None:
            logger.warning(
                "push_call failed for %s.%s on tunnel %s: %s",
                server_id,
                tool,
                tunnel_id,
                result.get("error"),
            )
            raise RuntimeError(
                f"Tunnel call failed for {server_id}.{tool}: {result.get('error')}"
            )
        return result.get("result")

    @classmethod
    async def get_servers(cls, tunnel_id: str, timeout: int = 30) -> list[dict[str, str]]:
        channel_layer = get_channel_layer()
        if not channel_layer:
            return []

        call_id = f"srv_{secrets.token_urlsafe(8)}"
        event = threading.Event()
        _relay_sync[call_id] = (None, event)

        asyncio.create_task(listen_for_call_result(call_id, timeout))

        await channel_layer.group_send(
            tunnel_group_name(tunnel_id),
            {
                "type": "tunnel_get_servers",
                "call_id": call_id,
            }
        )
        logger.debug("→ tunnel [%s] get_servers", tunnel_id)

        loop = asyncio.get_running_loop()
        wait_done = await loop.run_in_executor(None, event.wait, timeout)
        if not wait_done:
            _relay_sync.pop(call_id, None)
            logger.warning("get_servers timed out for tunnel %s", tunnel_id)
            return []
        # Re-read from dict after event is set — handler writes (msg, event) in place.
        result = _relay_sync.pop(call_id, (None, None))[0]
        return result.get("servers", []) if result else []

    @classmethod
    def claim_tunnel_sync(
            cls, tunnel_id: str, user_id: int, tunnel_api_key: str
    ) -> bool:
        try:
            with transaction.atomic():
                tunnel = Tunnel.objects.select_for_update(skip_locked=True).get(
                    tunnel_id=tunnel_id
                )
                if tunnel.tunnel_api_key != tunnel_api_key:
                    return False
                if tunnel.user and tunnel.user.id == user_id:
                    return True
                tunnel.user_id = user_id
                tunnel.save(update_fields=["user"])
            return True
        except Tunnel.DoesNotExist:
            return False
        except Exception:
            logger.exception("claim_tunnel error for tunnel_id=%s", tunnel_id)
            return False

    @classmethod
    async def claim_tunnel(cls, *args, **kwargs) -> bool:
        return cls.claim_tunnel_sync(*args, **kwargs)


def build_tunnel_tools(tunnel_ids, tunnel_servers_map: dict) -> list[BaseTool]:
    tunnel_tools: list[BaseTool] = []

    for tunnel_id in tunnel_ids:
        if not MCPTunnelManager.is_connected(tunnel_id):
            logger.warning("Skipping disconnected tunnel %s", tunnel_id)
            continue

        server_ids = tunnel_servers_map.get(tunnel_id, [])
        servers_data = None
        for attempt in range(2):
            try:
                if not MCPTunnelManager.is_connected(tunnel_id):
                    logger.warning("Skipping disconnected tunnel %s", tunnel_id)
                    break
                servers_data = async_to_sync(MCPTunnelManager.pull_tools)(
                    tunnel_id, server_ids, timeout=30
                )
                break
            except Exception as exc:
                logger.warning("Failed to pull tools from tunnel %s after retry: %s", tunnel_id, exc)
                servers_data = None

        if not servers_data:
            continue

        for server in servers_data:
            server_id = server.get("id", "")
            server_name = server.get("name", "")
            for tool_def in server.get("tools", []):
                tool_name = tool_def.get("name", "")
                tool_desc = tool_def.get("description", "")
                args_schema = tool_def.get("args_schema", {})

                async def _call_via_tunnel(
                        tunnel_id=tunnel_id, server_id=server_id, tool_name=tool_name, args=None, *extra_args, **kwargs
                ):
                    if args is None:
                        args = {}
                    if extra_args:
                        args = {**args, **dict(zip(['_'], extra_args))}
                    call_args = {**args, **kwargs}
                    logger.debug("  [tunnel %s] _call_via_tunnel tool=%s args=%s kwargs=%s", tunnel_id, tool_name, args,
                                 kwargs)
                    call_id = f"tc_{secrets.token_urlsafe(8)}"
                    result = await MCPTunnelManager.push_call(
                        tunnel_id, server_id, call_id, tool_name, call_args, timeout=120
                    )
                    logger.debug("  [tunnel %s] _call_via_tunnel result: %s", tunnel_id, result)
                    return result

                tool = StructuredTool.from_function(
                    name=tool_name,
                    description=f"[Desktop: {server_name}] {tool_desc}",
                    args_schema=args_schema,
                    func=lambda *_a, _c=_call_via_tunnel, **_kw: _c(*_a, **_kw),
                    coroutine=lambda *_a, _c=_call_via_tunnel, **_kw: _c(*_a, **_kw),
                )
                tunnel_tools.append(tool)

    return tunnel_tools
