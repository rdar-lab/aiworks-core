import asyncio
import json
import logging
from typing import Any

from .utils import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from django.utils import timezone

from .logic.mcp_tunnel import MCPTunnelManager, tunnel_group_name, call_group_name
from .models import Tunnel

logger = logging.getLogger(__name__)


class TunnelConsumer(AsyncWebsocketConsumer):
    tunnel_id: str | None = None
    _alive_task: asyncio.Task | None = None

    async def connect(self):
        self.tunnel_id = None
        await self.accept()

    async def send(self, text_data: str | None = None, bytes_data: bytes | None = None, **kwargs):
        if text_data:
            try:
                parsed = json.loads(text_data)
                logger.debug("→ tunnel [%s] %s", self.tunnel_id or "?", parsed.get("type", "?"))
            except Exception:
                logger.debug("→ tunnel [%s] %s", self.tunnel_id or "?", text_data[:100])
        await super().send(text_data=text_data, bytes_data=bytes_data, **kwargs)

    async def disconnect(self, close_code: int):
        if self.tunnel_id:
            await self._leave_group(self.tunnel_id)
            try:
                await MCPTunnelManager.disconnect(self.tunnel_id)
            except Exception:
                pass
            self.tunnel_id = None

        if self._alive_task:
            self._alive_task.cancel()
            self._alive_task = None

    async def _join_group(self, tunnel_id: str):
        channel_layer = get_channel_layer()
        if not channel_layer:
            raise RuntimeError("Channel layer not available")

        group_name = tunnel_group_name(tunnel_id)
        try:
            await channel_layer.group_add(
                group_name,
                self.channel_name,
            )
        except Exception as e:
            raise Exception(f"Failed to join channel group {group_name}: {e}")

    async def _leave_group(self, tunnel_id: str):
        channel_layer = get_channel_layer()
        if channel_layer:
            try:
                await channel_layer.group_discard(
                    tunnel_group_name(tunnel_id),
                    self.channel_name,
                )
            except Exception:
                pass

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None):
        if not text_data:
            return

        try:
            msg = json.loads(text_data)
        except json.JSONDecodeError:
            return

        logger.debug("← tunnel [%s] %s", self.tunnel_id or "?", msg.get("type", "?"))
        msg_type = msg.get("type")

        if self.tunnel_id and msg_type == "pong":
            try:
                await sync_to_async(
                    lambda: Tunnel.objects.filter(tunnel_id=self.tunnel_id).update(alive_beat=timezone.now())
                )()
            except Exception as e:
                logger.warning("Failed to refresh alive_beat for tunnel %s: %s", self.tunnel_id, e)

        if msg_type == "connect":
            await self._handle_connect(msg)
        elif msg_type == "ping":
            if self.tunnel_id:
                try:
                    await sync_to_async(
                        lambda: Tunnel.objects.filter(tunnel_id=self.tunnel_id).update(alive_beat=timezone.now())
                    )()
                except Exception as e:
                    logger.warning("Failed to refresh alive_beat for tunnel %s: %s", self.tunnel_id, e)
            await self.send(text_data=json.dumps({"type": "pong"}))
        elif msg.get("call_id") is not None:
            call_id = msg.get("call_id")
            group_name = call_group_name(call_id)
            channel_layer = get_channel_layer()
            if not channel_layer:
                raise RuntimeError("Channel layer not available")
            await channel_layer.group_send(
                group_name,
                {
                    "call_response": msg,
                }
            )
        else:
            logger.warning("Unknown message type from tunnel: %s", msg_type)

    async def _handle_connect(self, msg: dict[str, Any]):
        tunnel_api_key = msg.get("tunnel_api_key")
        tunnel_id = msg.get("tunnel_id")

        try:
            result = await MCPTunnelManager.connect(
                tunnel_api_key=tunnel_api_key,
                tunnel_id=tunnel_id,
            )
            self.tunnel_id = result["tunnel_id"]
            await self._join_group(self.tunnel_id)
            await self.send(
                text_data=json.dumps({"type": "connected", "tunnel_id": self.tunnel_id})
            )
        except Exception as e:
            logger.exception("Tunnel connect error: %s", e)
            await self.send(text_data=json.dumps({"type": "error", "message": str(e)}))
            await self.close()

    async def tunnel_pull_tools(self, event: dict[str, Any]):
        call_id = event["call_id"]
        server_ids = event["server_ids"]
        await self.send(text_data=json.dumps(
            {"type": "pull_tools", "call_id": call_id, "server_ids": server_ids}
        ))

    async def tunnel_call(self, event: dict[str, Any]):
        call_id = event["call_id"]
        server_id = event["server_id"]
        tool = event["tool"]
        args = event["args"]
        await self.send(text_data=json.dumps(
            {"type": "call", "call_id": call_id, "server_id": server_id, "tool": tool, "args": args}
        ))

    async def tunnel_get_servers(self, event: dict[str, Any]):
        call_id = event["call_id"]
        await self.send(text_data=json.dumps(
            {"type": "get_servers", "call_id": call_id}
        ))