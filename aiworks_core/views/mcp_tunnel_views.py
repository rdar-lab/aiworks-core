import logging

from ..utils import async_to_sync
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import Tunnel

logger = logging.getLogger(__name__)
User = get_user_model()


class MCPTunnelConnectView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        from ..logic.mcp_tunnel import MCPTunnelManager

        tunnel_api_key = request.data.get("tunnel_api_key")
        tunnel_id = request.data.get("tunnel_id")

        if not tunnel_api_key:
            return Response(
                {"error": "tunnel_api_key is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = MCPTunnelManager.connect_sync(
                tunnel_api_key=tunnel_api_key,
                tunnel_id=tunnel_id,
            )
            return Response(result)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Tunnel.DoesNotExist as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception("Tunnel connect error: %s", e)
            return Response(
                {"error": "Tunnel connection failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MCPTunnelListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from ..logic.mcp_tunnel import MCPTunnelManager

        tunnels = Tunnel.objects.filter(user=request.user).order_by("-created_at")
        result = []
        for tunnel in tunnels:
            servers = []
            if tunnel.is_connected:
                try:
                    servers = async_to_sync(MCPTunnelManager.get_servers)(tunnel.tunnel_id)
                except Exception:
                    pass
            result.append({
                "tunnel_id": tunnel.tunnel_id,
                "is_connected": tunnel.is_connected,
                "created_at": tunnel.created_at.isoformat() if tunnel.created_at else None,
                "servers": servers or [],
            })
        return Response({"tunnels": result})


class MCPTunnelClaimView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, tunnel_id):
        from ..logic.mcp_tunnel import MCPTunnelManager

        tunnel_api_key = request.data.get("tunnel_api_key")
        if not tunnel_api_key:
            return Response(
                {"error": "tunnel_api_key required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            success = MCPTunnelManager.claim_tunnel_sync(
                tunnel_id=tunnel_id,
                user_id=request.user.id,
                tunnel_api_key=tunnel_api_key,
            )
            if success:
                return Response({"success": True})
            return Response(
                {"error": "Tunnel API key mismatch or tunnel not found"},
                status=status.HTTP_403_FORBIDDEN,
            )
        except Exception as e:
            logger.exception("Tunnel claim error: %s", e)
            return Response(
                {"error": "Claim failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MCPTunnelServersView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, tunnel_id):
        from ..logic.mcp_tunnel import MCPTunnelManager

        try:
            tunnel = Tunnel.objects.get(tunnel_id=tunnel_id)
        except Tunnel.DoesNotExist:
            return Response(
                {"error": "Tunnel not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if tunnel.user_id != request.user.id:
            return Response(
                {"error": "Tunnel does not belong to this user"},
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            servers = async_to_sync(MCPTunnelManager.get_servers)(tunnel_id)
        except Exception as e:
            logger.exception("get_servers error: %s", e)
            return Response(
                {"servers": [], "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response({"servers": servers or []})


class MCPTunnelToolsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, tunnel_id):
        from ..logic.mcp_tunnel import MCPTunnelManager

        server_ids = request.data.get("server_ids", [])

        try:
            tunnel = Tunnel.objects.get(tunnel_id=tunnel_id)
        except Tunnel.DoesNotExist:
            return Response(
                {"error": "Tunnel not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if tunnel.user_id != request.user.id:
            return Response(
                {"error": "Tunnel does not belong to this user"},
                status=status.HTTP_403_FORBIDDEN,
            )

        tools = None
        try:
            tools = async_to_sync(MCPTunnelManager.pull_tools)(
                tunnel_id, server_ids, timeout=30
            )
        except Exception as e:
            logger.exception("pull_tools error: %s", e)
            return Response(
                {"error": f"Failed to pull tools: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response({"tools": tools or []})