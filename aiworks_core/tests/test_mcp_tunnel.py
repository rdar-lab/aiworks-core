from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from aiworks_core.logic.mcp_tunnel import MCPTunnelManager
from aiworks_core.models import Tunnel


class TestTunnelModel(TestCase):
    def test_tunnel_str(self):
        tunnel = Tunnel(
            tunnel_id="t_test123",
            tunnel_api_key="tk_secret",
        )
        self.assertIn("t_test123", str(tunnel))

    def test_tunnel_is_connected_true(self):
        tunnel = Tunnel(
            tunnel_id="t_test123",
            tunnel_api_key="tk_secret",
            alive_beat=timezone.now(),
        )
        self.assertTrue(tunnel.is_connected)

    def test_tunnel_is_connected_false_expired(self):
        tunnel = Tunnel.objects.create(
            tunnel_id="t_expired",
            tunnel_api_key="tk_secret",
        )
        Tunnel.objects.filter(tunnel_id=tunnel.tunnel_id).update(
            alive_beat=timezone.now() - timezone.timedelta(seconds=120)
        )
        refreshed = Tunnel.objects.get(tunnel_id=tunnel.tunnel_id)
        self.assertFalse(refreshed.is_connected)


class TestMCPTunnelManagerConnect(TestCase):
    def tearDown(self):
        Tunnel.objects.all().delete()

    def test_connect_creates_new_tunnel(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_new")
        self.assertIsNotNone(result["tunnel_id"])
        self.assertEqual(Tunnel.objects.count(), 1)
        tunnel = Tunnel.objects.get(tunnel_id=result["tunnel_id"])
        self.assertEqual(tunnel.tunnel_api_key, "tk_new")

    def test_connect_with_api_key_mismatch_on_reconnect(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_original")
        tunnel_id = result["tunnel_id"]

        with self.assertRaises(ValueError) as ctx:
            MCPTunnelManager.connect_sync(
                tunnel_api_key="tk_wrong",
                tunnel_id=tunnel_id,
            )
        self.assertIn("Tunnel API key mismatch", str(ctx.exception))

    def test_connect_reconnect_valid_key(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_same")
        tunnel_id = result["tunnel_id"]

        result2 = MCPTunnelManager.connect_sync(
            tunnel_api_key="tk_same",
            tunnel_id=tunnel_id,
        )
        self.assertEqual(result2["tunnel_id"], tunnel_id)
        self.assertEqual(Tunnel.objects.count(), 1)

    def test_connect_reconnect_wrong_key(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_real")
        tunnel_id = result["tunnel_id"]

        with self.assertRaises(ValueError) as ctx:
            MCPTunnelManager.connect_sync(
                tunnel_api_key="tk_wrong",
                tunnel_id=tunnel_id,
            )
        self.assertIn("Tunnel API key mismatch", str(ctx.exception))

    def test_connect_reconnect_not_found_establishes(self):
        result = MCPTunnelManager.connect_sync(
            tunnel_api_key="tk_new",
            tunnel_id="t_nonexistent",
        )
        tunnel_id = result["tunnel_id"]
        self.assertIsNotNone(tunnel_id)


class TestMCPTunnelManagerIsConnected(TestCase):
    def setUp(self):
        Tunnel.objects.all().delete()

    def tearDown(self):
        Tunnel.objects.all().delete()

    def test_is_connected_true(self):
        Tunnel.objects.create(
            tunnel_id="t_connected",
            tunnel_api_key="tk_test",
            alive_beat=timezone.now(),
        )
        self.assertTrue(MCPTunnelManager.is_connected("t_connected"))

    def test_is_connected_false_expired(self):
        t = Tunnel.objects.create(
            tunnel_id="t_expired",
            tunnel_api_key="tk_test",
        )
        Tunnel.objects.filter(tunnel_id=t.tunnel_id).update(
            alive_beat=timezone.now() - timezone.timedelta(seconds=121)
        )
        self.assertFalse(MCPTunnelManager.is_connected("t_expired"))

    def test_is_connected_false_not_found(self):
        self.assertFalse(MCPTunnelManager.is_connected("t_nonexistent"))


class TestMCPTunnelManagerClaim(TestCase):
    def tearDown(self):
        Tunnel.objects.all().delete()
        User = get_user_model()
        User.objects.all().delete()

    def test_claim_tunnel_success(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_claim")
        tunnel_id = result["tunnel_id"]

        User = get_user_model()
        user_obj = User.objects.create_user(username="testuser", password="pass")

        success = MCPTunnelManager.claim_tunnel_sync(
            tunnel_id=tunnel_id,
            user_id=user_obj.id,
            tunnel_api_key="tk_claim",
        )
        self.assertTrue(success)

        tunnel = Tunnel.objects.get(tunnel_id=tunnel_id)
        self.assertEqual(tunnel.user_id, user_obj.id)

    def test_claim_tunnel_wrong_key(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_real")
        tunnel_id = result["tunnel_id"]

        User = get_user_model()
        user_obj = User.objects.create_user(username="testuser_wrongkey", password="pass")

        success = MCPTunnelManager.claim_tunnel_sync(
            tunnel_id=tunnel_id,
            user_id=user_obj.id,
            tunnel_api_key="tk_wrong",
        )
        self.assertFalse(success)

    def test_claim_tunnel_not_found(self):
        User = get_user_model()
        user_obj = User.objects.create_user(username="testuser_notfound", password="pass")

        success = MCPTunnelManager.claim_tunnel_sync(
            tunnel_id="t_does_not_exist",
            user_id=user_obj.id,
            tunnel_api_key="tk_any",
        )
        self.assertFalse(success)


class TestMCPTunnelManagerDisconnect(TestCase):
    def tearDown(self):
        Tunnel.objects.all().delete()

    def test_disconnect_removes_consumer_but_keeps_record(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_disconnect")
        tunnel_id = result["tunnel_id"]

        MCPTunnelManager.disconnect_sync(tunnel_id)

        self.assertTrue(Tunnel.objects.filter(tunnel_id=tunnel_id).exists())

    def test_delete_tunnel_removes_record(self):
        result = MCPTunnelManager.connect_sync(tunnel_api_key="tk_disconnect")
        tunnel_id = result["tunnel_id"]

        MCPTunnelManager.delete_tunnel_sync(tunnel_id)

        self.assertFalse(Tunnel.objects.filter(tunnel_id=tunnel_id).exists())