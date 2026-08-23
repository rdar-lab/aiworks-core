"""
Unit tests for the notification views:
- GET /api/notifications/: list notifications, pagination, type filter, unread count
- PUT /api/notifications/: update notification status (read, dismissed)
- Authentication enforcement
- Ownership enforcement
"""

import uuid

from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from . import _make_user, _auth_header
from ..models import Notification


def _make_notification(user, notif_type="advisor_followup", notif_status="sent", **kwargs):
    """Create a test Notification record."""
    return Notification.objects.create(
        user=user,
        type=notif_type,
        title=kwargs.get("title", "Test notification"),
        body=kwargs.get("body", "Test body"),
        cta_action=kwargs.get("cta_action", "view_discussion"),
        cta_params=kwargs.get("cta_params", {}),
        status=notif_status,
        expires_at=kwargs.get("expires_at", timezone.now() + timezone.timedelta(days=7)),
    )


class NotificationListTests(APITestCase):
    """GET /api/notifications/ — list, pagination, type filter, unread count."""

    def setUp(self):
        self.user = _make_user(username="notif_user", email="notif@example.com")
        self.other_user = _make_user(
            username="notif_other", email="notif_other@example.com"
        )
        self.n1 = _make_notification(self.user, "advisor_followup", "sent")
        self.n2 = _make_notification(self.user, "deliberation_complete", "sent")
        _make_notification(self.other_user, "advisor_followup", "sent")

    def test_returns_200(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_returns_own_notifications_only(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        ids = [n["id"] for n in resp.data["data"]]
        self.assertIn(str(self.n1.id), ids)
        self.assertIn(str(self.n2.id), ids)
        self.assertEqual(len(ids), 2)

    def test_does_not_return_other_users_notifications(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        ids = [n["id"] for n in resp.data["data"]]
        # Other user's notification should NOT appear
        self.assertEqual(len(ids), 2)

    def test_requires_authentication(self):
        resp = self.client.get("/api/notifications/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_response_contains_meta(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        meta = resp.data["meta"]
        self.assertIn("unread_count", meta)
        self.assertIn("total", meta)
        self.assertIn("limit", meta)
        self.assertIn("offset", meta)
        self.assertIn("has_more", meta)

    def test_unread_count_counts_sent_notifications(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        # Both n1 and n2 are 'sent' (unread)
        self.assertEqual(resp.data["meta"]["unread_count"], 2)

    def test_unread_count_excludes_read_notifications(self):
        self.n1.status = "read"
        self.n1.read_at = timezone.now()
        self.n1.save(update_fields=["status", "read_at"])
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        self.assertEqual(resp.data["meta"]["unread_count"], 1)

    def test_unread_count_excludes_dismissed_notifications(self):
        self.n1.status = "dismissed"
        self.n1.dismissed_at = timezone.now()
        self.n1.save(update_fields=["status", "dismissed_at"])
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        self.assertEqual(resp.data["meta"]["unread_count"], 1)

    def test_type_filter_returns_matching_notifications(self):
        resp = self.client.get(
            "/api/notifications/?type=advisor_followup", **_auth_header(self.user)
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        ids = [n["id"] for n in resp.data["data"]]
        self.assertIn(str(self.n1.id), ids)
        self.assertNotIn(str(self.n2.id), ids)

    def test_type_filter_with_no_match_returns_empty(self):
        resp = self.client.get(
            "/api/notifications/?type=research_complete", **_auth_header(self.user)
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp.data["data"]), 0)

    def test_pagination_limit(self):
        # Create a third notification
        _make_notification(self.user, "advisor_followup", "sent")
        resp = self.client.get(
            "/api/notifications/?limit=2&offset=0", **_auth_header(self.user)
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp.data["data"]), 2)
        self.assertEqual(resp.data["meta"]["limit"], 2)
        self.assertEqual(resp.data["meta"]["offset"], 0)
        self.assertTrue(resp.data["meta"]["has_more"])

    def test_pagination_offset(self):
        _make_notification(self.user, "advisor_followup", "sent")
        resp = self.client.get(
            "/api/notifications/?limit=2&offset=2", **_auth_header(self.user)
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp.data["data"]), 1)
        self.assertFalse(resp.data["meta"]["has_more"])

    def test_total_meta_field(self):
        resp = self.client.get("/api/notifications/", **_auth_header(self.user))
        self.assertEqual(resp.data["meta"]["total"], 2)


class NotificationUpdateTests(APITestCase):
    """PUT /api/notifications/ — mark as read or dismissed."""

    def setUp(self):
        self.user = _make_user(username="notif_upd_user", email="notif_upd@example.com")
        self.other_user = _make_user(
            username="notif_upd_other", email="notif_upd_other@example.com"
        )
        self.notification = _make_notification(self.user, "advisor_followup", "sent")
        self.other_notification = _make_notification(
            self.other_user, "advisor_followup", "sent"
        )

    def test_mark_as_read_returns_200(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_mark_as_read_updates_status_in_db(self):
        self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.notification.refresh_from_db()
        self.assertEqual(self.notification.status, "read")

    def test_mark_as_read_sets_read_at_timestamp(self):
        self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.notification.refresh_from_db()
        self.assertIsNotNone(self.notification.read_at)

    def test_mark_as_dismissed_updates_status_in_db(self):
        self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "dismissed"},
            format="json",
            **_auth_header(self.user),
        )
        self.notification.refresh_from_db()
        self.assertEqual(self.notification.status, "dismissed")

    def test_mark_as_dismissed_sets_dismissed_at_timestamp(self):
        self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "dismissed"},
            format="json",
            **_auth_header(self.user),
        )
        self.notification.refresh_from_db()
        self.assertIsNotNone(self.notification.dismissed_at)

    def test_response_contains_updated_notification_data(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("data", resp.data)
        self.assertEqual(resp.data["data"]["status"], "read")

    def test_invalid_status_returns_400(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "expired"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_notification_id_returns_400(self):
        resp = self.client.put(
            "/api/notifications/",
            {"status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_status_returns_400(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id)},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_nonexistent_notification_id_returns_404(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(uuid.uuid4()), "status": "read"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_cannot_update_other_users_notification(self):
        """User cannot modify another user's notification (404 — not 403, to avoid info leak)."""
        resp = self.client.put(
            "/api/notifications/",
            {
                "notification_id": str(self.other_notification.id),
                "status": "read",
            },
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)
        # Verify other user's notification was NOT modified
        self.other_notification.refresh_from_db()
        self.assertEqual(self.other_notification.status, "sent")

    def test_requires_authentication(self):
        resp = self.client.put(
            "/api/notifications/",
            {"notification_id": str(self.notification.id), "status": "read"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
