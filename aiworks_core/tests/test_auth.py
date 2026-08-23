"""
Authentication views tests for the Ai-Works Core API.
"""

from datetime import timedelta
from unittest.mock import patch

from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import RefreshToken

from . import (
    _make_user,
    _auth_header,
)
from ..models import (
    User,
    SiteConfiguration,
    VerificationToken,
)


class RegisterViewTests(APITestCase):
    def test_register_creates_user_and_sends_verification_email(self):
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "strongpass99",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertIn("detail", resp.data)
        self.assertTrue(User.objects.filter(username="newuser").exists())
        user = User.objects.get(username="newuser")
        self.assertFalse(user.email_verified)
        self.assertTrue(
            VerificationToken.objects.filter(
                user=user, token_type=VerificationToken.TOKEN_TYPE_EMAIL
            ).exists()
        )

    def test_register_without_password(self):
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "nopassuser",
                "email": "nopass@example.com",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_duplicate_username_returns_400(self):
        _make_user(username="existinguser")
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "existinguser",
                "email": "other@example.com",
                "password": "pass123",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)
        self.assertIn("already taken", resp.data["error"])

    def test_register_duplicate_email_returns_400_with_message(self):
        _make_user(username="uniqueuser", email="taken@example.com")
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "brandnewuser",
                "email": "taken@example.com",
                "password": "strongpass99",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)
        self.assertIn("already registered", resp.data["error"])

    def test_register_invalid_username_returns_400_with_message(self):
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "bad username!",
                "email": "valid@example.com",
                "password": "strongpass99",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)
        self.assertIn("invalid", resp.data["error"])

    def test_register_stores_username_lowercase(self):
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "MixedCase",
                "email": "mixed@example.com",
                "password": "strongpass99",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(username="mixedcase").exists())
        self.assertFalse(User.objects.filter(username="MixedCase").exists())

    def test_register_case_variant_of_existing_username_returns_400(self):
        _make_user(username="alice")
        resp = self.client.post(
            "/api/auth/register/",
            {
                "username": "ALICE",
                "email": "alice2@example.com",
                "password": "strongpass99",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class LoginViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user(username="loginuser", password="mypassword")

    def test_login_valid_credentials_returns_tokens(self):
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": "loginuser",
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("tokens", resp.data)

    def test_login_invalid_password_returns_401(self):
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": "loginuser",
                "password": "wrongpass",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_missing_credentials_returns_400(self):
        resp = self.client.post("/api/auth/login/", {}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_disabled_account_returns_401(self):
        self.user.is_active = False
        self.user.save()
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": "loginuser",
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_case_insensitive_username(self):
        for variant in ("LOGINUSER", "LoginUser", "loginUSER"):
            resp = self.client.post(
                "/api/auth/login/",
                {
                    "username": variant,
                    "password": "mypassword",
                },
                format="json",
            )
            self.assertEqual(
                resp.status_code,
                status.HTTP_200_OK,
                f"Login should succeed for username variant '{variant}'",
            )
            self.assertIn("tokens", resp.data)

    def test_login_unverified_email_returns_403(self):
        self.user.email_verified = False
        self.user.save(update_fields=["email_verified"])
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": "loginuser",
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("error", resp.data)

    def test_login_with_email_returns_tokens(self):
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": self.user.email,
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("tokens", resp.data)

    def test_login_with_email_case_insensitive(self):
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": self.user.email.upper(),
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("tokens", resp.data)

    def test_login_with_wrong_email_returns_401(self):
        resp = self.client.post(
            "/api/auth/login/",
            {
                "username": "nonexistent@example.com",
                "password": "mypassword",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)


class GoogleAuthViewTests(APITestCase):
    VALID_ID_INFO = {
        "email": "googleuser@gmail.com",
        "email_verified": True,
        "sub": "google-sub-12345",
        "given_name": "Google",
        "family_name": "User",
        "picture": "https://example.com/photo.jpg",
    }

    def _post(self, credential="fake-google-token"):
        return self.client.post(
            "/api/auth/google/", {"credential": credential}, format="json"
        )

    @staticmethod
    def _set_google_client_id(value):
        cfg = SiteConfiguration.get_solo()
        cfg.google_client_id = value
        cfg.save(update_fields=["google_client_id"])

    def test_missing_credential_returns_400(self):
        self._set_google_client_id("test-client-id")
        resp = self.client.post("/api/auth/google/", {}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)

    def test_unconfigured_client_id_returns_503(self):
        self._set_google_client_id("")
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertIn("error", resp.data)

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_invalid_google_token_returns_400(self, mock_verify):
        self._set_google_client_id("test-client-id")
        mock_verify.side_effect = ValueError("bad token")
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_unverified_email_returns_400(self, mock_verify):
        self._set_google_client_id("test-client-id")
        mock_verify.return_value = {**self.VALID_ID_INFO, "email_verified": False}
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", resp.data)

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_new_user_is_created_and_tokens_returned(self, mock_verify):
        self._set_google_client_id("test-client-id")
        mock_verify.return_value = self.VALID_ID_INFO
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("tokens", resp.data)
        self.assertIn("user", resp.data)
        user = User.objects.get(email="googleuser@gmail.com")
        self.assertTrue(user.email_verified)
        self.assertEqual(user.provider, "google")
        self.assertFalse(user.has_usable_password())

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_existing_user_by_email_logs_in(self, mock_verify):
        self._set_google_client_id("test-client-id")
        existing = _make_user(
            username="existinguser", email="googleuser@gmail.com", password="pass123"
        )
        mock_verify.return_value = self.VALID_ID_INFO
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("tokens", resp.data)
        self.assertEqual(resp.data["user"]["email"], existing.email)

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_disabled_account_returns_403(self, mock_verify):
        self._set_google_client_id("test-client-id")
        existing = _make_user(
            username="disabledgoogle", email="googleuser@gmail.com", password="pass123"
        )
        existing.is_active = False
        existing.save()
        mock_verify.return_value = self.VALID_ID_INFO
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("error", resp.data)

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_username_collision_generates_unique_username(self, mock_verify):
        self._set_google_client_id("test-client-id")
        _make_user(username="googleuser", email="other@example.com", password="pass123")
        mock_verify.return_value = self.VALID_ID_INFO
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        created = User.objects.get(email="googleuser@gmail.com")
        self.assertNotEqual(created.username, "googleuser")

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_avatar_updated_when_blank(self, mock_verify):
        self._set_google_client_id("test-client-id")
        existing = _make_user(
            username="avataruser", email="googleuser@gmail.com", password="pass123"
        )
        existing.avatar = ""
        existing.save()
        mock_verify.return_value = self.VALID_ID_INFO
        resp = self._post()
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        existing.refresh_from_db()
        self.assertEqual(existing.avatar, self.VALID_ID_INFO["picture"])


class ServerSettingsViewTests(APITestCase):
    @staticmethod
    def _set_google_client_id(value):
        cfg = SiteConfiguration.get_solo()
        cfg.google_client_id = value
        cfg.save(update_fields=["google_client_id"])

    def test_returns_google_client_id_when_configured(self):
        self._set_google_client_id("my-client-id.apps.googleusercontent.com")
        resp = self.client.get("/api/server-settings/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(
            resp.data["googleClientId"], "my-client-id.apps.googleusercontent.com"
        )

    def test_returns_empty_string_when_not_configured(self):
        self._set_google_client_id("")
        resp = self.client.get("/api/server-settings/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["googleClientId"], "")

    def test_accessible_without_authentication(self):
        resp = self.client.get("/api/server-settings/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


class VerifyEmailViewTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="verifyuser", email="verify@example.com", password="pass123"
        )

    def test_verify_email_valid_token(self):
        token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_EMAIL,
            expires_at=timezone.now() + timedelta(hours=24),
        )
        resp = self.client.post(
            "/api/auth/verify-email/", {"token": str(token.token)}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("detail", resp.data)
        self.assertNotIn("tokens", resp.data)
        self.user.refresh_from_db()
        self.assertTrue(self.user.email_verified)
        self.assertFalse(VerificationToken.objects.filter(pk=token.pk).exists())

    def test_verify_email_invalid_token(self):
        resp = self.client.post(
            "/api/auth/verify-email/",
            {"token": "00000000-0000-0000-0000-000000000000"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_verify_email_malformed_token(self):
        resp = self.client.post(
            "/api/auth/verify-email/", {"token": "not-a-valid-uuid"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_verify_email_expired_token(self):
        token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_EMAIL,
            expires_at=timezone.now() - timedelta(hours=1),
        )
        resp = self.client.post(
            "/api/auth/verify-email/", {"token": str(token.token)}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.user.refresh_from_db()
        self.assertFalse(self.user.email_verified)

    def test_verify_email_missing_token(self):
        resp = self.client.post("/api/auth/verify-email/", {}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class PasswordResetViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user(
            username="resetuser", email="reset@example.com", password="oldpassword"
        )

    def test_request_password_reset_existing_email(self):
        resp = self.client.post(
            "/api/auth/password-reset/", {"email": "reset@example.com"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("detail", resp.data)
        self.assertTrue(
            VerificationToken.objects.filter(
                user=self.user, token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET
            ).exists()
        )

    def test_request_password_reset_unknown_email(self):
        resp = self.client.post(
            "/api/auth/password-reset/", {"email": "nobody@example.com"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_request_password_reset_replaces_existing_token(self):
        self.client.post(
            "/api/auth/password-reset/", {"email": "reset@example.com"}, format="json"
        )
        self.client.post(
            "/api/auth/password-reset/", {"email": "reset@example.com"}, format="json"
        )
        self.assertEqual(
            VerificationToken.objects.filter(
                user=self.user, token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET
            ).count(),
            1,
        )

    def test_confirm_password_reset_valid_token(self):
        token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
            expires_at=timezone.now() + timedelta(hours=1),
        )
        resp = self.client.post(
            "/api/auth/password-reset/confirm/",
            {
                "token": str(token.token),
                "password": "newpassword123",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("newpassword123"))
        self.assertFalse(VerificationToken.objects.filter(pk=token.pk).exists())

    def test_confirm_password_reset_invalid_token(self):
        resp = self.client.post(
            "/api/auth/password-reset/confirm/",
            {
                "token": "00000000-0000-0000-0000-000000000000",
                "password": "newpassword123",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_confirm_password_reset_malformed_token(self):
        resp = self.client.post(
            "/api/auth/password-reset/confirm/",
            {
                "token": "not-a-valid-uuid",
                "password": "newpassword123",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_confirm_password_reset_short_password(self):
        token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
            expires_at=timezone.now() + timedelta(hours=1),
        )
        resp = self.client.post(
            "/api/auth/password-reset/confirm/",
            {
                "token": str(token.token),
                "password": "short",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_confirm_password_reset_expired_token(self):
        token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
            expires_at=timezone.now() - timedelta(hours=1),
        )
        resp = self.client.post(
            "/api/auth/password-reset/confirm/",
            {
                "token": str(token.token),
                "password": "newpassword123",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class CurrentUserViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user()

    def test_authenticated_returns_user_data(self):
        resp = self.client.get("/api/auth/me/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["username"], self.user.username)

    def test_unauthenticated_returns_401(self):
        resp = self.client.get("/api/auth/me/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_patch_cannot_change_tier(self):
        resp = self.client.patch(
            "/api/auth/me/",
            {"tier": "ADMIN"},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertNotEqual(self.user.tier, "ADMIN")


class LightModePatchTests(APITestCase):
    def setUp(self):
        self.user = _make_user()

    def _patch(self, payload):
        return self.client.patch(
            "/api/auth/me/",
            payload,
            format="json",
            **_auth_header(self.user),
        )

    def test_get_returns_light_mode_field(self):
        resp = self.client.get("/api/auth/me/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("lightMode", resp.data)

    def test_patch_light_mode_true_persists(self):
        resp = self._patch({"lightMode": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.light_mode)

    def test_patch_light_mode_false_persists(self):
        self.user.light_mode = True
        self.user.save(update_fields=["light_mode"])
        resp = self._patch({"lightMode": False})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertFalse(self.user.light_mode)

    def test_patch_light_mode_response_contains_updated_value(self):
        resp = self._patch({"lightMode": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(resp.data["lightMode"])

    def test_patch_light_mode_does_not_affect_other_fields(self):
        original_username = self.user.username
        resp = self._patch({"lightMode": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertEqual(self.user.username, original_username)


class OnboardingBannerPatchTests(APITestCase):
    def setUp(self):
        self.user = _make_user()

    def _patch(self, payload):
        return self.client.patch(
            "/api/auth/me/",
            payload,
            format="json",
            **_auth_header(self.user),
        )

    def test_get_returns_has_seen_onboarding_field(self):
        resp = self.client.get("/api/auth/me/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("hasSeenOnboarding", resp.data)

    def test_patch_has_seen_onboarding_true_persists(self):
        resp = self._patch({"hasSeenOnboarding": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.has_seen_onboarding)

    def test_patch_has_seen_onboarding_false_persists(self):
        self.user.has_seen_onboarding = True
        self.user.save(update_fields=["has_seen_onboarding"])
        resp = self._patch({"hasSeenOnboarding": False})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertFalse(self.user.has_seen_onboarding)

    def test_patch_has_seen_onboarding_response_contains_updated_value(self):
        resp = self._patch({"hasSeenOnboarding": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(resp.data["hasSeenOnboarding"])

    def test_patch_has_seen_onboarding_does_not_affect_other_fields(self):
        original_username = self.user.username
        resp = self._patch({"hasSeenOnboarding": True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertEqual(self.user.username, original_username)


class TokenRefreshViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user(username="refreshuser", password="pass123")
        refresh = RefreshToken.for_user(self.user)
        self.refresh_token = str(refresh)

    def test_refresh_returns_new_access_token(self):
        resp = self.client.post(
            "/api/auth/refresh/",
            {"refresh": self.refresh_token},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn(
            "access", resp.data, "Refresh endpoint must return a new access token"
        )

    def test_refresh_does_not_return_refresh_token_when_rotation_disabled(self):
        resp = self.client.post(
            "/api/auth/refresh/",
            {"refresh": self.refresh_token},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertNotIn(
            "refresh",
            resp.data,
            "With ROTATE_REFRESH_TOKENS=False the backend must not send a new refresh token",
        )

    def test_invalid_refresh_token_returns_401(self):
        resp = self.client.post(
            "/api/auth/refresh/",
            {"refresh": "notavalidtoken"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)


class LogoutViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user(username="logoutuser", password="pass123")
        refresh = RefreshToken.for_user(self.user)
        self.refresh_token = str(refresh)
        self.access_token = str(refresh.access_token)

    def test_logout_with_valid_refresh_token_returns_200(self):
        resp = self.client.post(
            "/api/auth/logout/",
            {"refresh": self.refresh_token},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_logout_without_refresh_token_still_returns_200(self):
        resp = self.client.post(
            "/api/auth/logout/",
            {},
            format="json",
            **_auth_header(self.user),
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_logout_requires_authentication(self):
        resp = self.client.post(
            "/api/auth/logout/", {"refresh": self.refresh_token}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
