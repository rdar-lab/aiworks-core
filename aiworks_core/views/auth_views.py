import logging
import re
from datetime import timedelta
from typing import cast

import requests as http_requests
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import update_last_login
from django.core.exceptions import ValidationError
from django.utils import timezone
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
    throttle_classes,
)
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from ..models import (
    VerificationToken,
    SiteConfiguration,
    User,
)
from ..serializers import (
    UserSerializer,
    UserRegistrationSerializer,
)

# Precompiled regex for sanitizing usernames (keeps only lowercase alphanumeric, underscore, hyphen)
_USERNAME_SANITIZE_RE = re.compile(r"[^a-z0-9_-]")

logger = logging.getLogger(__name__)


class LoginRateThrottle(AnonRateThrottle):
    """Tight per-IP throttle applied to login and register endpoints."""

    scope = "login"


def _send_verification_email(user):
    """Send an email verification token to the newly registered user."""
    from ..logic.emails import EmailService

    token = VerificationToken.objects.create(
        user=user,
        token_type=VerificationToken.TOKEN_TYPE_EMAIL,
        expires_at=timezone.now() + timedelta(hours=24),
    )
    EmailService.send_verification_email(user, str(token.token))


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def register(request):
    """Register a new user and send a verification email."""
    serializer = UserRegistrationSerializer(data=request.data)
    serializer.provider = "email"
    if serializer.is_valid():
        user = serializer.save()
        logger.info("register | new user created: %s", user.username)

        _send_verification_email(user)

        return Response(
            {
                "detail": "Registration successful. Please check your email to verify your account."
            },
            status=status.HTTP_201_CREATED,
        )

    logger.warning("register | validation failed: %s", serializer.errors)
    errors = serializer.errors
    error_msg = None
    for field in ("email", "username", "password"):
        field_errors = errors.get(field)
        if field_errors:
            error_msg = str(field_errors[0])
            break
    if not error_msg:
        for field_errors in errors.values():
            if field_errors:
                error_msg = str(field_errors[0])
                break
    return Response(
        {"error": error_msg or "Registration failed."},
        status=status.HTTP_400_BAD_REQUEST,
    )


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def verify_email(request):
    """Verify a user's email address using the token from the verification email."""
    token_value = request.data.get("token")
    if not token_value:
        return Response(
            {"error": "Token is required"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        token = VerificationToken.objects.select_related("user").get(
            token=token_value,
            token_type=VerificationToken.TOKEN_TYPE_EMAIL,
        )
    except (VerificationToken.DoesNotExist, ValueError, ValidationError):
        return Response(
            {"error": "Invalid or expired verification token"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not token.is_valid():
        token.delete()
        return Response(
            {"error": "Invalid or expired verification token"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = token.user
    user.email_verified = True
    user.save(update_fields=["email_verified"])
    token.delete()
    logger.info("verify_email | email verified for user=%s", user.username)

    return Response({"detail": "Email verified successfully. You can now log in."})


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def login(request):
    """Login user with username and password"""
    username = request.data.get("username")
    password = request.data.get("password")

    if not username or not password:
        return Response(
            {"error": "Username and password are required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Resolve to stored username for case-insensitive lookup, also support email
    try:
        db_user = User.objects.get(username__iexact=username)
        username = db_user.username
    except User.DoesNotExist:
        try:
            db_user = User.objects.get(email__iexact=username)
            username = db_user.username
        except User.DoesNotExist:
            pass

    # Authenticate user
    user = cast(User, authenticate(username=username, password=password))

    if user is None:
        logger.warning("login | failed for username=%s", username)
        return Response(
            {"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED
        )

    if not user.is_active:
        logger.warning("login | account disabled: %s", username)
        return Response(
            {"error": "Account is disabled"}, status=status.HTTP_403_FORBIDDEN
        )

    if not user.email_verified:
        logger.warning("login | email not verified for username=%s", username)
        return Response(
            {"error": "Please verify your email address before logging in."},
            status=status.HTTP_403_FORBIDDEN,
        )

    # Generate JWT tokens
    refresh = RefreshToken.for_user(user)
    logger.info("login | success for username=%s", username)

    _bump_last_login(user)

    return Response(
        {
            "user": UserSerializer(user).data,
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            },
        }
    )


def _bump_last_login(user: User):
    # Update Django's last_login field (safe to wrap).
    # Only write if not already updated today — avoids a SELECT FOR UPDATE
    # on every /api/auth/me/ request.
    today = timezone.now().date()
    if user.last_login and user.last_login.date() == today:
        return
    # noinspection PyBroadException
    try:
        # noinspection PyTypeChecker
        update_last_login(None, user)
    except Exception:
        logger.exception(
            "bump_last_login | failed to update last_login for user=%s", user.username
        )


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def request_password_reset(request):
    """Send a password reset email to the user."""
    from ..logic.emails import EmailService

    email = request.data.get("email", "").strip()
    if not email:
        return Response(
            {"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Always return success to avoid leaking whether an email is registered
    try:
        user = User.objects.get(email__iexact=email)
    except User.DoesNotExist:
        logger.info("request_password_reset | no user found for email=%s", email)
        return Response(
            {
                "detail": "If that email is registered you will receive a reset token shortly."
            }
        )

    # Delete any existing password reset tokens for this user
    VerificationToken.objects.filter(
        user=user, token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET
    ).delete()

    token = VerificationToken.objects.create(
        user=user,
        token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
        expires_at=timezone.now() + timedelta(hours=1),
    )
    EmailService.send_password_reset_email(user, str(token.token))
    logger.info("request_password_reset | reset email sent to %s", user.email)
    return Response(
        {
            "detail": "If that email is registered you will receive a reset token shortly."
        }
    )


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def confirm_password_reset(request):
    """Reset the user's password using the token from the reset email."""
    token_value = request.data.get("token")
    new_password = request.data.get("password", "")

    if not token_value:
        return Response(
            {"error": "Token is required"}, status=status.HTTP_400_BAD_REQUEST
        )
    if len(new_password) < 8:
        return Response(
            {"error": "Password must be at least 8 characters"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        token = VerificationToken.objects.select_related("user").get(
            token=token_value,
            token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
        )
    except (VerificationToken.DoesNotExist, ValueError, ValidationError):
        return Response(
            {"error": "Invalid or expired reset token"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not token.is_valid():
        token.delete()
        return Response(
            {"error": "Invalid or expired reset token"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = token.user
    user.set_password(new_password)
    # Mark account as verified since password reset confirms email ownership
    user.email_verified = True
    user.provider = "email"
    user.save(update_fields=["provider", "password", "email_verified"])
    token.delete()
    logger.info("confirm_password_reset | password reset for user=%s", user.username)
    return Response({"detail": "Password reset successfully. You can now log in."})


@api_view(["GET"])
@permission_classes([AllowAny])
def server_settings(request):
    """Return public server-side settings consumed by the frontend."""
    config = SiteConfiguration.get_solo()
    return Response(
        {
            "googleClientId": config.google_client_id or "",
        }
    )


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def google_auth(request):
    """Authenticate or register a user via Google Sign-In.

    Supports two flows:
    - ID token flow (popup): pass ``credential`` (JWT issued by Google GIS).
    - Auth code flow (redirect): pass ``code`` + ``redirect_uri``.  The server
      exchanges the code for tokens using the configured client secret, then
      verifies the resulting ID token.  This flow is required on mobile Safari
      where ``window.open`` popups are blocked.
    """
    config = SiteConfiguration.get_solo()
    client_id = config.google_client_id
    if not client_id:
        return Response(
            {"error": "Google authentication is not configured"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    credential = request.data.get("credential")
    code = request.data.get("code")
    redirect_uri = request.data.get("redirect_uri")

    if code:
        # --- Auth code (redirect) flow ---
        client_secret = config.google_client_secret
        if not client_secret:
            return Response(
                {
                    "error": "Google authentication is not fully configured (missing client secret)"
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if not redirect_uri:
            return Response(
                {"error": "redirect_uri is required for the auth code flow"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            token_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                timeout=10,
            )
            token_data = token_response.json()
        except Exception as exc:
            logger.warning("google_auth | token exchange request failed: %s", exc)
            return Response(
                {"error": "Failed to exchange Google auth code"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        if not token_response.ok:
            error_description = token_data.get("error_description") or token_data.get(
                "error", "unknown"
            )
            logger.warning(
                "google_auth | token exchange failed (HTTP %s): %s",
                token_response.status_code,
                error_description,
            )
            return Response(
                {"error": "Google auth code exchange failed"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        credential = token_data.get("id_token")
        if not credential:
            logger.warning("google_auth | token exchange returned no id_token")
            return Response(
                {"error": "Google auth code exchange failed"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    elif not credential:
        return Response(
            {"error": "Either credential or code is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        id_info = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            client_id,
        )
    except ValueError as exc:
        logger.warning("google_auth | token verification failed: %s", exc)
        return Response(
            {"error": "Invalid Google token"}, status=status.HTTP_400_BAD_REQUEST
        )

    email = id_info.get("email", "").strip().lower()
    if not email or not id_info.get("email_verified"):
        return Response(
            {"error": "Google account email is not verified"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    first_name = id_info.get("given_name", "")
    last_name = id_info.get("family_name", "")
    avatar = id_info.get("picture", "")

    try:
        user = User.objects.get(email__iexact=email)
        _update_existing_user_from_oauth(
            user, first_name, last_name, avatar, provider="google"
        )
    except User.DoesNotExist:
        user = _create_new_user_from_oauth(
            email, first_name, last_name, avatar, provider="google"
        )
        logger.info("google_auth | new user created: %s", user.username)

    if not user.is_active:
        logger.warning("google_auth | account disabled: %s", user.username)
        return Response(
            {"error": "Account is disabled"}, status=status.HTTP_403_FORBIDDEN
        )

    refresh = RefreshToken.for_user(user)
    logger.info("google_auth | success for user=%s", user.username)

    _bump_last_login(user)

    return Response(
        {
            "user": UserSerializer(user).data,
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            },
        }
    )


def _create_new_user_from_oauth(email, first_name, last_name, avatar, provider) -> User:
    username = _calc_new_username(email, first_name, last_name)

    user = User(
        username=username,
        email=email,
        first_name=first_name,
        last_name=last_name,
        avatar=avatar,
        email_verified=True,
        provider=provider,
    )
    user.set_unusable_password()
    user.save()
    return user


def _update_existing_user_from_oauth(
        user: User, first_name, last_name, avatar, provider
):
    update_fields = []

    if first_name and first_name != user.first_name:
        user.first_name = first_name
        update_fields.append("first_name")

    if last_name and last_name != user.last_name:
        user.last_name = last_name
        update_fields.append("last_name")

    if user.provider != provider:
        user.provider = provider
        update_fields.append("provider")

    if avatar and avatar != user.avatar:
        user.avatar = avatar
        update_fields.append("avatar")
    # Mark email as verified since Google confirmed it
    if not user.email_verified:
        user.email_verified = True
        update_fields.append("email_verified")

    # Make sure to update the password field to prevent login with old password after switching to oauth
    if user.has_usable_password():
        user.set_unusable_password()
        update_fields.append("password")

    if update_fields:
        user.save(update_fields=update_fields)


def _calc_new_username(email, first_name, last_name) -> str:
    if first_name and last_name:
        username_candidate = f"{first_name}_{last_name}".lower()
    else:
        username_candidate = email.split("@")[0].lower()

    base_username = _USERNAME_SANITIZE_RE.sub("", username_candidate)
    if len(base_username) < 2:
        base_username = "user"

    username = base_username
    counter = 1
    while User.objects.filter(username__iexact=username).exists():
        username = f"{base_username}{counter}"
        counter += 1
    return username


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    """Logout user and blacklist the refresh token to prevent reuse."""
    refresh_token_str = request.data.get("refresh")
    if refresh_token_str:
        try:
            token = RefreshToken(refresh_token_str)
            token.blacklist()
            logger.info("logout | token blacklisted for user=%s", request.user.username)
        except TokenError as exc:
            logger.warning(
                "logout | invalid/expired refresh token for user=%s: %s",
                request.user.username,
                exc,
            )
    else:
        logger.warning(
            "logout | no refresh token provided for user=%s", request.user.username
        )
    return Response({"OK"}, status=status.HTTP_200_OK)


def _get_user_serializer_class():
    """Return the configured user serializer class, or the default."""
    serializer_class = getattr(settings, 'AIWORKS_CORE_USER_SERIALIZER_CLASS', None)
    if serializer_class:
        module_path, class_name = serializer_class.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    return UserSerializer


@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def current_user(request):
    """Get or update current user info"""
    serializer_class = _get_user_serializer_class()
    if request.method == "PATCH":
        data = request.data.copy()
        serializer = serializer_class(request.user, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        # Bump the last login, since the UI is also using this request to validate the user
        _bump_last_login(cast(User, request.user))
        return Response(serializer_class(request.user).data)
