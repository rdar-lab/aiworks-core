import re

from django.conf import settings
from rest_framework import serializers

from .models import (
    AttachedFile,
    KnowledgeBase,
    User,
    MCPServer,
    PredefinedMCPServer,
    Notification
)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""

    profileContext = serializers.CharField(
        source="profile_context", required=False, allow_blank=True
    )
    lightMode = serializers.BooleanField(source="light_mode", required=False)
    hasSeenOnboarding = serializers.BooleanField(
        source="has_seen_onboarding", required=False
    )
    isPro = serializers.BooleanField(source="is_pro", read_only=True)

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "provider",
            "tier",
            "isPro",
            "avatar",
            "profileContext",
            "lightMode",
            "hasSeenOnboarding",
        )
        read_only_fields = (
            "id",
            "tier",
            "isPro",
            "provider",
        )


# noinspection PyPep8Naming,PyMethodMayBeStatic
class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""

    password = serializers.CharField(write_only=True, required=True, min_length=8)

    class Meta:
        model = User
        fields = (
            "username",
            "email",
            "password",
            "first_name",
            "last_name",
            "provider",
        )
        extra_kwargs = {
            # Suppress auto-generated UniqueValidator and model-level validators so
            # our validate_* methods can return user-friendly messages instead.
            "email": {"validators": []},
            "username": {"validators": []},
        }

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError(
                "This email is already registered. Please use reset password instead."
            )
        return value

    def validate_username(self, value):
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise serializers.ValidationError(
                "The username is invalid, please select a username that only contains "
                "letters, numbers, underscores (_), or hyphens (-)."
            )
        normalized_username = value.lower()
        qs = User.objects.filter(username__iexact=value)
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError(
                "This username is already taken, please choose a different username."
            )
        return normalized_username

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        register_hook = getattr(settings, 'AIWORKS_CORE_USER_REGISTER_HOOK', None)
        if register_hook:
            module_path, function_name = register_hook.rsplit('.', 1)
            module = __import__(module_path, fromlist=[function_name])
            getattr(module, function_name)(user, validated_data)
        return user


class AttachedFileSerializer(serializers.ModelSerializer):
    """Serializer for AttachedFile model"""

    class Meta:
        model = AttachedFile
        fields = ("id", "name", "file_type")


class KnowledgeBaseFileSerializer(serializers.ModelSerializer):
    """Serializer for AttachedFile within a knowledge base (no content to keep responses slim)"""

    class Meta:
        model = AttachedFile
        fields = ("id", "name", "created_at")


# noinspection PyPep8Naming,PyMethodMayBeStatic
class KnowledgeBaseSerializer(serializers.ModelSerializer):
    """Serializer for KnowledgeBase model"""

    filesCount = serializers.SerializerMethodField()
    files = KnowledgeBaseFileSerializer(source="kb_files", many=True, read_only=True)

    def get_filesCount(self, obj):
        # Prefer the DB-level annotation added by get_queryset(); fall back to
        # a live count when the instance was not fetched through the viewset
        # queryset (e.g. immediately after create/update).
        if hasattr(obj, "files_count"):
            return obj.files_count
        return obj.kb_files.count()

    class Meta:
        model = KnowledgeBase
        fields = ("id", "name", "filesCount", "files", "created_at", "updated_at")
        read_only_fields = ("id", "filesCount", "files", "created_at", "updated_at")


class PredefinedMCPServerSerializer(serializers.ModelSerializer):
    """Read-only serializer for predefined MCP server catalogue entries."""

    class Meta:
        model = PredefinedMCPServer
        fields = [
            "id",
            "name",
            "url",
            "auth_type",
            "headers",
            "server_type",
            "openapi_spec_url",
            "openapi_spec",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class MCPServerSerializer(serializers.ModelSerializer):
    # Expose whether credentials are stored so the UI can show a "configured" indicator
    # without sending the actual secret values back.
    has_credentials = serializers.SerializerMethodField(read_only=True)
    oauth_metadata = serializers.JSONField(required=False, allow_null=True)
    predefined_server_id = serializers.PrimaryKeyRelatedField(
        source="predefined_server",
        queryset=PredefinedMCPServer.objects.all(),
        required=False,
        allow_null=True,
    )

    class Meta:
        model = MCPServer
        fields = [
            "id",
            "name",
            "url",
            "headers",
            "auth_type",
            "has_credentials",
            "predefined_server_id",
            "server_type",
            "openapi_spec_url",
            "openapi_spec",
            "oauth_metadata",
            # Write-only credential fields — never returned to the client.
            "username",
            "password",
            "token",
            "refresh_token",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "has_credentials", "created_at", "updated_at"]
        extra_kwargs = {
            "username": {"required": False, "write_only": True, "allow_blank": True},
            "password": {"required": False, "write_only": True, "allow_blank": True},
            "token": {"required": False, "write_only": True, "allow_blank": True},
            "refresh_token": {"required": False, "write_only": True, "allow_blank": True},
            "auth_type": {"required": False},
            "url": {"required": False, "allow_blank": True},
            "name": {"required": False, "allow_blank": True},
            "server_type": {"required": False},
            "openapi_spec_url": {"required": False, "allow_blank": True},
            "openapi_spec": {"required": False, "allow_blank": True},
        }

    def get_has_credentials(self, obj: MCPServer) -> bool:
        """Return True if any credential value is stored for this server."""
        return bool(obj.username or obj.password or obj.token or obj.refresh_token or obj.oauth_metadata)

    @staticmethod
    def __get_oauth_metadata(obj: MCPServer) -> dict | None:
        """Return Oauth metadata but strip the client secret out of it."""
        oauth_metadata = obj.oauth_metadata
        predefined = obj.predefined_server_id
        if predefined and oauth_metadata and "client_secret" in oauth_metadata:
            oauth_metadata = dict(oauth_metadata)
            del oauth_metadata["client_secret"]
        return oauth_metadata

    def to_representation(self, instance):
        """Override to_representation to conditionally include oauth_metadata with client_secret stripped."""
        ret = super().to_representation(instance)
        ret["oauth_metadata"] = self.__get_oauth_metadata(instance)
        return ret

    @staticmethod
    def _validate_auth_credentials(
            auth_type: str,
            has_username: bool,
            has_password: bool,
            has_token: bool,
            has_refresh: bool,
    ) -> None:
        """Raise ValidationError if *auth_type* is missing its required credentials."""
        if auth_type == "basic" and not (has_username and has_password):
            raise serializers.ValidationError(
                {"auth_type": "Basic Auth requires both username and password."}
            )
        if auth_type == "bearer" and not has_token:
            raise serializers.ValidationError(
                {"auth_type": "Bearer Token auth requires a token."}
            )
        if auth_type == "oauth" and not has_refresh:
            raise serializers.ValidationError(
                {
                    "auth_type": (
                        "OAuth requires completing the authentication flow first "
                        "(refresh token missing)."
                    )
                }
            )

    def validate(self, data):
        is_create = self.instance is None
        predefined_server = data.get("predefined_server")

        # Determine the effective auth_type and server_type:
        # from predefined server (if given) or field data.
        if predefined_server is not None:
            auth_type = predefined_server.auth_type
            server_type = predefined_server.server_type
        else:
            auth_type = data.get(
                "auth_type",
                getattr(self.instance, "auth_type", "none") if self.instance else "none",
            )
            server_type = data.get(
                "server_type",
                getattr(self.instance, "server_type", "http") if self.instance else "http",
            )

        # server_type="openapi" requires either openapi_spec_url or openapi_spec
        if server_type == "openapi":
            if predefined_server is not None:
                # For predefined servers, the spec fields are locked to the predefined values.
                openapi_spec_url = predefined_server.openapi_spec_url
                openapi_spec = predefined_server.openapi_spec
            else:
                openapi_spec_url = data.get(
                    "openapi_spec_url",
                    getattr(self.instance, "openapi_spec_url", "") if self.instance else "",
                )
                openapi_spec = data.get(
                    "openapi_spec",
                    getattr(self.instance, "openapi_spec", "") if self.instance else "",
                )
            if not openapi_spec_url and not openapi_spec:
                raise serializers.ValidationError(
                    {
                        "openapi_spec_url": (
                            "server_type=openapi requires either openapi_spec_url or openapi_spec."
                        )
                    }
                )

        if is_create:
            # Require url/name when not creating from a predefined server.
            if predefined_server is None:
                if not data.get("url"):
                    raise serializers.ValidationError({"url": "This field is required."})
                if not data.get("name"):
                    raise serializers.ValidationError({"name": "This field is required."})
            # Enforce required credentials per auth_type at creation time.
            self._validate_auth_credentials(
                auth_type,
                has_username=bool(data.get("username")),
                has_password=bool(data.get("password")),
                has_token=bool(data.get("token")),
                has_refresh=bool(data.get("refresh_token")),
            )
        else:
            # PATCH: empty string means "keep existing value" — remove from validated_data
            # so super().update() does not overwrite the stored credential.
            for field in ("username", "password", "token", "refresh_token"):
                if field in data and not data[field]:
                    data.pop(field)

            # When switching to an auth type that requires credentials, enforce them.
            # For predefined-derived servers, auth_type is locked to the predefined value.
            if predefined_server is None:
                new_auth_type = data.get("auth_type")
                if new_auth_type and new_auth_type != self.instance.auth_type:
                    self._validate_auth_credentials(
                        new_auth_type,
                        has_username=bool(data.get("username") or self.instance.username),
                        has_password=bool(data.get("password") or self.instance.password),
                        has_token=bool(data.get("token") or self.instance.token),
                        has_refresh=bool(data.get("refresh_token") or self.instance.refresh_token),
                    )

        return data

class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for Notification model."""

    ctaAction = serializers.CharField(source="cta_action", read_only=True)
    ctaParams = serializers.JSONField(source="cta_params", read_only=True)
    expiresAt = serializers.DateTimeField(
        source="expires_at", read_only=True, allow_null=True
    )
    status = serializers.CharField(read_only=True)
    readAt = serializers.DateTimeField(
        source="read_at", read_only=True, allow_null=True
    )
    dismissedAt = serializers.DateTimeField(
        source="dismissed_at", read_only=True, allow_null=True
    )
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = Notification
        fields = (
            "id",
            "type",
            "status",
            "title",
            "body",
            "ctaAction",
            "ctaParams",
            "expiresAt",
            "readAt",
            "dismissedAt",
            "createdAt",
        )
        read_only_fields = fields


class NotificationDetailSerializer(NotificationSerializer):
    """Detailed notification serializer."""

    pass
