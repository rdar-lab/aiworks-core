import uuid

from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.functions import Lower
from django.utils import timezone

# MCPServer server_type choices
MCP_SERVER_TYPE_HTTP = "http"
MCP_SERVER_TYPE_OPENAPI = "openapi"
MCP_SERVER_TYPE_CHOICES = [
    (MCP_SERVER_TYPE_HTTP, "HTTP MCP Server"),
    (MCP_SERVER_TYPE_OPENAPI, "OpenAPI Server"),
]


class User(AbstractUser):
    """Extended User model with additional fields"""

    TIER_FREE = "free"
    TIER_PRO = "pro"
    TIER_CHOICES = [
        (TIER_FREE, "Free"),
        (TIER_PRO, "Pro"),
    ]

    email = models.EmailField(unique=True)
    provider = models.CharField(max_length=50, default="email")
    tier = models.CharField(max_length=20, default=TIER_FREE, choices=TIER_CHOICES)
    avatar = models.URLField(blank=True, null=True)
    profile_context = models.TextField(blank=True)
    llm_generated_background = models.TextField(
        blank=True,
        default="",
        help_text=(
            "LLM-synthesised background paragraph derived from the user's profile "
            "context and memory entries.  Auto-cleared whenever profile_context or "
            "any MemoryEntry changes; lazily regenerated on next use."
        ),
    )
    email_verified = models.BooleanField(default=False)

    # Favorite session IDs for the user
    favorite_session_ids = models.JSONField(default=list, blank=True)
    # UI preference: light mode (True) or dark mode (False)
    light_mode = models.BooleanField(default=False)
    has_seen_onboarding = models.BooleanField(
        default=False,
        help_text="Whether the user has dismissed the first-time onboarding banner.",
    )
    # App-specific extra data stored as JSON (e.g., APP user profile fields)
    extra_data = models.JSONField(default=dict, blank=True)

    # noinspection PyUnresolvedReferences
    class Meta(AbstractUser.Meta):
        constraints = [
            models.UniqueConstraint(
                Lower("username"),
                name="user_username_case_insensitive_unique",
            ),
            models.CheckConstraint(
                condition=models.Q(tier__in=["free", "pro"]),
                name="user_tier_valid_value",
            ),
        ]
        indexes = []

    @property
    def is_pro(self):
        return self.tier == "pro"

    def __str__(self):
        return self.username


class VerificationToken(models.Model):
    """Token for email verification and password reset"""

    TOKEN_TYPE_EMAIL = "email_verify"
    TOKEN_TYPE_PASSWORD_RESET = "password_reset"

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="verification_tokens"
    )
    token = models.UUIDField(default=uuid.uuid4, unique=True, db_index=True)
    token_type = models.CharField(max_length=20, db_index=True)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["user", "token_type", "created_at"],
                name="vtoken_user_type_cr_idx",
            ),
        ]

    def is_valid(self):
        return timezone.now() < self.expires_at

    def __str__(self):
        return f"{self.token_type} token for {self.user.username}"


class Tunnel(models.Model):
    """Persistent WebSocket tunnel from desktop to cloud."""

    tunnel_id = models.CharField(max_length=100, primary_key=True)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tunnels",
    )
    tunnel_api_key = models.CharField(max_length=200)
    alive_beat = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["alive_beat"], name="tunnel_alive_beat_idx"),
        ]

    def __str__(self):
        return f"tunnel:{self.tunnel_id}"

    @property
    def is_connected(self) -> bool:
        return (timezone.now() - self.alive_beat).total_seconds() < 60


class KnowledgeBase(models.Model):
    """User-owned knowledge base for attaching documents and websites to sessions"""

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="knowledge_bases"
    )
    name = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = [("user", "name")]

    def __str__(self):
        return f"{self.name} ({self.user.username})"


class PredefinedMCPServer(models.Model):
    """Admin-managed catalogue of built-in MCP servers that users can add with one click"""

    AUTH_TYPE_NONE = "none"
    AUTH_TYPE_BASIC = "basic"
    AUTH_TYPE_BEARER = "bearer"
    AUTH_TYPE_OAUTH = "oauth"
    AUTH_TYPE_CHOICES = [
        ("none", "No Auth"),
        ("basic", "Basic Auth"),
        ("bearer", "Bearer Token"),
        ("oauth", "OAuth 2.1"),
    ]

    name = models.CharField(max_length=500, unique=True)
    url = models.CharField(
        max_length=2000, help_text="HTTP(S) or SSE endpoint URL of the MCP server"
    )
    auth_type = models.CharField(
        max_length=20,
        choices=AUTH_TYPE_CHOICES,
        default="none",
        help_text="Authentication method required by this server",
    )
    client_id = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="OAuth 2.1 client_id (used when dynamic client registration is not supported)",
    )
    client_secret = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="OAuth 2.1 client_secret (used when dynamic client registration is not supported)",
    )
    authorization_endpoint = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="Custom OAuth authorization endpoint URL (overrides auto-discovery). Only used when auth_type='oauth'.",
    )
    token_endpoint = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="Custom OAuth token endpoint URL (overrides auto-discovery). Only used when auth_type='oauth'.",
    )
    scope = models.CharField(
        max_length=1000,
        blank=True,
        default="",
        help_text="Custom OAuth scope string (overrides auto-discovered scopes_supported). Only used when auth_type='oauth'.",
    )
    custom_auth_params = models.JSONField(
        default=dict,
        blank=True,
        help_text="Custom URL query params appended to every OAuth authorization request. Only used when auth_type='oauth'.",
    )
    headers = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional additional HTTP headers sent with every request to this server",
    )
    server_type = models.CharField(
        max_length=20,
        default="http",
        choices=MCP_SERVER_TYPE_CHOICES,
        blank=False,
        help_text='"http" = MCP over HTTP; "openapi" = FastMCP wrapping OpenAPI spec',
    )
    openapi_spec_url = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="Admin-provided locked OpenAPI spec URL (for server_type=openapi)",
    )
    openapi_spec = models.TextField(
        blank=True,
        default="",
        help_text="Admin-provided locked OpenAPI spec content (for server_type=openapi)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Predefined MCP Server"
        verbose_name_plural = "Predefined MCP Servers"

    def __str__(self):
        return self.name


class MCPServer(models.Model):
    """User-owned MCP server configuration for attaching external tool providers to sessions"""

    AUTH_TYPE_NONE = "none"
    AUTH_TYPE_BASIC = "basic"
    AUTH_TYPE_BEARER = "bearer"
    AUTH_TYPE_OAUTH = "oauth"
    AUTH_TYPE_CHOICES = [
        ("none", "No Auth"),
        ("basic", "Basic Auth"),
        ("bearer", "Bearer Token"),
        ("oauth", "OAuth 2.1"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="mcp_servers")
    predefined_server = models.ForeignKey(
        PredefinedMCPServer,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="user_instances",
        help_text="The predefined server template this instance was created from, if any",
    )
    name = models.CharField(max_length=500)
    url = models.CharField(
        max_length=2000, help_text="HTTP(S) or SSE endpoint URL of the MCP server"
    )
    headers = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional additional HTTP headers sent with every request to this server",
    )
    auth_type = models.CharField(
        max_length=20,
        choices=AUTH_TYPE_CHOICES,
        default="none",
        help_text="Authentication method used to connect to this server",
    )
    username = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Username for Basic Auth",
    )
    password = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="Password for Basic Auth",
    )
    token = models.CharField(
        max_length=4000,
        blank=True,
        default="",
        help_text="Bearer token for Bearer Token auth",
    )
    refresh_token = models.CharField(
        max_length=4000,
        blank=True,
        default="",
        help_text="OAuth 2.1 refresh token for automatic token renewal",
    )
    oauth_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="OAuth 2.1 metadata: token_endpoint, client_id, scope, etc. stored at registration time",
    )
    server_type = models.CharField(
        max_length=20,
        default="http",
        choices=MCP_SERVER_TYPE_CHOICES,
        blank=False,
        help_text='"http" = MCP over HTTP; "openapi" = FastMCP wrapping OpenAPI spec',
    )
    openapi_spec_url = models.CharField(
        max_length=2000,
        blank=True,
        default="",
        help_text="Remote URL to OpenAPI JSON/YAML spec (for server_type=openapi)",
    )
    openapi_spec = models.TextField(
        blank=True,
        default="",
        help_text="Uploaded OpenAPI spec content as alternative to openapi_spec_url (for server_type=openapi)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = [("user", "name")]

    def __str__(self):
        return f"{self.name} ({self.user.username})"


class TextAttachedFile(models.Model):
    """Extracted text content from an AttachedFile binary."""

    attached_file = models.OneToOneField(
        "AttachedFile",
        on_delete=models.CASCADE,
        related_name="text_attached_file",
    )
    content = models.TextField()
    summary = models.TextField(default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"TextAttachedFile({self.attached_file.name})"


class AttachedFile(models.Model):
    """File attached to a session or knowledge base"""

    FILE_TYPE_INPUT = "input"
    FILE_TYPE_OUTPUT = "output"
    FILE_TYPE_CHOICES = [
        (FILE_TYPE_INPUT, "Input"),
        (FILE_TYPE_OUTPUT, "Output"),
    ]

    session = models.ForeignKey(
        "Session",
        on_delete=models.CASCADE,
        related_name="attached_files",
        null=True,
        blank=True,
    )
    knowledge_base = models.ForeignKey(
        KnowledgeBase,
        on_delete=models.CASCADE,
        related_name="kb_files",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=500)
    binary_content = models.BinaryField(null=True, blank=True)
    file_type = models.CharField(
        max_length=10,
        choices=FILE_TYPE_CHOICES,
        default=FILE_TYPE_INPUT,
        null=False,
        blank=False,
    )
    is_hidden = models.BooleanField(
        default=False,
        help_text="Hidden files are persisted but are not shown to the user.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                        models.Q(session__isnull=False, knowledge_base__isnull=True)
                        | models.Q(session__isnull=True, knowledge_base__isnull=False)
                ),
                name="attached_file_exactly_one_parent",
            ),
            models.UniqueConstraint(
                fields=["session", "name"],
                condition=models.Q(session__isnull=False),
                name="attached_file_unique_session_name",
            ),
            models.UniqueConstraint(
                fields=["knowledge_base", "name"],
                condition=models.Q(knowledge_base__isnull=False),
                name="attached_file_unique_kb_name",
            ),
        ]

    def clean(self):
        if not self.session_id and not self.knowledge_base_id:
            raise ValidationError(
                "AttachedFile must belong to a session or a knowledge base."
            )
        if self.session_id and self.knowledge_base_id:
            raise ValidationError(
                "AttachedFile cannot belong to both a session and a knowledge base."
            )

    def __str__(self):
        if self.session_id:
            return f"{self.name} - session:{self.session_id}"
        return f"{self.name} - kb:{self.knowledge_base_id}"


class Session(models.Model):
    """
    Base session model. All library models that need a session FK reference this class.

    For extension: apps should create a concrete Session model that inherits from this
    class via MTI. The ``session_type`` field is a CharField that identifies the
    session kind — each app can add its own session type values.
    """

    id = models.CharField(max_length=100, primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="sessions")
    include_user_context = models.BooleanField(default=False)
    is_research_needed = models.BooleanField(default=False)
    agent_result = models.TextField(blank=True)
    prev_agent_result = models.TextField(blank=True)
    is_research_online = models.BooleanField(
        default=False,
        help_text="When True, the research agent has access to internet search tools. When False with KBs attached, research uses only KB files.",
    )
    is_agent_finished = models.BooleanField(default=False)
    session_type = models.CharField(
        max_length=20
    )
    is_public = models.BooleanField(default=False)
    is_deleted = models.BooleanField(
        default=False,
        help_text="Logical deletion flag. Deleted sessions are hidden from users but retained in the database.",
    )
    knowledge_bases = models.ManyToManyField(
        KnowledgeBase,
        blank=True,
        related_name="sessions",
    )
    mcp_servers = models.ManyToManyField(
        MCPServer,
        blank=True,
        related_name="sessions",
    )
    attached_sessions = models.ManyToManyField(
        "self",
        symmetrical=False,
        related_name="attached_to_sessions",
        blank=True,
    )
    desktop_tunnel_servers = models.JSONField(
        default=dict,
        blank=True,
        help_text="Server IDs per tunnel: {tunnel_id: [server_id_1, ...]}.",
    )
    session_title = models.TextField()
    additional_fields = models.JSONField(default=dict, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["user", "-created_at"], name="session_user_created_idx"
            ),
            models.Index(
                fields=["user", "-updated_at"], name="session_user_updated_idx"
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(session_title=""),
                name="session_title_non_empty",
            ),
        ]

    def clean(self):
        if not self.id:
            raise ValidationError({"id": "Session ID cannot be empty."})

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.id} - {self.user.username} - [{self.session_type}] {self.session_title}"


class WorkerTask(models.Model):
    """Tracks the state of an asynchronous background worker running for a session."""

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_FATAL_FAILURE = "fatal_failure"

    MAX_RETRIES = 3

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_FATAL_FAILURE, "Fatal Failure"),
    ]

    session = models.OneToOneField(
        Session, on_delete=models.CASCADE, related_name="worker_task"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING
    )
    progress_step = models.TextField(blank=True)
    thinking = models.TextField(blank=True)
    last_tool_call = models.JSONField(default=dict, blank=True)
    step_tasks = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)
    resume_state = models.JSONField(null=True, blank=True)
    alive_beat = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    retry_count = models.IntegerField(default=0)
    executor_id = models.UUIDField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["session", "status"], name="workertask_session_status_idx"
            ),
            models.Index(
                fields=["status", "alive_beat"], name="workertask_status_beat_idx"
            ),
            models.Index(
                fields=["status", "started_at"], name="workertask_status_started_idx"
            ),
        ]

    @property
    def is_fatal(self):
        return self.status == self.STATUS_FATAL_FAILURE

    def __str__(self):
        return f"WorkerTask for {self.session.id} ({self.status})"


class SiteConfiguration(models.Model):
    """
    Singleton model for site-wide configuration managed via the Django admin.

    Only one record (pk=1) should exist.  Use ``get_solo()`` to fetch it.
    """

    EMAIL_BACKEND_CHOICES = [
        ("smtp", "SMTP"),
        ("console", "Console (stdout – development only)"),
        ("dummy", "Dummy (discard all emails)"),
    ]

    csrf_trusted_origins = models.TextField(
        blank=True,
        default="",
        help_text=(
            "Comma-separated list of origins that are allowed to access the "
            "Django admin and API (e.g. https://xxx.yyy.com). "
            "Set to * to allow all origins."
        ),
    )

    # ---------------------------------------------------------------------------
    # Email settings
    # ---------------------------------------------------------------------------

    email_backend = models.CharField(
        max_length=20,
        choices=EMAIL_BACKEND_CHOICES,
        default="console",
        help_text="Email delivery method.",
    )
    email_host = models.CharField(
        max_length=500,
        blank=True,
        default="localhost",
        help_text="SMTP server hostname (e.g. smtp.gmail.com).",
    )
    email_port = models.PositiveIntegerField(
        default=587,
        help_text="SMTP server port (typically 587 for TLS, 465 for SSL, 25 for plain).",
    )
    email_host_user = models.CharField(
        max_length=500,
        blank=True,
        help_text="Username / email address used to authenticate with the SMTP server.",
    )
    email_host_password = models.CharField(
        max_length=500,
        blank=True,
        help_text="Password or app-specific password for the SMTP account.",
    )
    email_use_tls = models.BooleanField(
        default=True,
        help_text="Use STARTTLS when connecting to the SMTP server (recommended for port 587).",
    )
    email_use_ssl = models.BooleanField(
        default=False,
        help_text="Use implicit SSL/TLS when connecting to the SMTP server (for port 465). "
                  "Mutually exclusive with TLS.",
    )
    default_from_email = models.CharField(
        max_length=500,
        blank=True,
        default="noreply@app.com",
        help_text='Default "From" address used when sending emails.',
    )
    site_url = models.CharField(
        max_length=500,
        blank=True,
        default="http://localhost:3000",
        help_text="Base URL of the frontend application (e.g. https://app.example.com).",
    )

    # ---------------------------------------------------------------------------
    # Authentication / Social login
    # ---------------------------------------------------------------------------

    google_client_id = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text=(
            "Google OAuth2 client ID (e.g. xxxx.apps.googleusercontent.com). "
            "Create credentials at https://console.cloud.google.com/apis/credentials. "
            "Leave blank to disable Google Sign-In."
        ),
    )

    google_client_secret = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text=(
            "Google OAuth2 client secret. "
            "Required for the redirect-based OAuth flow. "
            "Found in Google Cloud Console alongside the client ID."
        ),
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Site Configuration"
        verbose_name_plural = "Site Configuration"

    def __str__(self):
        return "Site Configuration"

    memory_generation_enabled = models.BooleanField(
        default=False,
        help_text=(
            "When enabled, the daily background job will extract memories from "
            "user sessions and store them as MemoryEntry records."
        ),
    )

    mem_worker_start = models.DateTimeField(
        null=True,
        blank=True,
        help_text=(
            "Timestamp when the memory generation job last started. "
            "Workers skip the job if this is set within the last hour (another worker is active)."
        ),
    )

    memory_last_run = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp of the last successful memory generation job run.",
    )

    memory_max_entries = models.IntegerField(
        default=50,
        help_text=(
            "Maximum number of memory entries per user before compression is triggered."
        ),
    )

    # ---------------------------------------------------------------------------
    # Cold storage settings
    # ---------------------------------------------------------------------------

    COLDSTORAGE_TYPE_NONE = "none"
    COLDSTORAGE_TYPE_LOCAL = "local"
    COLDSTORAGE_TYPE_S3 = "s3"
    COLDSTORAGE_TYPE_CHOICES = [
        (COLDSTORAGE_TYPE_NONE, "None"),
        (COLDSTORAGE_TYPE_LOCAL, "Local filesystem"),
        (COLDSTORAGE_TYPE_S3, "AWS S3"),
    ]

    coldstorage_type = models.CharField(
        max_length=20,
        choices=COLDSTORAGE_TYPE_CHOICES,
        default=COLDSTORAGE_TYPE_NONE,
        help_text=(
            "Where to store cold storage information."
            "'None' means data is ignored. "
            "'Local' writes to a directory on disk. "
            "'S3' uploads to an AWS S3 bucket."
        ),
    )

    coldstorage_offline = models.BooleanField(
        default=False,
        help_text=(
            "When enabled, cold storage access is disabled and raises an error. "
            "Use this to temporarily take cold storage offline for maintenance."
        ),
    )

    coldstorage_local_storage_location = models.CharField(
        max_length=1000,
        blank=True,
        default="",
        help_text=(
            "Absolute path to the local directory used for cold storage "
            "(only relevant when coldstorage_type is 'local'). "
            "Defaults to /tmp/cold_storage/ when blank."
        ),
    )

    coldstorage_s3_bucket_name = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="S3 bucket name for cold storage (only relevant when coldstorage_type is 's3').",
    )

    coldstorage_aws_access_key = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text=(
            "AWS access key ID for cold storage S3 access. "
            "Leave blank to use the default credential chain (IAM role, env vars, etc.)."
        ),
    )

    coldstorage_aws_secret_key = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="AWS secret access key for cold storage S3 access.",
    )

    coldstorage_aws_region = models.CharField(
        max_length=50,
        blank=True,
        default="us-east-1",
        help_text="AWS region for the cold storage S3 bucket (e.g. us-east-1).",
    )

    coldstorage_aws_endpoint = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text=(
            "S3-compatible endpoint URL "
            "(e.g. https://objectstorage.us-ashburn-1.oraclecloud.com for OCI support). "
            "Leave blank to use the default endpoint for the selected region."
        ),
    )

    # ---------------------------------------------------------------------------
    # Offload worker fields
    # ---------------------------------------------------------------------------
    offload_worker_start = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when the offload worker last claimed the daily offload job.",
    )
    offload_last_run = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when the offload job last completed successfully.",
    )

    # ---------------------------------------------------------------------------
    # Session purge settings
    # ---------------------------------------------------------------------------

    purge_session_cutoff_days = models.PositiveIntegerField(
        default=0,
        help_text=(
            "Number of days after which soft-deleted sessions are permanently purged. "
            "Set to 0 (default) to disable automatic purge."
        ),
    )

    @classmethod
    def get_solo(cls):
        """Return the singleton configuration record, creating it if absent."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


def _default_audio_voices():
    return ["eve", "ara", "rex", "sal", "leo"]


class LLMConfiguration(models.Model):
    """
    Singleton model storing all LLM provider configuration.

    Only one record (pk=1) should exist.  Use ``get_solo()`` to fetch it.
    All fields are stored in plain text; sensitive values such as API keys
    are intentionally kept editable via the Django admin interface.
    """

    LLM_PROVIDER_CHOICES = [
        ("not_configured", "Not Configured"),
        ("google", "Google Gemini"),
        ("openai", "OpenAI"),
        ("minimax", "MiniMax"),
        ("anthropic", "Anthropic"),
        ("azure", "Azure OpenAI"),
        ("aws", "AWS Bedrock"),
        ("ollama", "Ollama (local)"),
        ("huggingface", "HuggingFace"),
        ("deepseek", "DeepSeek"),
        ("openrouter", "OpenRouter"),
    ]

    # General
    provider = models.CharField(
        max_length=50, default="not_configured", choices=LLM_PROVIDER_CHOICES
    )
    fast_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for fast tasks. Leave blank to use the same provider as the smart model.",
    )
    reasoning_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for reasoning/research tasks. Leave blank to use the same provider as the smart model.",
    )
    ultra_fast_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for ultra-fast tasks. Leave blank to use the same provider as the smart model.",
    )
    ultra_smart_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for ultra-smart tasks. Leave blank to use the same provider as the smart model.",
    )
    image_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for image generation tasks. Leave blank to use the same provider as the smart model.",
    )
    video_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for video generation tasks. Leave blank to use the same provider as the smart model.",
    )
    audio_provider = models.CharField(
        max_length=50,
        blank=True,
        default="",
        choices=[("", "— same as main provider —")] + LLM_PROVIDER_CHOICES,
        help_text="Provider for audio generation. Leave blank to use the same provider as the smart model.",
    )

    # Google / Gemini
    google_api_key = models.CharField(
        max_length=500,
        blank=True,
        help_text="Google Gemini / Google AI API key (GOOGLE_API_KEY or GEMINI_API_KEY)",
    )
    google_model = models.CharField(
        max_length=200,
        blank=True,
        default="gemini-2.5-flash",
        help_text="Smart model for Google (default: gemini-2.5-flash).",
    )
    google_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="gemini-2.0-flash",
        help_text="Fast model for Google (default: gemini-2.0-flash).",
    )
    google_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="gemini-2.5-pro-preview-03-25",
        help_text="Reasoning model for Google (default: gemini-2.5-pro-preview-03-25).",
    )
    google_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for Google.",
    )
    google_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for Google.",
    )
    google_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for Google.",
    )
    google_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for Google.",
    )
    google_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for Google.",
    )

    # OpenAI
    openai_api_key = models.CharField(max_length=500, blank=True)
    openai_base_url = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Custom base URL for OpenAI (e.g. https://api.minimax.chat/v1). Leave blank to use the default OpenAI endpoint.",
    )
    openai_model = models.CharField(
        max_length=200,
        blank=True,
        default="gpt-4o",
        help_text="Smart model for OpenAI (default: gpt-4o).",
    )
    openai_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="gpt-4o-mini",
        help_text="Fast model for OpenAI (default: gpt-4o-mini).",
    )
    openai_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="o1",
        help_text="Reasoning model for OpenAI (default: o1).",
    )
    openai_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for OpenAI.",
    )
    openai_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for OpenAI.",
    )
    openai_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for OpenAI.",
    )
    openai_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for OpenAI.",
    )
    openai_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="x-ai/grok-voice-tts-1.0",
        help_text="Audio/TTS model for OpenAI (default: x-ai/grok-voice-tts-1.0).",
    )

    # MiniMax
    minimax_api_key = models.CharField(max_length=500, blank=True)
    minimax_model = models.CharField(
        max_length=200,
        blank=True,
        default="MiniMax-M2.7",
        help_text="Smart model for MiniMax (default: MiniMax-M2.7).",
    )
    minimax_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="MiniMax-M2.7",
        help_text="Fast model for MiniMax (default: MiniMax-M2.7).",
    )
    minimax_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="MiniMax-M2.7",
        help_text="Reasoning model for MiniMax (default: MiniMax-M2.7).",
    )
    minimax_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for MiniMax.",
    )
    minimax_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for MiniMax.",
    )
    minimax_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="image-01",
        help_text="Image model for MiniMax (default: image-01).",
    )
    minimax_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="MiniMax-Hailuo-2.3",
        help_text="Video model for MiniMax (default: MiniMax-Hailuo-2.3).",
    )
    minimax_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="speech-2.8-hd",
        help_text="Audio/TTS model for MiniMax (default: speech-2.8-hd).",
    )

    # Anthropic
    anthropic_api_key = models.CharField(max_length=500, blank=True)
    anthropic_base_url = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Custom base URL for Anthropic (e.g. https://api.minimax.chat/v1). Leave blank to use the default Anthropic endpoint.",
    )
    anthropic_model = models.CharField(
        max_length=200,
        blank=True,
        default="claude-sonnet-4-6",
        help_text="Smart model for Anthropic (default: claude-sonnet-4-6).",
    )
    anthropic_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="claude-haiku-4-5-20251001",
        help_text="Fast model for Anthropic (default: claude-haiku-4-5-20251001).",
    )
    anthropic_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="claude-opus-4-6",
        help_text="Reasoning model for Anthropic (default: claude-opus-4-6).",
    )
    anthropic_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for Anthropic.",
    )
    anthropic_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for Anthropic.",
    )
    anthropic_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for Anthropic.",
    )
    anthropic_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for Anthropic.",
    )
    anthropic_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for Anthropic.",
    )

    # Azure OpenAI
    azure_api_key = models.CharField(max_length=500, blank=True)
    azure_deployment = models.CharField(
        max_length=200, blank=True, help_text="Smart deployment name for Azure OpenAI."
    )
    azure_fast_deployment = models.CharField(
        max_length=200,
        blank=True,
        help_text="Fast deployment name for Azure OpenAI. Falls back to the smart deployment when blank.",
    )
    azure_reasoning_deployment = models.CharField(
        max_length=200,
        blank=True,
        help_text="Reasoning deployment name for Azure OpenAI. Falls back to the smart deployment when blank.",
    )
    azure_ultra_fast_deployment = models.CharField(
        max_length=200,
        blank=True,
        help_text="Ultra-fast deployment name for Azure OpenAI. Falls back to the smart deployment when blank.",
    )
    azure_ultra_smart_deployment = models.CharField(
        max_length=200,
        blank=True,
        help_text="Ultra-smart deployment name for Azure OpenAI. Falls back to the smart deployment when blank.",
    )
    azure_image_deployment = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image deployment name for Azure OpenAI.",
    )
    azure_video_deployment = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video deployment name for Azure OpenAI.",
    )
    azure_audio_deployment = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio deployment name for Azure OpenAI.",
    )
    azure_endpoint = models.CharField(max_length=500, blank=True)

    # AWS Bedrock
    aws_access_key_id = models.CharField(max_length=200, blank=True)
    aws_secret_access_key = models.CharField(max_length=500, blank=True)
    aws_session_token = models.CharField(max_length=500, blank=True)
    aws_region = models.CharField(max_length=50, blank=True, default="us-east-1")
    aws_profile_name = models.CharField(max_length=200, blank=True)
    aws_model = models.CharField(
        max_length=200,
        blank=True,
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help_text="Smart model for AWS Bedrock (default: anthropic.claude-3-5-sonnet-20241022-v2:0).",
    )
    aws_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help_text="Fast model for AWS Bedrock (default: anthropic.claude-3-haiku-20240307-v1:0).",
    )
    aws_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help_text="Reasoning model for AWS Bedrock (default: anthropic.claude-3-5-sonnet-20241022-v2:0).",
    )
    aws_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for AWS Bedrock.",
    )
    aws_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for AWS Bedrock.",
    )
    aws_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for AWS Bedrock.",
    )
    aws_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for AWS Bedrock.",
    )
    aws_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for AWS Bedrock.",
    )

    # Ollama
    ollama_base_url = models.CharField(
        max_length=500, blank=True, default="http://localhost:11434"
    )
    ollama_model = models.CharField(
        max_length=200,
        blank=True,
        default="llama3.2",
        help_text="Smart model for Ollama (default: llama3.2).",
    )
    ollama_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="llama3.2",
        help_text="Fast model for Ollama (default: llama3.2).",
    )
    ollama_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="llama3.2",
        help_text="Reasoning model for Ollama (default: llama3.2).",
    )
    ollama_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for Ollama.",
    )
    ollama_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for Ollama.",
    )
    ollama_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for Ollama.",
    )
    ollama_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for Ollama.",
    )
    ollama_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for Ollama.",
    )

    # HuggingFace
    huggingface_api_key = models.CharField(max_length=500, blank=True)
    huggingface_model = models.CharField(
        max_length=200,
        blank=True,
        default="microsoft/Phi-3-mini-4k-instruct",
        help_text="Smart model for HuggingFace (default: microsoft/Phi-3-mini-4k-instruct).",
    )
    huggingface_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="microsoft/Phi-3-mini-4k-instruct",
        help_text="Fast model for HuggingFace (default: microsoft/Phi-3-mini-4k-instruct).",
    )
    huggingface_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="microsoft/Phi-3-mini-4k-instruct",
        help_text="Reasoning model for HuggingFace (default: microsoft/Phi-3-mini-4k-instruct).",
    )
    huggingface_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for HuggingFace.",
    )
    huggingface_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for HuggingFace.",
    )
    huggingface_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for HuggingFace.",
    )
    huggingface_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for HuggingFace.",
    )
    huggingface_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for HuggingFace.",
    )

    # DeepSeek
    deepseek_api_key = models.CharField(max_length=500, blank=True)
    deepseek_model = models.CharField(
        max_length=200,
        blank=True,
        default="deepseek-chat",
        help_text="Smart model for DeepSeek (default: deepseek-chat).",
    )
    deepseek_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="deepseek-chat",
        help_text="Fast model for DeepSeek (default: deepseek-chat).",
    )
    deepseek_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="deepseek-reasoner",
        help_text="Reasoning model for DeepSeek (default: deepseek-reasoner).",
    )
    deepseek_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for DeepSeek.",
    )
    deepseek_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for DeepSeek.",
    )
    deepseek_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for DeepSeek.",
    )
    deepseek_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for DeepSeek.",
    )
    deepseek_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Audio/TTS model for DeepSeek.",
    )

    # OpenRouter
    openrouter_api_key = models.CharField(max_length=500, blank=True)
    openrouter_model = models.CharField(
        max_length=200,
        blank=True,
        default="x-ai/grok-4.1-fast",
        help_text="Smart model for OpenRouter (default: x-ai/grok-4.1-fast).",
    )
    openrouter_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="deepseek/deepseek-chat",
        help_text="Fast model for OpenRouter (default: deepseek/deepseek-chat).",
    )
    openrouter_reasoning_model = models.CharField(
        max_length=200,
        blank=True,
        default="xiaomi/mimo-v2-pro",
        help_text="Reasoning model for OpenRouter (default: xiaomi/mimo-v2-pro).",
    )
    openrouter_ultra_fast_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-fast model for OpenRouter.",
    )
    openrouter_ultra_smart_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Ultra-smart model for OpenRouter.",
    )
    openrouter_image_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Image model for OpenRouter.",
    )
    openrouter_video_model = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Video model for OpenRouter.",
    )
    openrouter_audio_model = models.CharField(
        max_length=200,
        blank=True,
        default="x-ai/grok-voice-tts-1.0",
        help_text="Audio/TTS model for OpenRouter (default: x-ai/grok-voice-tts-1.0).",
    )
    audio_voices = models.JSONField(
        default=_default_audio_voices,
        blank=True,
        help_text='Available voice IDs for TTS (e.g. ["eve", "ara", "rex", "sal", "leo"] for OpenAI x-ai/grok-voice-tts-1.0). LLM picks from this list per video.',
    )
    OPENROUTER_REASONING_EFFORT_CHOICES = [
        ("", "— disabled —"),
        ("minimal", "Minimal"),
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("xhigh", "Extra High"),
    ]
    openrouter_reasoning_effort = models.CharField(
        max_length=10,
        blank=True,
        default="medium",
        choices=OPENROUTER_REASONING_EFFORT_CHOICES,
        help_text=(
            "Reasoning effort for the OpenRouter reasoning model (llm_type=reasoning). "
            "Leave blank to disable reasoning."
        ),
    )

    # Tavily (web search tool)
    tavily_api_key = models.CharField(max_length=500, blank=True)

    # Prompt compression
    compress_prompts = models.BooleanField(
        default=False,
        help_text="When enabled, long prompts are compressed using LLMLingua before being sent to the LLM.",
    )
    compress_prompts_model = models.CharField(
        max_length=500,
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        help_text="HuggingFace model used for LLMLingua prompt compression. Changes require a server restart.",
    )
    compress_prompts_target_token_rate = models.FloatField(
        default=0.6,
        help_text="Target ratio of tokens to retain after compression (0.0–1.0).",
    )
    compress_prompts_min_length = models.PositiveIntegerField(
        default=1000,
        help_text="Minimum character length a prompt must exceed before compression is attempted.",
    )
    compress_prompts_force_tokens = models.JSONField(
        null=True,
        help_text='Tokens that must always be preserved during compression (JSON list, e.g. [".", "\\n"]).',
    )
    compress_prompts_chunk_end_tokens = models.JSONField(
        null=True,
        help_text='Tokens treated as chunk boundaries during compression (JSON list, e.g. [".", "\\n"]).',
    )
    compress_prompts_pool_size = models.PositiveIntegerField(
        default=5,
        help_text="Number of compressor instances to keep in the worker pool. Changes require a server restart.",
    )
    compress_prompts_cache_size = models.PositiveIntegerField(
        default=256,
        help_text="Maximum number of prompt/compressed-prompt pairs to cache in memory. Changes require a server restart.",
    )
    compress_prompts_use_llm = models.BooleanField(
        default=False,
        help_text=(
            "When enabled, use the configured LLM to compress prompts instead of LLMLingua. "
            "The LLM compression uses the prompt template below and is also cached."
        ),
    )
    compress_prompts_llm_prompt = models.TextField(
        default=(
            "Compress the following text to reduce token count while preserving all meaning, "
            "key facts, numbers, dates, and named entities verbatim. "
            "Return only the compressed text without any explanation.\n\nTEXT:\n{text}"
        ),
        help_text=(
            "Prompt template used when LLM-based compression is enabled. "
            "Must contain a {text} placeholder for the text to compress."
        ),
    )

    # Debugging
    llm_debug = models.BooleanField(
        default=False,
        help_text="When enabled, every LLM input and output is recorded in the LLM Debug Log for troubleshooting.",
    )

    # Advanced
    enable_subagents = models.BooleanField(
        default=True,
        help_text="When disabled, the subagent tool is not available to deep agents.",
    )
    enable_playwright = models.BooleanField(
        default=True,
        help_text="When disabled, the Playwright browser automation tool is not available.",
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "LLM Configuration"
        verbose_name_plural = "LLM Configuration"

    def __str__(self):
        return f"LLM Configuration ({self.provider})"

    @classmethod
    def get_solo(cls):
        """Return the singleton configuration record, creating it if absent."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class LLMDebugLog(models.Model):
    """
    Records every LLM call.  Metadata (operation, provider, model, token counts,
    agent flags) is always captured.  Raw payload fields (input_messages,
    raw_output, agent_history, agent_tasks) are only populated when
    ``LLMConfiguration.llm_debug`` is enabled or the call failed.

    Entries are written by ``invoke_llm`` and are read-only in the Django admin
    interface.
    """

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    operation_name = models.CharField(
        max_length=200,
        blank=True,
        db_index=True,
        help_text="Logical operation identifier passed to invoke_llm (e.g. my_llm_op, synthesis).",
    )
    call_label = models.CharField(
        max_length=500,
        blank=True,
        help_text='Template name or "(custom messages)" identifying the call.',
    )
    provider = models.CharField(
        max_length=100,
        blank=True,
        db_index=True,
        help_text="LLM provider used for this call (e.g. openai, anthropic, google).",
    )
    model = models.CharField(
        max_length=200,
        blank=True,
        help_text="Model name / deployment used for this call.",
    )
    is_agent = models.BooleanField(
        default=False,
        db_index=True,
        help_text="True when invoke_llm was called with is_agent=True.",
    )
    is_deep_agent = models.BooleanField(
        default=False,
        help_text="True when invoke_llm was called with is_deep_agent=True.",
    )
    input_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Number of input/prompt tokens consumed (None when unavailable).",
    )
    output_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Number of output/completion tokens generated (None when unavailable).",
    )
    input_messages = models.TextField(
        blank=True,
        help_text=(
            "Pretty-printed JSON list of LangChain messages sent to the LLM. "
            "Only populated when llm_debug is enabled or the call failed."
        ),
    )
    raw_output = models.TextField(
        blank=True,
        help_text=(
            "Raw text returned by the LLM before JSON parsing. "
            "Only populated when llm_debug is enabled or the call failed."
        ),
    )
    error = models.TextField(
        blank=True,
        help_text="Exception message if the call failed, empty otherwise.",
    )
    agent_history = models.TextField(
        blank=True,
        help_text=(
            "Pretty-printed JSON list of ordered agent events (thoughts and tool runs). "
            'Each entry has a "type" key ("thought" or "tool_run") plus type-specific fields: '
            'thoughts have "content"; tool runs have "tool", "input", and "output". '
            "Only populated when llm_debug is enabled or the call failed."
        ),
    )
    agent_tasks = models.TextField(
        blank=True,
        help_text=(
            "Pretty-printed JSON list of todos/tasks from the last write_todos call by the agent. "
            "Only populated when llm_debug is enabled or the call failed."
        ),
    )

    class Meta:
        verbose_name = "LLM Debug Log"
        verbose_name_plural = "LLM Debug Logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"LLMDebugLog({self.pk}) [{self.created_at}] {self.operation_name or self.call_label}"


class MemoryEntry(models.Model):
    """A single memory item extracted from user sessions by the memory generation job."""

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="memory_entries"
    )
    content = models.TextField(help_text="The extracted memory text.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Memory Entry"
        verbose_name_plural = "Memory Entries"
        ordering = ["-created_at"]

    def __str__(self):
        return f"MemoryEntry(user={self.user_id}, created={self.created_at:%Y-%m-%d}): {self.content[:60]}"


class SessionSnapshot(models.Model):
    """A point-in-time snapshot of a session's state and attached files.

    Snapshots are created automatically after every successful execution of
    a session.  They capture the full
    session state (text fields) and all attached files (input, output, and
    hidden) so the session can be restored to any previous version.

    The actual file data is stored in ColdStorage as a ZIP archive under the
    key ``session_snapshots/{session_id}/{snapshot_id}.zip``.  When cold
    storage is disabled (``coldstorage_type=none``) snapshots are created as
    metadata-only records and the ZIP is not persisted — restoration will
    restore session field values but cannot restore attached files.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name="snapshots",
    )
    label = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Optional human-readable label for the snapshot.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["session", "-created_at"],
                name="snapshot_session_created_idx",
            ),
        ]

    def __str__(self):
        return f"Snapshot {self.id} for session {self.session_id} at {self.created_at}"


class LLMOperationConfig(models.Model):
    """
    Per-operation LLM override.  When a record exists for an operation name,
    its fields take precedence over the global LLMConfiguration defaults.
    """

    operation_name = models.CharField(
        max_length=100,
        primary_key=True,
        help_text="Identifier for this operation (e.g. my_llm_op, synthesis).",
    )
    description = models.TextField(blank=True, default="")

    override_provider = models.CharField(
        max_length=50,
        blank=True,
        choices=[("", "— system default —")] + LLMConfiguration.LLM_PROVIDER_CHOICES,
    )

    selected_llm_type = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ("", "— use override_model instead —"),
            ("ultra-fast", "Ultra Fast"),
            ("fast", "Fast"),
            ("smart", "Smart"),
            ("ultra-smart", "Ultra Smart"),
            ("reasoning", "Reasoning"),
            ("image", "Image"),
            ("video", "Video"),
            ("audio", "Audio"),
        ],
        help_text="LLM tier. Required unless override_model is set.",
    )
    override_model = models.CharField(
        max_length=200,
        blank=True,
        help_text=(
            "Exact model name (e.g. gpt-4o-mini). "
            "When set, selected_llm_type must be blank and override_provider must be set."
        ),
    )

    is_enabled = models.BooleanField(
        default=True,
        help_text="Disable to fall back to system defaults for this operation.",
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "LLM Operation Config"
        verbose_name_plural = "LLM Operation Configs"
        ordering = ["operation_name"]

    def __str__(self):
        return self.operation_name

    def clean(self):
        if self.override_model and self.selected_llm_type:
            raise ValidationError(
                "override_model and selected_llm_type are mutually exclusive. "
                "Set only one."
            )
        if self.override_model and not self.override_provider:
            raise ValidationError(
                "override_provider must be set when override_model is specified."
            )

# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------


class Notification(models.Model):
    """User-facing alert from the  system."""

    STATUS_CHOICES = [
        ("sent", "Sent"),
        ("read", "Read"),
        ("dismissed", "Dismissed"),
        ("expired", "Expired"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="notifications"
    )
    type = models.CharField(max_length=30)
    title = models.CharField(max_length=200)
    body = models.TextField(max_length=500, blank=True, default="")
    cta_action = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Navigation action for CTA button.",
    )
    cta_params = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="sent",
    )
    read_at = models.DateTimeField(null=True, blank=True)
    dismissed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "status"], name="notif_user_status_idx"),
            models.Index(
                fields=["user", "-created_at"], name="notif_user_created_idx"
            ),
        ]

    def __str__(self):
        return f"[{self.type}] {self.title}"
