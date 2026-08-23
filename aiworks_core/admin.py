import csv
import datetime
import json
import logging
import mimetypes
from urllib.parse import quote

from django.contrib import admin
from django.contrib.admin.sites import site
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.http import HttpResponse, Http404
from django.urls import path
from django.utils import timezone
from django.utils.html import format_html, format_html_join, escape
from django.utils.safestring import mark_safe

from .models import (
    User,
    AttachedFile,
    TextAttachedFile,
    LLMConfiguration,
    LLMOperationConfig,
    SiteConfiguration,
    WorkerTask,
    LLMDebugLog,
    KnowledgeBase,
    VerificationToken,
    MCPServer,
    PredefinedMCPServer,
    MemoryEntry,
    Tunnel,
    SessionSnapshot,
    Notification
)

logger = logging.getLogger(__name__)


# noinspection PyDeprecation
def _format_tasks_view(tasks) -> str:
    if not tasks:
        return "—"
    status_icons = {"pending": "○", "in_progress": "⏳", "completed": "✓"}
    rows = format_html_join(
        "",
        '<li style="margin-bottom:4px">'
        '<span style="font-family:monospace;margin-right:6px">{} [{}]</span>{}'
        "</li>",
        (
            (
                status_icons.get(t.get("status", ""), "?"),
                t.get("status", "?"),
                t.get("content", ""),
            )
            for t in tasks
        ),
    )
    return format_html('<ul style="margin:0;padding-left:18px">{}</ul>', rows)


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = (
        "id",
        "username",
        "provider",
        "email",
        "first_name",
        "last_name",
        "email_verified",
        "tier",
        "date_joined",
        "last_login",
    )
    list_filter = ("date_joined", "last_login", "provider", "tier")
    fieldsets = [
        *BaseUserAdmin.fieldsets,
        (
            "Additional Info",
            {"fields": ("provider", "email_verified", "avatar", "has_seen_onboarding")},
        ),
        (
            "Subscription",
            {
                "fields": (
                    "tier",
                )
            },
        ),
        ("Profile", {"fields": ("profile_context", "llm_generated_background", "extra_data")}),
        ("UI Preferences", {"fields": ("light_mode",)}),
    ]


@admin.register(VerificationToken)
class VerificationTokenAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "token_type",
        "expires_at",
        "is_valid_display",
        "created_at",
    )
    list_filter = ("token_type", "created_at")
    search_fields = ("user__email", "user__username", "token_type")
    readonly_fields = ("user", "token", "token_type", "expires_at", "created_at")

    def is_valid_display(self, obj):
        return obj.is_valid()

    is_valid_display.boolean = True
    is_valid_display.short_description = "Valid?"

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "name", "files_count", "created_at", "updated_at")
    list_filter = ("created_at", "updated_at")
    search_fields = ("name", "user__email", "user__username")
    readonly_fields = ("created_at", "updated_at")

    def files_count(self, obj):
        return obj.kb_files.count()

    files_count.short_description = "Files"


@admin.register(MCPServer)
class MCPServerAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "name", "url", "predefined_server", "created_at", "updated_at")
    list_filter = ("created_at", "updated_at", "predefined_server")
    search_fields = ("name", "url", "user__email", "user__username")
    readonly_fields = ("created_at", "updated_at")
    actions = ["enumerate_mcp_servers", "execute_mcp_tool"]

    def change_view(self, request, object_id, form_url="", extra_context=None):
        extra_context = extra_context or {}
        extra_context["execute_tool_url"] = f"/admin/aiworks_core/mcpserver/{object_id}/execute-tool/"
        return super().change_view(request, object_id, form_url, extra_context)

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "<int:pk>/execute-tool/",
                site.admin_view(self.execute_tool_view),
                name="api_mcpserver_execute_tool",
            ),
        ]
        return custom + urls

    @staticmethod
    def execute_tool_view(request, pk):
        from .admin_helpers.admin_mcp_helper import execute_tool_view
        return execute_tool_view(request, pk)

    @admin.action(description="Enumerate selected MCP servers and tools")
    def enumerate_mcp_servers(self, request, queryset):
        from .admin_helpers.admin_mcp_helper import enumerate_mcp_servers
        return enumerate_mcp_servers(self, request, queryset)


@admin.register(PredefinedMCPServer)
class PredefinedMCPServerAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "url", "auth_type", "server_type", "created_at", "updated_at")
    list_filter = ("auth_type", "server_type", "created_at", "updated_at")
    search_fields = ("name", "url")
    readonly_fields = ("created_at", "updated_at")


@admin.register(Tunnel)
class TunnelAdmin(admin.ModelAdmin):
    list_display = ("tunnel_id", "user", "alive_beat", "is_connected", "created_at")
    list_filter = ("created_at", "alive_beat")
    search_fields = ("tunnel_id", "user__email", "user__username")
    readonly_fields = ("created_at", "alive_beat", "is_connected")
    actions = ["enumerate_tunnel_servers", "execute_tunnel_tool"]

    def change_view(self, request, object_id, form_url="", extra_context=None):
        extra_context = extra_context or {}
        extra_context["execute_tool_url"] = f"/admin/aiworks_core/tunnel/{object_id}/execute-tool/"
        return super().change_view(request, object_id, form_url, extra_context)

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "<str:pk>/execute-tool/",
                site.admin_view(self.execute_tool_view),
                name="api_tunnel_execute_tool",
            ),
        ]
        return custom + urls

    @staticmethod
    def execute_tool_view(request, pk):
        from .admin_helpers.admin_tunnel_helper import execute_tool_view
        return execute_tool_view(request, pk)

    def enumerate_tunnel_servers(self, request, queryset):
        from .admin_helpers.admin_tunnel_helper import enumerate_tunnel_servers
        return enumerate_tunnel_servers(self, request, queryset)


@admin.register(SessionSnapshot)
class SessionSnapshotAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "label", "created_at")
    list_filter = ("created_at",)
    search_fields = ("session__id", "session__session_title", "label")
    readonly_fields = ("created_at",)
    date_hierarchy = "created_at"

    def has_change_permission(self, request, obj=None):
        return False

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return True


# noinspection PyDeprecation
@admin.register(AttachedFile)
class AttachedFileAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "session",
        "knowledge_base",
        "name",
        "file_type",
        "is_hidden",
        "binary_preview",
        "download_link",
        "created_at",
    )
    list_filter = ("file_type", "is_hidden", "created_at")
    search_fields = ("name",)
    readonly_fields = ("created_at", "download_link")

    def binary_preview(self, obj):
        bc = obj.binary_content
        if bc:
            preview = bc[:100] if len(bc) > 100 else bc
            try:
                text = preview.decode("utf-8", errors="replace")
                if len(bc) > 100:
                    text = text + "..."
            except Exception:
                text = preview.hex()
            return mark_safe('<div dir="auto">' + escape(text[:200]) + '</div>')
        return "(offloaded to cold storage)"

    binary_preview.short_description = "binary_content"

    def download_link(self, obj):
        from .logic.files import get_binary_data

        data = get_binary_data(obj)
        if data is None:
            return "—"
        app_label = obj._meta.app_label
        url = f"/admin/{app_label}/attachedfile/{obj.pk}/download/"
        return format_html('<a href="{}">Download</a>', url)

    download_link.short_description = "File"

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "<int:pk>/download/",
                self.admin_site.admin_view(self.download_view),
                name="api_attachedfile_download",
            ),
        ]
        return custom + urls

    @staticmethod
    def download_view(_request, pk):
        from .logic.files import get_binary_data

        try:
            obj = AttachedFile.objects.get(pk=pk)
        except AttachedFile.DoesNotExist:
            raise Http404
        data = get_binary_data(obj)
        if data is None:
            raise Http404
        mime_type, _ = mimetypes.guess_type(obj.name)
        mime_type = mime_type or "application/octet-stream"
        response = HttpResponse(data, content_type=mime_type)
        response["Content-Disposition"] = "attachment; filename*=UTF-8''{}".format(
            quote(obj.name)
        )
        return response


@admin.register(TextAttachedFile)
class TextAttachedFileAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "attached_file",
        "content_preview",
        "summary_preview",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("content", "summary", "attached_file__name")
    readonly_fields = ("created_at",)

    def content_preview(self, obj):
        content = obj.content[:500] + "..." if len(obj.content) > 500 else obj.content
        return mark_safe(
            '<div dir="auto" style="max-width:600px;overflow-x:auto;">' + escape(content) + '</div>'
        )

    content_preview.short_description = "content"

    def summary_preview(self, obj):
        if not obj.summary:
            return "—"
        return mark_safe(
            '<div dir="auto">' + escape(obj.summary[:200]) + '</div>'
        )

    summary_preview.short_description = "summary"


# noinspection PyDeprecation
@admin.register(WorkerTask)
class WorkerTaskAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "session",
        "status",
        "progress_step",
        "error_message",
        "executor_id",
        "alive_beat",
        "started_at",
        "completed_at",
    )
    list_filter = ("status", "started_at")
    search_fields = ("status", "progress_step", "error_message", "executor_id")
    readonly_fields = (
        "alive_beat",
        "started_at",
        "completed_at",
        "executor_id",
        "step_tasks_display",
    )

    def step_tasks_display(self, obj):
        """Render the step_tasks JSON as a formatted, readable list for troubleshooting."""
        tasks = obj.step_tasks
        return _format_tasks_view(tasks)

    step_tasks_display.short_description = "Step Tasks"


@admin.register(LLMConfiguration)
class LLMConfigurationAdmin(admin.ModelAdmin):
    list_display = (
        "provider",
        "fast_provider",
        "reasoning_provider",
        "ultra_fast_provider",
        "ultra_smart_provider",
        "image_provider",
        "video_provider",
        "audio_provider",
        "updated_at",
    )
    readonly_fields = ("updated_at",)
    fieldsets = (
        (
            "General",
            {
                "fields": (
                    "provider",
                    "fast_provider",
                    "reasoning_provider",
                    "ultra_fast_provider",
                    "ultra_smart_provider",
                    "image_provider",
                    "video_provider",
                    "audio_provider",
                ),
            },
        ),
        (
            "Google Gemini",
            {
                "fields": (
                    "google_api_key",
                    "google_model",
                    "google_fast_model",
                    "google_reasoning_model",
                    "google_ultra_fast_model",
                    "google_ultra_smart_model",
                    "google_image_model",
                    "google_video_model",
                    "google_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "OpenAI",
            {
                "fields": (
                    "openai_api_key",
                    "openai_base_url",
                    "openai_model",
                    "openai_fast_model",
                    "openai_reasoning_model",
                    "openai_ultra_fast_model",
                    "openai_ultra_smart_model",
                    "openai_image_model",
                    "openai_video_model",
                    "openai_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "MiniMax",
            {
                "fields": (
                    "minimax_api_key",
                    "minimax_model",
                    "minimax_fast_model",
                    "minimax_reasoning_model",
                    "minimax_ultra_fast_model",
                    "minimax_ultra_smart_model",
                    "minimax_image_model",
                    "minimax_video_model",
                    "minimax_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Anthropic",
            {
                "fields": (
                    "anthropic_api_key",
                    "anthropic_base_url",
                    "anthropic_model",
                    "anthropic_fast_model",
                    "anthropic_reasoning_model",
                    "anthropic_ultra_fast_model",
                    "anthropic_ultra_smart_model",
                    "anthropic_image_model",
                    "anthropic_video_model",
                    "anthropic_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Azure OpenAI",
            {
                "fields": (
                    "azure_api_key",
                    "azure_deployment",
                    "azure_fast_deployment",
                    "azure_reasoning_deployment",
                    "azure_ultra_fast_deployment",
                    "azure_ultra_smart_deployment",
                    "azure_image_deployment",
                    "azure_video_deployment",
                    "azure_audio_deployment",
                    "azure_endpoint",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "AWS Bedrock",
            {
                "fields": (
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_session_token",
                    "aws_region",
                    "aws_profile_name",
                    "aws_model",
                    "aws_fast_model",
                    "aws_reasoning_model",
                    "aws_ultra_fast_model",
                    "aws_ultra_smart_model",
                    "aws_image_model",
                    "aws_video_model",
                    "aws_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Ollama",
            {
                "fields": (
                    "ollama_base_url",
                    "ollama_model",
                    "ollama_fast_model",
                    "ollama_reasoning_model",
                    "ollama_ultra_fast_model",
                    "ollama_ultra_smart_model",
                    "ollama_image_model",
                    "ollama_video_model",
                    "ollama_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "HuggingFace",
            {
                "fields": (
                    "huggingface_api_key",
                    "huggingface_model",
                    "huggingface_fast_model",
                    "huggingface_reasoning_model",
                    "huggingface_ultra_fast_model",
                    "huggingface_ultra_smart_model",
                    "huggingface_image_model",
                    "huggingface_video_model",
                    "huggingface_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "DeepSeek",
            {
                "fields": (
                    "deepseek_api_key",
                    "deepseek_model",
                    "deepseek_fast_model",
                    "deepseek_reasoning_model",
                    "deepseek_ultra_fast_model",
                    "deepseek_ultra_smart_model",
                    "deepseek_image_model",
                    "deepseek_video_model",
                    "deepseek_audio_model",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "OpenRouter",
            {
                "fields": (
                    "openrouter_api_key",
                    "openrouter_model",
                    "openrouter_fast_model",
                    "openrouter_reasoning_model",
                    "openrouter_ultra_fast_model",
                    "openrouter_ultra_smart_model",
                    "openrouter_image_model",
                    "openrouter_video_model",
                    "openrouter_audio_model",
                    "openrouter_reasoning_effort",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Audio / TTS",
            {
                "fields": (
                    "audio_voices",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Web Search (Tavily)",
            {
                "fields": ("tavily_api_key",),
                "classes": ("collapse",),
            },
        ),
        (
            "Compression",
            {
                "fields": (
                    "compress_prompts",
                    "compress_prompts_model",
                    "compress_prompts_target_token_rate",
                    "compress_prompts_min_length",
                    "compress_prompts_force_tokens",
                    "compress_prompts_chunk_end_tokens",
                    "compress_prompts_pool_size",
                    "compress_prompts_cache_size",
                    "compress_prompts_use_llm",
                    "compress_prompts_llm_prompt",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "LLM Debug",
            {
                "fields": ("llm_debug",),
                "classes": ("collapse",),
            },
        ),
        (
            "Advanced",
            {
                "fields": (
                    "enable_subagents",
                    "enable_playwright",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("updated_at",),
            },
        ),
    )

    def has_add_permission(self, request):
        # Only allow adding if no configuration exists yet
        return not LLMConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(LLMOperationConfig)
class LLMOperationConfigAdmin(admin.ModelAdmin):
    list_display = (
        "operation_name",
        "selected_llm_type",
        "override_provider",
        "override_model",
        "is_enabled",
        "updated_at",
    )
    list_filter = ("is_enabled", "selected_llm_type", "override_provider")
    search_fields = ("operation_name", "description")
    readonly_fields = ("updated_at",)
    fieldsets = (
        (
            "Operation",
            {
                "fields": (
                    "operation_name",
                    "description",
                    "is_enabled",
                ),
            },
        ),
        (
            "LLM Override",
            {
                "fields": (
                    "override_provider",
                    "selected_llm_type",
                    "override_model",
                ),
                "description": (
                    "Leave all three blank to use system defaults. "
                    "override_provider and selected_llm_type can both be set. "
                    "override_model requires override_provider and cannot be combined with selected_llm_type."
                ),
            },
        ),
        ("Metadata", {"fields": ("updated_at",)}),
    )

    def has_add_permission(self, request):
        return True

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(SiteConfiguration)
class SiteConfigurationAdmin(admin.ModelAdmin):
    list_display = ("__str__", "updated_at")
    readonly_fields = ("updated_at",)
    fieldsets = (
        (
            "General",
            {
                "fields": (
                    "site_url",
                ),
            },
        ),
        (
            "Memory Generation",
            {
                "fields": (
                    "memory_generation_enabled",
                    "mem_worker_start",
                    "memory_last_run",
                    "memory_max_entries",
                ),
            },
        ),
        (
            "Cold storage offload",
            {
                "fields": (
                    "offload_worker_start",
                    "offload_last_run",
                ),
            },
        ),
        (
            "CSRF / Security",
            {
                "fields": ("csrf_trusted_origins",),
            },
        ),
        (
            "Authentication",
            {
                "fields": ("google_client_id", "google_client_secret"),
            },
        ),
        (
            "Email",
            {
                "fields": (
                    "email_backend",
                    "email_host",
                    "email_port",
                    "email_host_user",
                    "email_host_password",
                    "email_use_tls",
                    "email_use_ssl",
                    "default_from_email",
                ),
            },
        ),
        (
            "Cold Storage",
            {
                "fields": (
                    "coldstorage_type",
                    "coldstorage_offline",
                    "coldstorage_local_storage_location",
                    "coldstorage_s3_bucket_name",
                    "coldstorage_aws_access_key",
                    "coldstorage_aws_secret_key",
                    "coldstorage_aws_region",
                    "coldstorage_aws_endpoint",
                ),
            },
        ),
        (
            "Session Purge",
            {
                "fields": ("purge_session_cutoff_days",),
            },
        ),
        (
            "Metadata",
            {
                "fields": ("updated_at",),
            },
        ),
    )

    actions = [
        "run_memory_generation_job",
    ]

    # noinspection PyUnusedLocal
    @admin.action(description="Run Memory Generation Job Now")
    def run_memory_generation_job(self, request, queryset):
        config = SiteConfiguration.get_solo()
        try:
            SiteConfiguration.objects.filter(pk=config.pk).update(
                mem_worker_start=None,
                memory_last_run=timezone.now() - datetime.timedelta(days=1),
            )
            self.message_user(request, "Usage stats job reset completed successfully.")
        except Exception as exc:
            logger.exception(exc)
            self.message_user(
                request, f"Memory generation job reset failed: {exc}", level="error"
            )

    def has_add_permission(self, request):
        return not SiteConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(MemoryEntry)
class MemoryEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "short_content", "created_at")
    list_filter = ("created_at", "user")
    search_fields = ("user__username", "user__email", "content")
    readonly_fields = ("created_at",)

    def short_content(self, obj):
        return (obj.content or "")[:80]

    short_content.short_description = "Content"


# noinspection PyDeprecation
@admin.register(LLMDebugLog)
class LLMDebugLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "created_at",
        "operation_name",
        "call_label",
        "provider",
        "model",
        "is_agent",
        "is_deep_agent",
        "input_tokens",
        "output_tokens",
        "has_error",
        "has_agent_data",
    )
    list_filter = ("created_at", "provider", "is_agent", "is_deep_agent")
    search_fields = ("operation_name", "call_label", "provider", "model", "error")
    readonly_fields = (
        "created_at",
        "operation_name",
        "call_label",
        "provider",
        "model",
        "is_agent",
        "is_deep_agent",
        "input_tokens",
        "output_tokens",
        "formatted_input_messages",
        "formatted_raw_output",
        "error",
        "formatted_agent_history",
        "formatted_agent_tasks",
    )
    # noinspection PyUnresolvedReferences
    fields = (
        "created_at",
        "operation_name",
        "call_label",
        "provider",
        "model",
        "is_agent",
        "is_deep_agent",
        "input_tokens",
        "output_tokens",
        "error",
        "formatted_input_messages",
        "formatted_raw_output",
        "formatted_agent_history",
        "formatted_agent_tasks",
    )
    actions = ["export_metadata_csv"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return True

    @admin.display(boolean=True, description="Error?")
    def has_error(self, obj):
        return bool(obj.error)

    @admin.display(boolean=True, description="Agent data?")
    def has_agent_data(self, obj):
        return bool(obj.agent_history or obj.agent_tasks)

    @admin.action(description="Export selected log metadata as CSV (no raw payloads)")
    def export_metadata_csv(self, request, queryset):
        """Download selected rows as a CSV containing only metadata fields.

        Raw payload fields (input_messages, raw_output, agent_history,
        agent_tasks) are intentionally excluded so the export is always safe
        to share.
        """
        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = (
            'attachment; filename="llm_debug_log_metadata.csv"'
        )
        writer = csv.writer(response)
        writer.writerow(
            [
                "id",
                "created_at",
                "operation_name",
                "call_label",
                "provider",
                "model",
                "is_agent",
                "is_deep_agent",
                "input_tokens",
                "output_tokens",
            ]
        )
        for obj in queryset.order_by("-created_at"):
            writer.writerow(
                [
                    obj.pk,
                    obj.created_at.isoformat() if obj.created_at else "",
                    obj.operation_name,
                    obj.call_label,
                    obj.provider,
                    obj.model,
                    obj.is_agent,
                    obj.is_deep_agent,
                    obj.input_tokens if obj.input_tokens is not None else "",
                    obj.output_tokens if obj.output_tokens is not None else "",
                ]
            )
        return response

    @admin.display(description="Input messages")
    def formatted_input_messages(self, obj):
        if not obj.input_messages:
            return ""
        try:
            messages = json.loads(obj.input_messages)
            return format_html_join(
                "",
                '<div style="margin-bottom:1em;border-left:3px solid #ccc;padding-left:0.5em;">'
                "<strong>{}</strong>"
                '<pre style="white-space:pre-wrap;word-wrap:break-word;margin-top:0.25em;">{}</pre>'
                "</div>",
                (
                    (msg.get("type", "Unknown"), msg.get("content", ""))
                    for msg in messages
                ),
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            return format_html(
                '<pre style="white-space:pre-wrap;word-wrap:break-word;">{}</pre>',
                obj.input_messages,
            )

    @admin.display(description="Raw output")
    def formatted_raw_output(self, obj):
        return format_html(
            '<pre style="white-space: pre-wrap; word-wrap: break-word;">{}</pre>',
            obj.raw_output,
        )

    @admin.display(description="Agent history")
    def formatted_agent_history(self, obj):
        if not obj.agent_history:
            return "—"
        try:
            history = json.loads(obj.agent_history)
            if not history:
                return "—"
            parts = []
            thought_counter = 0
            tool_run_counter = 0
            for entry in history:
                entry_type = entry.get("type", "")
                if entry_type == "thought":
                    thought_counter += 1
                    parts.append(
                        format_html(
                            '<div style="margin-bottom:0.75em;border-left:3px solid #6c9;padding-left:0.5em;">'
                            "<strong>💭 Thought {}</strong>"
                            '<pre style="white-space:pre-wrap;word-wrap:break-word;margin-top:0.25em;">{}</pre>'
                            "</div>",
                            thought_counter,
                            entry.get("content", ""),
                        )
                    )
                elif entry_type == "tool_run":
                    tool_run_counter += 1
                    parts.append(
                        format_html(
                            '<div style="margin-bottom:0.75em;border-left:3px solid #69c;padding-left:0.5em;">'
                            "<strong>🔧 Tool run {} — {}</strong>"
                            '<div style="padding-left:3em;"><strong>input:</strong> <pre style="white-space:pre-wrap;word-wrap:break-word;margin-top:0.25em;">{}</pre></div>'
                            '<div style="padding-left:3em;"><strong>output:</strong> <pre style="white-space:pre-wrap;word-wrap:break-word;margin-top:0.25em;">{}</pre></div>'
                            "</div>",
                            tool_run_counter,
                            entry.get("tool", ""),
                            entry.get("input", ""),
                            entry.get("output", ""),
                        )
                    )
            if not parts:
                return "—"
            return format_html("{}" * len(parts), *parts)
        except (json.JSONDecodeError, TypeError):
            return format_html(
                '<pre style="white-space:pre-wrap;word-wrap:break-word;">{}</pre>',
                obj.agent_history,
            )

    @admin.display(description="Agent tasks")
    def formatted_agent_tasks(self, obj):
        if not obj.agent_tasks:
            return "—"
        try:
            tasks = json.loads(obj.agent_tasks)
            return _format_tasks_view(tasks)
        except (json.JSONDecodeError, TypeError):
            return format_html(
                '<pre style="white-space:pre-wrap;word-wrap:break-word;">{}</pre>',
                obj.agent_tasks,
            )


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "type", "status", "title", "created_at")
    list_filter = ("type", "status", "created_at")
    search_fields = ("user__email", "title", "body")
    readonly_fields = ("created_at",)

    fieldsets = (
        (
            "Notification",
            {
                "fields": (
                    "user",
                    "type",
                    "status",
                    "title",
                    "body",
                    "cta_action",
                    "cta_params",
                )
            },
        ),
        ("Tracking", {"fields": ("expires_at", "read_at", "dismissed_at")}),
        ("Timestamp", {"fields": ("created_at",)}),
    )
