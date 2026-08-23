# aiworks-core — Django Library for AI-Powered Features

aiworks-core is a self-contained Django library providing AI orchestration, knowledge management, and LLM-powered features for applications. It ships as a Django app (`aiworks_core`) with models, REST API endpoints, background workers, and service layers that can be dropped into any Django project.

---

## Table of Contents

1. [Installation](#1-installation)
2. [App Configuration](#2-app-configuration)
3. [URL Routing](#3-url-routing)
4. [Models](#4-models)
5. [REST API Endpoints](#5-rest-api-endpoints)
6. [Core Services](#6-core-services)
7. [Background Workers & Watchdog](#7-background-workers--watchdog)
8. [Testing](#8-testing)
9. [Integration Patterns (Lessons from Production Migrations)](#9-integration-patterns-lessons-from-production-migrations)
   - [9.1 Import Paths — Never Use `api.*` Paths](#91-import-paths--never-use-apipaths)
   - [9.2 URL Deduplication — Use `api_urlpatterns` from aiworks_core](#92-url-deduplication--use-api_urlpatterns-from-aiworks_core)
   - [9.3 Multi-Table Inheritance (MTI) for Extending the Session Model](#93-multi-table-inheritance-mti-for-extending-the-session-model)
   - [9.4 User Data in `extra_data` JSONField — Plain Python Wrapper Pattern](#94-user-data-in-extradata-jsonfield--plain-python-wrapper-pattern)
   - [9.5 Session Formatting and Download Callbacks](#95-session-formatting-and-download-callbacks)
   - [9.6 Admin Integration — Use aiworks-core Admins Directly](#96-admin-integration--use-aiworks-core-admins-directly)
   - [9.7 SiteConfiguration Singleton — App-Specific Subclass](#97-siteconfiguration-singleton--app-specific-subclass)
   - [9.8 Key Gotchas](#98-key-gotchas)
   - [9.9 Project Structure for a New aiworks-core Integration](#99-project-structure-for-a-new-aiworks-core-integration)
10. [Example: Integrating Into a New Project](#10-example-integrating-into-a-new-project)

---

## 1. Installation

### 1.1 Dependencies

aiworks-core requires Python 3.11+ and Django 5+. Install it into your project:

```bash
pip install aiworks-core
```

Or add to `requirements.in` and compile:

```bash
uv pip compile requirements.in -o requirements.txt
```

### 1.2 Django Settings

Add `aiworks_core` to `INSTALLED_APPS` **before** your project apps so that aiworks-core models and migrations take precedence:

```python
INSTALLED_APPS = [
    "aiworks_core",          # ← must come before project apps
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "channels",
    # ... your other apps
]
```

### 1.2.1 Template Directory

aiworks-core ships admin templates from `aiworks_core/resources/templates/`. Django does not auto-discover templates in `resources/` subdirectories — only the standard `app_name/templates/` layout. Add the library's template directory to your `TEMPLATES` DIRS in your Django settings file:

```python
from pathlib import Path
import aiworks_core

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            BASE_DIR / "myapp" / "templates",           # your project's templates
            Path(aiworks_core.__file__).parent / "resources" / "templates",  # aiworks-core admin templates
        ],
        "APP_DIRS": True,
        ...
    },
]
```

Without this, admin features that rely on template overrides (e.g., the "Execute Tool" button on the MCPServer change view) will silently fall back to the default admin template and appear missing.

### 1.3 User Model

aiworks-core provides a `User` model. Configure it as the auth user model:

```python
AUTH_USER_MODEL = "aiworks_core.User"
```

### 1.4 URL Configuration

Include the library's URL patterns at your desired API prefix. All library endpoints are prefixed with `api/` by default:

```python
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("aiworks_core.urls")),
]
```

### 1.5 Required Settings

#### Core Settings

| Setting | Required | Description |
|---------|----------|-------------|
| `AIWORKS_CORE_PROMPTS_FILE` | Yes | Path to a `prompts.yaml` file. See [Prompts](#61-prompts). |
| `JWT_SIGNING_KEY` | Yes | Secret key for signing JWT tokens. |

#### Optional General Settings

| Setting | Required | Description |
|---------|----------|-------------|
| `AIWORKS_CORE_APP_NAME` | No | Display name used in email templates. Defaults to `"AiWorksCore"`. |
| `AIWORKS_CORE_SNAPSHOT_SESSION_FIELDS` | No | List of session field names to include in session snapshots. |
| `AIWORKS_CORE_SNAPSHOT_FIELDS_TO_CLEAR` | No | List of session field names to clear on snapshot restore. |
| `AIWORKS_CORE_SNAPSHOT_SUPPORTED_SESSION_TYPES` | No | List of session types that support snapshots. Example: `["type_a", "type_b", "type_c", "type_d"]`. |

#### User Integration Hooks

These hooks allow a host project to customize user registration and serialization without subclassing aiworks-core models.

| Setting | Required | Description |
|---------|----------|-------------|
| `AIWORKS_CORE_USER_REGISTER_HOOK` | No | Dotted path to a function called when a new user is registered (e.g., `"myapp.serializers.on_user_register"`). The function receives the `User` instance after creation and should initialize `user.extra_data`. See [User `extra_data` Pattern](#94-user-data-in-extradata-jsonfield--plain-python-wrapper-pattern). |
| `AIWORKS_CORE_USER_SERIALIZER_CLASS` | No | Dotted path to a custom serializer class extending `aiworks_core.serializers.UserSerializer` (e.g., `"myapp.serializers.MyUserSerializer"`). Used by `/api/auth/me/` to include `extra_data` fields. See [User Serializers](#94-user-data-in-extradata-jsonfield--plain-python-wrapper-pattern). |

#### Session Output Hooks

These hooks allow a host project to customize session formatting and download ZIP generation.

| Setting | Required | Description |
|---------|----------|-------------|
| `AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK` | No | Dotted path to a function with signature `(session) -> str` that formats a session as text for downloads and exports. Example: `"myapp.logic.session_helper.format_session_as_text"`. |
| `AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK` | No | Dotted path to a function with signature `(zip_file: zipfile.ZipFile, session, included_files: list[str]) -> None` that builds the download ZIP. |
| `AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK` | No | Dotted path to a function with signature `(session) -> list[AttachedFile]` that returns additional output files to include in session exports. |

### 1.6 Database Migrations

Run migrations to create the library's tables:

```bash
python manage.py migrate aiworks_core
```

Or apply all at once:

```bash
python manage.py migrate
```

### 1.7 Channels (Optional — for WebSocket support)

aiworks-core uses Django Channels for WebSocket consumers. Add a channel layer to `settings.py`:

```python
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}
```

For production, use Redis:

```python
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [("redis://localhost:6379", 6379)]},
    }
}
```

---

## 2. App Configuration

### 2.1 AiWorksCoreConfig

The library's Django app config (`aiworks_core.apps.AiWorksCoreConfig`) handles initialization:

- Registers the `SiteConfiguration` singleton on first access
- Connects session pre-save signals to block updates to deleted sessions
- Connects user post-save signals to trigger background generation when profile changes
- On `ready()`: starts the watchdog background thread (see [Background Workers](#7-background-workers--watchdog))

### 2.2 SiteConfiguration Singleton

The `SiteConfiguration` model is a singleton accessed via `SiteConfiguration.get_solo()`. It holds:

- **Email settings**: backend, host, port, credentials, TLS/SSL, default from address, site URL
- **Google OAuth**: client ID and secret for Google Sign-In
- **Memory generation**: enabled flag, worker start timestamp, last run, max entries
- **Cold storage**: type, offline flag, local path or S3 credentials
- **CSRF trusted origins**: comma-separated list of allowed CORS origins

Configure via Django admin at `/admin/aiworks_core/siteconfiguration/`.

### 2.3 LLM Configuration

`LLMConfiguration` is a singleton (`LLMConfiguration.get_solo()`) storing:

- **Provider selection**: default provider, fast provider, reasoning provider, ultra-fast/smart providers
- **Per-provider model names**: for smart, fast, reasoning, ultra-fast, ultra-smart model tiers across 9 providers (Google, OpenAI, Anthropic, Azure, AWS, Ollama, HuggingFace, DeepSeek, OpenRouter)
- **API keys**: stored encrypted at rest, per-provider
- **Prompt compression**: enabled flag
- **Debug mode**: LLM debug logging flag

Configure via Django admin at `/admin/aiworks_core/llmconfiguration/`.

### 2.4 Per-Operation LLM Overrides

`LLMOperationConfig` allows per-operation LLM overrides (e.g., use `"ultra-fast"` for a specific operation). Fields:

- `operation_name`: unique identifier
- `selected_llm_type`: one of `ultra-fast`, `fast`, `smart`, `ultra-smart`, `reasoning`
- `override_provider` + `override_model`: full override (mutually exclusive with `selected_llm_type`)
- `is_enabled`: whether the override is active

---

## 3. URL Routing

All library URLs are mounted under `/api/`. With `path("api/", include("aiworks_core.urls"))` in your root URL config:

| Path | View / Handler |
|------|----------------|
| `api/knowledge-bases/` | `KnowledgeBaseViewSet` (DRF ViewSet) |
| `api/mcp-servers/` | `MCPServerViewSet` (DRF ViewSet) |
| `api/predefined-mcp-servers/` | `PredefinedMCPServerViewSet` (DRF ViewSet) |
| `api/server-settings/` | `auth_views.server_settings` |
| `api/auth/register/` | `auth_views.register` |
| `api/auth/login/` | `auth_views.login` |
| `api/auth/logout/` | `auth_views.logout` |
| `api/auth/refresh/` | `TokenRefreshView` |
| `api/auth/me/` | `auth_views.current_user` |
| `api/auth/verify-email/` | `auth_views.verify_email` |
| `api/auth/password-reset/` | `auth_views.request_password_reset` |
| `api/auth/password-reset/confirm/` | `auth_views.confirm_password_reset` |
| `api/auth/google/` | `auth_views.google_auth` |
| `api/help-chat/` | `help_views.help_chat` |
| `api/memory/` | `memory_views.memory_list` |
| `api/memory/import/` | `memory_views.memory_import` |
| `api/memory/<id>/` | `memory_views.memory_delete` |
| `api/mcp-tunnel/connect/` | `MCPTunnelConnectView` |
| `api/mcp-tunnel/` | `MCPTunnelListView` |
| `api/mcp-tunnel/<tunnel_id>/claim/` | `MCPTunnelClaimView` |
| `api/mcp-tunnel/<tunnel_id>/servers/` | `MCPTunnelServersView` |
| `api/mcp-tunnel/<tunnel_id>/tools/` | `MCPTunnelToolsView` |
| `admin/aiworks_core/attachedfile/<id>/download/` | `AttachedFileAdmin.download_view` (staff only) |

---

## 4. Models

### 4.1 User

```python
aiworks_core.User
```

Represents an application user. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `email` | EmailField | Unique email address |
| `username` | CharField | Unique username |
| `tier` | CharField | `"free"` or `"pro"` |
| `provider` | CharField | Auth provider (e.g., `"email"`, `"google"`) |
| `email_verified` | BooleanField | Whether email is verified |
| `profile_context` | TextField | Free-text user background (e.g., role, goals) |
| `llm_generated_background` | TextField | Cached LLM-synthesised background |
| `favorite_session_ids` | JSONField | List of bookmarked session IDs |
| `light_mode` | BooleanField | UI light/dark mode preference |
| `has_seen_onboarding` | BooleanField | Whether the onboarding banner has been dismissed |
| `avatar` | URLField | Avatar image URL |
| `extra_data` | JSONField | Host-project extensions (e.g., per-feature counters, preferences) |

**VerificationToken** (separate model): `user` FK, `token`, `created_at`

### 4.2 Session

```python
aiworks_core.Session
```

Represents a user session for async agent tasks. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | CharField (PK) | Session identifier |
| `user` | ForeignKey | Owner |
| `session_type` | CharField | Type: `type_a`, `type_b`, `type_c`, `type_d`, `type_e`, `type_f`, or other host-project session types |
| `session_title` | TextField | Title (non-empty constraint) |
| `include_user_context` | BooleanField | Whether to include user background in context |
| `is_research_needed` | BooleanField | Whether research phase is required |
| `is_research_online` | BooleanField | Whether research has internet access |
| `is_agent_finished` | BooleanField | Whether agent has completed |
| `agent_result` | TextField | Agent's final output |
| `prev_agent_result` | TextField | Previous run's output |
| `is_public` | BooleanField | Public session flag |
| `is_deleted` | BooleanField | Soft-delete flag |
| `knowledge_bases` | ManyToMany | Attached knowledge bases |
| `mcp_servers` | ManyToMany | Attached MCP servers |
| `attached_sessions` | ManyToMany | Attached sessions |
| `desktop_tunnel_servers` | JSONField | Desktop tunnel server IDs per tunnel |
| `created_at` / `updated_at` | DateTimeField | Timestamps |

### 4.3 KnowledgeBase

```python
aiworks_core.KnowledgeBase
```

A named collection of files used as context for research and other session types.

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `user` | ForeignKey | Owner |
| `name` | CharField | Unique per user |
| `created_at` / `updated_at` | DateTimeField | Timestamps |

Files are stored as `AttachedFile` records via the `kb_files` reverse relation.

### 4.4 MCPServer

```python
aiworks_core.MCPServer
```

A Model Context Protocol (MCP) server connection. Supports HTTP servers and OpenAPI-based servers.

| Field | Type | Description |
|-------|------|-------------|
| `user` | ForeignKey | Owner |
| `name` | CharField | Display name |
| `server_type` | CharField | `"http"` or `"openapi"` |
| `url` | URLField | Server URL |
| `auth_type` | CharField | `"none"`, `"basic"`, `"bearer"`, `"oauth"` |
| `username` / `password` | CharField | Basic auth credentials |
| `token` / `refresh_token` | CharField | OAuth/bearer tokens |
| `oauth_metadata` | JSONField | OAuth client ID, client secret, auth params, token endpoint |
| `predefined_server` | ForeignKey | Link to a `PredefinedMCPServer` |
| `headers` | JSONField | Custom HTTP headers |
| `openapi_spec_url` / `openapi_spec` | TextField | OpenAPI spec URL or inline spec |
| `enabled` | BooleanField | Whether server is active |

### 4.5 PredefinedMCPServer

```python
aiworks_core.PredefinedMCPServer
```

Admin-managed catalogue of known MCP servers (e.g., GitHub, Slack). Users create `MCPServer` records derived from these. Read-only via API.

### 4.6 Persona

```python
aiworks_core.Persona
```

An AI persona (e.g., "Devil's Advocate", "Financial Analyst").

| Field | Type | Description |
|-------|------|-------------|
| `name` | CharField (PK) | Unique identifier |
| `role` | CharField | Display role |
| `description` | TextField | Persona description |
| `custom_prompt` | TextField | Custom system prompt |
| `is_system` | BooleanField | System-defined persona |
| `is_agent` | BooleanField | Agentic persona |
| `created_by` | ForeignKey | User who created it (null for system) |

### 4.7 AttachedFile

```python
aiworks_core.AttachedFile
```

A file attached to a session or knowledge base. Content can be stored in the DB (`binary_content`) or offloaded to cold storage.

| Field | Type | Description |
|-------|------|-------------|
| `session` | ForeignKey | Owning session (nullable) |
| `knowledge_base` | ForeignKey | Owning KB (nullable) |
| `name` | CharField | Filename |
| `binary_content` | BinaryField | Raw file bytes (can be offloaded to cold storage) |
| `file_type` | CharField | `"input"` or `"output"` |
| `is_hidden` | BooleanField | Hidden from UI (used for `report.md`, `memory.md`) |
| `created_at` | DateTimeField | Creation timestamp |

### 4.8 TextAttachedFile

```python
aiworks_core.TextAttachedFile
```

Extracted text content from an `AttachedFile`. Created lazily on first access.

| Field | Type | Description |
|-------|------|-------------|
| `attached_file` | OneToOneField | Parent AttachedFile |
| `content` | TextField | Extracted plain text |
| `summary` | TextField | LLM-generated summary |
| `created_at` | DateTimeField | Creation timestamp |

### 4.9 WorkerTask

```python
aiworks_core.WorkerTask
```

Tracks the state of an asynchronous background worker for a session.

| Field | Type | Description |
|-------|------|-------------|
| `session` | OneToOneField | Associated session |
| `status` | CharField | `pending`, `running`, `completed`, `failed`, `fatal_failure` |
| `progress_step` | CharField | Current step description |
| `alive_beat` | DateTimeField | Last heartbeat timestamp |
| `executor_id` | CharField | ID of the executor running this task |
| `retry_count` | IntegerField | Number of retry attempts |
| `error_message` | TextField | Last error message |
| `step_tasks` | JSONField | List of completed steps |
| `started_at` | DateTimeField | When task started |

### 4.10 MemoryEntry

```python
aiworks_core.MemoryEntry
```

A user memory extracted from sessions by the daily memory generation job.

| Field | Type | Description |
|-------|------|-------------|
| `user` | ForeignKey | Owner |
| `content` | TextField | Memory text |
| `created_at` | DateTimeField | Creation timestamp |

### 4.11 LLMConfiguration / LLMOperationConfig

See [App Configuration](#2-app-configuration).

### 4.12 SessionSnapshot

```python
aiworks_core.SessionSnapshot
```

A point-in-time snapshot of a session for restore/rerun. Stored as a ZIP in cold storage.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUIDField | Primary key |
| `session` | ForeignKey | Associated session |
| `label` | CharField | Snapshot label |
| `created_at` | DateTimeField | Creation timestamp |

### 4.13 Integration

```python
aiworks_core.Integration
```

OAuth integration for external services (LinkedIn, Netlify, Trello).

| Field | Type | Description |
|-------|------|-------------|
| `user` | ForeignKey | Owner |
| `integration_type` | CharField | Type: `"linkedin"`, `"netlify"`, `"trello"` |
| `token` / `refresh_token` | CharField | OAuth tokens |
| `oauth_metadata` | JSONField | Token endpoint, client ID |
| `config` | JSONField | Service-specific config |

---

## 5. REST API Endpoints

All endpoints require JWT authentication (Bearer token) unless noted. Responses are JSON.

### 5.1 Authentication

**POST `/api/auth/register/`** — Register a new user
```json
{ "username": "...", "email": "...", "password": "..." }
```

**POST `/api/auth/login/`** — Login
```json
{ "username": "...", "password": "..." }
// Response: { "access": "...", "refresh": "..." }
```

**POST `/api/auth/logout/`** — Logout (requires refresh token)
```json
{ "refresh": "..." }
```

**POST `/api/auth/refresh/`** — Refresh access token (no auth required)
```json
{ "refresh": "..." }
```

**GET `/api/auth/me/`** — Current user info

**PATCH `/api/auth/me/`** — Update current user

**POST `/api/auth/google/`** — Google OAuth authentication
```json
{ "credential": "<id_token>" }
// or
{ "code": "<auth_code>", "redirect_uri": "..." }
```

**POST `/api/auth/verify-email/`** — Verify email token
```json
{ "token": "..." }
```

**POST `/api/auth/password-reset/`** — Request password reset

**POST `/api/auth/password-reset/confirm/`** — Confirm password reset

### 5.2 Knowledge Bases

**GET `/api/knowledge-bases/`** — List user's KBs

**POST `/api/knowledge-bases/`** — Create KB
```json
{ "name": "My KB" }
```

**GET `/api/knowledge-bases/<id>/`** — Get KB detail

**PUT `/api/knowledge-bases/<id>/`** — Update KB

**DELETE `/api/knowledge-bases/<id>/`** — Delete KB

**POST `/api/knowledge-bases/<id>/upload_files/`** — Upload files to KB (multipart)

**DELETE `/api/knowledge-bases/<id>/files/<file_id>/`** — Remove file from KB

### 5.3 MCP Servers

**GET `/api/mcp-servers/`** — List user's MCP servers

**POST `/api/mcp-servers/`** — Create MCP server
```json
{
    "name": "GitHub",
    "server_type": "http",
    "url": "https://api.github.com",
    "auth_type": "bearer",
    "token": "ghp_..."
}
```

**GET `/api/mcp-servers/<id>/`** — Get MCP server detail

**PUT `/api/mcp-servers/<id>/`** — Update MCP server

**DELETE `/api/mcp-servers/<id>/`** — Delete MCP server

**POST `/api/mcp-servers/<id>/validate_connection/`** — Validate server connection

**POST `/api/mcp-servers/discover-oauth/`** — Discover OAuth endpoints for a server

**POST `/api/mcp-servers/oauth-exchange/`** — Exchange OAuth code for tokens

**GET `/api/predefined-mcp-servers/`** — List predefined (admin) MCP servers (read-only)

### 5.4 Memory

**GET `/api/memory/`** — List user's memory entries (most recent first)

**POST `/api/memory/import/`** — Import memories from text (Pro only)
```json
{ "text": "## Instructions\nAlways use markdown.\n\n## Identity\nJohn, 35, London." }
```

**DELETE `/api/memory/<id>/`** — Delete a memory entry

### 5.5 Help Chat

**POST `/api/help-chat/`** — Chat with help assistant
```json
{
    "messages": [
        { "role": "user", "content": "How do I create a session?" }
    ],
    "manual_content": "# App Manual\nUse the dashboard."
}
```

### 5.6 Server Settings

**GET `/api/server-settings/`** — Public server settings (no auth required)
```json
{
    "googleClientId": "xxx.apps.googleusercontent.com"
}
```

### 5.7 MCP Tunnel (Desktop CLI)

**GET `/api/mcp-tunnel/`** — List user's tunnels

**POST `/api/mcp-tunnel/connect/`** — Connect a new desktop tunnel

**POST `/api/mcp-tunnel/<tunnel_id>/claim/`** — Claim a tunnel for the current session

**GET `/api/mcp-tunnel/<tunnel_id>/servers/`** — Get servers available on a tunnel

**GET `/api/mcp-tunnel/<tunnel_id>/tools/`** — Get tools available on a tunnel

---

## 6. Core Services

### 6.1 Prompts

aiworks-core uses a `prompts.yaml` file referenced by `AIWORKS_CORE_PROMPTS_FILE`. The file contains prompt templates keyed by operation name. Each template has:

- `system_message`: System prompt (supports `{{variable}}` interpolation)
- `prompt_template`: User message template

Key operations:

| Operation | Description |
|-----------|-------------|
| `memory_extraction` | Extract memories from session history |
| `memory_compression` | Compress memories when limit reached |
| `memory_import` | Parse imported memory text |
| `user_background_synthesis` | Synthesise user background from sessions |
| `summarize_document` | Summarise uploaded documents |
| `image_tool` | Image generation prompt |
| `help_chat` | Help assistant chat (system_message receives `manual_content`) |
| `quality_check` | QA gate for deep agent output |
| `deep_agent` | Deep agent system prompt |

### 6.2 LLM Invocation (`invoke_llm`)

The central LLM entry point is `aiworks_core.logic.llm.invoke_llm()`:

```python
from aiworks_core.logic.llm import invoke_llm

result = invoke_llm(
    "my_operation",           # operation name — looks up LLMOperationConfig then prompts.yaml
    messages=[...],          # list of LangChain messages (optional, overrides template)
    system_message_template_name="...",  # override system prompt template
    user_message_template_name="...",     # override user prompt template
    template_params={...},   # variables for template interpolation
    temperature=0.7,
    parse_json=True,         # parse response as JSON
    tools=[...],             # LangChain tools for agentic mode
    is_agent=True,           # run in agent mode with tools
)
```

`invoke_llm` is async. From sync code use `aiworks_core.utils.async_to_sync`:

```python
from aiworks_core.utils import async_to_sync
result = async_to_sync(invoke_llm)("operation_name", ...)
```

### 6.3 Memory Service

```python
from aiworks_core.logic.memory import (
    generate_memories_for_user,
    run_memory_generation_job,
    get_effective_context,
)
```

- `generate_memories_for_user(user)`: Extract memories from recent sessions
- `run_memory_generation_job()`: Daily job — processes all users with sessions
- `get_effective_context(session)`: Returns user's background as `"User Background\n{bg}"` or `""`

### 6.4 User Background

```python
from aiworks_core.logic.user_background import (
    generate_user_background,
    get_effective_user_background,
)
```

- `generate_user_background(user)`: LLM-synthesises background from sessions and profile
- `get_effective_user_background(user)`: Returns cached background, generating lazily if empty

### 6.5 Knowledge Base & RAG

```python
from aiworks_core.logic.rag import RagIndex
```

`RagIndex` provides vector search over KB files. Methods:
- `add_text(collection, text, metadata)`: Add a chunk with metadata
- `search(collection, query, top_k)`: Search for similar chunks

### 6.6 MCP Tools

```python
from aiworks_core.logic.mcp_tools import (
    build_mcp_tools_from_servers,
    validate_ssrf_safe_url,
)
```

- `build_mcp_tools_from_servers(servers)`: Build LangChain tools from MCP server list
- `validate_ssrf_safe_url(url)`: Validate URL is safe for SSRF (blocks private IPs, etc.)

### 6.7 Cold Storage

```python
from aiworks_core.logic.cold_storage import (
    ColdStorageManager,
    store_attached_file,
    get_attached_file_data,
)
```

`ColdStorageManager` backends: `NoneColdStorageManager` (no-op), `LocalColdStorageManager`, `S3StorageManager`. Get instance via `ColdStorageManager.get_instance()`.

`store_attached_file(attached_file)`: Offloads `attached_file.binary_content` to cold storage.
`get_attached_file_data(attached_file)`: Returns bytes (from DB or cold storage).

### 6.8 Email Service

```python
from aiworks_core.logic.emails import EmailService
```

Static methods:
- `EmailService.send_verification_email(user, token)`
- `EmailService.send_password_reset_email(user, token)`

Email templates live at `templates/emails/` (e.g., `verify_email.txt`, `verify_email.html`, `password_reset.txt`, `password_reset.html`). Templates receive context: `user`, `token`, `site_url`, `app_name`.

### 6.9 Schema Validation

```python
from aiworks_core.logic.schema_validation_utils import (
    validate_json_with_schema_file,
    validate_json_schema,
)
```

- `validate_json_with_schema_file(content, schema_file_path)`: Validate JSON string against a schema file
- `validate_json_schema(content, schema)`: Validate JSON string against a schema string

Schema files are resolved relative to `settings.BASE_DIR`, falling back to `aiworks_core/resources/schema/` if not found.

### 6.10 File Management

```python
from aiworks_core.logic.files import (
    get_binary_data,
    get_text_content,
    set_binary_content,
    offload_to_cold_storage,
    can_extract_text,
    get_file_summary,
)
```

- `get_binary_data(attached_file)`: Get raw bytes (DB or cold storage)
- `get_text_content(attached_file)`: Get extracted text (creates `TextAttachedFile` lazily)
- `set_binary_content(attached_file, data)`: Set bytes, delete any cached text
- `offload_to_cold_storage(attached_file)`: Move `binary_content` to cold storage
- `can_extract_text(filename)`: Check if file type supports text extraction
- `get_file_summary(attached_file)`: Get or generate LLM summary of text content

### 6.11 Session Snapshots

```python
from aiworks_core.logic.session_snapshot import (
    create_snapshot,
    restore_snapshot,
)
```

- `create_snapshot(session)`: Create a ZIP snapshot of session state + files in cold storage
- `restore_snapshot(session, snapshot)`: Restore session from snapshot

### 6.12 Document Parsing

```python
from aiworks_core.logic.document_parser import (
    parse_file,
    parse_files_from_request,
)
```

- `parse_file(file_bytes, filename, max_size_mb=50)`: Parse PDF, DOCX, Excel, PPT files
- `parse_files_from_request(request, session, max_files=10, max_size_mb=50)`: Parse uploaded files from a Django request

### 6.13 Export

```python
from aiworks_core.logic.export import generate_pdf, generate_docx
```

- `generate_pdf(content, title)`: Convert HTML/markdown content to PDF
- `generate_docx(content, title)`: Convert HTML/markdown content to DOCX

### 6.14 Image Generation

```python
from aiworks_core.logic.generate_image_tool import CreateGenerateImageTool
```

Factory for creating LangChain image generation tools per session.

---

## 7. Background Workers & Watchdog

### 7.1 Watchdog

The `AiWorksCoreConfig` starts a watchdog background thread when Django finishes loading (`ready()`). The watchdog:

- Scans every 30 seconds for pending `WorkerTask` records
- Claims tasks atomically via `SELECT FOR UPDATE SKIP LOCKED`
- Resets stale tasks (no heartbeat in 60s) back to `pending`
- Retries failed tasks up to 3 times (max 3 × 60s delay)
- Escalates to `fatal_failure` after max retries
- Runs scheduled session reruns
- Unschedules sessions for users downgraded from Pro to Free
- Purges soft-deleted sessions older than 30 days
- Runs daily memory generation job (when enabled in `SiteConfiguration`)
- Runs daily cold storage offload job

### 7.2 Bounded Execution

Session executions use a bounded thread pool (default 4 concurrent). The pool is a `bounded semaphore` acquired before queuing a task. If the pool is full, the task stays `pending`.

### 7.3 WorkerTask State Machine

```
pending → running → completed
                  → failed → (retry_count < 3) → pending → running
                          → (retry_count >= 3) → fatal_failure
```

### 7.4 Running Without the Watchdog

In server environments where you run the watchdog as a separate process, set `RUN_WATCHDOG=false` as an environment variable to disable the embedded watchdog thread.

---

## 8. Testing

### 8.1 Test Settings

Tests use `aiworks_core.tests.settings`. Configure via `DJANGO_SETTINGS_MODULE`:

```python
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = aiworks_core.tests.settings
```

Key test settings:
- `RUNNING_TESTS = True` — enables in-memory DB, disables throttling
- `AIWORKS_CORE_PROMPTS_FILE` — points to `aiworks_core/tests/prompts.yaml`
- `AIWORKS_CORE_COLDSTORAGE_TYPE = "none"` — disables cold storage
- `EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"` — captures emails in `mail.outbox`

### 8.2 Test Utilities

```python
from aiworks_core.tests import (
    _make_user,      # Create a test user
    _make_session,   # Create a test session
    _auth_header,    # JWT auth header dict for APIClient
)
```

### 8.3 Running Tests

```bash
cd /path/to/aiworks-core
source .venv/bin/activate
pytest aiworks_core/tests/ -v
```

### 8.4 Test Fixtures

Minimal prompt templates are provided in `aiworks_core/tests/prompts.yaml` covering all operations used by tests. Email templates are provided in `aiworks_core/templates/emails/` for test email sending.

### 8.5 Writing Tests

When adding features to the library, ensure:

1. Tests are added in `aiworks_core/tests/`
2. Use `_make_user()` and `_make_session()` helpers
3. Use `async_to_sync` wrapper for calling async library functions from sync tests
4. Mock `invoke_llm` using `unittest.mock.patch` on `"aiworks_core.logic.{module}.invoke_llm"`
5. URL endpoints use the `/api/` prefix (configured in `urls.py`)
6. For admin endpoints, use `/admin/aiworks_core/...` (based on `app_label`)

---

## 9. Integration Patterns (Lessons from Production Migrations)

This section captures hard-won patterns for integrating aiworks-core into an existing project. Study these before starting — they prevent the most common integration mistakes.

---

### 9.1 Import Paths — Never Use `api.*` Paths

aiworks-core exposes its modules directly. **Never assume or create `api.*` import paths** — that prefix belongs to the host project's URL configuration, not to the library.

```python
# ✅ Correct — aiworks_core provides these directly
from aiworks_core.models import Session, User, WorkerTask, AttachedFile
from aiworks_core.logic.llm import invoke_llm
from aiworks_core.logic.files import get_binary_data, get_text_content
from aiworks_core.logic.session_snapshot import create_snapshot, restore_snapshot
from aiworks_core.logic.cold_storage import ColdStorageManager
from aiworks_core.logic.emails import EmailService
from aiworks_core.logic.deepagent_utils import run_deep_agent_on_session
from aiworks_core.logic.schema_validation_utils import validate_json_with_schema_file
from aiworks_core.views.jwt import create_token
from aiworks_core.utils import async_to_sync

# ❌ Wrong — these paths do not exist in aiworks_core
from api.models import Session
from api.logic.llm import invoke_llm
from api.logic.files import get_binary_data
```

**Rule:** Always import from `aiworks_core.{subpackage}`. If a module is not at the top level of aiworks_core, it is not a public API.

---

### 9.2 URL Deduplication — Use `api_urlpatterns` from aiworks_core

Do NOT copy aiworks-core URL patterns into your project's `urls.py`. Import `api_urlpatterns` and include it:

```python
# myproject/urls.py
from django.urls import path, include
from aiworks_core.urls import api_urlpatterns  # ← import the list, not a module

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include(api_urlpatterns)),  # ← includes all aiworks_core endpoints
    # ... your project's additional routes below
]
```

`api_urlpatterns` is a plain list of URL patterns (without the `/api/` prefix). This avoids duplicating ~20 URL patterns (knowledge-bases, mcp-servers, auth endpoints, memory, mcp-tunnel, help-chat).

If you need to add project-specific routes alongside aiworks-core routes, include `api_urlpatterns` first, then add your own routes in the same `urlpatterns` list.

---

### 9.3 Multi-Table Inheritance (MTI) for Extending the Session Model

If your project needs additional fields on the `Session` model, extend `aiworks_core.Session` via MTI:

```python
# myapp/models.py
from django.db import models
from aiworks_core.models import Session


class MyProjectSession(Session):
    """Extended session with project-specific fields."""

    class Meta:
        proxy = False  # False = real child table with MTI

    my_custom_field = models.JSONField(default=list, blank=True)
    my_custom_field = models.CharField(max_length=255, blank=True)

# Backward-compatibility alias if needed
Session = MyProjectSession
```

**⚠️ Critical Gotcha — `WorkerTask.session` FK returns the parent class:**

`WorkerTask.session` is defined on `aiworks_core.Session`. When you access `worker_task.session`, Django returns the **parent** `aiworks_core.Session` instance, not your child class. Any fields specific to your child model will not be on that object.

**Fix:** Always re-fetch using your child model when you need child-specific fields:

```python
# ❌ Wrong — returns aiworks_core.Session, not MyProjectSession
session = worker_task.session
print(session.my_custom_field)  # AttributeError

# ✅ Correct — re-fetch with your child model
MyProjectSession = type(worker_task.session)  # or hardcode your child model
session = MyProjectSession.objects.select_related("user").get(id=worker_task.session.id)
print(session.my_custom_field)  # works
```

**Apply this pattern for any code that calls EmailService, sends notifications, or accesses child-specific fields after fetching via `WorkerTask.session`:**

```python
# Always re-fetch before EmailService / Notification calls
my_session = MyProjectSession.objects.select_related("user").get(id=session.id)
EmailService.send_my_email(my_session.user, my_session)
```

---

### 9.4 User Data in `extra_data` JSONField — Plain Python Wrapper Pattern

aiworks-core provides `User.extra_data = models.JSONField(...)` on the built-in `User` model. **Do not create a separate Django model** (like `UserProfile`) for user preferences and counters. Instead, use a plain Python wrapper class.

**Why:** A separate `UserProfile` model creates MTI complications, requires its own migrations, and is harder to maintain. The JSONField approach stores all user-specific data directly on the User record.

**Pattern — Plain Python wrapper class:**

```python
# myapp/models.py


class UserProfile:
    """Plain Python class (not a Django model) wrapping User.extra_data.

    Provides typed property access to JSON fields.
    """

    def __init__(self, user):
        self.user = user
        self._data = user.extra_data or {}

    # --- preferences ---
    @property
    def my_feature_enabled(self):
        return self._data.get("my_feature_enabled", True)

    @my_feature_enabled.setter
    def my_feature_enabled(self, value):
        self._data["my_feature_enabled"] = value

    @property
    def my_feature_disabled(self):
        return self._data.get("my_feature_disabled", [])

    @my_feature_disabled.setter
    def my_feature_disabled(self, value):
        self._data["my_feature_disabled"] = value

    # --- daily counters ---
    @property
    def my_feature_daily_count(self):
        return self._data.get("my_feature_daily_count", 0)

    @my_feature_daily_count.setter
    def my_feature_daily_count(self, value):
        self._data["my_feature_daily_count"] = value

    @property
    def my_feature_last_used_date(self):
        return self._data.get("my_feature_last_used_date")

    @my_feature_last_used_date.setter
    def my_feature_last_used_date(self, value):
        self._data["my_feature_last_used_date"] = (
            value.isoformat() if hasattr(value, "isoformat") else value
        )

    # --- effective limits ---
    def get_effective_my_feature_limit(self):
        override = self._data.get("my_feature_daily_limit")
        if override:
            return override
        return 3 if self.user.tier == "free" else 20

    # Add similar patterns for other features

    def save(self):
        self.user.save(update_fields=["extra_data"])
```

**Usage throughout your codebase:**

```python
# Reading
profile = UserProfile(user)
if profile.my_feature_daily_count >= profile.get_effective_my_feature_limit():
    raise PermissionDenied("Daily limit reached")

# Writing
profile = UserProfile(user)
profile.my_feature_daily_count = profile.my_feature_daily_count + 1
profile.my_feature_last_used_date = date.today()
profile.save()
```

**Two required hooks for full integration:**

1. **`AIWORKS_CORE_USER_REGISTER_HOOK`** — initializes `extra_data` when a new user is created:

```python
# myapp/serializers.py
def on_user_register(user):
    """Called by aiworks_core.UserRegistrationSerializer.create() after user is saved."""
    user.extra_data = {
        "my_feature_enabled": True,
        "my_feature_disabled": [],
        "my_feature_daily_count": 0,
        # ... other defaults
    }
    user.save(update_fields=["extra_data"])
```

```python
# settings.py
AIWORKS_CORE_USER_REGISTER_HOOK = "myapp.serializers.on_user_register"
```

2. **`AIWORKS_CORE_USER_SERIALIZER_CLASS`** — injects your custom serializer for `/api/auth/me`:

```python
# myapp/serializers.py
class MyUserSerializer(aiworks_core.serializers.UserSerializer):
    my_feature_enabled = serializers.BooleanField(required=False)
    my_feature_disabled = serializers.ListField(child=serializers.CharField(), required=False)

    class Meta(aiworks_core.serializers.UserSerializer.Meta):
        fields = list(aiworks_core.serializers.UserSerializer.Meta.fields) + [
            "my_feature_enabled", "my_feature_disabled"
        ]
```

```python
# settings.py
AIWORKS_CORE_USER_SERIALIZER_CLASS = "myapp.serializers.MyUserSerializer"
```

---

### 9.5 Session Formatting and Download Callbacks

aiworks-core provides four settings hooks for customizing session output:

| Setting | Type | Purpose |
|---------|------|---------|
| `AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK` | `"module.path:function"` string | Override session text formatting for downloads/exports |
| `AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK` | `"module.path:function"` string | Override how the ZIP download is built |
| `AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK` | `"module.path:function"` string | Inject additional files into session output |
| `AIWORKS_CORE_SNAPSHOT_SUPPORTED_SESSION_TYPES` | list of strings | Limit which session types support snapshots |

All callbacks use `importlib.import_module` to resolve the function at runtime. Use the colon separator:

```python
# settings.py
AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK = "myapp.session_helper.format_session_as_text"
AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK = "myapp.session_helper.inject_files_to_zip"
AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK = "myapp.session_helper.get_session_output_files"
AIWORKS_CORE_SNAPSHOT_SUPPORTED_SESSION_TYPES = ["type_a", "type_b"]
```

**Callback signatures:**

```python
# AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK
def format_session_as_text(session) -> str:
    """Return a formatted text representation of the session."""
    ...

# AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK
def inject_files_to_zip(zip_file: zipfile.ZipFile, session, included_files: list[str]) -> None:
    """Add project-specific files to the download ZIP."""
    ...

# AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK
def get_session_output_files(session) -> list[AttachedFile]:
    """Return additional output files for the session."""
    ...
```

---

### 9.6 Admin Integration — Use aiworks-core Admins Directly

**Do NOT copy aiworks-core admin classes into your project.** Register the admin models and let aiworks-core's admin classes handle them. Only add customizations where your project genuinely differs.

```python
# myapp/admin.py

# ✅ Correct — register aiworks_core models with aiworks_core's admin classes
from aiworks_core.admin import (
    KnowledgeBaseAdmin,
    MCPServerAdmin,
    PredefinedMCPServerAdmin,
    TunnelAdmin,
)
from django.contrib import admin
from aiworks_core.models import KnowledgeBase, MCPServer, PredefinedMCPServer, Tunnel

admin.site.register(KnowledgeBase, KnowledgeBaseAdmin)
admin.site.register(MCPServer, MCPServerAdmin)
admin.site.register(PredefinedMCPServer, PredefinedMCPServerAdmin)
admin.site.register(Tunnel, TunnelAdmin)
```

**Do NOT call `admin.site.unregister()`** — that is only needed if you were overriding an admin with a custom class. If you use aiworks-core's admins directly, no unregister is needed.

**If you need to add fields** (e.g., to `UserAdmin` for `extra_data`):

```python
from aiworks_core.admin import UserAdmin as AiWorksUserAdmin


class MyUserAdmin(AiWorksUserAdmin):
    fieldsets = (
        AiWorksUserAdmin.fieldsets[0],  # keep existing
        ("Profile", {"fields": ("extra_data",)}),  # add extra_data
    )


admin.site.register(User, MyUserAdmin)
```

---

### 9.7 SiteConfiguration Singleton — App-Specific Subclass

If your project extends `SiteConfiguration` with additional fields, use `models.SingletonModel` (or the same pattern aiworks-core uses) and ensure:

1. The model has a proper migration (test with `python manage.py makemigrations myapp` — it should say "no changes detected", not produce a new migration)
2. Use the **project-specific** singleton accessor, not `aiworks_core.SiteConfiguration.get_solo()` directly:

```python
# myapp/models.py
from aiworks_core.models import SiteConfiguration


class MySiteConfiguration(SiteConfiguration):
    """Extended site config with project-specific fields."""

    my_custom_field = models.CharField(max_length=255, blank=True)

    class Meta:
        verbose_name = "Site Configuration"
        verbose_name_plural = "Site Configuration"


# myapp/apps.py
class MyAppConfig(DjangoAppConfig):
    def ready(self):
        # Ensure singleton exists
        cfg, _ = MySiteConfiguration.objects.get_or_create(pk=1)
```

Then use `MySiteConfiguration.get_solo()` in your code — not `SiteConfiguration.get_solo()` which returns the parent class.

---

### 9.8 Key Gotchas

**1. Migrations are disposable.**
Do not hand-craft or preserve old migrations. If a migration is problematic, delete it and regenerate with `makemigrations`. Do not worry about preserving migration history for backward compatibility.

**2. Never patch `async_to_sync` or `sync_to_async` in tests.**
These are asgiref bridge functions. Patching them is fragile and breaks across Django/Python version updates. Instead, patch the underlying async function directly:

```python
# ✅ Correct — patch the actual logic function
with patch("myapp.logic.my_module.invoke_llm", new_callable=AsyncMock) as mock_invoke:
    mock_invoke.return_value = {"result": "ok"}
    result = async_to_sync(my_function)()

# ❌ Wrong — patching the bridge
with patch("asgiref.sync.async_to_sync", ...):
    ...
```

**3. `select_for_update(skip_locked=True)` is required for atomic task claiming.**
Never claim a `WorkerTask` without `select_for_update(skip_locked=True)` inside `transaction.atomic()`. This prevents two nodes from claiming the same task in a multi-pod environment.

**4. Inline imports in views only.**
To keep Django startup fast, only use inline imports (`from .logic import func`) inside view functions. All other files (logic, models, middleware, tests) must use top-level imports.

**5. Always use relative imports within the `api` package.**
When inside the `backend/` Django project, use relative imports for intra-package imports:

```python
# ✅ Correct — relative import
from .models import MyModel
from .logic import my_helper

# ❌ Wrong — absolute path from project root
from api.models import MyModel
```

The exception is any place a string is used as a reference (mock patch targets, logger names, Django model labels in migrations) — those must use the full `aiworks_core.path` string because they are resolved at runtime.

---

### 9.9 Project Structure for a New aiworks-core Integration

When starting a new project that uses aiworks-core, organize your code like this:

```
myproject/
├── myproject/
│   ├── settings.py          # INSTALLED_APPS, AIWORKS_CORE_* hooks
│   └── urls.py              # include(api_urlpatterns) — no duplication
├── myapp/
│   ├── models.py            # MTI session model, UserProfile wrapper (plain class)
│   ├── serializers.py        # Custom user serializer, register hook
│   ├── admin.py              # Register aiworks_core admins; add UserAdmin only
│   ├── logic/                # Project-specific logic (adapters, overrides)
│   │   ├── emails.py         # EmailService adapter (if needed)
│   │   ├── session_helper.py # Callback implementations
│   │   └── executors/        # Project-specific executors if needed
│   ├── views/
│   │   └── myapp_views.py    # Project-specific API views
│   └── tests/
│       └── test_myapp.py
└── prompts.yaml             # Your prompt templates
```

**Key files you must create:**
- `UserProfile` wrapper class in `myapp/models.py`
- `on_user_register` hook in `myapp/serializers.py`
- Custom `UserSerializer` in `myapp/serializers.py` (if you need `extra_data` fields in `/api/auth/me`)
- Settings entries: `AIWORKS_CORE_USER_REGISTER_HOOK`, `AIWORKS_CORE_USER_SERIALIZER_CLASS`, `AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK`, etc.

**Files you should NOT create (they exist in aiworks_core — use them directly):**
- `User` model (use `aiworks_core.User` or your own via MTI)
- `Session` model (extend via MTI if needed)
- `WorkerTask`, `AttachedFile`, `KnowledgeBase`, `MCPServer`, `Persona`
- `logic/emails.py` (use `aiworks_core.logic.emails.EmailService` directly)
- `logic/files.py`, `logic/cold_storage.py`, `logic/session_snapshot.py`

---

## 10. Example: Integrating Into a New Project

```python
# settings.py
INSTALLED_APPS = [
    "aiworks_core",
    "rest_framework",
    "rest_framework_simplejwt",
    "channels",
    "django.contrib.admin",
    "django.contrib.auth",
    # ...
]
AUTH_USER_MODEL = "aiworks_core.User"
ROOT_URLCONF = "myproject.urls"
CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}

AIWORKS_CORE_PROMPTS_FILE = "/path/to/prompts.yaml"
JWT_SIGNING_KEY = "my-secret-key"
```

```python
# urls.py
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("aiworks_core.urls")),
]
```

```bash
# Apply migrations
python manage.py migrate aiworks_core

# Configure LLM provider
python manage.py shell
>>> from aiworks_core.models import LLMConfiguration
>>> cfg = LLMConfiguration.get_solo()
>>> cfg.default_provider = "openai"
>>> cfg.openai_api_key = "sk-..."
>>> cfg.save()
```
