# AGENTS.md — aiworks-core

Read this file at the start of every session. It contains project-specific conventions, architectural context, and hard-won feedback that must be followed.

---

## ✅ End-of-Task Checklist

**Before marking any task as done, go through every item below:**

- [ ] **Unit tests** — every change must have tests covering the new/changed behaviour. Add them before finishing.
- [ ] **Ruff** — run `ruff check` on all changed Python files. Fix any issues.
- [ ] **Lint** — run `cd aiworks_core && .venv/bin/python -m mypy aiworks_core` if mypy is configured. Fix any issues.
- [ ] **Tests pass** — run `pytest aiworks_core/tests/ -q --tb=short`. All must pass.
- [ ] **MANUAL.md** — did this change affect models, endpoints, services, or API? If yes, update the relevant section(s) of `MANUAL.md` now.
- [ ] **README.md** — did this change affect installation, configuration, features, or architecture? If yes, update `README.md` now.
- [ ] **AGENTS.md** — did this change affect architecture, models, endpoints, logic, data flows, infrastructure, or test structure? If yes, update the relevant section(s) of this file now.

Do not skip any item. A task is not done until this checklist is complete.

---

## Project Structure

```
aiworks_core/
├── __init__.py
├── apps.py                  # AiWorksCoreConfig — singleton init, signals, watchdog
├── models.py               # All ORM models (User, Session, KnowledgeBase, MCPServer, etc.)
├── admin.py                # Django admin registrations
├── urls.py                 # URL routing
├── middleware.py            # DynamicCsrfMiddleware
├── signals.py              # Django signals (session pre-save, user post-save)
├── logic/
│   ├── llm.py              # invoke_llm — single LLM choke point
│   ├── memory.py           # Memory generation job
│   ├── user_background.py   # User background synthesis
│   ├── rag.py              # ChromaDB RAG index
│   ├── mcp_tools.py        # MCP tool adapter + SSRF validation
│   ├── cold_storage.py      # ColdStorageManager (None/Local/S3)
│   ├── emails.py            # EmailService (verification, password reset)
│   ├── files.py            # File management helpers
│   ├── session_snapshot.py  # Snapshot creation and restoration
│   ├── document_parser.py   # PDF/DOCX/Excel/PPT parsing
│   ├── schema_validation_utils.py  # JSON schema validation
│   ├── export.py           # PDF/DOCX generation
│   ├── generate_image_tool.py    # Image generation tool factory
│   ├── deepagent_utils.py  # Deep agent run + quality gate + recovery
│   ├── executor.py         # WorkerExecutor base class
│   ├── executor_factory.py  # create_executor() factory (host must override)
│   ├── session_helper.py   # Session CRUD helpers
│   ├── help.py / help_logic.py  # Help chat
│   ├── oauth.py            # Generic OAuth 2.1 / PKCE helpers
│   ├── mcp_tunnel.py       # Desktop MCP tunnel manager
│   └── *.py                # Other utilities (audio, video, image, etc.)
├── views/
│   ├── auth_views.py        # Auth endpoints
│   ├── kb_views.py          # Knowledge base CRUD + file upload
│   ├── mcp_views.py        # MCP server CRUD + OAuth discovery
│   ├── memory_views.py      # Memory list/delete
│   ├── help_views.py       # Help chat
│   ├── mcp_tunnel_views.py # Desktop MCP tunnel endpoints
│   └── views_utils.py      # Shared view utilities
├── resources/
│   └── schemas/            # JSON schemas for schema validation
├── templates/
│   └── emails/             # Email templates (verify_email, password_reset)
├── tests/
│   ├── __init__.py          # Test helpers: _make_user, _make_session, _auth_header
│   ├── settings.py          # Test Django settings
│   ├── prompts.yaml         # Minimal prompts for tests
│   ├── conftest.py          # pytest fixtures
│   └── test_*.py           # Test modules
└── management/
    └── commands/           # Management commands
```

---

## Core Conventions

### Python Imports

**Relative imports are mandatory** within the `aiworks_core` package. Never use absolute `aiworks_core.*` imports.

```python
# CORRECT — relative import
from .models import User
from .logic.llm import invoke_llm
from ..some.nested.module import something

# WRONG — absolute import (breaks IDE when opened at repo root)
from aiworks_core.models import User
from aiworks_core.logic.llm import invoke_llm
```

**Exception — anywhere a string is used as a reference to a module, class, or logger:** Mock target strings, logger name strings, and Django lazy model reference strings must use the absolute `aiworks_core.*` path because they are resolved at runtime as strings, not Python imports:

```python
unittest.mock.patch("aiworks_core.logic.invoke_llm")          # mock target string
logging.getLogger("aiworks_core.llm")                       # logger name string
ForeignKey("aiworks_core.User")                             # Django lazy reference
self.assertLogs("aiworks_core.llm", ...)                   # logger name string
AUTH_USER_MODEL = "aiworks_core.User"                       # Django setting
```

### Inline Imports — Views-to-Logic Layer ONLY

**Inline imports are permitted in `views.py` and any file within the `views/` package** for `.logic`, `.mcp_tools`, or any other logic/LLM module — place them inside the function/method body, never at the top of the file.

**Every other Python file** (`models.py`, `logic/*.py`, `middleware.py`, `apps.py`, `tests/*.py`, etc.) must use **top-level** imports only. No inline imports inside functions, ever.

```python
# WRONG — in logic/*.py, models.py, apps.py, tests/*.py
def some_function():
    from .llm import invoke_llm        # NEVER inline import outside views/
    result = invoke_llm(...)
```

```python
# CORRECT — in views/ package files, inside a function:
def my_view(request):
    from .logic.llm import invoke_llm   # deferred until called
    result = invoke_llm(...)

# CORRECT — in any other file, at top of file:
from .llm import invoke_llm             # top-level, always
from .models import Session
```

### Managing Python Dependencies

**Never edit `requirements.txt` or `requirements-dev.txt` directly.** These files are auto-generated and any manual changes will be overwritten.

**How to apply:**
1. Add or change a production dependency → edit `requirements.in`
2. Add or change a dev/test-only dependency → edit `requirements-dev.in`
3. Recompile the lock files:
   ```bash
   uv pip compile requirements.in -o requirements.txt
   uv pip compile requirements-dev.in -o requirements-dev.txt
   ```
4. Install into the local venv:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## Architecture

### The Single LLM Choke Point

All LLM calls — regardless of whether they come from executors, help chat, memory generation, or any other feature — must go through `invoke_llm()` defined in `aiworks_core/logic/llm.py`. Never call LLM provider SDKs directly.

```python
from .llm import invoke_llm

result = invoke_llm(
    "my_operation",      # operation name — resolved via LLMOperationConfig then prompts.yaml
    messages=[...],
    template_params={...},
    temperature=0.7,
    parse_json=True,
)
```

`invoke_llm` is async. From sync code (views, tests) use `aiworks_core.utils.async_to_sync`:

```python
from aiworks_core.utils import async_to_sync
result = async_to_sync(invoke_llm)("operation_name", ...)
```

### invoke_llm Resolution Order

1. Check `LLMOperationConfig` for a per-operation override (if `is_enabled=True`)
2. Otherwise, look up the operation in `prompts.yaml` to get the system/user template
3. Interpolate template variables from `template_params`
4. Resolve `llm_type` (`None`/smart · `fast` · `reasoning` · `ultra-fast` · `ultra-smart`) using `LLMConfiguration`
5. Resolve provider + model name for the selected `llm_type`
6. Call the provider SDK with the resolved model, temperature, and parsed response

### Prompts (`prompts.yaml`)

All prompt templates live in the file pointed to by `AIWORKS_CORE_PROMPTS_FILE`. The file contains keyed templates:

```yaml
my_operation:
  system_message: |
    You are a helpful assistant. {{user_name}} is a {{user_role}}.
  user_message_template: |
    Help them with: {{question}}
```

Variables use `{{}}` interpolation. Never hardcode prompts in Python code.

### Per-Operation LLM Overrides

`LLMOperationConfig` allows switching LLM type (e.g., `ultra-fast`) or even full provider+model for specific operations without changing code. This is the correct way to tune performance/cost trade-offs.

### Deep Agent Quality Gates

Executors that use `run_deep_agent_on_session` (deep agent executors) run a quality-check LLM call after producing output. The `quality_check_prompt_key` parameter names the quality check prompt in `prompts.yaml`. The QA agent reads `memory.md` and the generated output, then returns `{is_approved: bool, feedback: str}`. If `is_approved=false` and a `recovery_prompt_key` is set, `recover_deep_agent_run` is called with the `recovery_reason` injected into the recovery prompt template. Recovery attempts are capped at `_RECOVERY_MAX_ATTEMPTS=3`.

### Executors

Executors are the background task units. All extend `WorkerExecutor` (defined in `logic/executor.py`) and are spawned exclusively via `create_executor(session)` factory (`logic/executor_factory.py`). Never instantiate an executor directly.

Available executors:

| Component | Purpose |
|-----------|---------|
| `WorkerExecutor` | Base class for all async executors; `create_executor(session)` spawns the right one |
| `create_executor(session)` | Factory function — **host project must override** to map `session.session_type` to the appropriate `WorkerExecutor` subclass |
| `invoke_llm` | Single LLM choke point; all executors use this |
| `deepagent_utils` | `run_deep_agent_on_session`, quality gates, subagent spawning — for executors that run deep agents |
| `AiWorksCoreConfig.ready()` | Starts the watchdog background thread |

### WorkerTask State Machine

```
pending → running → completed
                  → failed → (retry_count < 3) → pending → running
                          → (retry_count >= 3) → fatal_failure
```

- **Atomic claiming**: always use `select_for_update(skip_locked=True)` inside `transaction.atomic()` when transitioning `pending → running`. This is the only mechanism that prevents two workers from picking up the same task.
- **Heartbeat**: executors write `alive_beat` every few seconds. The watchdog resets any task whose `alive_beat` is older than 60s back to `pending`.
- **Bounded pool**: a semaphore (default size 4) limits concurrent executions. If the pool is full, the task stays `pending`.

### Watchdog

The `AiWorksCoreConfig.ready()` starts a watchdog background thread. The watchdog:

- Scans every 30 seconds for pending `WorkerTask` records
- Resets stale tasks (>60s heartbeat) to `pending`
- Retries failed tasks up to 3 times
- Escalates to `fatal_failure` after max retries
- Runs scheduled reruns
- Unschedules sessions for users downgraded from Pro to Free
- Purges soft-deleted sessions older than 30 days
- Runs daily memory generation job (when enabled in `SiteConfiguration`)
- Runs daily cold storage offload job

**Running without the watchdog**: set `RUN_WATCHDOG=false` environment variable to disable the embedded watchdog thread. Run the watchdog as a separate process instead.

---

## Models

### User Model

`aiworks_core.User` is the auth user model. It has:

- `email` + `username` (both unique)
- `tier`: `"free"` or `"pro"`
- `email_verified` + `verification_token` for email verification
- `profile_context`: free-text user background
- `llm_generated_background`: cached LLM-synthesised background
- `favorite_session_ids`: JSON list of bookmarked session IDs
- `light_mode`: UI theme preference
- `has_seen_onboarding`: whether the onboarding banner has been dismissed
- `avatar`: URL field
- `extra_data`: JSONField for host-project extensions (host projects store arbitrary user-specific fields here)

### Session Model

`aiworks_core.Session` is the base async task container. Host projects extend it via MTI with additional fields:

- `id`: primary key
- `user`: owner FK
- `session_type`: arbitrary string; host project maps to executors
- `session_title`: non-empty constraint enforced at DB level
- `include_user_context`: whether to include user background in context
- `is_research_needed` / `is_research_online`: whether the session needs research and whether it should use online tools
- `knowledge_bases` / `mcp_servers` / `attached_sessions`: M2M relations for context
- `desktop_tunnel_servers`: JSON mapping tunnel_id → [server_id, ...]
- `is_public` / `is_deleted`: visibility and soft-delete
- `agent_result` / `prev_agent_result`: executor output storage
- `is_agent_finished`: signals completion to frontend polling
- `additional_fields`: JSON field for host-project extensions

### Cold Storage

`AttachedFile.binary_content` stores raw bytes in the DB. The `ColdStorageManager` abstraction supports three backends:

1. **`NoneColdStorageManager`** — no-op (default in settings)
2. **`LocalColdStorageManager`** — stores files on local filesystem at `coldstorage_local_storage_location`
3. **`S3StorageManager`** — stores files in S3 bucket

The daily offload job (`_offload_attached_files_to_cold_storage`) moves `AttachedFile.binary_content` to cold storage and clears the DB field. Reads via `get_attached_file_data()` transparently fall back to cold storage.

### Session Snapshots

Snapshots store a ZIP archive of session state (session fields + files) in cold storage at `session_snapshots/{session_id}/{snapshot_id}.zip`. Key methods:

```python
from aiworks_core.logic.session_snapshot import create_snapshot, restore_snapshot

snapshot = create_snapshot(session)   # creates ZIP in cold storage
restore_snapshot(session, snapshot)    # restores from ZIP
```

---

## REST API Conventions

### Authentication

All endpoints require JWT authentication (Bearer token) except:

- `POST /api/auth/register/`
- `POST /api/auth/login/`
- `POST /api/auth/refresh/`
- `POST /api/auth/password-reset/`
- `POST /api/auth/password-reset/confirm/`
- `POST /api/auth/google/`
- `GET /api/server-settings/`

### URL Routing

Views are DRF ViewSets registered via Django REST Framework's router. Custom endpoints (auth, help-chat, memory, tunnels) are registered as individual URL patterns. All library URLs are under `/api/` when included with `path("api/", include("aiworks_core.urls"))`.

### Response Format

All API responses are JSON. Errors return `{"detail": "..."}` with the appropriate HTTP status code.

---

## Testing

### Running Tests

```bash
cd /path/to/aiworks-core
source .venv/bin/activate
pytest aiworks_core/tests/ -v
```

### Test Settings

Tests use `aiworks_core.tests.settings`. Set via environment or `pytest.ini`:

```ini
[pytest]
DJANGO_SETTINGS_MODULE = aiworks_core.tests.settings
```

Key test settings:
- `RUNNING_TESTS=1` — enables in-memory DB, disables throttling
- `AIWORKS_CORE_PROMPTS_FILE` — points to `aiworks_core/tests/prompts.yaml`
- `AIWORKS_CORE_COLDSTORAGE_TYPE="none"` — disables cold storage
- `EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend"` — captures emails in `mail.outbox`

### Test Utilities

```python
from aiworks_core.tests import (
    _make_user,      # Create a test user
    _make_session,   # Create a test session (session_type="research" by default)
    _auth_header,    # JWT auth header dict for APIClient
)
```

### Mocking invoke_llm

Never patch `async_to_sync` or `sync_to_async`. Instead patch the specific module's `invoke_llm`:

```python
with unittest.mock.patch("aiworks_core.logic.memory.invoke_llm", new_callable=AsyncMock) as mock:
    mock.return_value = "synthesised memory text"
    run_memory_generation_job()
```

For deep agent flows, patch `aiworks_core.logic.deepagent_utils.invoke_llm` (the parent `invoke_llm` call inside `run_deep_agent`).

### Writing Tests

1. Use `_make_user()` and `_make_session()` helpers
2. Use `async_to_sync` wrapper for calling async library functions from sync tests
3. Mock `invoke_llm` using `unittest.mock.patch` on the specific module path
4. URL endpoints use `/api/` prefix
5. Admin endpoints use `/admin/aiworks_core/...`

---

## Key Invariants (Don't Break These)

- **Single LLM choke point** — all LLM calls go through `invoke_llm()`. Never call LLM provider SDKs directly.
- **Atomic task claiming** — always use `select_for_update(skip_locked=True)` inside `transaction.atomic()` when transitioning `WorkerTask` from `pending → running`.
- **`invoke_llm` is async** — use `async_to_sync` wrapper from sync code. Never patch the asgiref bridges.
- **Relative imports** — always relative within the package; absolute only for runtime-resolved strings (mock targets, logger names, Django model references).
- **No inline imports outside views/** — logic files, models, and tests must have all imports at the top level.
- **`AUTH_USER_MODEL = "aiworks_core.User"`** — the User model is tightly coupled; swapping it requires careful migration of all user-dependent logic.
- **Session `session_title` non-empty** — enforced at DB level with a CheckConstraint.
- **Cold storage reads are transparent** — `get_attached_file_data()` and `get_binary_data()` always return bytes regardless of storage backend.
- **Singleton models** — `SiteConfiguration` and `LLMConfiguration` must be accessed via `get_solo()`, not `objects.get()`.
- **Test prompts.yaml** — `aiworks_core/tests/prompts.yaml` is a minimal file; all test operations that call `invoke_llm` must have entries here. Keep it in sync with what tests exercise.

---

## Django Admin

Models are registered in `admin.py`. The admin is primarily for operators/admins to configure:

- `SiteConfiguration` — email, OAuth, memory job, cold storage, CSRF origins
- `LLMConfiguration` — provider, model names, API keys
- `LLMOperationConfig` — per-operation LLM overrides
- `PredefinedMCPServer` — admin-managed MCP server catalogue
- `User` — user management, tier assignment
- `Session` — session inspection
- `WorkerTask` — task state inspection
- `AttachedFile` + `TextAttachedFile` — file inspection
- `KnowledgeBase` + `MCPServer` — resource management

The `AttachedFileAdmin` provides a staff-only download view at `/admin/aiworks_core/attachedfile/<id>/download/`.

---

## Configuration Checklist for New Projects

When integrating aiworks-core into a host project, verify:

- [ ] `aiworks_core` in `INSTALLED_APPS` before project apps
- [ ] `AUTH_USER_MODEL = "aiworks_core.User"` set
- [ ] `AIWORKS_CORE_PROMPTS_FILE` pointing to a valid `prompts.yaml`
- [ ] `JWT_SIGNING_KEY` set to a real secret
- [ ] `ROOT_URLCONF` includes `path("api/", include("aiworks_core.urls"))`
- [ ] `CHANNEL_LAYERS` configured (for WebSocket/tunnel support)
- [ ] `migrate aiworks_core` applied
- [ ] `LLMConfiguration` singleton configured via admin or shell
- [ ] Cold storage settings if using large file attachments
- [ ] Email backend configured in `SiteConfiguration` (or Django `EMAIL_*` settings as fallback)
- [ ] `SiteConfiguration` singleton has `site_url` set for email templates

---

## Forbidden Commands

The following commands must never be run under any circumstances:

- **`git revert`** — do not use; instead, manually undo the change by editing the file back to its previous state
- **`git stash`** — do not use; changes should be committed or discarded explicitly
- **`rm -r`** — **NEVER use recursive delete**. Always delete files individually. Recursive deletes have destroyed workspace directories.
- **`find -delete`** — **NEVER use**. Always delete files individually. Using `find -delete` is as dangerous as `rm -r`.

---

## Iterative Code Review

At the end of each fix or feature implementation, the agent must:

1. Spawn a code review sub-agent to review the changes
2. Address and fix all code review comments
3. Spawn another code review sub-agent to verify the fixes
4. Repeat until the code review returns completely green (no issues)

**Do not skip iterations.** A "mostly green" review is not green. Keep iterating until the reviewer returns a clean report.
