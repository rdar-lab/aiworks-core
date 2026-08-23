# aiworks-core

A self-contained Django library for AI-powered features: LLM orchestration, knowledge management, MCP tool integration, memory generation, and more.

---

## What is aiworks-core?

aiworks-core is a Django app you drop into any project to get a suite of AI features out of the box:

- **Knowledge Bases + RAG** — Upload documents (PDF, DOCX, Excel, PPT) into knowledge bases. Vector search via ChromaDB for context retrieval.
- **MCP Tool Integration** — Connect to Model Context Protocol (MCP) servers. Supports HTTP servers, OpenAPI specs, and OAuth authentication flows.
- **Memory Generation** — Daily background job extracts insights from session history into user memory entries. Synthesises a persistent user background.
- **Deep Agent Execution** — Async background tasks with quality gates, recovery loops, and structured output (JSON, markdown, files).
- **Session Snapshots** — Capture and restore session state for reproducible reruns.
- **Cold Storage** — Offload large file blobs to local disk or AWS S3 to keep the database lean.
- **Email Service** — Transactional emails (verification, password reset) with multipart HTML/plain-text rendering.
- **Desktop MCP Tunnel** — WebSocket-based tunnel for desktop CLI tools to expose local MCP servers to the cloud app.

---

## Quick Start

```bash
pip install aiworks-core
```

### 1. Django Settings

```python
INSTALLED_APPS = [
    "aiworks_core",   # ← must come before your apps
    "rest_framework",
    "rest_framework_simplejwt",
    "channels",
    # ...
]

AUTH_USER_MODEL = "aiworks_core.User"
ROOT_URLCONF = "myproject.urls"

AIWORKS_CORE_PROMPTS_FILE = "/path/to/prompts.yaml"
JWT_SIGNING_KEY = "replace-with-a-real-secret"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}
```

### 2. URL Configuration

```python
# urls.py
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("aiworks_core.urls")),
]
```

### 3. Apply Migrations

```bash
python manage.py migrate aiworks_core
```

### 4. Configure LLM Provider

```python
# Django shell: python manage.py shell
from aiworks_core.models import LLMConfiguration
cfg = LLMConfiguration.get_solo()
cfg.default_provider = "openai"
cfg.openai_api_key = "sk-..."
cfg.save()
```

All endpoints require JWT authentication. See [MANUAL.md](MANUAL.md) for complete API documentation.

---

## Features

| Feature | Description |
|---------|-------------|
| **User model** | Email+username auth, daily usage limits per tier, profile context, memory |
| **Knowledge Bases** | Named collections of files with RAG vector search |
| **MCP Servers** | Connect HTTP/OpenAPI MCP servers with OAuth, bearer, or basic auth |
| **Sessions** | General-purpose session container for async agent tasks |
| **Memory** | Daily extraction of insights from sessions into user memory |
| **Cold Storage** | Offload AttachedFile blobs to local disk or S3 |
| **Session Snapshots** | ZIP archive of session state for restore/rerun |
| **Email** | Verification, password reset, multipart HTML+plain |
| **Help Chat** | LLM-powered help assistant with manual content injection |
| **Desktop Tunnel** | WebSocket tunnel for desktop MCP tool bridging |

---

## Architecture Highlights

- **Single LLM choke point** — All LLM calls go through `invoke_llm()` with per-operation config, retries, and prompt resolution.
- **Bounded execution pool** — Concurrency-limited thread pool prevents runaway resource usage.
- **Atomic task claiming** — `SELECT FOR UPDATE SKIP LOCKED` ensures only one worker picks up a task.
- **Quality gates** — Deep agent outputs pass through an LLM QA check before being accepted.
- **Watchdog thread** — Background scanner handles pending tasks, retries, scheduled reruns, and cleanup.
- **Cold storage abstraction** — File blobs transparently migrate to cold storage; reads fall back automatically.

---

## Documentation

Detailed documentation is in [MANUAL.md](MANUAL.md), covering:

- Every model and all fields
- Every REST API endpoint (request/response format)
- All service layers (`invoke_llm`, cold storage, RAG, email, etc.)
- Background worker / watchdog behaviour
- Testing utilities and test settings
- Example integration into a new project

---

## Requirements

- Python 3.11+
- Django 5+
- Django REST Framework
- `rest_framework_simplejwt`
- Channels (optional, for WebSocket/tunnel support)
- ChromaDB (optional, for RAG)

See `requirements.txt` for the full pinned dependency list.
