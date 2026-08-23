import atexit
import datetime
import importlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from django.apps import AppConfig
from django.conf import settings
from django.db import (
    OperationalError,
    ProgrammingError,
    transaction,
)
from django.db.backends.signals import connection_created
from django.db.models import Q
from django.db.models.signals import post_delete, post_save, pre_save
from django.utils import timezone
from .utils import chunked, close_connections

logger = logging.getLogger(__name__)

_WATCHDOG_INTERVAL = 30  # seconds between watchdog scans
_ALIVE_BEAT_TIMEOUT = 60  # seconds before a running task is considered dead
_FAILED_TASK_RETRY_DELAY = 60  # seconds to wait before retrying a failed task

# Bounded thread pool — caps concurrent session executions regardless of how many
# sessions are pending. Python's ThreadPoolExecutor uses an unbounded internal queue,
# so we pair it with a BoundedSemaphore to enforce the cap. If the semaphore cannot
# be acquired (all workers busy), the task is left as pending so another node can
# pick it up on the next watchdog cycle.
_POOL_MAX_WORKERS = 10
_executor_pool = ThreadPoolExecutor(max_workers=_POOL_MAX_WORKERS)
_pool_semaphore = threading.BoundedSemaphore(_POOL_MAX_WORKERS)


def _shutdown_executor_pool():
    """Gracefully shutdown the executor pool on process exit.

    Called via atexit.register() below. Threads running tasks will be interrupted
    when the process exits (SIGTERM/SIGINT), which is fine — task state is
    preserved in the DB and the watchdog will restart any incomplete tasks.
    """
    logger.debug("shutting down executor pool (max_workers=%d)", _POOL_MAX_WORKERS)
    _executor_pool.shutdown(wait=False, cancel_futures=True)
    logger.debug("executor pool shutdown complete")


atexit.register(_shutdown_executor_pool)


# ---------------------------------------------------------------------------
# Signal handlers — auto-clear llm_generated_background on data changes
# ---------------------------------------------------------------------------


def _clear_user_background(user_pk: int) -> None:
    """Clear llm_generated_background for the user identified by *user_pk*.

    Uses a targeted UPDATE so that we never trigger further signals and never
    load the full User object unnecessarily.
    """
    from .models import (
        User as _User,
    )  # deferred import to avoid circular deps at module load

    _User.objects.filter(pk=user_pk).update(llm_generated_background="")


def _on_user_pre_save(sender, instance, update_fields=None, **kwargs):
    """Clear the generated background when profile_context is being changed."""
    if instance.pk is None:
        # New user — nothing to clear
        return
    if update_fields is not None and "profile_context" not in update_fields:
        return
    try:
        from .models import User as _User

        previous = (
            _User.objects.filter(pk=instance.pk)
            .values_list("profile_context", flat=True)
            .first()
        )
        if previous is not None and previous != instance.profile_context:
            _clear_user_background(instance.pk)
            instance.llm_generated_background = ""
    except Exception:
        logger.exception(
            "user_background | pre_save signal error for user pk=%s", instance.pk
        )


def _on_memory_entry_change(sender, instance, **kwargs):
    """Clear the generated background whenever a MemoryEntry is created/saved/deleted."""
    try:
        _clear_user_background(instance.user_id)
    except Exception:
        logger.exception(
            "user_background | memory_entry signal error for user pk=%s",
            instance.user_id,
        )


def _on_session_pre_save(sender, instance, update_fields=None, **kwargs):
    """Block updates to deleted sessions. Allows create, restore (is_deleted: True→False), and soft-delete."""
    from .models import Session as _Session

    if instance.pk is None:
        return  # Creating new — allowed

    try:
        old = _Session.objects.get(pk=instance.pk)
    except _Session.DoesNotExist:
        return  # New record — allowed

    if not old.is_deleted:
        return  # Not deleted — allow any save

    # update_fields=None means full form submit (e.g. Django admin) — allow restore
    if update_fields is None:
        if not instance.is_deleted:
            return  # Restore via admin — allowed
        raise PermissionError(
            "Cannot update a deleted session. Restore it via Django admin first."
        )

    # update_fields is set — only allow if is_deleted is the field being changed
    if update_fields == {"is_deleted"}:
        return  # delete or restore — allowed

    raise PermissionError(
        "Cannot update a deleted session. Restore it via Django admin first."
    )


def _submit_work(executor) -> bool:
    """Submit an executor to the thread pool if a slot is available.

    Acquires a semaphore slot before submitting so that the pool never has more
    than _POOL_MAX_WORKERS tasks queued or running at any time.  This is
    necessary because Python's ThreadPoolExecutor uses an unbounded internal
    queue — without the semaphore, tasks would accumulate in memory and every
    node would believe it had claimed the work, defeating the multi-node
    pending-requeue logic.

    The semaphore slot is always released when the executor finishes, even if
    it raises.  If submit() itself raises (e.g. pool already shut down), the
    slot is released immediately and the exception is re-raised so the caller
    can handle it.

    Returns:
        True  — task was accepted and is now running (or queued in the pool).
        False — all slots are busy; the task was not submitted and should be
                left as pending so another node can pick it up on the next
                watchdog cycle.
    """
    if not _pool_semaphore.acquire(blocking=False):
        return False

    def _run(e=executor):
        try:
            e.run()
        finally:
            try:
                close_connections()
            except Exception as exp:
                logger.warning(f"Was unable to release connections, due to {repr(exp)}")

            _pool_semaphore.release()

    try:
        _executor_pool.submit(_run)
    except Exception:
        _pool_semaphore.release()
        raise

    return True


def _pool_available_slots() -> int:
    """Return the number of free slots in the executor pool.

    Reads the semaphore's internal counter, which equals _POOL_MAX_WORKERS minus
    the number of tasks that are currently running or queued for execution.
    Safe to call from the watchdog thread to limit how many tasks are fetched
    from the DB in a single cycle — avoids claiming tasks that cannot be started
    and having to immediately revert them back to pending.
    """
    return _pool_semaphore._value  # type: ignore[attr-defined]


def _get_executor_factory():
    """Resolve the executor factory function from AIWORKS_CORE_EXECUTOR_FACTORY setting.

    The setting value is a "module.path:function_name" string, e.g.
    "myapp.logic.executor_factory:create_executor".

    The resolved function must accept a Session and return a WorkerExecutor instance.
    """
    factory_path = getattr(settings, "AIWORKS_CORE_EXECUTOR_FACTORY", None)
    if not factory_path:
        from .logic.executor_factory import create_executor as _default

        return _default
    module_path, func_name = factory_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


# noinspection PyUnusedLocal
def _enable_sqlite_wal(sender, connection, **kwargs):
    """Set SQLite PRAGMAs to reduce lock contention.

    WAL (Write-Ahead Logging) allows concurrent reads alongside a single writer,
    which dramatically reduces "database is locked" errors when the watchdog thread
    and web workers write at the same time. busy_timeout controls how long a write
    waits for the lock before failing. Has no effect on PostgreSQL.
    """
    if connection.vendor == "sqlite":
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=30000;")


connection_created.connect(_enable_sqlite_wal)


class AiWorksCoreConfig(AppConfig):
    name = "aiworks_core"

    def ready(self):
        """Launch the background watchdog that restarts dead session tasks.

        The watchdog is only started when the process is actually serving HTTP
        requests (``runserver``, gunicorn, uvicorn, …).  Management commands
        such as ``migrate``, ``shell``, or ``collectstatic`` must not trigger
        background executor threads because the interpreter exits immediately
        after the command finishes, causing ``RuntimeError: cannot schedule new
        futures after interpreter shutdown``.

        In deployed environments the ``RUN_WATCHDOG`` environment variable can
        be set to ``false`` to disable the watchdog so that the API process
        only serves HTTP requests while a dedicated execution-worker process
        handles task execution.
        """
        # Register signals unconditionally so they work in all processes
        # (HTTP server, management commands, tests).
        from .models import Session, User, MemoryEntry

        pre_save.connect(_on_user_pre_save, sender=User, weak=False)
        post_save.connect(_on_memory_entry_change, sender=MemoryEntry, weak=False)
        post_delete.connect(_on_memory_entry_change, sender=MemoryEntry, weak=False)

        # Connect to Session and all its subclasses (MTI children in host apps)
        pre_save.connect(_on_session_pre_save, sender=Session, weak=False)
        from django.apps import apps

        for model in apps.get_models():
            if model is not Session and issubclass(model, Session):
                pre_save.connect(_on_session_pre_save, sender=model, weak=False)

        if not self._is_http_server_process():
            return
        if os.environ.get("RUN_WATCHDOG", "true").lower() == "false":
            return

        t = threading.Thread(target=self._watchdog, daemon=True)
        t.start()

    @staticmethod
    def _is_http_server_process():
        """Return True when the current process is serving HTTP requests.

        Checks ``sys.argv`` to distinguish server entry points (``runserver``,
        gunicorn, uvicorn) from management commands (``migrate``, ``shell``,
        ``collectstatic``, etc.).
        """
        argv0 = sys.argv[0] if sys.argv else ""
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""

        return (
            cmd == "runserver"  # Django development server
            or cmd == "runserver_ws"  # Daphne ASGI server
            or "gunicorn" in argv0  # Gunicorn WSGI server
            or "uvicorn" in argv0  # Uvicorn ASGI server
            or "daphne" in argv0  # Daphne ASGI server
            or "hypercorn" in argv0  # Hypercorn ASGI server
        )

    @staticmethod
    def _watchdog():
        """Continuously monitor for orphaned or dead session tasks and restart them.

        The watchdog runs forever in a daemon thread, sleeping
        ``_WATCHDOG_INTERVAL`` seconds between iterations.  On each pass it
        looks for:

        * ``pending`` tasks — created but never picked up (e.g. the server
          crashed between ``get_or_create`` and the executor thread starting).
        * ``running`` tasks whose ``alive_beat`` is stale (> ``_ALIVE_BEAT_TIMEOUT``
          seconds ago) or has never been set — the executor died without
          updating its heartbeat.

        ``select_for_update(skip_locked=True)`` inside an ``atomic()`` block
        ensures each task is claimed by at most one node when multiple workers
        run simultaneously.  Claimed tasks are set directly to ``running`` with
        a fresh ``alive_beat`` inside the same transaction so that other
        watchdog nodes cannot re-pick them before the new executor thread has
        started its own heartbeat.
        """

        # Small initial delay so Django finishes setup before the first scan.
        time.sleep(5)

        while True:
            try:
                AiWorksCoreConfig._run_watchdog_step()
            except Exception as exp:
                logger.exception("watchdog | error: %s", exp)

            time.sleep(_WATCHDOG_INTERVAL)

    @staticmethod
    def _run_watchdog_step():
        try:
            close_connections()

            try:
                AiWorksCoreConfig._handle_failed_sessions()
            except Exception as exp:
                logger.warning("watchdog | error handling failed sessions: %s", exp)

            try:
                AiWorksCoreConfig._handle_pending_sessions()
            except Exception as exp:
                logger.warning("watchdog | error handling pending sessions: %s", exp)

            try:
                AiWorksCoreConfig._handle_tokens()
            except Exception as exp:
                logger.warning("watchdog | error handling tokens: %s", exp)

            try:
                AiWorksCoreConfig._handle_memory_generation()
            except Exception as exp:
                logger.warning("watchdog | error handling memory generation: %s", exp)

            try:
                AiWorksCoreConfig._purge_deleted_sessions()
            except Exception as exp:
                logger.warning("watchdog | error purging deleted sessions: %s", exp)

            try:
                AiWorksCoreConfig._offload_attached_files_to_cold_storage()
            except Exception as exp:
                logger.warning(
                    "watchdog | error offloading attached files: %s", exp
                )

            try:
                AiWorksCoreConfig._purge_deleted_attached_files()
            except Exception as exp:
                logger.warning(
                    "watchdog | error purging deleted attached files: %s", exp
                )

            try:
                AiWorksCoreConfig._run_watchdog_hooks()
            except Exception as exp:
                logger.warning("watchdog | error running hooks: %s", exp)

        except (OperationalError, ProgrammingError):
            # DB tables may not exist yet (e.g., before migrations run)
            logger.warning("watchdog | database not ready, will retry...")
            pass
        except Exception as exc:
            logger.exception("watchdog | error: %s", exc)
        finally:
            close_connections()

    @staticmethod
    def _run_watchdog_hooks():
        """Run optional watchdog hooks registered via AIWORKS_CORE_WATCHDOG_HOOKS setting.

        Each hook is a "module.path:function_name" string. Hooks are called after all
        core watchdog steps, with exceptions caught and logged so a failing hook cannot
        break the watchdog cycle.
        """
        hooks = getattr(settings, "AIWORKS_CORE_WATCHDOG_HOOKS", []) or []
        for hook_path in hooks:
            try:
                module_path, func_name = hook_path.rsplit(":", 1)
                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name)
                func()
            except Exception as exc:
                logger.warning("watchdog hook %s failed: %s", hook_path, exc)

    @staticmethod
    def _handle_tokens():
        """Delete expired VerificationTokens (email verification / password reset)"""
        from .models import VerificationToken

        deleted_vt, _ = VerificationToken.objects.filter(
            expires_at__lt=timezone.now()
        ).delete()
        if deleted_vt:
            logger.info(
                "watchdog | deleted %d expired verification token(s)", deleted_vt
            )

        # Flush expired SimpleJWT outstanding tokens (and cascaded blacklisted tokens)
        from rest_framework_simplejwt.token_blacklist.models import OutstandingToken

        deleted_jwt, _ = OutstandingToken.objects.filter(
            expires_at__lte=timezone.now()
        ).delete()
        if deleted_jwt:
            logger.info(
                "watchdog | deleted %d expired outstanding JWT token(s)", deleted_jwt
            )

    @staticmethod
    def _purge_deleted_sessions():
        """Hard-delete sessions that have been soft-deleted (is_deleted=True) for more than purge_session_cutoff_days."""
        from .models import Session
        from .logic.session_snapshot import SNAPSHOTS_FOLDER
        from .logic.cold_storage import ColdStorageManager
        from .models import SiteConfiguration

        config = SiteConfiguration.get_solo()
        cutoff_days = config.purge_session_cutoff_days
        if cutoff_days <= 0:
            return

        cutoff = timezone.now() - datetime.timedelta(days=cutoff_days)
        deleted_sessions = list(
            Session.objects.filter(
                is_deleted=True,
                updated_at__lt=cutoff,
            ).only("id")
        )
        if not deleted_sessions:
            return

        purged_count = 0
        for session in deleted_sessions:
            try:
                if ColdStorageManager.is_enabled():
                    try:
                        manager = ColdStorageManager.get_instance()
                        folder = f"{SNAPSHOTS_FOLDER}/{session.id}"
                        for file_name in manager.list(folder):
                            manager.remove(folder, file_name)
                    except Exception as exp:
                        logger.exception(
                            "watchdog | failed to purge session snapshots from cold storage=%s: %s",
                            session.id,
                            exp,
                        )

                Session.objects.filter(id=session.id, is_deleted=True).delete()
                purged_count += 1
            except Exception as exp:
                logger.exception(
                    "watchdog | failed to purge deleted session=%s: %s",
                    session.id,
                    exp,
                )

        if purged_count:
            logger.info("watchdog | purged %d deleted session(s)", purged_count)

    @staticmethod
    def _handle_failed_sessions():
        """Reset failed tasks that are eligible for retry, or mark them as fatally failed."""
        from .models import WorkerTask

        now = timezone.now()
        retry_threshold = now - datetime.timedelta(seconds=_FAILED_TASK_RETRY_DELAY)
        with transaction.atomic():
            # Find failed tasks eligible for retry (alive_beat or started_at
            # older than retry_threshold, so we wait at least
            # _FAILED_TASK_RETRY_DELAY seconds before retrying).
            # This runs first so that tasks reset to pending are immediately
            # picked up by _handle_pending_sessions in the same cycle.
            failed_tasks = list(
                WorkerTask.objects.select_for_update(skip_locked=True)
                .filter(
                    Q(
                        status=WorkerTask.STATUS_FAILED,
                        alive_beat__lt=retry_threshold,
                    )
                    | Q(
                        status=WorkerTask.STATUS_FAILED,
                        alive_beat__isnull=True,
                        started_at__lt=retry_threshold,
                    )
                )
                .filter(session__is_deleted=False)
                .select_related("session")
            )
            for task in failed_tasks:
                if task.retry_count < WorkerTask.MAX_RETRIES:
                    WorkerTask.objects.filter(id=task.id).update(
                        status=WorkerTask.STATUS_PENDING,
                        retry_count=task.retry_count + 1,
                        progress_step="",
                        error_message="",
                        step_tasks=[],
                        alive_beat=None,
                        completed_at=None,
                        thinking="",
                        last_tool_call={},
                    )
                    logger.info(
                        "watchdog | retrying failed task=%s session=%s retry_count=%d",
                        task.id,
                        task.session.id,
                        task.retry_count + 1,
                    )
                else:
                    WorkerTask.objects.filter(id=task.id).update(
                        status=WorkerTask.STATUS_FATAL_FAILURE,
                    )
                    logger.info(
                        "watchdog | fatal failure task=%s session=%s retry_count=%d",
                        task.id,
                        task.session.id,
                        task.retry_count,
                    )

    @staticmethod
    def _handle_pending_sessions():
        """Pick up pending or stale-running tasks and restart their executor threads."""
        from .models import WorkerTask

        create_executor = _get_executor_factory()
        available_slots = _pool_available_slots()
        if available_slots == 0:
            return

        now = timezone.now()
        stale_threshold = now - datetime.timedelta(seconds=_ALIVE_BEAT_TIMEOUT)
        with transaction.atomic():
            dead_tasks = list(
                WorkerTask.objects.select_for_update(skip_locked=True)
                .filter(
                    Q(status=WorkerTask.STATUS_PENDING)
                    | Q(
                        status=WorkerTask.STATUS_RUNNING,
                        alive_beat__lt=stale_threshold,
                    )
                    | Q(
                        status=WorkerTask.STATUS_RUNNING,
                        alive_beat__isnull=True,
                        started_at__lt=stale_threshold,
                    )
                )
                .filter(session__is_deleted=False)
                .select_related("session")[:available_slots]
            )

            tasks_to_run = []
            for task in dead_tasks:
                if task.session.desktop_tunnel_servers:
                    # Note: each tunnel_id results in a DB query here. In normal
                    # usage a user has at most a handful of active tunnel connections
                    # (one per desktop machine), so this is not a performance concern.
                    # If that assumption ever breaks, batch the is_connected checks
                    # into a single query via Tunnel.objects.filter(tunnel_id__in=...).
                    from .logic.mcp_tunnel import MCPTunnelManager

                    tunnel_ids = list(task.session.desktop_tunnel_servers.keys())
                    all_connected = all(
                        MCPTunnelManager.is_connected(tid)
                        for tid in tunnel_ids
                    )
                    if not all_connected:
                        logger.info(
                            "watchdog | skipping session %s — not all desktop tunnels are connected",
                            task.session.id,
                        )
                        continue
                tasks_to_run.append(task)

            # Atomically claim each dead task by marking it as running
            # with a fresh alive_beat.  This prevents other watchdog
            # threads/processes from re-picking the same task in the
            # window between this transaction committing and the new
            # executor thread stamping its own heartbeat.
            for task in tasks_to_run:
                WorkerTask.objects.filter(id=task.id).update(
                    status=WorkerTask.STATUS_RUNNING,
                    alive_beat=now,
                    progress_step="",
                    error_message="",
                    executor_id=None,
                    thinking="",
                    last_tool_call={},
                )

        for task in tasks_to_run:
            session = task.session
            logger.info(
                "watchdog | restarting task=%s session=%s prev_status=%s",
                task.id,
                session.id,
                task.status,
            )
            executor = create_executor(task.session)
            if not _submit_work(executor):
                # Pool is at capacity — revert to pending so watchdog can pick it up later
                WorkerTask.objects.filter(id=task.id).update(
                    status=WorkerTask.STATUS_PENDING,
                    alive_beat=None,
                )
                logger.warning(
                    "watchdog | task=%s session=%s rejected (pool full), set back to pending",
                    task.id,
                    session.id,
                )

    @staticmethod
    def _run_job_and_update(job_name, job_run_method, last_run_field):
        from .models import SiteConfiguration

        cfg = SiteConfiguration.get_solo()

        # noinspection PyBroadException
        def _thread_execution():
            try:
                start_time = timezone.now()

                try:
                    logger.info(f"watchdog | starting {job_name} job")
                    job_run_method()
                    logger.info(f"watchdog | {job_name} job finished successfully")
                except Exception:
                    logger.exception(f"watchdog | error running {job_name} job")
                    return

                with transaction.atomic():
                    SiteConfiguration.objects.filter(pk=cfg.pk).update(
                        **{last_run_field: start_time}
                    )
            finally:
                close_connections()

        # Start the execution in a new thread
        thread = threading.Thread(target=_thread_execution)
        thread.start()

    @staticmethod
    def _should_memory_worker_run(cfg=None):
        if not cfg.memory_generation_enabled:
            return False

        now = timezone.now()
        if cfg.memory_last_run and (now - cfg.memory_last_run).total_seconds() < 86400:
            return False
        if cfg.mem_worker_start and (now - cfg.mem_worker_start).total_seconds() < 3600:
            logger.debug(
                "watchdog | memory generation already started on another node (started %s), skipping",
                cfg.mem_worker_start,
            )
            return False

        return True

    @staticmethod
    def _handle_memory_generation():
        """Run the daily memory generation job if enabled and due.

        Uses a dedicated "started at" field on SiteConfiguration as a lightweight
        claim marker. If another worker set it within the last hour, this node
        skips the run. The winning worker updates ``memory_last_run`` only on
        success; a crashed worker leaves ``mem_worker_start`` set so subsequent
        runs are deferred for up to 1 hour rather than silently skipping work.
        """
        from .models import SiteConfiguration
        from .logic.memory import run_memory_generation_job, run_user_background_calculation

        cfg = SiteConfiguration.get_solo()

        if not AiWorksCoreConfig._should_memory_worker_run(cfg):
            return

        try:
            with transaction.atomic():
                cfg = SiteConfiguration.objects.select_for_update(nowait=True).get(
                    pk=cfg.pk
                )
                if not AiWorksCoreConfig._should_memory_worker_run(cfg):
                    return

                since_dt = cfg.memory_last_run or timezone.make_aware(
                    datetime.datetime(1970, 1, 1)
                )
                SiteConfiguration.objects.filter(pk=cfg.pk).update(
                    mem_worker_start=timezone.now(),
                )
        except OperationalError:
            logger.debug(
                "watchdog | memory generation already running on another node, skipping"
            )
            return

        def _run_memory_job():
            run_memory_generation_job(since_dt)

            # Rerun memory calculation for all users to update their backgrounds based on new memories and profile context changes.
            run_user_background_calculation()

        AiWorksCoreConfig._run_job_and_update(
            "run_memory_job", _run_memory_job, "memory_last_run"
        )

    @staticmethod
    def _should_offload_run(cfg) -> bool:
        """Check if the daily offload job should run.

        Runs once per calendar day. Uses offload_worker_start as a claim marker —
        if another worker set it within the last hour, skip.
        """
        now = timezone.now()
        if cfg.offload_last_run and cfg.offload_last_run.date() == now.date():
            return False
        if (
            cfg.offload_worker_start
            and (now - cfg.offload_worker_start).total_seconds() < 3600
        ):
            logger.debug(
                "watchdog | offload already started on another node (started %s), skipping",
                cfg.offload_worker_start,
            )
            return False
        return True

    @staticmethod
    def _offload_attached_files_to_cold_storage():
        """Daily job: move AttachedFile.binary_content to cold storage and clear DB.

        Runs once per calendar day. Skips entirely if cold storage is disabled.
        Uses SELECT FOR UPDATE NOWAIT for 1-hour exclusivity claim.
        """
        from .models import SiteConfiguration
        from .logic.files import offload_to_cold_storage

        cfg = SiteConfiguration.get_solo()
        if not AiWorksCoreConfig._should_offload_run(cfg):
            return

        try:
            with transaction.atomic():
                cfg = SiteConfiguration.objects.select_for_update(nowait=True).get(
                    pk=cfg.pk
                )
                if not AiWorksCoreConfig._should_offload_run(cfg):
                    return
                SiteConfiguration.objects.filter(pk=cfg.pk).update(
                    offload_worker_start=timezone.now(),
                )
        except OperationalError:
            logger.debug(
                "watchdog | offload already running on another node, skipping"
            )
            return

        def run_offload_job():
            offload_to_cold_storage()

        AiWorksCoreConfig._run_job_and_update(
            "offload_to_cold_storage", run_offload_job, "offload_last_run"
        )

    @staticmethod
    def _purge_deleted_attached_files() -> int:
        """Remove orphan cold-storage binaries that have no corresponding DB record.

        Iterates over all files in cold storage in chunks of 1000.  For each chunk,
        collects the integer IDs, queries the DB for matching ``AttachedFile`` PKs,
        and removes any cold-storage file whose ID is not present in the DB — these
        are true orphans caused by hard-delete of the parent Session via CASCADE,
        or binaries stored without a corresponding DB record.

        Best-effort: errors are logged but never raised, so failures do not stop
        the purge.

        Returns:
            The number of binary objects actually removed from cold storage.
        """
        from .logic.cold_storage import ColdStorageManager, ATTACHED_FILES_FOLDER
        from .models import AttachedFile

        if not ColdStorageManager.is_enabled():
            return 0

        manager = ColdStorageManager.get_instance()

        try:

            chunk_size = 1000
            removed = 0

            for batch in chunked(manager.list(ATTACHED_FILES_FOLDER), chunk_size):
                cold_ids: set[int] = set()
                for name in batch:
                    try:
                        cold_ids.add(int(name))
                    except (ValueError, TypeError):
                        logger.debug("cold_storage.purge_deleted | invalid file name in cold storage attached files folder, ignoring: %s", name)

                if not cold_ids:
                    continue

                existing_in_db: set[int] = set(
                    AttachedFile.objects.filter(pk__in=cold_ids).values_list("pk", flat=True)
                )
                orphan_ids = cold_ids - existing_in_db

                for pk in orphan_ids:
                    try:
                        if manager.exists(ATTACHED_FILES_FOLDER, str(pk)):
                            manager.remove(ATTACHED_FILES_FOLDER, str(pk))
                            removed += 1
                            logger.debug(
                                "cold_storage.purge_deleted | removed orphan id=%s from cold storage",
                                pk,
                            )
                    except Exception as exp:
                        logger.exception(
                            "cold_storage.purge_deleted | failed to remove id=%s from cold storage | exp=%s",
                            pk,
                            repr(exp)
                        )

            if removed:
                logger.info(
                    "cold_storage.purge_deleted | removed %d binary object(s) from cold storage",
                    removed,
                )
            return removed

        except Exception:
            logger.exception("cold_storage.purge_deleted | unexpected error")
            return 0
