import abc
import logging
import threading
import uuid
from typing import Optional

from django.utils import timezone
from .purge_unused_images import purge_unused_images
from .session_snapshot import create_snapshot
from ..models import WorkerTask, Session
from ..utils import close_connections

logger = logging.getLogger(__name__)


class WorkerExecutor(abc.ABC):
    """Abstract base class for all session executor types."""

    def __init__(self, session_id):
        self.executor_id = uuid.uuid4()
        self.session_id = session_id
        self.task_id = None  # Set at the beginning of run()
        self._current_progress_step: Optional[str] = (
            None  # Tracks the current step for auto-clearing
        )

    def run(self):
        """
        Execute the full session processing lifecycle.

        Designed to be passed as the ``target`` of a daemon
        ``threading.Thread``.  Reads all session data from the DB, writes
        progress and the final result back to the associated
        ``SessionTask`` record, and always closes stale DB connections
        on exit.

        A heartbeat daemon thread updates ``alive_beat`` on the task row
        every 5 seconds so the watchdog in ``apps.py`` can detect crashed
        executors and restart them.
        """

        close_connections()
        stop_heartbeat = threading.Event()

        try:
            try:
                session = Session.objects.select_related("user").get(id=self.session_id)
            except Session.DoesNotExist:
                logger.error(
                    "worker_executor | session=%s | session not found, cannot run",
                    self.session_id,
                )
                return False

            try:
                # noinspection PyUnresolvedReferences
                task = session.worker_task
            except WorkerTask.DoesNotExist:
                logger.error(
                    "worker_executor | session=%s | task not found, cannot run",
                    self.session_id,
                )
                return False

            self.task_id = task.id

            # Atomic claim: only one executor may own a task at a time.
            # The watchdog resets executor_id=None before spawning a thread,
            # so a fresh run can always claim.  If the conditional UPDATE
            # affects 0 rows, another executor already claimed it → bail out.
            claimed = WorkerTask.objects.filter(
                id=self.task_id,
                executor_id__isnull=True,
            ).update(executor_id=self.executor_id)
            if not claimed:
                logger.warning(
                    "worker_executor | session=%s | task=%s | duplicate executor detected, aborting",
                    self.session_id,
                    self.task_id,
                )
                return False
            logger.info(
                "worker_executor | session=%s | task=%s | executor claimed | executor_id=%s",
                self.session_id,
                self.task_id,
                self.executor_id,
            )

            # Start heartbeat: update alive_beat every 5 s so the watchdog
            # can detect a crashed executor (stale alive_beat → dead task).
            def _heartbeat():
                while not stop_heartbeat.wait(5):
                    # noinspection PyBroadException
                    try:
                        WorkerTask.objects.filter(
                            id=self.task_id,
                            executor_id=self.executor_id,
                            status=WorkerTask.STATUS_RUNNING).update(
                            alive_beat=timezone.now()
                        )
                    except Exception:
                        pass
                    finally:
                        close_connections()

            threading.Thread(target=_heartbeat, daemon=True).start()

            success = self.run_inner(session)

            if success:
                # Re-fetch session to capture any field updates made by run_inner
                # before creating the snapshot so the ZIP reflects the latest state.
                session.refresh_from_db()

                try:
                    purge_unused_images(session)
                except Exception:
                    logger.exception("worker_executor | session=%s | purge_unused_images failed", self.session_id)

                self._create_snapshot_if_eligible(session)

                self._update_task(
                    progress_step="",
                    step_tasks=[],
                    status=WorkerTask.STATUS_COMPLETED,
                    completed_at=timezone.now(),
                )
            else:
                self._update_task(status=WorkerTask.STATUS_FAILED)
            return success
        except Exception as error:
            logger.exception(
                "worker_executor | session=%s | unhandled error: %s",
                self.session_id,
                error,
            )
            if self.task_id is not None:
                self._update_task(
                    status=WorkerTask.STATUS_FAILED,
                    error_message=repr(error),
                )
        finally:
            stop_heartbeat.set()
            close_connections()

    @staticmethod
    def _create_snapshot_if_eligible(session) -> None:
        """Create a session snapshot after a successful run (best-effort)."""
        try:
            create_snapshot(session)
        except Exception as exc:
            logger.warning(
                "worker_executor | session=%s | snapshot creation failed (non-fatal): %s",
                session.id,
                exc,
            )

    @abc.abstractmethod
    def run_inner(self, session):
        """Execute the full session processing lifecycle."""

    # ------------------------------------------------------------------
    # Task status helper
    # ------------------------------------------------------------------

    def _update_task(self, **kwargs):
        """Write progress/status fields to the persisted SessionTask row.

        Progress update failures are non-fatal: the session continues so
        it still completes; the SSE stream detects the final status on the
        next poll cycle.

        When ``progress_step`` is included in the update and its value differs
        from the current step, ``step_tasks`` is automatically reset to an
        empty list (unless explicitly provided).  This ensures the previous
        step's task list is cleared whenever the session moves to a new
        phase, but avoids spurious clears on re-entrant updates to the same
        step.
        """
        # Guard: if task was cancelled (e.g. session deleted) while we were
        # running, bail out instead of overwriting the terminal state.
        try:
            task = WorkerTask.objects.get(id=self.task_id)

            if task.executor_id != self.executor_id:
                raise PermissionError("Task is not executed by current execution thread")

            if task.status == WorkerTask.STATUS_FATAL_FAILURE:
                raise PermissionError("Task was cancelled during execution")

            new_progress_step = kwargs.get("progress_step")
            if (
                    new_progress_step is not None
                    and new_progress_step != self._current_progress_step
                    and "step_tasks" not in kwargs
            ):
                kwargs["step_tasks"] = []
                if "thinking" not in kwargs:
                    kwargs["thinking"] = ""
                if "last_tool_call" not in kwargs:
                    kwargs["last_tool_call"] = {}

            if new_progress_step is not None:
                self._current_progress_step = new_progress_step

            logger.info("worker_task | task=%s | update: %s", self.task_id, kwargs)
            try:
                WorkerTask.objects.filter(id=self.task_id, executor_id=self.executor_id).update(**kwargs)
            except Exception as exc:
                logger.warning(
                    "worker_task | task=%s | DB update failed: %s", self.task_id, exc
                )
        finally:
            close_connections()

    def _get_resume_state(self):
        """Get resume_state from the WorkerTask."""
        try:
            task = WorkerTask.objects.get(id=self.task_id)
            return task.resume_state or {}
        except WorkerTask.DoesNotExist:
            raise Exception(f"executor | task=%s | task {self.task_id} does not exist")

    def _update_resume_state(self, updates):
        """Merge updates into the WorkerTask resume_state."""
        try:
            current = self._get_resume_state()
            current.update(updates)
            WorkerTask.objects.filter(id=self.task_id, executor_id=self.executor_id).update(resume_state=current)
        except Exception as e:
            logger.warning("executor | task=%s | failed to update resume_state: %s", self.task_id, e)
