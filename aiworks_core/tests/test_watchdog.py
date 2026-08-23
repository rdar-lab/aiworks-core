"""
Watchdog tests for the Ai-Works Core API.
"""

import datetime
import shutil
import sys
import tempfile
import uuid
from datetime import timedelta
from unittest.mock import patch

from django.apps import apps as django_apps
from django.test import TestCase
from django.utils import timezone
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
from . import _make_session, _make_user
from .test_cold_storage import _set_coldstorage_config
from ..apps import AiWorksCoreConfig
from ..logic.cold_storage import ColdStorageManager, store_attached_file, ATTACHED_FILES_FOLDER
from ..models import (
    MemoryEntry,
    Session,
    SiteConfiguration,
    VerificationToken,
    WorkerTask, AttachedFile,
)


class WatchdogTokenCleanupTests(TestCase):
    """Watchdog step deletes expired VerificationTokens and flushes expired JWT tokens."""

    def setUp(self):
        self.user = _make_user(username="wdog_user", password="pass")
        now = timezone.now()
        # Expired token
        self.expired_token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_EMAIL,
            expires_at=now - timedelta(hours=1),
        )
        # Valid token
        self.valid_token = VerificationToken.objects.create(
            user=self.user,
            token_type=VerificationToken.TOKEN_TYPE_PASSWORD_RESET,
            expires_at=now + timedelta(hours=1),
        )

    def test_watchdog_deletes_expired_verification_tokens(self):
        AiWorksCoreConfig._run_watchdog_step()
        self.assertFalse(
            VerificationToken.objects.filter(pk=self.expired_token.pk).exists()
        )

    def test_watchdog_preserves_valid_verification_tokens(self):
        AiWorksCoreConfig._run_watchdog_step()
        self.assertTrue(
            VerificationToken.objects.filter(pk=self.valid_token.pk).exists()
        )

    def test_watchdog_flushes_expired_jwt_tokens(self):
        # Create an expired outstanding token directly in the DB
        OutstandingToken.objects.create(
            user=self.user,
            jti="test-jti-expired",
            token="dummy.token.value",
            created_at=timezone.now() - timedelta(days=10),
            expires_at=timezone.now() - timedelta(days=3),
        )
        AiWorksCoreConfig._run_watchdog_step()
        self.assertFalse(
            OutstandingToken.objects.filter(jti="test-jti-expired").exists()
        )

    def test_watchdog_preserves_valid_jwt_tokens(self):
        # Create a valid (non-expired) outstanding token directly in the DB
        OutstandingToken.objects.create(
            user=self.user,
            jti="test-jti-valid",
            token="dummy.token.value",
            created_at=timezone.now() - timedelta(days=1),
            expires_at=timezone.now() + timedelta(days=3),
        )
        AiWorksCoreConfig._run_watchdog_step()
        self.assertTrue(OutstandingToken.objects.filter(jti="test-jti-valid").exists())


class WatchdogReadyServerDetectionTests(TestCase):
    """AiWorksCoreConfig.ready() must only start the watchdog when the process is serving HTTP requests."""

    @staticmethod
    def _is_server(argv):
        original_argv = sys.argv
        try:
            sys.argv = argv
            return AiWorksCoreConfig._is_http_server_process()
        finally:
            sys.argv = original_argv

    @staticmethod
    def _thread_started(argv):
        original_argv = sys.argv
        try:
            sys.argv = argv
            with patch("threading.Thread") as mock_thread:
                django_apps.get_app_config("aiworks_core").ready()
                if mock_thread.call_count == 0:
                    return False
                mock_thread.return_value.start.assert_called_once()
                return True
        finally:
            sys.argv = original_argv

    # --- _is_http_server_process detection ---

    def test_detects_runserver(self):
        self.assertTrue(self._is_server(["manage.py", "runserver"]))

    def test_detects_gunicorn(self):
        self.assertTrue(
            self._is_server(
                ["/usr/local/bin/gunicorn", "aiworks_core.wsgi:application"]
            )
        )

    def test_detects_uvicorn(self):
        self.assertTrue(
            self._is_server(
                ["/usr/local/bin/uvicorn", "aiworks_core.asgi:application"]
            )
        )

    def test_rejects_migrate(self):
        self.assertFalse(self._is_server(["manage.py", "migrate"]))

    def test_rejects_makemigrations(self):
        self.assertFalse(self._is_server(["manage.py", "makemigrations"]))

    def test_rejects_shell(self):
        self.assertFalse(self._is_server(["manage.py", "shell"]))

    def test_rejects_collectstatic(self):
        self.assertFalse(self._is_server(["manage.py", "collectstatic"]))

    # --- ready() starts (or skips) the thread ---

    def test_ready_starts_and_runs_thread_for_runserver(self):
        self.assertTrue(self._thread_started(["manage.py", "runserver"]))

    def test_ready_starts_and_runs_thread_for_gunicorn(self):
        self.assertTrue(
            self._thread_started(
                ["/usr/local/bin/gunicorn", "aiworks_core.wsgi:application"]
            )
        )

    def test_ready_starts_and_runs_thread_for_uvicorn(self):
        self.assertTrue(
            self._thread_started(
                ["/usr/local/bin/uvicorn", "aiworks_core.asgi:application"]
            )
        )

    def test_ready_skips_thread_for_migrate(self):
        self.assertFalse(self._thread_started(["manage.py", "migrate"]))


class WatchdogSessionRetryTests(TestCase):
    """Tests for the watchdog's session failure retry and fatal-failure escalation."""

    def setUp(self):
        self.user = _make_user()
        self.session = _make_session(self.user, "sess_wd")
        # Timestamp old enough to satisfy the retry threshold (>60 s ago).
        self._old_ts = timezone.now() - timedelta(seconds=65)

    def _make_failed_task(self, retry_count=0, **kwargs):
        """Create a failed task whose alive_beat is past the retry threshold."""
        task = WorkerTask.objects.create(
            session=self.session,
            status=WorkerTask.STATUS_FAILED,
            retry_count=retry_count,
            error_message="Some error",
            progress_step="Some step",
            step_tasks=["step1"],
            alive_beat=self._old_ts,
            **kwargs,
        )
        return task

    def test_failed_task_below_max_retries_is_reset_to_pending_then_running(self):
        """A failed task with retry_count < MAX_RETRIES is reset to pending
        and immediately claimed as running in the same watchdog cycle."""
        task = self._make_failed_task(retry_count=0)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        # Pending → running happens in the same cycle (failed block runs first).
        self.assertEqual(task.status, WorkerTask.STATUS_RUNNING)

    def test_failed_task_retry_count_incremented(self):
        task = self._make_failed_task(retry_count=0)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.retry_count, 1)

    def test_second_retry_increments_retry_count_to_two(self):
        task = self._make_failed_task(retry_count=1)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.retry_count, 2)

    def test_failed_task_fields_cleared_on_retry(self):
        """progress_step, error_message, step_tasks, and completed_at are cleared on retry."""
        task = self._make_failed_task(retry_count=0)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.progress_step, "")
        self.assertEqual(task.error_message, "")
        self.assertEqual(task.step_tasks, [])
        self.assertIsNone(task.completed_at)

    def test_failed_task_at_max_retries_escalated_to_fatal_failure(self):
        task = self._make_failed_task(retry_count=WorkerTask.MAX_RETRIES)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.status, WorkerTask.STATUS_FATAL_FAILURE)

    def test_fatal_failure_task_is_fatal(self):
        """Task escalated to fatal_failure has is_fatal == True."""
        task = self._make_failed_task(retry_count=WorkerTask.MAX_RETRIES)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertTrue(task.is_fatal)

    def test_failed_task_not_yet_past_retry_threshold_is_skipped(self):
        """Failed task with a recent alive_beat is not retried before the delay elapses."""
        task = WorkerTask.objects.create(
            session=self.session,
            status=WorkerTask.STATUS_FAILED,
            retry_count=0,
            alive_beat=timezone.now(),  # just now — not yet eligible
        )
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.status, WorkerTask.STATUS_FAILED)
        self.assertEqual(task.retry_count, 0)

    def test_failed_task_null_alive_beat_old_started_at_is_retried(self):
        """A failed task with alive_beat=None is retried when started_at is past the threshold."""
        task = WorkerTask.objects.create(
            session=self.session,
            status=WorkerTask.STATUS_FAILED,
            retry_count=0,
            alive_beat=None,
        )
        # Force started_at to be old enough (auto_now_add requires an ORM update).
        WorkerTask.objects.filter(id=task.id).update(started_at=self._old_ts)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertNotEqual(task.status, WorkerTask.STATUS_FAILED)
        self.assertEqual(task.retry_count, 1)

    def test_fatal_failure_task_not_modified_by_watchdog(self):
        """A task in fatal_failure state is terminal and not touched by the watchdog."""
        task = WorkerTask.objects.create(
            session=self.session,
            status=WorkerTask.STATUS_FATAL_FAILURE,
            retry_count=WorkerTask.MAX_RETRIES,
            alive_beat=self._old_ts,
        )
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.status, WorkerTask.STATUS_FATAL_FAILURE)
        self.assertEqual(task.retry_count, WorkerTask.MAX_RETRIES)

    def test_retried_task_immediately_picked_up_in_same_cycle(self):
        """Tasks reset to pending by the retry block are claimed as running
        by the dead_task block within the same watchdog call."""
        task = self._make_failed_task(retry_count=0)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        # If the dead_task block ran after the retry block in the same transaction,
        # the task transitions all the way to running (not just pending).
        self.assertEqual(task.status, WorkerTask.STATUS_RUNNING)

    def test_failed_task_with_retry_count_two_still_retried(self):
        """retry_count=2 is still below MAX_RETRIES=3, so it should be retried."""
        task = self._make_failed_task(retry_count=2)
        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()
        task.refresh_from_db()
        self.assertEqual(task.retry_count, 3)
        # At this point it's been reset to running (not yet fatal_failure).
        self.assertEqual(task.status, WorkerTask.STATUS_RUNNING)


class WatchdogPurgeDeletedSessionsTests(TestCase):
    """Tests for the watchdog purging of soft-deleted sessions after 30 days."""

    def setUp(self):
        self.user = _make_user(username="purge_wdog", email="purge@example.com")
        SiteConfiguration.objects.all().delete()
        self.cfg = SiteConfiguration.get_solo()
        self.cfg.purge_session_cutoff_days = 30
        self.cfg.save()

    def _make_deleted_session(self, updated_at_days_ago):
        """Create a soft-deleted session with updated_at set to the given days in the past."""
        session = _make_session(
            self.user,
            session_id=f"purge_{uuid.uuid4().hex[:8]}",
        )
        session.is_deleted = True
        session.save(update_fields=["is_deleted"])
        old_date = timezone.now() - datetime.timedelta(days=updated_at_days_ago)
        Session.objects.filter(pk=session.pk).update(updated_at=old_date)
        session.refresh_from_db()
        return session

    def test_watchdog_purges_deleted_sessions_older_than_30_days(self):
        """Sessions soft-deleted > 30 days ago are hard-deleted by the watchdog."""
        old_session = self._make_deleted_session(updated_at_days_ago=31)
        recent_session = self._make_deleted_session(updated_at_days_ago=29)

        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()

        self.assertFalse(Session.objects.filter(pk=old_session.pk).exists())
        self.assertTrue(Session.objects.filter(pk=recent_session.pk).exists())

    def test_watchdog_does_not_purge_recently_deleted_sessions(self):
        """Sessions soft-deleted < 30 days ago are preserved."""
        session = self._make_deleted_session(updated_at_days_ago=15)

        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()

        self.assertTrue(Session.objects.filter(pk=session.pk).exists())

    def test_watchdog_does_not_purge_non_deleted_sessions(self):
        """Active (non-deleted) sessions are never affected by purge."""
        active_session = _make_session(
            self.user,
            session_id=f"active_{uuid.uuid4().hex[:8]}",
        )
        old_date = timezone.now() - datetime.timedelta(days=60)
        Session.objects.filter(pk=active_session.pk).update(updated_at=old_date)

        with patch("aiworks_core.apps.threading.Thread"):
            AiWorksCoreConfig._run_watchdog_step()

        self.assertTrue(Session.objects.filter(pk=active_session.pk).exists())


class WatchdogMemoryAndBackgroundRecalculationTests(TestCase):
    """Watchdog runs memory generation followed by background recalculation in one flow."""

    def setUp(self):
        SiteConfiguration.objects.all().delete()
        self.cfg = SiteConfiguration.get_solo()
        self.cfg.memory_generation_enabled = True
        self.cfg.memory_last_run = None
        self.cfg.mem_worker_start = None
        self.cfg.save()
        self.user = _make_user(username="mem_bg_wdog", email="membg@example.com")
        self.session = _make_session(self.user, session_id="mem_bg_sess")
        self.since_dt = timezone.now() - datetime.timedelta(days=1)

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.filter(pk=self.session.pk).delete()
        SiteConfiguration.objects.all().delete()
        self.user.delete()

    def test_memory_job_error_does_not_break_watchdog(self):
        with patch("aiworks_core.logic.memory.run_memory_generation_job") as mock_mem_gen:
            mock_mem_gen.side_effect = RuntimeError("Memory LLM failed")
            with patch("aiworks_core.apps.threading.Thread"):
                AiWorksCoreConfig._run_watchdog_step()


# ---------------------------------------------------------------------------
# Purge deleted attached files
# ---------------------------------------------------------------------------

class PurgeDeletedAttachedFilesTests(TestCase):
    """Tests for purge_deleted_attached_files()."""

    def setUp(self):
        super().setUp()
        ColdStorageManager.reset()
        # Ensure a clean SiteConfiguration singleton.
        SiteConfiguration.objects.all().delete()
        self._tmpdir = tempfile.mkdtemp()
        self._user = _make_user()

    def tearDown(self):
        SiteConfiguration.objects.all().delete()
        ColdStorageManager.reset()
        super().tearDown()
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def test_returns_zero_when_cold_storage_disabled(self):
        _set_coldstorage_config(coldstorage_type="none")
        session = _make_session(self._user, session_id="purge_disabled")
        af = AttachedFile.objects.create(
            session=session,
            name="orphaned.txt",
            binary_content=b"orphaned",
        )
        store_attached_file(af)

        removed = AiWorksCoreConfig._purge_deleted_attached_files()
        self.assertEqual(removed, 0)

    def test_purges_orphan_when_db_record_deleted(self):
        _set_coldstorage_config(
            coldstorage_type="local",
            coldstorage_local_storage_location=self._tmpdir,
        )
        session = _make_session(self._user, session_id="purge_orphan_session")
        af = AttachedFile.objects.create(
            session=session,
            name="orphaned.txt",
            binary_content=b"orphan content",
        )
        store_attached_file(af)

        manager = ColdStorageManager.get_instance()
        pk_str = str(af.pk)
        self.assertTrue(
            manager.exists(ATTACHED_FILES_FOLDER, pk_str),
            "Binary should exist before purge",
        )

        af.delete()

        removed = AiWorksCoreConfig._purge_deleted_attached_files()
        self.assertEqual(removed, 1)
        self.assertFalse(
            manager.exists(ATTACHED_FILES_FOLDER, pk_str),
            "Binary should be gone after purge",
        )

    def test_noop_when_all_cold_files_have_db_record(self):
        _set_coldstorage_config(
            coldstorage_type="local",
            coldstorage_local_storage_location=self._tmpdir,
        )
        session = _make_session(self._user, session_id="purge_active_session")
        af = AttachedFile.objects.create(
            session=session,
            name="active.txt",
            binary_content=b"should stay",
        )
        store_attached_file(af)

        manager = ColdStorageManager.get_instance()
        self.assertTrue(manager.exists(ATTACHED_FILES_FOLDER, str(af.pk)))

        removed = AiWorksCoreConfig._purge_deleted_attached_files()
        self.assertEqual(removed, 0)
        self.assertTrue(
            manager.exists(ATTACHED_FILES_FOLDER, str(af.pk)),
        )

    def test_purges_multiple_orphan_batches(self):
        _set_coldstorage_config(
            coldstorage_type="local",
            coldstorage_local_storage_location=self._tmpdir,
        )
        manager = ColdStorageManager.get_instance()
        pks_to_delete = []
        for i in range(5):
            af = AttachedFile.objects.create(
                session=_make_session(self._user, session_id=f"batch_orphan_{i}"),
                name=f"orphan{i}.txt",
                binary_content=f"orphan content {i}".encode(),
            )
            store_attached_file(af)
            pks_to_delete.append(af.pk)

        self.assertTrue(
            all(manager.exists(ATTACHED_FILES_FOLDER, str(pk)) for pk in pks_to_delete)
        )

        AttachedFile.objects.filter(pk__in=pks_to_delete).delete()

        removed = AiWorksCoreConfig._purge_deleted_attached_files()
        self.assertEqual(removed, 5)
        self.assertFalse(
            any(manager.exists(ATTACHED_FILES_FOLDER, str(pk)) for pk in pks_to_delete)
        )

    def test_returns_zero_when_no_orphaned_files(self):
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)

        removed = AiWorksCoreConfig._purge_deleted_attached_files()
        self.assertEqual(removed, 0)


class WatchdogHooksTests(TestCase):
    """Watchdog hooks mechanism — runs registered hooks after core steps."""

    def test_runs_registered_hook(self):
        """A hook registered via AIWORKS_CORE_WATCHDOG_HOOKS is called."""
        called = []

        def my_hook():
            called.append(True)

        with patch("aiworks_core.apps.AiWorksCoreConfig._run_watchdog_hooks", my_hook):
            AiWorksCoreConfig._run_watchdog_hooks()

        self.assertEqual(called, [True])

    def test_runs_multiple_hooks_in_order(self):
        """Multiple hooks are called in registration order."""
        call_order = []

        def hook_a():
            call_order.append("a")

        def hook_b():
            call_order.append("b")

        with patch("aiworks_core.apps.settings") as mock_settings:
            mock_settings.AIWORKS_CORE_WATCHDOG_HOOKS = [
                "test_watchdog:hook_a",
                "test_watchdog:hook_b",
            ]
            import sys
            test_module = type(sys)("test_watchdog")
            test_module.hook_a = hook_a
            test_module.hook_b = hook_b
            with patch.dict(sys.modules, {"test_watchdog": test_module}):
                AiWorksCoreConfig._run_watchdog_hooks()

        self.assertEqual(call_order, ["a", "b"])

    def test_hook_failure_does_not_raise(self):
        """A failing hook logs a warning but does not propagate the exception."""
        import logging
        from unittest.mock import MagicMock

        _logger = logging.getLogger("aiworks_core.apps")

        def bad_hook():
            raise RuntimeError("hook failed")

        with patch("builtins.__import__") as mock_import:
            mock_mod = MagicMock()
            mock_mod.bad_hook = bad_hook
            mock_import.return_value = mock_mod

            with patch("aiworks_core.apps.settings") as mock_settings:
                mock_settings.AIWORKS_CORE_WATCHDOG_HOOKS = ["foo:bad_hook"]
                # Should not raise
                AiWorksCoreConfig._run_watchdog_hooks()

    def test_empty_hooks_setting_runs_without_error(self):
        """When AIWORKS_CORE_WATCHDOG_HOOKS is empty or None, no error occurs."""
        with patch("aiworks_core.apps.settings") as mock_settings:
            mock_settings.AIWORKS_CORE_WATCHDOG_HOOKS = []
            AiWorksCoreConfig._run_watchdog_hooks()  # Should not raise
