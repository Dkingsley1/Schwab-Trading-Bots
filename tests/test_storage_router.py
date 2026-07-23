import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import storage_router


class StorageRouterTests(unittest.TestCase):
    def _set_env(self, updates: dict[str, str]) -> dict[str, str | None]:
        previous: dict[str, str | None] = {}
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        return previous

    def _restore_env(self, previous: dict[str, str | None]) -> None:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def test_split_brain_conflict_blocks_initial_failback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            external_root = Path(td) / 'external'
            local_root = root / 'local_fallback_storage'

            self._write_text(local_root / 'logs' / 'state.json', 'local-only-delta')
            self._write_text(external_root / 'logs' / 'state.json', 'external-copy')

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(local_root),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '0',
                    'BOT_LOGS_BLOCK_SPLIT_BRAIN': '1',
                }
            )
            try:
                result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'local_fallback_split_brain')
            self.assertEqual(result.active_root, local_root)
            self.assertEqual(result.split_brain_conflicts, 1)
            self.assertEqual(
                storage_router._resolve_link_target(root / 'logs'),
                (local_root / 'logs').resolve(strict=False),
            )

    def test_split_brain_conflict_does_not_revert_existing_external_cutover(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            external_root = Path(td) / 'external'
            local_root = root / 'local_fallback_storage'

            self._write_text(local_root / 'logs' / 'state.json', 'local-only-delta')
            self._write_text(external_root / 'logs' / 'state.json', 'external-copy')
            (root / 'logs').symlink_to(external_root / 'logs')

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(local_root),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '0',
                    'BOT_LOGS_BLOCK_SPLIT_BRAIN': '1',
                }
            )
            try:
                result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'external')
            self.assertEqual(result.active_root, external_root)
            self.assertEqual(result.split_brain_conflicts, 1)
            self.assertEqual(
                storage_router._resolve_link_target(root / 'logs'),
                (external_root / 'logs').resolve(strict=False),
            )

    def test_low_free_external_space_falls_back_to_local_storage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            external_root = Path(td) / 'external'
            external_root.mkdir(parents=True, exist_ok=True)
            local_root = root / 'local_fallback_storage'
            usage = shutil.disk_usage(td)

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(local_root),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '0',
                    'BOT_LOGS_EXTERNAL_MIN_FREE_BYTES': '100',
                }
            )
            try:
                with mock.patch.object(
                    storage_router.shutil,
                    'disk_usage',
                    return_value=type(usage)(usage.total, usage.used, 50),
                ):
                    result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'local_fallback')
            self.assertEqual(result.active_root, local_root)
            self.assertEqual(
                storage_router._resolve_link_target(root / 'logs'),
                (local_root / 'logs').resolve(strict=False),
            )

    def test_passthrough_data_dir_reconciles_existing_nested_sqlite_links_to_active_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            (root / 'data').mkdir()
            external_root = Path(td) / 'external'
            external_root.mkdir(parents=True, exist_ok=True)
            local_root = root / 'local_fallback_storage'
            self._write_text(local_root / 'data' / 'jsonl_link.sqlite3', 'local-primary')
            self._write_text(local_root / 'data' / 'snapshot_context.sqlite3', 'local-snapshot')
            self._write_text(external_root / 'data' / 'bot_channel_queue.sqlite3', 'external-queue')
            (root / 'data' / 'jsonl_link.sqlite3').symlink_to(external_root / 'data' / 'jsonl_link.sqlite3')
            (root / 'data' / 'jsonl_link.sqlite3-wal').symlink_to(external_root / 'data' / 'jsonl_link.sqlite3-wal')
            (root / 'data' / 'snapshot_context.sqlite3').symlink_to(external_root / 'data' / 'snapshot_context.sqlite3')
            (root / 'data' / 'bot_channel_queue.sqlite3').symlink_to(external_root / 'data' / 'bot_channel_queue.sqlite3')

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(local_root),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '0',
                    'BOT_LOGS_EXTERNAL_MIN_FREE_BYTES': '100',
                }
            )
            usage = shutil.disk_usage(td)
            try:
                with mock.patch.object(
                    storage_router.shutil,
                    'disk_usage',
                    return_value=type(usage)(usage.total, usage.used, 50),
                ):
                    result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'local_fallback')
            self.assertEqual(
                storage_router._resolve_link_target(root / 'data' / 'jsonl_link.sqlite3'),
                (local_root / 'data' / 'jsonl_link.sqlite3').resolve(strict=False),
            )
            self.assertEqual(
                storage_router._resolve_link_target(root / 'data' / 'jsonl_link.sqlite3-wal'),
                (local_root / 'data' / 'jsonl_link.sqlite3-wal').resolve(strict=False),
            )
            self.assertEqual(
                storage_router._resolve_link_target(root / 'data' / 'snapshot_context.sqlite3'),
                (local_root / 'data' / 'snapshot_context.sqlite3').resolve(strict=False),
            )
            self.assertEqual(
                storage_router._resolve_link_target(root / 'data' / 'bot_channel_queue.sqlite3'),
                (external_root / 'data' / 'bot_channel_queue.sqlite3').resolve(strict=False),
            )
            self.assertIn('data/jsonl_link.sqlite3', result.switched_links)
            self.assertIn('nested_sqlite_skipped:data/bot_channel_queue.sqlite3', result.passthrough_paths)

    def test_auto_sync_records_copy_error_details(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_root = root / 'local'
            external_root = root / 'external'
            self._write_text(local_root / 'logs' / 'state.json', 'hello')
            external_root.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(storage_router.shutil, 'copy2', side_effect=OSError('disk I/O failed')):
                copied, errors, pruned, details = storage_router._auto_sync_local_to_external(
                    local_root=local_root,
                    external_root=external_root,
                    link_dirs=('logs',),
                    prune_local=False,
                    max_copy_files=10,
                )

            self.assertEqual(copied, 0)
            self.assertEqual(errors, 1)
            self.assertEqual(pruned, 0)
            self.assertEqual(len(details), 1)
            self.assertIn('logs/state.json', details[0])

    def test_route_runtime_storage_skips_autosync_under_free_space_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            external_root = Path(td) / 'external'
            external_root.mkdir(parents=True, exist_ok=True)
            local_root = root / 'local_fallback_storage'
            self._write_text(local_root / 'logs' / 'state.json', 'local-backlog')
            usage = shutil.disk_usage(td)

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(local_root),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '1',
                    'BOT_LOGS_AUTO_SYNC_MIN_FREE_BYTES': '100',
                    'BOT_LOGS_BLOCK_SPLIT_BRAIN': '0',
                }
            )
            try:
                with mock.patch.object(
                    storage_router.shutil,
                    'disk_usage',
                    return_value=type(usage)(usage.total, usage.used, 50),
                ):
                    result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'external')
            self.assertEqual(result.autosync_copied_files, 0)
            self.assertEqual(result.autosync_free_bytes, 50)
            self.assertEqual(result.autosync_min_free_bytes, 100)
            self.assertIn('autosync_skipped_external_low_space', result.autosync_skipped_reason)
            self.assertTrue((local_root / 'logs' / 'state.json').exists())
            self.assertFalse((external_root / 'logs' / 'state.json').exists())

    def test_auto_sync_skips_sqlite_sidecars_for_failback_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_root = root / 'local'
            external_root = root / 'external'
            self._write_text(local_root / 'data' / 'jsonl_link.sqlite3-wal', 'wal-bytes')
            self._write_text(local_root / 'data' / 'jsonl_link.sqlite3-shm', 'shm-bytes')
            self._write_text(local_root / 'data' / 'bot_channel_queue.sqlite3-wal', 'queue-wal')
            self._write_text(local_root / 'data' / 'snapshot_context.sqlite3-shm', 'snapshot-shm')
            external_root.mkdir(parents=True, exist_ok=True)

            copied, errors, pruned, details = storage_router._auto_sync_local_to_external(
                local_root=local_root,
                external_root=external_root,
                link_dirs=('data',),
                prune_local=True,
                max_copy_files=10,
            )

            self.assertEqual(copied, 0)
            self.assertEqual(errors, 0)
            self.assertEqual(pruned, 0)
            self.assertEqual(details, [])
            self.assertTrue((local_root / 'data' / 'jsonl_link.sqlite3-wal').exists())
            self.assertFalse((external_root / 'data' / 'jsonl_link.sqlite3-wal').exists())

    def test_route_runtime_storage_records_route_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            external_root = Path(td) / 'external'
            external_root.mkdir(parents=True, exist_ok=True)
            ops_db = root / 'governance' / 'ops_data_plane.sqlite3'

            previous = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': str(external_root),
                    'BOT_LOGS_LOCAL_FALLBACK_ROOT': str(root / 'local_fallback_storage'),
                    'BOT_LOGS_AUTO_SYNC_ON_RECONNECT': '0',
                    'BOT_OPS_CONTROL_DB': str(ops_db),
                }
            )
            try:
                result = storage_router.route_runtime_storage(root, link_dirs=('logs',))
            finally:
                self._restore_env(previous)

            self.assertEqual(result.mode, 'external')
            self.assertTrue(result.ops_event_recorded)
            with sqlite3.connect(str(ops_db)) as conn:
                row = conn.execute(
                    "SELECT mode, active_root FROM storage_route_events ORDER BY id DESC LIMIT 1"
                ).fetchone()

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], 'external')
            self.assertEqual(row[1], str(external_root))


if __name__ == '__main__':
    unittest.main()
