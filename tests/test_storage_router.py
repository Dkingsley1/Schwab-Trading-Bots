import os
import shutil
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


if __name__ == '__main__':
    unittest.main()
