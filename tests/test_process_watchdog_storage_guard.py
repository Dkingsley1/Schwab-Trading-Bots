import os
import tempfile
import unittest
from pathlib import Path

from scripts.ops import process_watchdog as pw


class ProcessWatchdogStorageGuardTests(unittest.TestCase):
    def _set_env(self, updates: dict[str, str]) -> dict[str, str | None]:
        prev: dict[str, str | None] = {}
        for key, val in updates.items():
            prev[key] = os.environ.get(key)
            os.environ[key] = val
        return prev

    def _restore_env(self, prev: dict[str, str | None]) -> None:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_resolve_external_storage_paths_prefers_configured_root(self) -> None:
        prev = self._set_env(
            {
                'BOT_LOGS_EXTERNAL_MOUNT': '/tmp/mount_override',
                'BOT_LOGS_EXTERNAL_PROJECT_ROOT': '/tmp/custom_external_root',
                'BOT_LOGS_EXTERNAL_PROJECT_DIR': 'ignored_here',
            }
        )
        try:
            mount_root, external_root = pw._resolve_external_storage_paths()
            self.assertEqual(str(mount_root), '/tmp/mount_override')
            self.assertEqual(str(external_root), '/tmp/custom_external_root')
        finally:
            self._restore_env(prev)

    def test_probe_storage_mount_reports_available_when_mount_and_root_writable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mount_root = Path(td) / 'mnt'
            project_dir = 'project_a'
            external_root = mount_root / project_dir
            external_root.mkdir(parents=True, exist_ok=True)

            prev = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_MOUNT': str(mount_root),
                    'BOT_LOGS_EXTERNAL_PROJECT_DIR': project_dir,
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': '',
                }
            )
            try:
                probe = pw._probe_storage_mount()
                self.assertTrue(probe['mount_present'])
                self.assertTrue(probe['external_root_exists'])
                self.assertTrue(probe['external_root_writable'])
                self.assertTrue(probe['external_available'])
                self.assertEqual(probe['mount_root'], str(mount_root))
                self.assertEqual(probe['external_root'], str(external_root))
            finally:
                self._restore_env(prev)

    def test_probe_storage_mount_reports_unavailable_when_mount_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing_mount = Path(td) / 'missing_mount'
            prev = self._set_env(
                {
                    'BOT_LOGS_EXTERNAL_MOUNT': str(missing_mount),
                    'BOT_LOGS_EXTERNAL_PROJECT_DIR': 'project_b',
                    'BOT_LOGS_EXTERNAL_PROJECT_ROOT': '',
                }
            )
            try:
                probe = pw._probe_storage_mount()
                self.assertFalse(probe['mount_present'])
                self.assertFalse(probe['external_available'])
            finally:
                self._restore_env(prev)

    def test_evaluate_storage_mount_transition(self) -> None:
        self.assertEqual(pw._evaluate_storage_mount_transition(True, True), {})
        self.assertEqual(pw._evaluate_storage_mount_transition(False, False), {})
        self.assertEqual(pw._evaluate_storage_mount_transition(True, False), {'from': True, 'to': False})
        self.assertEqual(pw._evaluate_storage_mount_transition(False, True), {'from': False, 'to': True})
        self.assertEqual(pw._evaluate_storage_mount_transition(None, True), {})
        self.assertEqual(pw._evaluate_storage_mount_transition(None, False), {'from': 'unknown', 'to': False})


if __name__ == '__main__':
    unittest.main()
