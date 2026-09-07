import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "daily_state_snapshot_drill.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "daily_state_snapshot_drill", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load daily_state_snapshot_drill module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DailyStateSnapshotDrillTests(unittest.TestCase):
    def test_large_file_metadata_is_not_restore_or_hash_proof(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)

            large_file = root / "large.bin"
            large_file.write_bytes(b"A" * 128)

            out_root = root / "exports" / "state_snapshot_drills"
            argv = [
                "daily_state_snapshot_drill.py",
                "--out-root",
                str(out_root),
                "--targets",
                str(large_file),
                "--max-copy-bytes",
                "16",
                "--json",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = module.main()

            self.assertEqual(rc, 2)
            latest = json.loads((out_root / "latest.json").read_text(encoding="utf-8"))
            self.assertTrue(latest["latest_write_verified"])
            self.assertEqual(latest["latest_file"], str(out_root / "latest.json"))
            row = latest["rows"][0]
            self.assertEqual(row["copy_mode"], "metadata_only_large_file")
            self.assertEqual(row["snapshot"], "")
            self.assertEqual(row["restored"], "")
            self.assertFalse(row["restore_ok"])
            self.assertFalse(row["restore_verified"])
            self.assertEqual(row["sha256"], "")
            self.assertEqual(row["snapshot_sha256"], "")
            self.assertEqual(row["restore_sha256"], "")
            self.assertEqual(row["metadata_observation"]["size_bytes"], 128)
            self.assertFalse(latest["full_platform_restore_verified"])

    def test_small_file_copy_restore_mode(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)

            small_file = root / "small.txt"
            small_file.write_text("hello", encoding="utf-8")

            out_root = root / "exports" / "state_snapshot_drills"
            argv = [
                "daily_state_snapshot_drill.py",
                "--out-root",
                str(out_root),
                "--targets",
                str(small_file),
                "--max-copy-bytes",
                "1024",
                "--json",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = module.main()

            self.assertEqual(rc, 0)
            latest = json.loads((out_root / "latest.json").read_text(encoding="utf-8"))
            self.assertTrue(latest["latest_write_verified"])
            self.assertEqual(latest["latest_file"], str(out_root / "latest.json"))
            row = latest["rows"][0]
            self.assertEqual(row["copy_mode"], "full_copy_restore")
            self.assertNotEqual(row["snapshot"], "")
            self.assertNotEqual(row["restored"], "")
            self.assertTrue(row["restore_ok"])

    def test_broken_routed_sqlite_uses_local_fallback_source(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)

            routed_db = root / "data" / "jsonl_link.sqlite3"
            missing_external_db = (
                Path(td) / "missing_bot_logs" / "data" / "jsonl_link.sqlite3"
            )
            fallback_db = (
                root / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
            )
            routed_db.parent.mkdir(parents=True, exist_ok=True)
            fallback_db.parent.mkdir(parents=True, exist_ok=True)
            routed_db.symlink_to(missing_external_db)
            with sqlite3.connect(fallback_db) as conn:
                conn.execute("CREATE TABLE observations (value TEXT)")
                conn.execute("INSERT INTO observations VALUES ('ready')")

            out_root = root / "exports" / "state_snapshot_drills"
            argv = [
                "daily_state_snapshot_drill.py",
                "--out-root",
                str(out_root),
                "--targets",
                str(routed_db),
                "--max-copy-bytes",
                "1048576",
                "--json",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = module.main()

            self.assertEqual(rc, 0)
            latest = json.loads((out_root / "latest.json").read_text(encoding="utf-8"))
            self.assertTrue(latest["latest_write_verified"])
            self.assertEqual(latest["latest_file"], str(out_root / "latest.json"))
            row = latest["rows"][0]
            self.assertEqual(row["requested_source"], str(routed_db))
            self.assertEqual(row["effective_source"], str(fallback_db.resolve()))
            self.assertEqual(row["source"], str(fallback_db.resolve()))
            self.assertTrue(row["restore_ok"])


def _run_drill(root, targets, *extra):
    module = _load_module()
    module.PROJECT_ROOT = root
    out_root = root / "exports" / "state_snapshot_drills"
    argv = [
        "drill",
        "--out-root",
        str(out_root),
        "--targets",
        *map(str, targets),
        *extra,
        "--json",
    ]
    with mock.patch.object(sys, "argv", argv):
        code = module.main()
    return code, json.loads((out_root / "latest.json").read_text())


def test_empty_target_set_does_not_pass(tmp_path):
    code, report = _run_drill(tmp_path, [])
    assert code == 2
    assert report["files_restore_verified"] == 0


def test_sqlite_restore_contains_committed_wal_rows(tmp_path):
    database = tmp_path / "active.sqlite3"
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE observations (value TEXT)")
        conn.execute("INSERT INTO observations VALUES ('in_wal')")
        conn.commit()
        assert Path(str(database) + "-wal").stat().st_size > 0
        code, report = _run_drill(tmp_path, [database])
        row = report["rows"][0]
        assert code == 0
        assert row["copy_mode"] == "online_sqlite_backup_restore"
        assert row["sqlite_integrity_verified"] is True
        assert row["sha256"] == ""
        assert row["hash_scope"] == "consistent_sqlite_snapshot"
        with sqlite3.connect(row["restored"]) as restored:
            assert (
                restored.execute("SELECT value FROM observations").fetchone()[0]
                == "in_wal"
            )
    finally:
        conn.close()


def test_sqlite_copy_budget_includes_pages_not_checkpointed_to_main_file(tmp_path):
    database = tmp_path / "active.sqlite3"
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE observations (value BLOB)")
        conn.execute("INSERT INTO observations VALUES (zeroblob(100000))")
        conn.commit()
        assert database.stat().st_size <= 8192
        code, report = _run_drill(tmp_path, [database], "--max-copy-bytes", "8192")
        assert code == 2
        assert report["rows"][0]["error"] == "sqlite_snapshot_exceeds_copy_budget"
    finally:
        conn.close()


def test_keep_zero_preserves_current_verified_restore(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"ready":true}')
    code, report = _run_drill(tmp_path, [target], "--keep-runs", "0")
    assert code == 0
    assert report["retention"]["keep_runs"] == 1
    assert Path(report["rows"][0]["restored"]).is_file()


def test_retention_only_removes_owned_completed_runs(tmp_path):
    module = _load_module()
    current = tmp_path / "20260907_120000_123456"
    old = tmp_path / "20260906_120000"
    unrelated = tmp_path / "20260905_120000"
    linked = tmp_path / "20260904_120000"
    for directory in (current, old, unrelated):
        directory.mkdir()
    (old / "manifest.json").write_text(json.dumps({"run_dir": str(old)}))
    linked.symlink_to(unrelated, target_is_directory=True)
    assert module._prune_old_runs(tmp_path, 0, current_run=current) == 1
    assert current.exists() and unrelated.exists() and linked.is_symlink()
    assert not old.exists()


def test_protected_symlink_source_is_rejected_without_target_access(tmp_path):
    target = tmp_path / "protected.json"
    target.symlink_to("/Volumes/VIDEO/state.json")
    original = Path.lstat

    def guarded_lstat(path, *args, **kwargs):
        assert not str(path).casefold().startswith("/volumes/video")
        return original(path, *args, **kwargs)

    with mock.patch.object(Path, "lstat", guarded_lstat):
        code, report = _run_drill(tmp_path, [target])
    assert code == 2
    assert report["missing_files"] == [str(target)]


def test_regular_file_mutation_during_copy_fails_verification(tmp_path):
    module = _load_module()
    module.PROJECT_ROOT = tmp_path
    target = tmp_path / "mutable.txt"
    target.write_text("before")
    real_copy = module.shutil.copy2

    def mutate_after_copy(source, destination):
        result = real_copy(source, destination)
        if source == target:
            target.write_text("changed while copying")
        return result

    output = tmp_path / "snapshots"
    with mock.patch.object(module.shutil, "copy2", side_effect=mutate_after_copy):
        with mock.patch.object(
            sys, "argv", ["drill", "--out-root", str(output), "--targets", str(target)]
        ):
            assert module.main() == 2
    report = json.loads((output / "latest.json").read_text())
    assert report["rows"][0]["error"] == "source_changed_during_snapshot"


if __name__ == "__main__":
    unittest.main()
