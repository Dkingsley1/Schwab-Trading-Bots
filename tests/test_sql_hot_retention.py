import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sql_hot_retention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sql_hot_retention", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load sql_hot_retention module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, source_rel TEXT, line_no INTEGER)"
    )
    conn.commit()
    conn.close()


def _insert_rows(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT INTO jsonl_records (id, ingested_at, source_rel, line_no) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
        return int(row[0] if row and row[0] is not None else 0)
    finally:
        conn.close()


def _run_main(module, argv: list[str]) -> tuple[int, dict]:
    buf = StringIO()
    with tempfile.TemporaryDirectory() as td:
        missing_override = Path(td) / "missing_swap_override.env"
        with mock.patch.object(module, "SWAP_OVERRIDE_PATH", missing_override):
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(buf):
                    rc = module.main()
    return rc, json.loads(buf.getvalue().strip())


class SqlHotRetentionTests(unittest.TestCase):

    def test_noop_archive_retention_never_requests_writable_connection(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "retained ?# archive.sqlite3"
            _init_db(archive)
            _insert_rows(
                archive, [(1, datetime.now(timezone.utc).isoformat(), "source", 1)]
            )
            before = archive.read_bytes()
            with mock.patch.object(
                module,
                "_connect",
                side_effect=AssertionError("no writable archive admission"),
            ):
                result = module._prune_archive_storage(
                    archive_db=archive,
                    archive_root=None,
                    archive_retention_days=30,
                    archive_prune_vacuum=True,
                    cold_export_root=None,
                    cold_export_format="parquet",
                    cold_export_batch_size=1000,
                    cold_export_compression="zstd",
                )
                self.assertEqual(module._count_archive_rows(archive), 1)
            self.assertEqual(result["pruned_rows"], 0)
            self.assertFalse(result["errors"])
            self.assertEqual(archive.read_bytes(), before)

    def test_readonly_archive_probe_sees_committed_wal_without_write_authority(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "retained.sqlite3"
            _init_db(archive)
            writer = sqlite3.connect(archive)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute(
                    "INSERT INTO jsonl_records VALUES (1, '2026-09-08', 'source', 1)"
                )
                writer.commit()
                self.assertEqual(module._count_archive_rows(archive), 1)
                reader = module._connect_readonly(archive)
                try:
                    with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                        reader.execute("DELETE FROM jsonl_records")
                finally:
                    reader.close()
            finally:
                writer.close()

    def test_conflicting_archive_id_does_not_delete_source(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            db, archive = Path(td) / "hot.sqlite3", Path(td) / "archive.sqlite3"
            for path in (db, archive):
                _init_db(path)
            _insert_rows(db, [(1, "2000-01-01T00:00:00+00:00", "source", 1)])
            _insert_rows(archive, [(1, "2000-01-01T00:00:00+00:00", "different", 1)])
            with self.assertRaisesRegex(
                RuntimeError, "archive_copy_verification_failed"
            ):
                _run_main(
                    module,
                    [
                        "retention",
                        "--db",
                        str(db),
                        "--archive-db",
                        str(archive),
                        "--json",
                    ],
                )
            self.assertEqual(_count_rows(db), 1)
            with sqlite3.connect(archive) as conn:
                self.assertEqual(
                    conn.execute("SELECT source_rel FROM jsonl_records").fetchone()[0],
                    "different",
                )

    def test_identical_archive_retry_is_verified_before_source_release(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            db, archive = Path(td) / "hot.sqlite3", Path(td) / "archive.sqlite3"
            for path in (db, archive):
                _init_db(path)
                _insert_rows(path, [(1, "2000-01-01T00:00:00+00:00", "source", 1)])
            rc, payload = _run_main(
                module,
                [
                    "retention",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive),
                    "--archive-retention-days",
                    "0",
                    "--json",
                ],
            )
            self.assertEqual(rc, 0)
            self.assertEqual(payload["moved_rows"], 1)
            self.assertEqual(_count_rows(db), 0)
            self.assertEqual(_count_rows(archive), 1)

    def test_requested_vacuum_reclaims_pages_without_newly_expired_rows(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "hot.sqlite3"
            _init_db(db)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE old_payload (payload BLOB)")
                conn.execute("INSERT INTO old_payload VALUES (zeroblob(1048576))")
                conn.commit()
                conn.execute("DROP TABLE old_payload")
            before = db.stat().st_size
            rc, payload = _run_main(module, ["retention", "--db", str(db), "--archive-db", str(root / "archive.sqlite3"), "--vacuum", "--json"])
            self.assertEqual(rc, 0)
            self.assertEqual(payload["moved_rows"], 0)
            self.assertLess(db.stat().st_size, before)

    def test_corrupt_archive_is_preserved_while_healthy_archive_is_pruned(self) -> None:
        module = _load_module()
        for broken_name in ("latest.sqlite3", "jsonl_link_archive_2000_01_01.sqlite3"):
            with self.subTest(broken_name=broken_name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                broken = root / broken_name
                broken.write_bytes(b"corrupt archive retained for recovery")
                healthy = root / "jsonl_link_archive_2000_01_02.sqlite3"
                _init_db(healthy)
                _insert_rows(healthy, [(1, "2000-01-02T00:00:00+00:00", "test.jsonl", 1)])
                result = module._prune_archive_storage(
                    archive_db=root / "latest.sqlite3", archive_root=root,
                    archive_retention_days=10, archive_prune_vacuum=False,
                    cold_export_root=None, cold_export_format="parquet",
                    cold_export_batch_size=1000, cold_export_compression="zstd",
                )
                self.assertIn(str(broken), result["errors"])
                self.assertEqual(broken.read_bytes(), b"corrupt archive retained for recovery")
                self.assertIn(str(healthy), result["deleted_archive_files"])
                self.assertEqual(result["pruned_rows"], 1)

    def test_archive_failure_emits_partial_progress_and_nonzero_status(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "hot.sqlite3"
            _init_db(db)
            archive = root / "latest.sqlite3"
            archive.write_bytes(b"broken archive")
            rc, payload = _run_main(module, ["retention", "--db", str(db), "--archive-db", str(archive), "--archive-retention-days", "10", "--json"])
            self.assertEqual(rc, 2)
            self.assertFalse(payload["ok"])
            self.assertIn(str(archive), payload["archive_pruning"]["errors"])
            self.assertEqual(payload["moved_rows"], 0)

    def test_swap_pressure_pause_skips_hot_retention_without_touching_rows(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            _init_db(db)
            now = datetime.now(timezone.utc)
            _insert_rows(db, [(1, (now - timedelta(days=30)).isoformat(), "old.jsonl", 1)])

            with mock.patch.dict(
                os.environ,
                {
                    "RETENTION_MAINTENANCE_PAUSED_FOR_SWAP": "1",
                    "SWAP_PRESSURE_TIER": "pause_research",
                    "SWAP_PRESSURE_SWAP_USED_GB": "19.1",
                },
                clear=False,
            ):
                rc, payload = _run_main(
                    module,
                    [
                        "sql_hot_retention.py",
                        "--db",
                        str(db),
                        "--archive-db",
                        str(archive_db),
                        "--hot-days",
                        "1",
                        "--json",
                    ],
                )

            self.assertEqual(rc, 0)
            self.assertTrue(payload["skipped"])
            self.assertEqual(payload["reason"], "swap_pressure_pause")
            self.assertEqual(payload["moved_rows"], 0)
            self.assertEqual(_count_rows(db), 1)
            self.assertFalse(archive_db.exists())

    def test_monthly_archive_files_older_than_retention_are_deleted(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            archive_root = root / "jsonl_link_archives"
            _init_db(db)

            now = datetime.now(timezone.utc)
            _insert_rows(
                db,
                [
                    (1, (now - timedelta(days=140)).isoformat(), "old.jsonl", 1),
                    (2, (now - timedelta(days=60)).isoformat(), "newer.jsonl", 2),
                ],
            )

            rc, payload = _run_main(
                module,
                [
                    "sql_hot_retention.py",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive_db),
                    "--archive-root",
                    str(archive_root),
                    "--archive-period",
                    "month",
                    "--hot-days",
                    "1",
                    "--archive-retention-days",
                    "90",
                    "--json",
                ],
            )

            self.assertEqual(rc, 0)
            self.assertEqual(payload["moved_rows"], 2)
            pruning = payload["archive_pruning"]
            self.assertTrue(pruning["enabled"])
            self.assertEqual(pruning["pruned_rows"], 1)
            self.assertEqual(len(pruning["deleted_archive_files"]), 1)
            archive_files = sorted(archive_root.glob("jsonl_link_archive_*.sqlite3"))
            self.assertEqual(len(archive_files), 1)
            self.assertEqual(_count_rows(archive_files[0]), 1)
            self.assertEqual(_count_rows(db), 0)

    def test_single_archive_db_prunes_old_rows_by_retention_cutoff(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            _init_db(db)

            now = datetime.now(timezone.utc)
            _insert_rows(
                db,
                [
                    (1, (now - timedelta(days=140)).isoformat(), "old.jsonl", 1),
                    (2, (now - timedelta(days=40)).isoformat(), "newer.jsonl", 2),
                ],
            )

            rc, payload = _run_main(
                module,
                [
                    "sql_hot_retention.py",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive_db),
                    "--archive-period",
                    "single",
                    "--hot-days",
                    "1",
                    "--archive-retention-days",
                    "90",
                    "--json",
                ],
            )

            self.assertEqual(rc, 0)
            self.assertEqual(payload["moved_rows"], 2)
            pruning = payload["archive_pruning"]
            self.assertTrue(pruning["enabled"])
            self.assertEqual(pruning["pruned_rows"], 1)
            self.assertEqual(pruning["deleted_archive_files"], [])
            self.assertEqual(_count_rows(archive_db), 1)
            self.assertEqual(_count_rows(db), 0)

    def test_monthly_archive_files_export_to_cold_storage_before_delete(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except Exception:
            self.skipTest("pyarrow not installed")

        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            archive_root = root / "jsonl_link_archives"
            cold_root = root / "cold_archives"
            _init_db(db)

            now = datetime.now(timezone.utc)
            _insert_rows(
                db,
                [
                    (1, (now - timedelta(days=140)).isoformat(), "old.jsonl", 1),
                    (2, (now - timedelta(days=40)).isoformat(), "newer.jsonl", 2),
                ],
            )

            rc, payload = _run_main(
                module,
                [
                    "sql_hot_retention.py",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive_db),
                    "--archive-root",
                    str(archive_root),
                    "--archive-period",
                    "month",
                    "--hot-days",
                    "1",
                    "--archive-retention-days",
                    "90",
                    "--cold-export-root",
                    str(cold_root),
                    "--json",
                ],
            )

            self.assertEqual(rc, 0)
            pruning = payload["archive_pruning"]
            cold_export = pruning["cold_archive_export"]
            self.assertTrue(cold_export["enabled"])
            self.assertEqual(len(cold_export["exported_files"]), 1)
            exported_path = Path(next(iter(cold_export["output_files"].values())))
            self.assertTrue(exported_path.exists())
            deleted_path = Path(pruning["deleted_archive_files"][0])
            self.assertFalse(deleted_path.exists())

    def test_hot_hours_archives_older_same_day_rows(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            _init_db(db)

            now = datetime.now(timezone.utc)
            _insert_rows(
                db,
                [
                    (1, (now - timedelta(hours=3)).isoformat(), "old.jsonl", 1),
                    (2, (now - timedelta(minutes=20)).isoformat(), "fresh.jsonl", 2),
                ],
            )

            rc, payload = _run_main(
                module,
                [
                    "sql_hot_retention.py",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive_db),
                    "--hot-hours",
                    "1",
                    "--json",
                ],
            )

            self.assertEqual(rc, 0)
            self.assertEqual(payload["hot_window"]["unit"], "hours")
            self.assertEqual(payload["hot_window"]["value"], 1)
            self.assertEqual(payload["moved_rows"], 1)
            self.assertEqual(_count_rows(archive_db), 1)
            self.assertEqual(_count_rows(db), 1)

    def test_skip_remaining_count_keeps_large_retention_pass_bounded(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "jsonl_link.sqlite3"
            archive_db = root / "jsonl_link_archive.sqlite3"
            _init_db(db)

            now = datetime.now(timezone.utc)
            _insert_rows(
                db,
                [
                    (1, (now - timedelta(days=10)).isoformat(), "old.jsonl", 1),
                    (2, now.isoformat(), "fresh.jsonl", 2),
                ],
            )

            rc, payload = _run_main(
                module,
                [
                    "sql_hot_retention.py",
                    "--db",
                    str(db),
                    "--archive-db",
                    str(archive_db),
                    "--hot-days",
                    "7",
                    "--skip-remaining-count",
                    "--json",
                ],
            )

            self.assertEqual(rc, 0)
            self.assertEqual(payload["moved_rows"], 1)
            self.assertEqual(payload["remaining_rows"], -1)
            self.assertTrue(payload["remaining_count_skipped"])
            self.assertEqual(_count_rows(archive_db), 1)
            self.assertEqual(_count_rows(db), 1)


if __name__ == "__main__":
    unittest.main()
