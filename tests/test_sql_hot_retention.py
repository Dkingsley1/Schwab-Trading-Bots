import importlib.util
import json
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
    with mock.patch.object(sys, "argv", argv):
        with redirect_stdout(buf):
            rc = module.main()
    return rc, json.loads(buf.getvalue().strip())


class SqlHotRetentionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
