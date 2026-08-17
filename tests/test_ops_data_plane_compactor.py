import gzip
import json
from pathlib import Path

from scripts import ops_data_plane
from scripts.ops import ops_data_plane_compactor as src


def _seed_legacy_events(project_root: Path) -> Path:
    db_path = project_root / "governance" / "ops_data_plane.sqlite3"
    with ops_data_plane.connect(project_root) as conn:
        conn.executemany(
            """
            INSERT INTO schema_drift_events(
                lane, source_rel, line_no, observed_schema_version,
                expected_schema_version, drift_kind, payload_sha256,
                payload_json, run_id, iter_id, recorded_utc, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "sqlite",
                    "decisions/demo.jsonl",
                    line_no,
                    0,
                    7,
                    "missing_log_schema_version",
                    f"hash-{line_no}",
                    json.dumps({"line": line_no, "payload": "x" * 4096}),
                    "run-1",
                    f"iter-{line_no}",
                    f"2026-08-11T12:00:0{line_no}+00:00",
                    "{}",
                )
                for line_no in (1, 2)
            ],
        )
        conn.commit()
    return db_path


def test_compactor_archives_rollups_and_reclaims_legacy_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    db_path = _seed_legacy_events(project_root)
    archive_root = tmp_path / "cold"

    payload = src.compact_ops_data_plane(
        project_root,
        archive_root=archive_root,
        apply=True,
        require_stack_stopped=False,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["legacy_snapshot"]["row_count"] == 2
    assert payload["legacy_after"]["row_count"] == 0
    assert payload["rollup_rows"] == 1
    assert payload["rollup_occurrences"] == 2
    archive_path = Path(payload["archive"]["archive_path"])
    assert archive_path.exists()
    assert Path(payload["archive"]["manifest_path"]).exists()
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        archived = json.loads(handle.readline())
    assert archived["occurrence_count"] == 2
    assert archived["first_payload_sha256"] == "hash-1"
    assert archived["last_payload_sha256"] == "hash-2"

    with ops_data_plane.connect(project_root) as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM schema_drift_events").fetchone()[0]) == 0
        row = conn.execute(
            "SELECT occurrence_count, sample_payload_json FROM schema_drift_rollups"
        ).fetchone()
    assert row == (2, "")
    assert db_path.stat().st_size < 512 * 1024


def test_compactor_requires_quiesced_runtime_for_apply(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_legacy_events(project_root)

    payload = src.compact_ops_data_plane(project_root, apply=True)

    assert payload["ok"] is False
    assert payload["blockers"] == ["runtime_must_be_quiesced_before_ops_data_plane_compaction"]


def test_compactor_rejects_protected_archive_volume_without_accessing_it(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_legacy_events(project_root)

    payload = src.compact_ops_data_plane(
        project_root,
        archive_root=Path("/Volumes/VIDEO/schwab_trading_bot_cold"),
        apply=True,
        require_stack_stopped=False,
    )

    assert payload["ok"] is False
    assert payload["blockers"] == ["protected_archive_volume_rejected"]
