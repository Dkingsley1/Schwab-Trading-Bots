import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_route_chaos_drill as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_storage_route_chaos_drill_records_readiness_event(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "storage_mount_guard_latest.json", {"ok": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    monkeypatch.setenv("BOT_OPS_CONTROL_DB", str(project_root / "governance" / "ops_data_plane.sqlite3"))

    payload = src.build_payload(project_root, scenario="external_unavailable")

    assert payload["ok"] is True
    with sqlite3.connect(str(project_root / "governance" / "ops_data_plane.sqlite3")) as conn:
        row = conn.execute(
            "SELECT mode FROM storage_route_events ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert row == ("chaos_drill",)
