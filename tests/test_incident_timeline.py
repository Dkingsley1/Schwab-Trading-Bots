import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_timeline as timeline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def test_auth_start_and_success_events_without_ok_field_are_info(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    events = project_root / "governance" / "events"

    _write_jsonl(
        events / "auth_events_20260521.jsonl",
        [
            {"timestamp_utc": "2026-05-21T14:26:00+00:00", "event": "auth_start", "status": "started"},
            {"timestamp_utc": "2026-05-21T14:26:01+00:00", "event": "auth_success", "status": "ok"},
        ],
    )
    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "ready"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready"})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"overall_status": "ready"})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})

    payload = timeline.build_payload(project_root, recent_limit=10)

    assert payload["overall_status"] == "ready"
    assert payload["incident_counts"]["by_severity"].get("info") == 2
    assert payload["incident_counts"]["by_severity"].get("warning", 0) == 0
    assert payload["open_incident_count"] == 0
