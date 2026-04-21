import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_timeline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_incident_timeline_rolls_up_recent_events_and_open_surfaces(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    events_root = project_root / "governance" / "events"
    watchdog_root = project_root / "governance" / "watchdog"

    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"overall_status": "blocked", "reason": "shared_host_contention"},
    )
    _write_json(
        health_root / "auth_lease_manager_latest.json",
        {"overall_status": "degraded", "lease_state": "warning"},
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {"overall_status": "needs_coverage", "coverage_shortfall_bots": 2},
    )
    _write_json(
        health_root / "process_watchdog_latest.json",
        {"restart_storms": [{"name": "paper"}], "alerts": [{"name": "storage"}]},
    )
    _write_jsonl(
        events_root / "auth_events_20260421.jsonl",
        [
            {
                "timestamp_utc": "2026-04-21T14:00:00+00:00",
                "summary": "interactive_refresh_required",
                "ok": False,
            }
        ],
    )
    _write_jsonl(
        watchdog_root / "shadow_watchdog_halt_recovery_events.jsonl",
        [
            {
                "timestamp_utc": "2026-04-21T14:10:00+00:00",
                "message": "recovered from halt after feed reset",
                "status": "ok",
            }
        ],
    )

    payload = incident_timeline.build_payload(project_root, files_per_pattern=2, rows_per_file=10, recent_limit=10)

    assert payload["overall_status"] == "blocked"
    assert payload["recent_incident_count"] == 2
    assert payload["open_incident_count"] >= 3
    assert payload["incident_counts"]["by_category"]["auth_lease"] == 1
    assert payload["intervention_counts"]["recovery_events"] == 1
    assert any(row["surface"] == "runtime_separation" for row in payload["open_surfaces"])
