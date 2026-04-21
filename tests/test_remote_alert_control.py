import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import remote_alert_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_remote_alert_control_treats_imessage_bridge_as_channel_and_ignores_resolved_old_noise(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "mac_notification_watch_state.json", {"imessage_enabled": True, "imessage_recipient_configured": True, "max_alert_age_seconds": 900.0})
    _write_json(health / "incident_timeline_latest.json", {"open_surfaces": []})
    _write_json(project_root / "governance" / "watchdog" / "remote_alert_ack_state.json", {"events": {}})
    _write_jsonl(
        project_root / "governance" / "watchdog" / "pager_alerts.jsonl",
        [
            {
                "timestamp_utc": "2026-04-21T10:00:00+00:00",
                "severity": "critical",
                "event": "reboot_resilience_recovery_failed",
                "message": "Reboot resilience failed to recover: com.dankingsley.all_sleeves",
                "sent": False,
                "suppressed": True,
            }
        ],
    )

    payload = src.build_payload(
        project_root,
        hours=24,
        ack_state_path=project_root / "governance" / "watchdog" / "remote_alert_ack_state.json",
    )

    assert payload["channels"]["imessage_bridge"] is True
    assert payload["channels"]["any_configured"] is True
    assert payload["overall_status"] == "ready"
    assert payload["backlog_compaction"]["grouped_active_count"] == 0
