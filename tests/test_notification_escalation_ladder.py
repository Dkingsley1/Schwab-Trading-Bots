import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import notification_escalation_ladder as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_notification_escalation_ladder_surfaces_degraded_ack_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "portable_brain_contract_latest.json", {"host_contract": {"system": "Darwin"}})
    _write_json(health / "process_watchdog_latest.json", {"status": [{"name": "mac_notification_watch", "running": 1}]})
    _write_json(
        health / "mac_notification_watch_state.json",
        {"imessage_enabled": True, "imessage_recipient_configured": True},
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {
            "overall_status": "degraded",
            "channels": {"any_configured": True},
            "critical_backlog": {"unacked_count": 2, "unsent_count": 0},
            "backlog_compaction": {"grouped_unacked_count": 1, "grouped_unsent_count": 0, "dedupe_ratio": 2.0},
            "recommended_actions": ["acknowledge critical alerts explicitly"],
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["remote_pager_ready"] is True
    assert payload["attended_runtime_ready"] is True
    assert payload["unattended_runtime_ready"] is False
    assert payload["critical_backlog"]["unacked_count"] == 2
    assert payload["critical_backlog"]["grouped_unacked_count"] == 1
    assert any(step["step"] == "imessage_operator_bridge" and step["ready"] for step in payload["steps"])


def test_notification_escalation_ladder_treats_local_operator_bridge_as_attended_fallback(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "portable_brain_contract_latest.json", {"host_contract": {"system": "Darwin"}})
    _write_json(health / "process_watchdog_latest.json", {"status": [{"name": "mac_notification_watch", "running": 1}]})
    _write_json(
        health / "mac_notification_watch_state.json",
        {"imessage_enabled": True, "imessage_recipient_configured": True},
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {
            "overall_status": "blocked",
            "channels": {"any_configured": False},
            "critical_backlog": {"unacked_count": 0, "unsent_count": 0},
            "backlog_compaction": {"grouped_unacked_count": 0, "grouped_unsent_count": 0, "dedupe_ratio": 1.0},
            "recommended_actions": ["configure remote pager"],
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["attended_runtime_ready"] is True
    assert payload["unattended_runtime_ready"] is False
