#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "notification_escalation_ladder_latest.json"


def _process_names(process_watchdog: dict[str, Any]) -> list[str]:
    rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    return [str((row or {}).get("name") or "").strip().lower() for row in rows if isinstance(row, dict)]


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    remote_alert = load_json(health_root / "remote_alert_control_latest.json")
    portable = load_json(health_root / "portable_brain_contract_latest.json")
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")
    mac_notification_state = load_json(health_root / "mac_notification_watch_state.json")

    names = _process_names(process_watchdog)
    host_contract = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
    system_name = str(host_contract.get("system") or "")
    desktop_ready = system_name == "Darwin" or any("notification" in name for name in names)
    imessage_enabled = bool(mac_notification_state.get("imessage_enabled", False))
    imessage_recipient_configured = bool(mac_notification_state.get("imessage_recipient_configured", False))
    imessage_ready = bool((imessage_enabled and imessage_recipient_configured) or any(token in name for name in names for token in ("mac_notification", "notification")))
    channels = remote_alert.get("channels") if isinstance(remote_alert.get("channels"), dict) else {}
    pager_ready = bool(channels.get("any_configured", False))
    backlog = remote_alert.get("critical_backlog") if isinstance(remote_alert.get("critical_backlog"), dict) else {}
    compaction = remote_alert.get("backlog_compaction") if isinstance(remote_alert.get("backlog_compaction"), dict) else {}
    unacked = int(backlog.get("unacked_count", 0) or 0)
    unsent = int(backlog.get("unsent_count", 0) or 0)
    grouped_unacked = int(compaction.get("grouped_unacked_count", unacked) or 0)
    grouped_unsent = int(compaction.get("grouped_unsent_count", unsent) or 0)
    dedupe_ratio = float(compaction.get("dedupe_ratio", 1.0) or 1.0)
    attended_runtime_ready = bool(desktop_ready and imessage_ready and grouped_unsent == 0)
    unattended_runtime_ready = bool(pager_ready and grouped_unsent == 0 and grouped_unacked == 0)

    steps = [
        {"step": "desktop_banner", "ready": desktop_ready, "status": ("ready" if desktop_ready else "missing"), "reason": "local_desktop_path_available" if desktop_ready else "desktop_notification_path_missing"},
        {
            "step": "imessage_operator_bridge",
            "ready": imessage_ready,
            "status": ("ready" if imessage_ready else "degraded"),
            "reason": (
                "imessage_operator_bridge_configured"
                if imessage_enabled and imessage_recipient_configured
                else ("mac_notification_watch_present" if imessage_ready else "local_operator_bridge_not_detected")
            ),
        },
        {"step": "remote_pager", "ready": pager_ready, "status": ("ready" if pager_ready else "blocked"), "reason": "remote_channel_configured" if pager_ready else "no_remote_channel_configured"},
        {"step": "ack_backlog", "ready": grouped_unsent == 0, "status": ("ready" if grouped_unsent == 0 and grouped_unacked == 0 else "degraded"), "reason": f"grouped_unacked={grouped_unacked} grouped_unsent={grouped_unsent} dedupe_ratio={dedupe_ratio:.2f}"},
    ]

    overall_status = "ready"
    if grouped_unsent > 0 or (not pager_ready and not attended_runtime_ready):
        overall_status = "blocked"
    elif grouped_unacked > 0 or not desktop_ready or not imessage_ready or not pager_ready:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        list(remote_alert.get("recommended_actions") or [])[:3]
        + [
            "keep the Mac notification watcher alive so local desktop and iMessage alerts stay part of the ladder" if not imessage_ready else "",
            "run notify-start with iMessage enabled so the operator bridge can act as the attended fallback pager on macOS" if not imessage_ready else "",
            "treat unsent remote critical alerts as a hard blocker for unattended runtime" if grouped_unsent > 0 else "",
            "configure at least one remote pager channel before relying on unattended runtime even if the local operator bridge is healthy" if not pager_ready else "",
            "route duplicate alert storms through the grouped backlog view before paging so operator load tracks unique incidents" if dedupe_ratio > 1.25 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "steps": steps,
        "desktop_ready": desktop_ready,
        "imessage_ready": imessage_ready,
        "remote_pager_ready": pager_ready,
        "attended_runtime_ready": attended_runtime_ready,
        "unattended_runtime_ready": unattended_runtime_ready,
        "critical_backlog": {
            "unacked_count": unacked,
            "unsent_count": unsent,
            "grouped_unacked_count": grouped_unacked,
            "grouped_unsent_count": grouped_unsent,
            "dedupe_ratio": round(dedupe_ratio, 3),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the notification escalation ladder across local and remote alert paths.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "notification_escalation_ladder "
            f"overall_status={payload.get('overall_status', '')} "
            f"remote_pager_ready={int(bool(payload.get('remote_pager_ready', False)))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
