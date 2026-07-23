#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, load_recent_jsonl, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, load_recent_jsonl, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "remote_alert_control_latest.json"
DEFAULT_ACK_STATE_PATH = PROJECT_ROOT / "governance" / "watchdog" / "remote_alert_ack_state.json"
PLACEHOLDER_WEBHOOK_HOSTS = {"example.com", "example.invalid", "localhost.invalid"}


def _webhook_url() -> str:
    return str(os.getenv("OPS_ALERT_WEBHOOK_URL", "")).strip()


def _webhook_config_error(url: str | None = None) -> str:
    raw = _webhook_url() if url is None else str(url or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if any(token in lowered for token in ("<", ">", "your_", "changeme", "placeholder")):
        return "placeholder_webhook_url"
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "invalid_webhook_url"
    host = (parsed.hostname or "").strip().lower()
    if host in PLACEHOLDER_WEBHOOK_HOSTS or host.endswith(".invalid"):
        return "placeholder_webhook_url"
    return ""


def _channel_config_errors() -> dict[str, str]:
    webhook_error = _webhook_config_error()
    return {"webhook": webhook_error} if webhook_error else {}


def _configured_channels() -> dict[str, bool]:
    return {
        "webhook": bool(_webhook_url() and not _webhook_config_error()),
        "pushover": bool(
            str(os.getenv("OPS_ALERT_PUSHOVER_TOKEN", "")).strip() and str(os.getenv("OPS_ALERT_PUSHOVER_USER_KEY", "")).strip()
        ),
    }


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_category(event: str) -> str:
    normalized = str(event or "").strip().lower()
    if any(token in normalized for token in ("token", "auth", "schwab")):
        return "auth_lease"
    if any(token in normalized for token in ("write_failure", "snapshot", "writer", "backlog")):
        return "data_plane"
    if any(token in normalized for token in ("restart", "reboot_resilience", "watchdog")):
        return "operations"
    if any(token in normalized for token in ("macro", "hearing", "transcript")):
        return "macro"
    return "operations"


def _load_ack_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    return state if isinstance(state.get("events"), dict) else {"events": {}}


def _save_ack_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _alert_signature(row: dict[str, Any]) -> str:
    material = "|".join(
        [
            str(row.get("event") or "generic").strip().lower(),
            str(row.get("severity") or "info").strip().lower(),
            str(row.get("message") or "").strip().lower(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    hours: int = 24,
    ack_state_path: Path = DEFAULT_ACK_STATE_PATH,
) -> dict[str, Any]:
    log_path = project_root / "governance" / "watchdog" / "pager_alerts.jsonl"
    health_root = project_root / "governance" / "health"
    rows = load_recent_jsonl(log_path, limit=max(int(hours), 1) * 200)
    ack_state = _load_ack_state(ack_state_path)
    ack_events = ack_state.get("events") if isinstance(ack_state.get("events"), dict) else {}
    channels = _configured_channels()
    channel_config_errors = _channel_config_errors()
    mac_state = load_json(health_root / "mac_notification_watch_state.json")
    timeline = load_json(health_root / "incident_timeline_latest.json")
    imessage_bridge = bool(mac_state.get("imessage_enabled", False) and mac_state.get("imessage_recipient_configured", False))
    channels["remote_pager_configured"] = bool(channels.get("webhook", False) or channels.get("pushover", False))
    channels["imessage_bridge"] = imessage_bridge
    open_categories = {
        str((row or {}).get("category") or "").strip().lower()
        for row in (timeline.get("open_surfaces") or [])
        if isinstance(row, dict) and str((row or {}).get("category") or "").strip()
    }
    active_window_seconds = max(int(float(mac_state.get("max_alert_age_seconds", 3600.0) or 3600.0)), 300)
    open_surface_window_seconds = max(active_window_seconds * 4, 3600)
    now = datetime.now(timezone.utc)

    severity_counts = {"info": 0, "warn": 0, "critical": 0}
    unacked_critical: list[dict[str, Any]] = []
    unsent_critical: list[dict[str, Any]] = []
    grouped_backlog: dict[str, dict[str, Any]] = {}
    for row in rows:
        severity = str(row.get("severity") or "info").strip().lower()
        if severity in severity_counts:
            severity_counts[severity] += 1
        if severity != "critical":
            continue
        event = str(row.get("event") or "generic").strip() or "generic"
        signature = _alert_signature(row)
        acked = isinstance(ack_events.get(event), dict)
        group = grouped_backlog.setdefault(
            signature,
            {
                "signature": signature,
                "event": event,
                "category": _event_category(event),
                "message": str(row.get("message") or ""),
                "count": 0,
                "unacked": False,
                "unsent": False,
                "suppressed_count": 0,
                "latest_timestamp_utc": "",
                "latest_age_seconds": None,
                "active": False,
            },
        )
        group["count"] = int(group.get("count", 0) or 0) + 1
        group["unacked"] = bool(group.get("unacked", False) or (not acked))
        group["unsent"] = bool(group.get("unsent", False) or (not bool(row.get("sent", False))))
        if bool(row.get("suppressed", False)):
            group["suppressed_count"] = int(group.get("suppressed_count", 0) or 0) + 1
        timestamp = _parse_ts(row.get("timestamp_utc"))
        if timestamp is not None:
            age_seconds = max((now - timestamp).total_seconds(), 0.0)
            if not group.get("latest_timestamp_utc") or age_seconds < float(group.get("latest_age_seconds") or 1e18):
                group["latest_timestamp_utc"] = timestamp.isoformat()
                group["latest_age_seconds"] = round(float(age_seconds), 3)
        if not acked:
            unacked_critical.append(
                {
                    "event": event,
                    "message": str(row.get("message") or ""),
                    "signature": signature,
                    "sent": bool(row.get("sent", False)),
                    "suppressed": bool(row.get("suppressed", False)),
                }
            )
        if not bool(row.get("sent", False)):
            unsent_critical.append({"event": event, "message": str(row.get("message") or ""), "signature": signature})

    grouped_rows = sorted(
        grouped_backlog.values(),
        key=lambda row: (
            not bool(row.get("unsent", False)),
            not bool(row.get("unacked", False)),
            -int(row.get("count", 0) or 0),
            str(row.get("event") or ""),
        ),
    )
    for row in grouped_rows:
        latest_age = row.get("latest_age_seconds")
        recent_enough = latest_age is not None and float(latest_age) <= float(active_window_seconds)
        open_surface_recent = (
            latest_age is not None
            and str(row.get("category") or "") in open_categories
            and float(latest_age) <= float(open_surface_window_seconds)
        )
        row["active"] = bool(recent_enough or open_surface_recent)
        row["effective_unsent"] = bool(row.get("unsent", False) and not imessage_bridge)
    grouped_active = [row for row in grouped_rows if bool(row.get("active", False))]
    grouped_unacked = [row for row in grouped_active if bool(row.get("unacked", False))]
    grouped_unsent = [row for row in grouped_active if bool(row.get("effective_unsent", False))]
    dedupe_ratio = round((len(rows) / max(len(grouped_rows), 1)), 3) if rows else 1.0

    any_channel = any(channels.values())
    overall_status = "ready"
    if not any_channel or grouped_unsent:
        overall_status = "blocked"
    elif grouped_unacked:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "configure at least one remote pager channel before relying on multi-week unattended runtime" if not any_channel else "",
            "replace placeholder OPS_ALERT_WEBHOOK_URL with a real phone/pager webhook or remove it" if channel_config_errors.get("webhook") else "",
            (
                "iMessage bridge is configured for phone delivery; add Pushover or a real webhook only for unattended pager coverage"
                if imessage_bridge
                else "configure Pushover or a real webhook for phone delivery"
            )
            if not channels.get("remote_pager_configured", False)
            else "",
            "acknowledge critical alerts explicitly so the escalation backlog does not blur together" if grouped_unacked else "",
            "treat unsent critical alerts as a hard operational blocker" if grouped_unsent else "",
            "compact duplicate alert storms by signature before escalating so pager volume reflects unique incidents" if dedupe_ratio > 1.25 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "channels": {**channels, "any_configured": any_channel},
        "channel_config_errors": channel_config_errors,
        "alert_window_hours": int(hours),
        "severity_counts": severity_counts,
        "critical_backlog": {
            "unacked_count": len(grouped_unacked),
            "unsent_count": len(grouped_unsent),
            "raw_unacked_count": len(unacked_critical),
            "raw_unsent_count": len(unsent_critical),
            "unacked_events": unacked_critical[:12],
            "unsent_events": unsent_critical[:12],
        },
        "backlog_compaction": {
            "grouped_count": len(grouped_rows),
            "grouped_active_count": len(grouped_active),
            "grouped_unacked_count": len(grouped_unacked),
            "grouped_unsent_count": len(grouped_unsent),
            "dedupe_ratio": dedupe_ratio,
            "top_grouped_events": grouped_rows[:12],
        },
        "ack_state_path": str(ack_state_path),
        "infra_bots": ["remote_alert_control", "pager_alert_router", "process_watchdog"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track remote alert routing, acknowledgement, and escalation readiness.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--ack-state-path", default=str(DEFAULT_ACK_STATE_PATH))
    parser.add_argument("--ack-event", default="")
    parser.add_argument("--ack-note", default="")
    parser.add_argument("--ack-all-critical", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    ack_state_path = Path(args.ack_state_path).expanduser()
    if args.ack_event or args.ack_all_critical:
        state = _load_ack_state(ack_state_path)
        events = state.get("events") if isinstance(state.get("events"), dict) else {}
        payload = build_payload(root, hours=int(args.hours), ack_state_path=ack_state_path)
        critical_rows = payload.get("critical_backlog") if isinstance(payload.get("critical_backlog"), dict) else {}
        grouped = payload.get("backlog_compaction") if isinstance(payload.get("backlog_compaction"), dict) else {}
        if args.ack_all_critical:
            ack_names: set[str] = set()
            for row in critical_rows.get("unacked_events") or []:
                if not isinstance(row, dict):
                    continue
                ack_names.add(str(row.get("event") or "generic"))
            for row in grouped.get("top_grouped_events") or []:
                if not isinstance(row, dict) or not bool(row.get("unacked", False)):
                    continue
                ack_names.add(str(row.get("event") or "generic"))
            for event_name in sorted(name for name in ack_names if str(name).strip()):
                events[str(event_name).strip() or "generic"] = {"acknowledged_at_utc": iso_now(), "note": str(args.ack_note or "")}
        else:
            events[str(args.ack_event).strip() or "generic"] = {"acknowledged_at_utc": iso_now(), "note": str(args.ack_note or "")}
        state["events"] = events
        _save_ack_state(ack_state_path, state)

    payload = build_payload(root, hours=int(args.hours), ack_state_path=ack_state_path)
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "remote_alert_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"unacked_critical={int(((payload.get('critical_backlog') or {}).get('unacked_count', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
