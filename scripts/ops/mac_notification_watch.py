#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
ALERTS_DIR = PROJECT_ROOT / "governance" / "alerts"
DEFAULT_STATE_PATH = HEALTH_DIR / "mac_notification_watch_state.json"
DEFAULT_PID_PATH = HEALTH_DIR / "mac_notification_watch.pid"
TRIPWIRE_PATH = HEALTH_DIR / "shadow_watchdog_tripwire_latest.json"
PROCESS_WATCHDOG_PATH = HEALTH_DIR / "process_watchdog_latest.json"
STORAGE_GUARD_PATH = HEALTH_DIR / "storage_mount_guard_latest.json"
GLOBAL_HALT_PATH = HEALTH_DIR / "GLOBAL_TRADING_HALT.flag"
INCIDENT_AUTO_HALT_PATH = ALERTS_DIR / "incident_auto_halt_latest.json"
PREFLIGHT_CRITICAL_PATH = ALERTS_DIR / "preflight_critical_latest.json"
IMESSAGE_ENABLED_ENV = "MAC_NOTIFICATION_WATCH_IMESSAGE_ENABLED"
IMESSAGE_RECIPIENT_ENV = "MAC_NOTIFICATION_WATCH_IMESSAGE_RECIPIENT"
MAX_ALERT_AGE_SECONDS_ENV = "MAC_NOTIFICATION_WATCH_MAX_ALERT_AGE_SECONDS"
IMESSAGE_MIN_SEVERITY_ENV = "MAC_NOTIFICATION_WATCH_IMESSAGE_MIN_SEVERITY"
IMESSAGE_EVENT_ALLOWLIST_ENV = "MAC_NOTIFICATION_WATCH_IMESSAGE_EVENT_ALLOWLIST"
DEFAULT_MAX_ALERT_AGE_SECONDS = 900.0
DEFAULT_IMESSAGE_MIN_SEVERITY = "warn"
DEFAULT_IMESSAGE_EVENT_ALLOWLIST = ""
PMSET_POWER_LOG_TAIL_LINES = 240
PMSET_POWER_LOG_TIMEOUT_SECONDS = 8.0
PMSET_LINE_RE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})\s+"
    r"(?P<kind>\S+)\s+(?P<message>.*)$"
)
SEVERITY_RANK = {"info": 0, "warn": 1, "warning": 1, "critical": 2}


def _clean_token(value: str) -> str:
    return str(value or "").strip().replace("_", " ")


def _title_token(value: str) -> str:
    cleaned = _clean_token(value)
    return cleaned.title() if cleaned else "Unknown"


def _compact_context(profile: str, broker: str) -> str:
    return f"{_title_token(profile)} / {_title_token(broker)}"


def _human_event_label(event: str) -> str:
    labels = {
        "lane_kill_switch_engaged": "Lane Kill Switch",
        "options_margin_guard": "Margin Guard",
        "critical_alert": "Critical Alert",
    }
    key = str(event or "").strip().lower()
    return labels.get(key, _title_token(key))


def _normalize_severity(value: str, default: str = "warn") -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "warning":
        normalized = "warn"
    return normalized if normalized in SEVERITY_RANK else default


def _severity_at_least(current: str, minimum: str) -> bool:
    current_rank = SEVERITY_RANK[_normalize_severity(current, "info")]
    minimum_rank = SEVERITY_RANK[_normalize_severity(minimum, DEFAULT_IMESSAGE_MIN_SEVERITY)]
    return current_rank >= minimum_rank


def _event_family(key: str) -> str:
    normalized_key = str(key or "").strip().lower()
    if normalized_key.startswith("critical_alert:"):
        return "critical_alert"
    if normalized_key.startswith("tripwire:"):
        return "tripwire"
    if normalized_key.startswith("restart_storm:"):
        return "restart_storm"
    return normalized_key


def _parse_imessage_event_allowlist(value: str) -> set[str]:
    tokens = {str(part or "").strip().lower() for part in str(value or "").split(",")}
    tokens.discard("")
    if not tokens:
        return set()
    if tokens & {"*", "all", "any"}:
        return {"*"}
    return tokens


def _imessage_event_allowed(key: str, allowlist: set[str]) -> bool:
    if not allowlist or "*" in allowlist:
        return True
    normalized_key = str(key or "").strip().lower()
    family = _event_family(normalized_key)
    return normalized_key in allowlist or family in allowlist


def _event_severity(key: str, message: str) -> str:
    normalized_key = str(key or "").strip().lower()
    if normalized_key.startswith("power_clamshell_sleep:"):
        return "critical"
    if normalized_key.startswith("power_lid_open:"):
        return "info"
    if normalized_key.startswith("critical_alert:"):
        parts = normalized_key.split(":", 2)
        if len(parts) >= 3:
            parsed = _normalize_severity(parts[1], "")
            if parsed:
                return parsed
        lowered = str(message or "").lower()
        if lowered.startswith("margin guard") or lowered.startswith("futures margin guard") or lowered.startswith("warning"):
            return "warn"
        return "critical"
    if normalized_key.startswith("tripwire:") or normalized_key.startswith("restart_storm:"):
        return "critical"
    if normalized_key in {"tripwire", "all_sleeves_down", "global_halt", "incident_auto_halt", "preflight_critical", "storage_mount_missing"}:
        return "critical"
    return "warn"


def _notification_heading(key: str, message: str) -> Tuple[str, str]:
    severity = _event_severity(key, message)
    if key.startswith("power_clamshell_sleep:"):
        return ("Trading Bot Critical", "Laptop Closed")
    if key.startswith("power_lid_open:"):
        return ("Trading Bot Incident", "Laptop Opened")
    if key.startswith("critical_alert:"):
        if severity == "warn":
            return ("Trading Bot Warning", "Guardrail Warning")
        return ("Trading Bot Critical", "Critical Guardrail")
    if key == "tripwire":
        return ("Trading Bot Critical", "Tripwire")
    if key.startswith("tripwire:"):
        return ("Trading Bot Critical", "Tripwire")
    if key.startswith("restart_storm:"):
        return ("Trading Bot Critical", "Restart Storm")
    if key == "all_sleeves_down":
        return ("Trading Bot Critical", "Sleeves Down")
    if key == "global_halt":
        return ("Trading Bot Critical", "Global Halt")
    if key == "incident_auto_halt":
        return ("Trading Bot Critical", "Auto Halt")
    if key == "preflight_critical":
        return ("Trading Bot Critical", "Preflight")
    if key == "storage_mount_missing":
        return ("Trading Bot Critical", "Storage Route")
    if severity == "critical":
        return ("Trading Bot Critical", "Trading Bot Alert")
    if severity == "warn":
        return ("Trading Bot Warning", "Trading Bot Alert")
    return ("Trading Bot Incident", "Trading Bot Alert")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if value >= 0.0 else default


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _notify_mac(title: str, body: str, subtitle: str = "Trading Bot Alert") -> Dict[str, Any]:
    script = 'display notification "{}" with title "{}" subtitle "{}"'.format(
        _escape_applescript_string(body),
        _escape_applescript_string(title),
        _escape_applescript_string(subtitle),
    )
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    return {
        "channel": "mac",
        "returncode": int(proc.returncode),
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
def _compose_imessage_text(title: str, body: str) -> str:
    return f"{title}\n{body}"


def _notify_imessage(title: str, body: str, recipient: str) -> Dict[str, Any]:
    applescript = """on run argv
set targetRecipient to item 1 of argv
set targetMessage to item 2 of argv

tell application "Messages"
  if not running then
    launch
    delay 1
  end if
  try
    set targetParticipant to participant targetRecipient
    send targetMessage to targetParticipant
  on error
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy targetRecipient of targetService
    send targetMessage to targetBuddy
  end try
end tell
end run
"""
    proc = subprocess.run(
        ["osascript", "-", recipient, _compose_imessage_text(title, body)],
        input=applescript,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "channel": "imessage",
        "recipient": recipient,
        "returncode": int(proc.returncode),
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
def _notify(
    title: str,
    body: str,
    subtitle: str = "Trading Bot Alert",
    *,
    imessage_enabled: bool = False,
    imessage_recipient: str = "",
    imessage_min_severity: str = DEFAULT_IMESSAGE_MIN_SEVERITY,
    severity: str = "warn",
) -> Dict[str, Any]:
    mac_result = _notify_mac(title, body, subtitle=subtitle)
    out: Dict[str, Any] = {
        "mac": mac_result,
        "imessage_attempted": False,
        "imessage": None,
    }
    if imessage_enabled and imessage_recipient.strip() and _severity_at_least(severity, imessage_min_severity):
        out["imessage_attempted"] = True
        out["imessage"] = _notify_imessage(title, body, imessage_recipient.strip())
    return out
def _parse_timestamp(payload: Dict[str, Any]) -> datetime | None:
    raw = str(payload.get("timestamp_utc", "")).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _is_recent(payload: Dict[str, Any], max_age_seconds: float) -> bool:
    if max_age_seconds <= 0.0:
        return True
    ts = _parse_timestamp(payload)
    if ts is None:
        return False
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return 0.0 <= age <= max_age_seconds


def _parse_pmset_timestamp(raw: str) -> datetime | None:
    try:
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S %z").astimezone(timezone.utc)
    except Exception:
        return None


def _recent_pmset_lines(limit: int = PMSET_POWER_LOG_TAIL_LINES) -> List[str]:
    tail = max(int(limit), 1)
    proc = subprocess.run(
        ["/bin/zsh", "-lc", f"/usr/bin/pmset -g log | tail -n {tail}"],
        capture_output=True,
        text=True,
        timeout=PMSET_POWER_LOG_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def _power_event_candidates(max_age_seconds: float) -> List[Tuple[str, str]]:
    latest_close: Tuple[str, str, datetime] | None = None
    latest_open: Tuple[str, str, datetime] | None = None
    for line in _recent_pmset_lines():
        match = PMSET_LINE_RE.match(line)
        if match is None:
            continue
        ts = _parse_pmset_timestamp(match.group("stamp"))
        if ts is None:
            continue
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age < 0.0 or age > max_age_seconds:
            continue
        kind = match.group("kind").strip().lower()
        message = match.group("message").strip()
        local_stamp = ts.astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z")
        if kind == "sleep" and "Clamshell Sleep" in message and "Entering Sleep state" in message:
            latest_close = (
                f"power_clamshell_sleep:{ts.isoformat()}",
                "MacBook lid closed\n"
                f"Sleep: {local_stamp}\n"
                "Live bot collection/training pauses while the laptop sleeps.",
                ts,
            )
        elif "lidopen" in message:
            latest_open = (
                f"power_lid_open:{ts.isoformat()}",
                "MacBook lid opened\n"
                f"Wake: {local_stamp}",
                ts,
            )
    out: List[Tuple[str, str]] = []
    if latest_close is not None:
        out.append((latest_close[0], latest_close[1]))
    if latest_open is not None:
        out.append((latest_open[0], latest_open[1]))
    return out


def _tripwire_event(payload: Dict[str, Any]) -> Tuple[str, str] | None:
    if not bool(payload.get("active", False)):
        return None
    incidents = payload.get("active_incidents", []) if isinstance(payload.get("active_incidents"), list) else []
    targets = ",".join(str(x.get("target", "")) for x in incidents if str(x.get("target", "")).strip()) or "unknown"
    return (f"tripwire:{targets}", f"Tripwire triggered for {targets}")


def _global_halt_event() -> Tuple[str, str] | None:
    if not GLOBAL_HALT_PATH.exists():
        return None
    return ("global_halt", "GLOBAL_TRADING_HALT is set")


def _restart_storm_event(payload: Dict[str, Any]) -> Tuple[str, str] | None:
    storms = payload.get("restart_storms", [])
    if not isinstance(storms, list) or not storms:
        return None
    names = ",".join(str(x.get("name", "")) for x in storms if str(x.get("name", "")).strip()) or "unknown"
    return (f"restart_storm:{names}", f"Restart storm on {names}")


def _all_sleeves_down_event(payload: Dict[str, Any]) -> Tuple[str, str] | None:
    for row in payload.get("status", []) if isinstance(payload.get("status"), list) else []:
        if str(row.get("name", "")) == "all_sleeves" and int(row.get("running", 0) or 0) == 0:
            if (not bool(row.get("heartbeat_ok", False))) and int(row.get("alt_running", 0) or 0) == 0:
                return ("all_sleeves_down", "All sleeves are down and not heartbeating")
    return None


def _storage_event(payload: Dict[str, Any]) -> Tuple[str, str] | None:
    if payload and (not bool(payload.get("external_available", False))):
        root = str(payload.get("mount_root", "/Volumes/BOT_LOGS"))
        reason = str(payload.get("external_unavailable_reason", "")).strip().lower()
        if reason == 'low_space':
            return ("storage_mount_missing", f"Storage route unavailable: {root} (external low space)")
        return ("storage_mount_missing", f"Storage route unavailable: {root}")
    return None


def _critical_alert_events(max_age_seconds: float) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for path in sorted(ALERTS_DIR.glob("critical_latest_*.json")):
        payload = _read_json(path)
        if not payload or not _is_recent(payload, max_age_seconds):
            continue
        severity = str(payload.get("severity", "critical")).strip().lower() or "critical"
        event = str(payload.get("event", "critical_alert")).strip() or "critical_alert"
        message = str(payload.get("message", "")).strip() or event
        profile = str(payload.get("profile", "default")).strip() or "default"
        broker = str(payload.get("broker", "unknown")).strip() or "unknown"
        key = f"critical_alert:{_normalize_severity(severity, 'critical')}:{path.stem}"
        label = _human_event_label(event)
        context = _compact_context(profile, broker)
        out.append((key, f"{label} [{context}]\n{message}"))
    return out


def _incident_auto_halt_event(payload: Dict[str, Any], max_age_seconds: float) -> Tuple[str, str] | None:
    if not payload or not _is_recent(payload, max_age_seconds):
        return None
    failed_checks = payload.get("failed_checks", []) if isinstance(payload.get("failed_checks"), list) else []
    halt = bool(payload.get("halt", False))
    ok = bool(payload.get("ok", True))
    if ok and not halt and not failed_checks:
        return None
    checks = ",".join(str(x) for x in failed_checks if str(x).strip()) or "unknown"
    state = "HALTED" if halt else "FAILED"
    return ("incident_auto_halt", f"Incident auto-halt {state.lower()}\nChecks: {checks}")


def _preflight_critical_event(payload: Dict[str, Any], max_age_seconds: float) -> Tuple[str, str] | None:
    if not payload or not _is_recent(payload, max_age_seconds):
        return None
    failed_checks = payload.get("failed_checks", []) if isinstance(payload.get("failed_checks"), list) else []
    if not failed_checks:
        return None
    names: List[str] = []
    for item in failed_checks:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    broker = str(payload.get("broker", "unknown")).strip() or "unknown"
    summary = ", ".join(names) or "unknown"
    return ("preflight_critical", f"Preflight critical [{_title_token(broker)}]\nChecks: {summary}")


def _load_state(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        data = {}
    data.setdefault("sent", {})
    return data


def _event_candidates(max_age_seconds: float) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    out.extend(_power_event_candidates(max_age_seconds))
    for event in (
        _tripwire_event(_read_json(TRIPWIRE_PATH)),
        _global_halt_event(),
        _restart_storm_event(_read_json(PROCESS_WATCHDOG_PATH)),
        _all_sleeves_down_event(_read_json(PROCESS_WATCHDOG_PATH)),
        _storage_event(_read_json(STORAGE_GUARD_PATH)),
        _incident_auto_halt_event(_read_json(INCIDENT_AUTO_HALT_PATH), max_age_seconds),
        _preflight_critical_event(_read_json(PREFLIGHT_CRITICAL_PATH), max_age_seconds),
    ):
        if event is not None:
            out.append(event)
    out.extend(_critical_alert_events(max_age_seconds))
    return out


def _run_watch_loop(
    state_path: Path,
    poll_seconds: float,
    *,
    imessage_enabled: bool = False,
    imessage_recipient: str = "",
    imessage_min_severity: str = DEFAULT_IMESSAGE_MIN_SEVERITY,
    imessage_event_allowlist: str = DEFAULT_IMESSAGE_EVENT_ALLOWLIST,
) -> int:
    state = _load_state(state_path)
    sent: Dict[str, str] = dict((state.get("sent") or {}))
    last_delivery = state.get("last_delivery")
    max_age_seconds = _env_float(MAX_ALERT_AGE_SECONDS_ENV, DEFAULT_MAX_ALERT_AGE_SECONDS)
    normalized_imessage_min_severity = _normalize_severity(imessage_min_severity, DEFAULT_IMESSAGE_MIN_SEVERITY)
    parsed_imessage_event_allowlist = _parse_imessage_event_allowlist(imessage_event_allowlist)
    while True:
        active_keys = set()
        for key, message in _event_candidates(max_age_seconds):
            active_keys.add(key)
            if sent.get(key) != message:
                title, subtitle = _notification_heading(key, message)
                severity = _event_severity(key, message)
                delivery = _notify(
                    title,
                    message,
                    subtitle=subtitle,
                    imessage_enabled=bool(imessage_enabled and _imessage_event_allowed(key, parsed_imessage_event_allowlist)),
                    imessage_recipient=imessage_recipient,
                    imessage_min_severity=normalized_imessage_min_severity,
                    severity=severity,
                )
                last_delivery = delivery
                print(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event_key": key,
                    "message": message,
                    "severity": severity,
                    "delivery": delivery,
                }, ensure_ascii=True), flush=True)
                sent[key] = message
        for key in list(sent.keys()):
            if key not in active_keys:
                sent.pop(key, None)
        _write_json(
            state_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sent": sent,
                "imessage_enabled": bool(imessage_enabled),
                "imessage_recipient_configured": bool(imessage_recipient.strip()),
                "imessage_min_severity": normalized_imessage_min_severity,
                "imessage_event_allowlist": (["*"] if "*" in parsed_imessage_event_allowlist else sorted(parsed_imessage_event_allowlist)),
                "max_alert_age_seconds": max_age_seconds,
                "last_delivery": last_delivery,
            },
        )
        time.sleep(max(poll_seconds, 2.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Send macOS notifications for bot crashes/halts.")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--pid-file", default=str(DEFAULT_PID_PATH))
    parser.add_argument("--poll-seconds", type=float, default=8.0)
    parser.add_argument("--imessage-recipient", default=os.environ.get(IMESSAGE_RECIPIENT_ENV, "").strip())
    parser.add_argument(
        "--imessage-min-severity",
        default=os.environ.get(IMESSAGE_MIN_SEVERITY_ENV, DEFAULT_IMESSAGE_MIN_SEVERITY).strip() or DEFAULT_IMESSAGE_MIN_SEVERITY,
        choices=["info", "warn", "critical"],
    )
    parser.add_argument(
        "--imessage-event-allowlist",
        default=os.environ.get(IMESSAGE_EVENT_ALLOWLIST_ENV, DEFAULT_IMESSAGE_EVENT_ALLOWLIST).strip(),
        help="Comma-separated event families/keys allowed to send iMessage alerts.",
    )
    parser.add_argument("--enable-imessage", dest="imessage_enabled", action="store_true")
    parser.add_argument("--disable-imessage", dest="imessage_enabled", action="store_false")
    parser.add_argument("--test", action="store_true")
    parser.set_defaults(imessage_enabled=_env_flag(IMESSAGE_ENABLED_ENV, False))
    args = parser.parse_args()

    pid_path = Path(args.pid_file).expanduser()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        if args.test:
            _notify(
                "Trading Bot Incident",
                "Notification watcher test",
                imessage_enabled=bool(args.imessage_enabled),
                imessage_recipient=str(args.imessage_recipient),
                imessage_min_severity=str(args.imessage_min_severity),
                severity="critical",
            )
            return 0
        return _run_watch_loop(
            Path(args.state_file).expanduser(),
            float(args.poll_seconds),
            imessage_enabled=bool(args.imessage_enabled),
            imessage_recipient=str(args.imessage_recipient),
            imessage_min_severity=str(args.imessage_min_severity),
            imessage_event_allowlist=str(args.imessage_event_allowlist),
        )
    finally:
        try:
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
