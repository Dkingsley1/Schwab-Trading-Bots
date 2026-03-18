import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "governance" / "watchdog" / "pager_alert_state.json"
PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _webhook_url() -> str:
    return os.getenv("OPS_ALERT_WEBHOOK_URL", "").strip()


def _pushover_config() -> Dict[str, str]:
    return {
        "token": os.getenv("OPS_ALERT_PUSHOVER_TOKEN", "").strip(),
        "user": os.getenv("OPS_ALERT_PUSHOVER_USER_KEY", "").strip(),
        "device": os.getenv("OPS_ALERT_PUSHOVER_DEVICE", "").strip(),
        "priority": os.getenv("OPS_ALERT_PUSHOVER_PRIORITY", "0").strip() or "0",
        "sound": os.getenv("OPS_ALERT_PUSHOVER_SOUND", "").strip(),
    }


def _configured_channels() -> Dict[str, bool]:
    pushover = _pushover_config()
    return {
        "webhook": bool(_webhook_url()),
        "pushover": bool(pushover["token"] and pushover["user"]),
    }


def _post_json(url: str, payload: dict) -> bool:
    try:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=4):
            pass
        return True
    except Exception:
        return False


def _severity_title(severity: str) -> str:
    sev = str(severity or "info").strip().upper()
    return f"Trading Bot {sev}"


def _send_webhook(payload: dict) -> bool:
    url = _webhook_url()
    if not url:
        return False
    return _post_json(url, payload)


def _send_pushover(payload: dict) -> bool:
    cfg = _pushover_config()
    if not (cfg["token"] and cfg["user"]):
        return False

    title = _severity_title(payload.get("severity", "info"))
    event = str(payload.get("event", "")).strip()
    if event:
        title = f"{title}: {event}"

    message = {
        "token": cfg["token"],
        "user": cfg["user"],
        "title": title,
        "message": str(payload.get("message", "")).strip() or event or "generic",
        "priority": cfg["priority"],
    }
    if cfg["device"]:
        message["device"] = cfg["device"]
    if cfg["sound"]:
        message["sound"] = cfg["sound"]

    try:
        body = urllib.parse.urlencode(message).encode("utf-8")
        req = urllib.request.Request(PUSHOVER_API_URL, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=4):
            pass
        return True
    except Exception:
        return False


def send(payload: dict) -> Dict[str, Any]:
    configured = _configured_channels()
    results = {
        "webhook": False,
        "pushover": False,
        "configured": configured,
        "any_configured": any(configured.values()),
    }
    if configured["webhook"]:
        results["webhook"] = _send_webhook(payload)
    if configured["pushover"]:
        results["pushover"] = _send_pushover(payload)
    results["any_sent"] = bool(results["webhook"] or results["pushover"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Route pager alerts to webhook and optional phone push with dedupe/suppression.")
    parser.add_argument("--severity", default="info", choices=["info", "warn", "critical"])
    parser.add_argument("--event", default="generic")
    parser.add_argument("--message", default="")
    parser.add_argument("--suppress-seconds", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    state = _load(STATE_PATH)
    key = f"{args.severity}:{args.event}:{args.message}"[:300]

    can_send = True
    if not args.force and key in state:
        try:
            prev = datetime.fromisoformat(str(state[key]).replace("Z", "+00:00")).astimezone(timezone.utc)
            age = (now - prev).total_seconds()
            if age < args.suppress_seconds:
                can_send = False
        except Exception:
            pass

    payload = {
        "timestamp_utc": now_iso,
        "severity": args.severity,
        "event": args.event,
        "message": args.message,
        "suppressed": not can_send,
    }

    channel_results: Dict[str, Any] = {
        "webhook": False,
        "pushover": False,
        "configured": _configured_channels(),
        "any_configured": any(_configured_channels().values()),
        "any_sent": False,
    }
    if can_send:
        channel_results = send(payload)
        state[key] = now_iso
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    log_path = PROJECT_ROOT / "governance" / "watchdog" / "pager_alerts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {**payload, **channel_results, "sent": bool(channel_results.get("any_sent", False))}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps(row, ensure_ascii=True))
    if not can_send:
        return 0
    if not channel_results.get("any_configured", False):
        return 0
    return 0 if channel_results.get("any_sent", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
