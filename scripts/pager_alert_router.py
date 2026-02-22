import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "governance" / "watchdog" / "pager_alert_state.json"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def send(payload: dict) -> bool:
    url = os.getenv("OPS_ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return False
    try:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=4):
            pass
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Route pager alerts to webhook with dedupe/suppression.")
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

    ok = False
    if can_send:
        ok = send(payload)
        state[key] = now_iso
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    log_path = PROJECT_ROOT / "governance" / "watchdog" / "pager_alerts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**payload, "sent": ok}, ensure_ascii=True) + "\n")

    print(json.dumps({**payload, "sent": ok}, ensure_ascii=True))
    webhook_set = bool(os.getenv("OPS_ALERT_WEBHOOK_URL", "").strip())
    return 0 if (ok or not webhook_set or not can_send) else 2


if __name__ == "__main__":
    raise SystemExit(main())
