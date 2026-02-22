import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser(description="Route pager alerts to webhook.")
    parser.add_argument("--severity", default="info", choices=["info", "warn", "critical"])
    parser.add_argument("--event", default="generic")
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "severity": args.severity,
        "event": args.event,
        "message": args.message,
    }

    ok = send(payload)
    log_path = PROJECT_ROOT / "governance" / "watchdog" / "pager_alerts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**payload, "sent": ok}, ensure_ascii=True) + "\n")

    print(json.dumps({**payload, "sent": ok}, ensure_ascii=True))
    return 0 if ok or not os.getenv("OPS_ALERT_WEBHOOK_URL", "").strip() else 2


if __name__ == "__main__":
    raise SystemExit(main())
