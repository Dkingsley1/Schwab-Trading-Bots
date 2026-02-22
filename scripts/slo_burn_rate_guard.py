import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _notify(payload: dict) -> None:
    url = os.getenv("OPS_ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        import urllib.request
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=4):
            pass
    except Exception:
        pass


def _append_event(payload: dict) -> None:
    p = PROJECT_ROOT / "governance" / "watchdog" / "slo_burn_events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="SLO burn-rate style health alerts.")
    parser.add_argument("--max-heartbeat-age-sec", type=float, default=180.0)
    parser.add_argument("--max-pending-lines", type=int, default=25000)
    parser.add_argument("--max-watchdog-restarts", type=int, default=3)
    args = parser.parse_args()

    ready = _load(PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json")
    ingest = _load(PROJECT_ROOT / "governance" / "health" / "ingestion_backpressure_latest.json")
    one = _load(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json")

    checks = []
    hb_age = 1e9
    for c in ready.get("checks", []):
        if c.get("name") == "heartbeat_freshness":
            d = str(c.get("details", ""))
            if "heartbeat_age_sec=" in d:
                try:
                    hb_age = float(d.split("heartbeat_age_sec=")[-1])
                except Exception:
                    pass
    checks.append({"name": "heartbeat_age", "ok": hb_age <= args.max_heartbeat_age_sec, "value": hb_age})

    pending = int(ingest.get("pending_lines", 0) or 0)
    checks.append({"name": "ingest_pending_lines", "ok": pending <= args.max_pending_lines, "value": pending})

    restarts = int(float(one.get("watchdog_restarts", 0) or 0)) if one else 0
    checks.append({"name": "watchdog_restarts", "ok": restarts <= args.max_watchdog_restarts, "value": restarts})

    burn = sum(0 if c["ok"] else 1 for c in checks)
    severity = "ok" if burn == 0 else ("warn" if burn == 1 else "critical")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slo_burn_score": burn,
        "severity": severity,
        "checks": checks,
    }

    out = PROJECT_ROOT / "governance" / "health" / "slo_burn_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _append_event(payload)
    if severity in {"warn", "critical"}:
        _notify({"event": "slo_burn", "severity": severity, "payload": payload})
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if severity == "ok" else (2 if severity == "warn" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
