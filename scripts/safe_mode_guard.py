import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.halt_flags import write_halt_flag_atomic

HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "health_gates_latest.json"
STATE_PATH = PROJECT_ROOT / "governance" / "health" / "safe_mode_state.json"
HALT_FLAG = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
EVENT_LOG = PROJECT_ROOT / "governance" / "watchdog" / "safe_mode_events.jsonl"


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
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=4):
            pass
    except Exception:
        pass


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Escalate to global halt after repeated hard gates.")
    parser.add_argument("--trip-streak", type=int, default=3)
    parser.add_argument("--clear-streak", type=int, default=2)
    parser.add_argument("--force-clear", action="store_true")
    parser.add_argument("--auto-clear", action="store_true", default=os.getenv("SAFE_MODE_AUTO_CLEAR", "0") == "1")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    state = _load(STATE_PATH)
    state.setdefault("trip_streak", 0)
    state.setdefault("clear_streak", 0)

    if args.force_clear:
        if HALT_FLAG.exists():
            HALT_FLAG.unlink()
        state["trip_streak"] = 0
        state["clear_streak"] = 0
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
        _append_jsonl(EVENT_LOG, {"timestamp_utc": now, "event": "halt_force_cleared"})
        print("safe_mode:halt_cleared")
        return 0

    health = _load(HEALTH_PATH)
    tripped = bool(health.get("hard_gate_triggered", False))

    if tripped:
        state["trip_streak"] = int(state.get("trip_streak", 0)) + 1
        state["clear_streak"] = 0
    else:
        state["clear_streak"] = int(state.get("clear_streak", 0)) + 1
        state["trip_streak"] = 0

    event = {
        "timestamp_utc": now,
        "hard_gate_triggered": tripped,
        "trip_streak": state["trip_streak"],
        "clear_streak": state["clear_streak"],
    }

    if state["trip_streak"] >= max(args.trip_streak, 1):
        write_halt_flag_atomic(
            HALT_FLAG,
            {"timestamp_utc": now, "reason": "repeated_hard_gates"},
            project_root=str(PROJECT_ROOT),
            source="safe_mode_guard",
        )
        event["event"] = "halt_set"
        _notify({"event": "halt_set", "timestamp_utc": now, "trip_streak": state["trip_streak"], "reason": "repeated_hard_gates"})
    elif HALT_FLAG.exists() and args.auto_clear and state["clear_streak"] >= max(args.clear_streak, 1):
        HALT_FLAG.unlink()
        event["event"] = "halt_cleared"
    else:
        event["event"] = "state_update"

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    _append_jsonl(EVENT_LOG, event)

    print(f"safe_mode_event={event['event']} trip_streak={state['trip_streak']} clear_streak={state['clear_streak']} halt={HALT_FLAG.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
