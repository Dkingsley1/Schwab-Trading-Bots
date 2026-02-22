import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "governance" / "watchdog" / "failover_events.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _proc_alive(match: str) -> bool:
    p = subprocess.run(["ps", "-ax", "-o", "command="], capture_output=True, text=True, check=False)
    return any(match in line for line in (p.stdout or "").splitlines())


def _heartbeat_age_sec(path: Path) -> float:
    if not path.exists():
        return 1e9
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ts = str(payload.get("timestamp_utc", "")).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        return max((datetime.now(timezone.utc) - dt).total_seconds(), 0.0)
    except Exception:
        return 1e9


def _start_cmd(cmd: str) -> bool:
    try:
        subprocess.Popen(shlex.split(cmd), cwd=str(PROJECT_ROOT))
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Hot-standby failover monitor for shadow runtime.")
    parser.add_argument("--primary-match", default="scripts/run_parallel_shadows.py")
    parser.add_argument("--primary-heartbeat", default=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_conservative_equities.json"))
    parser.add_argument("--max-heartbeat-age-sec", type=float, default=150.0)
    parser.add_argument("--standby-start-cmd", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=20)
    args = parser.parse_args()

    standby_cmd = args.standby_start_cmd.strip()
    if not standby_cmd:
        standby_cmd = (
            f"{PROJECT_ROOT}/.venv312/bin/python {PROJECT_ROOT}/scripts/run_parallel_shadows.py "
            "--simulate"
        )

    while True:
        alive = _proc_alive(args.primary_match)
        hb_age = _heartbeat_age_sec(Path(args.primary_heartbeat))
        stale = hb_age > args.max_heartbeat_age_sec
        event = {
            "timestamp_utc": _now_iso(),
            "primary_alive": alive,
            "heartbeat_age_sec": hb_age,
            "stale": stale,
            "action": "none",
        }

        if (not alive) or stale:
            ok = _start_cmd(standby_cmd)
            event["action"] = "standby_start_attempt"
            event["standby_ok"] = ok
            event["standby_cmd"] = standby_cmd

        _append(event)
        print(json.dumps(event, ensure_ascii=True))

        if args.once:
            return 0
        time.sleep(max(args.interval_seconds, 5))


if __name__ == "__main__":
    raise SystemExit(main())
