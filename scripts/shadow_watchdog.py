import argparse
import glob
import json
import shlex
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
PARALLEL_SHADOW_SCRIPT = PROJECT_ROOT / "scripts" / "run_parallel_shadows.py"
SHADOW_LOOP_SCRIPT = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"


@dataclass
class Target:
    name: str
    match: str
    start_cmd: Optional[str]
    required: bool = True
    restart_times: Deque[float] = field(default_factory=deque)
    heartbeat_glob: Optional[str] = None
    heartbeat_stale_seconds: int = 0
    min_healthy_heartbeats: int = 1


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _event_log_path(day: Optional[str] = None) -> Path:
    d = day or _now_utc().strftime("%Y%m%d")
    return WATCHDOG_DIR / f"watchdog_events_{d}.jsonl"


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _scan_process_rows() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout or ""
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append((pid, parts[1]))
    return rows


def _find_matching_rows(rows: list[tuple[int, str]], match: str) -> list[tuple[int, str]]:
    return [(pid, cmd) for pid, cmd in rows if match in cmd]


def _terminate_pids(pids: list[int], timeout_seconds: float = 8.0) -> None:
    if not pids:
        return
    for pid in pids:
        subprocess.run(["kill", "-TERM", str(pid)], check=False)

    start = time.time()
    alive = set(pids)
    while alive and (time.time() - start) < timeout_seconds:
        for pid in list(alive):
            probe = subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode != 0:
                alive.discard(pid)
        if alive:
            time.sleep(0.2)

    for pid in alive:
        subprocess.run(["kill", "-KILL", str(pid)], check=False)


def _prune_restart_times(history: Deque[float], now_ts: float, window_seconds: int) -> None:
    while history and (now_ts - history[0]) > window_seconds:
        history.popleft()


def _can_restart(target: Target, now_ts: float, max_restarts: int, window_seconds: int) -> bool:
    _prune_restart_times(target.restart_times, now_ts, window_seconds)
    return len(target.restart_times) < max_restarts


def _start_target(start_cmd: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        args = shlex.split(start_cmd)
        subprocess.Popen(args, cwd=str(PROJECT_ROOT))
        return True
    except Exception:
        return False


def _build_default_schwab_cmd(simulate: bool) -> str:
    base = f"{VENV_PY} {PARALLEL_SHADOW_SCRIPT}"
    if simulate:
        return base + " --simulate"
    return base


def _build_default_coinbase_cmd() -> str:
    return (
        f"{VENV_PY} {SHADOW_LOOP_SCRIPT} "
        "--broker coinbase "
        "--symbols BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD "
        "--interval-seconds 60"
    )


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _heartbeat_health(target: Target) -> tuple[bool, int, Optional[float]]:
    if not target.heartbeat_glob or target.heartbeat_stale_seconds <= 0:
        return True, 0, None

    now = _now_utc()
    healthy = 0
    latest_age: Optional[float] = None

    for fp in glob.glob(target.heartbeat_glob):
        path = Path(fp)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts = _parse_ts(str(payload.get("timestamp_utc", "")))
        if ts is None:
            continue
        age = max((now - ts).total_seconds(), 0.0)
        latest_age = age if latest_age is None else min(latest_age, age)
        if age <= target.heartbeat_stale_seconds:
            healthy += 1

    return healthy >= max(target.min_healthy_heartbeats, 1), healthy, latest_age


def _status_payload(entries: list[dict]) -> dict:
    return {
        "timestamp_utc": _now_iso(),
        "targets": entries,
    }


def _run_iteration(
    targets: list[Target],
    max_restarts_per_window: int,
    restart_window_seconds: int,
    dry_run: bool,
    emit_json: bool,
    event_log_path: Optional[Path],
) -> int:
    rows = _scan_process_rows()
    now_ts = time.time()
    overall_rc = 0
    entries: list[dict] = []

    for target in targets:
        matches = _find_matching_rows(rows, target.match)
        pids = [pid for pid, _ in matches]
        proc_live = len(matches) > 0

        hb_ok, hb_count, hb_age = _heartbeat_health(target)
        hb_required = bool(target.heartbeat_glob and target.heartbeat_stale_seconds > 0)
        live = proc_live and (hb_ok if hb_required else True)

        note_parts = []
        if proc_live:
            note_parts.append("process_live")
        else:
            note_parts.append("process_missing")
        if hb_required:
            note_parts.append(f"heartbeat_ok={hb_ok}")
            note_parts.append(f"heartbeat_count={hb_count}")
            if hb_age is not None:
                note_parts.append(f"heartbeat_age_s={hb_age:.1f}")

        entry: Dict[str, object] = {
            "name": target.name,
            "required": target.required,
            "match": target.match,
            "live": live,
            "process_live": proc_live,
            "match_count": len(matches),
            "match_pids": pids,
            "action": "none",
            "note": ",".join(note_parts),
        }

        if live:
            pass
        elif not target.required:
            entry["note"] = entry["note"] + ",optional_target_missing"
        elif not target.start_cmd:
            overall_rc = 1
            entry["action"] = "error"
            entry["note"] = entry["note"] + ",missing_start_command"
        elif not _can_restart(target, now_ts, max_restarts_per_window, restart_window_seconds):
            overall_rc = 1
            entry["action"] = "throttled"
            entry["note"] = entry["note"] + ",restart_rate_limit"
        else:
            if proc_live:
                _terminate_pids(pids)
            ok = _start_target(target.start_cmd, dry_run=dry_run)
            if ok:
                target.restart_times.append(now_ts)
                entry["action"] = "restart"
                entry["note"] = entry["note"] + ",restart_attempted"
                entry["start_cmd"] = target.start_cmd
            else:
                overall_rc = 1
                entry["action"] = "error"
                entry["note"] = entry["note"] + ",restart_failed"
                entry["start_cmd"] = target.start_cmd

        entries.append(entry)

    payload = _status_payload(entries)

    if event_log_path is not None:
        _append_jsonl(event_log_path, payload)

    if emit_json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"[{payload['timestamp_utc']}] watchdog check")
        for e in entries:
            print(
                " - {name}: live={live} process_live={process_live} matches={match_count} action={action} note={note}".format(
                    name=e["name"],
                    live=e["live"],
                    process_live=e["process_live"],
                    match_count=e["match_count"],
                    action=e["action"],
                    note=e["note"],
                )
            )
    return overall_rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watchdog for shadow bot processes (Schwab parallel + optional Coinbase + heartbeat staleness)."
    )
    parser.add_argument("--once", action="store_true", help="Run one check and exit.")
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--max-restarts-per-window", type=int, default=4)
    parser.add_argument("--restart-window-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")

    parser.add_argument("--simulate-schwab", action="store_true", help="Default Schwab start command adds --simulate.")
    parser.add_argument("--schwab-start-cmd", default=None)
    parser.add_argument("--coinbase-start-cmd", default=None)
    parser.add_argument("--watch-coinbase", action="store_true")
    parser.add_argument("--coinbase-optional", action="store_true")

    parser.add_argument("--schwab-heartbeat-stale-seconds", type=int, default=120)
    parser.add_argument("--coinbase-heartbeat-stale-seconds", type=int, default=180)
    parser.add_argument("--schwab-min-heartbeats", type=int, default=2)
    parser.add_argument("--coinbase-min-heartbeats", type=int, default=1)

    parser.add_argument(
        "--event-log-path",
        default=str(_event_log_path()),
        help="JSONL path for watchdog events (default: governance/watchdog/watchdog_events_YYYYMMDD.jsonl).",
    )
    parser.add_argument("--no-event-log", action="store_true")
    args = parser.parse_args()

    schwab_cmd = args.schwab_start_cmd or _build_default_schwab_cmd(simulate=args.simulate_schwab)
    coinbase_cmd = args.coinbase_start_cmd or _build_default_coinbase_cmd()

    targets: list[Target] = [
        Target(
            name="schwab_parallel",
            match="scripts/run_parallel_shadows.py",
            start_cmd=schwab_cmd,
            required=True,
            heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*_equities_schwab_*.json"),
            heartbeat_stale_seconds=max(args.schwab_heartbeat_stale_seconds, 30),
            min_healthy_heartbeats=max(args.schwab_min_heartbeats, 1),
        )
    ]

    if args.watch_coinbase:
        targets.append(
            Target(
                name="coinbase_shadow",
                match="scripts/run_shadow_training_loop.py --broker coinbase",
                start_cmd=coinbase_cmd,
                required=not args.coinbase_optional,
                heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*_crypto_coinbase_*.json"),
                heartbeat_stale_seconds=max(args.coinbase_heartbeat_stale_seconds, 30),
                min_healthy_heartbeats=max(args.coinbase_min_heartbeats, 1),
            )
        )

    interval = max(args.interval_seconds, 5)
    max_restarts = max(args.max_restarts_per_window, 1)
    window_seconds = max(args.restart_window_seconds, 60)
    event_log_path = None if args.no_event_log else Path(args.event_log_path)

    if args.once:
        return _run_iteration(
            targets=targets,
            max_restarts_per_window=max_restarts,
            restart_window_seconds=window_seconds,
            dry_run=args.dry_run,
            emit_json=args.json,
            event_log_path=event_log_path,
        )

    while True:
        rc = _run_iteration(
            targets=targets,
            max_restarts_per_window=max_restarts,
            restart_window_seconds=window_seconds,
            dry_run=args.dry_run,
            emit_json=args.json,
            event_log_path=event_log_path,
        )
        if rc != 0 and args.dry_run:
            return rc
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
