import argparse
import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pgrep_count(pattern: str) -> int:
    try:
        proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return 0
        return len([x for x in (proc.stdout or "").splitlines() if x.strip()])
    except Exception:
        return 0


def _fresh_minutes(path: Path) -> float:
    if not path.exists():
        return 1e9
    try:
        mt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return max((datetime.now(timezone.utc) - mt).total_seconds() / 60.0, 0.0)
    except Exception:
        return 1e9


def _count_keyword(path: Path, keyword: str, tail_lines: int = 2000) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-tail_lines:]
        k = keyword.lower()
        return sum(1 for ln in lines if k in ln.lower())
    except Exception:
        return 0


def _latest_existing(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _latest_glob(pattern: str) -> Path | None:
    rows = [Path(p) for p in glob.glob(pattern)]
    return _latest_existing(rows)


def _resolve_watchdog_log() -> Path | None:
    home_logs = Path.home() / "Library" / "Logs" / "schwab_trading_bot"
    candidates = [
        home_logs / "shadow_watchdog.out.log",
        PROJECT_ROOT / "logs" / "shadow_watchdog.out.log",
        Path("/private/tmp/com.dankingsley.shadow_watchdog.out.log"),
        _latest_glob(str(PROJECT_ROOT / "logs" / "shadow_watchdog_manual_*.log")),
    ]
    return _latest_existing([p for p in candidates if p is not None])


def _resolve_all_sleeves_log() -> Path | None:
    home_logs = Path.home() / "Library" / "Logs" / "schwab_trading_bot"
    candidates = [
        home_logs / "all_sleeves.out.log",
        PROJECT_ROOT / "logs" / "all_sleeves.out.log",
        Path("/private/tmp/com.dankingsley.all_sleeves.out.log"),
        _latest_glob(str(PROJECT_ROOT / "logs" / "all_sleeves_*.log")),
    ]
    return _latest_existing([p for p in candidates if p is not None])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Nightly resilience health check (watchdog/loops/feed freshness).")
    ap.add_argument("--max-log-age-minutes", type=float, default=20.0)
    ap.add_argument("--require-watchdog", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-loop", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "nightly_resilience_latest.json"))
    ap.add_argument("--event-file", default=str(PROJECT_ROOT / "governance" / "events" / f"nightly_resilience_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    watchdog_count = _pgrep_count("scripts/shadow_watchdog.py")
    loop_count = _pgrep_count("run_shadow_training_loop.py")
    all_sleeves_count = _pgrep_count("scripts/run_all_sleeves.py")

    watchdog_log = _resolve_watchdog_log()
    all_sleeves_log = _resolve_all_sleeves_log()
    wd_log_age = _fresh_minutes(watchdog_log) if watchdog_log is not None else 1e9
    sleeves_log_age = _fresh_minutes(all_sleeves_log) if all_sleeves_log is not None else 1e9
    restart_mentions = _count_keyword(watchdog_log, "restart") if watchdog_log is not None else 0

    failed = []
    if args.require_watchdog and watchdog_count <= 0:
        failed.append("watchdog_not_running")
    if args.require_loop and loop_count <= 0:
        failed.append("shadow_loop_not_running")
    if wd_log_age > float(args.max_log_age_minutes):
        failed.append("watchdog_log_stale")
    # Only enforce an all-sleeves launcher log when that wrapper is in use.
    if all_sleeves_count > 0 and sleeves_log_age > float(args.max_log_age_minutes):
        failed.append("all_sleeves_log_stale")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "metrics": {
            "watchdog_process_count": int(watchdog_count),
            "shadow_loop_process_count": int(loop_count),
            "all_sleeves_process_count": int(all_sleeves_count),
            "watchdog_log_age_minutes": round(float(wd_log_age), 4),
            "all_sleeves_log_age_minutes": round(float(sleeves_log_age), 4),
            "watchdog_restart_mentions_tail": int(restart_mentions),
        },
        "thresholds": {"max_log_age_minutes": float(args.max_log_age_minutes)},
        "evidence": {
            "watchdog_log": str(watchdog_log) if watchdog_log is not None else "",
            "all_sleeves_log": str(all_sleeves_log) if all_sleeves_log is not None else "",
        },
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    _append_jsonl(Path(args.event_file), out)

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"nightly_resilience_ok={int(out['ok'])} failed_checks={','.join(failed) if failed else 'none'}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
