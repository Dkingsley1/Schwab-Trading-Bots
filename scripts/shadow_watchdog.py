import argparse
import glob
import json
import os
import shlex
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
PARALLEL_SHADOW_SCRIPT = PROJECT_ROOT / "scripts" / "run_parallel_shadows.py"
PARALLEL_AGGRESSIVE_SCRIPT = PROJECT_ROOT / "scripts" / "run_parallel_aggressive_modes.py"
SHADOW_LOOP_SCRIPT = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
DIVIDEND_SHADOW_SCRIPT = PROJECT_ROOT / "scripts" / "run_dividend_shadow.py"
BOND_SHADOW_SCRIPT = PROJECT_ROOT / "scripts" / "run_bond_shadow.py"
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
GLOBAL_HALT_FLAG = HEALTH_DIR / "GLOBAL_TRADING_HALT.flag"
OPERATOR_STOP_FLAG = HEALTH_DIR / "OPERATOR_STOP.flag"
HALT_RECOVERY_LATEST = HEALTH_DIR / "shadow_watchdog_halt_recovery_latest.json"
HALT_RECOVERY_EVENTS = WATCHDOG_DIR / "shadow_watchdog_halt_recovery_events.jsonl"
HEALTH_PRUNE_LATEST = HEALTH_DIR / "shadow_watchdog_health_prune_latest.json"


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
    exclude_matches: tuple[str, ...] = ()


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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_reason_set(raw: str) -> Set[str]:
    return {str(x).strip().lower() for x in str(raw or "").split(",") if str(x).strip()}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_latest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _halt_flag_age_seconds(path: Path) -> Optional[float]:
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except Exception:
        return None


def _evaluate_halt_auto_clear(
    *,
    halt_active: bool,
    halt_reason: str,
    halt_age_seconds: Optional[float],
    operator_stop_active: bool,
    auto_clear_enabled: bool,
    min_age_seconds: int,
    allowed_reasons: Set[str],
    require_paper_only: bool,
    market_data_only: bool,
    allow_order_execution: bool,
) -> tuple[bool, str]:
    if not halt_active:
        return False, "halt_not_set"
    if not auto_clear_enabled:
        return False, "auto_clear_disabled"
    if operator_stop_active:
        return False, "operator_stop_active"
    if require_paper_only and (not market_data_only or allow_order_execution):
        return False, "paper_only_guard_failed"

    min_age = max(int(min_age_seconds), 0)
    if halt_age_seconds is not None and halt_age_seconds < float(min_age):
        return False, f"cooldown_not_elapsed:{halt_age_seconds:.1f}s<{min_age}s"

    normalized_reason = str(halt_reason or "").strip().lower()
    if allowed_reasons and normalized_reason not in allowed_reasons:
        return False, f"reason_not_allowed:{normalized_reason or 'unknown'}"

    return True, "eligible"


def _auto_clear_global_halt(
    *,
    auto_clear_enabled: bool,
    min_age_seconds: int,
    allowed_reasons: Set[str],
    require_paper_only: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    halt_active = GLOBAL_HALT_FLAG.exists()
    operator_stop_active = OPERATOR_STOP_FLAG.exists()
    halt_payload = _load_json(GLOBAL_HALT_FLAG) if halt_active else {}
    halt_reason = str(halt_payload.get("reason", "")).strip()
    halt_age_seconds = _halt_flag_age_seconds(GLOBAL_HALT_FLAG) if halt_active else None

    market_data_only = _env_flag("MARKET_DATA_ONLY", "1")
    allow_order_execution = _env_flag("ALLOW_ORDER_EXECUTION", "0")

    should_clear, decision_reason = _evaluate_halt_auto_clear(
        halt_active=halt_active,
        halt_reason=halt_reason,
        halt_age_seconds=halt_age_seconds,
        operator_stop_active=operator_stop_active,
        auto_clear_enabled=auto_clear_enabled,
        min_age_seconds=min_age_seconds,
        allowed_reasons=allowed_reasons,
        require_paper_only=require_paper_only,
        market_data_only=market_data_only,
        allow_order_execution=allow_order_execution,
    )

    action = "halt_not_set" if not halt_active else "halt_auto_clear_skipped"
    error = ""
    if should_clear:
        if dry_run:
            action = "halt_auto_clear_dry_run"
        else:
            try:
                GLOBAL_HALT_FLAG.unlink()
                halt_active = False
                action = "halt_auto_cleared"
            except Exception as exc:
                action = "halt_auto_clear_error"
                error = f"{type(exc).__name__}:{exc}"

    payload: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "action": action,
        "decision_reason": decision_reason,
        "halt_active": bool(halt_active),
        "halt_reason": halt_reason,
        "halt_age_seconds": (round(float(halt_age_seconds), 2) if halt_age_seconds is not None else None),
        "operator_stop_active": bool(operator_stop_active),
        "market_data_only": bool(market_data_only),
        "allow_order_execution": bool(allow_order_execution),
        "auto_clear_enabled": bool(auto_clear_enabled),
        "auto_clear_min_age_seconds": max(int(min_age_seconds), 0),
        "auto_clear_allowed_reasons": sorted(allowed_reasons),
        "auto_clear_require_paper_only": bool(require_paper_only),
        "error": error,
    }

    _write_latest(HALT_RECOVERY_LATEST, payload)
    if action != "halt_not_set":
        _append_jsonl(HALT_RECOVERY_EVENTS, payload)

    return payload


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


def _find_matching_rows(rows: list[tuple[int, str]], match: str, exclude_matches: Iterable[str] = ()) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    excludes = [x for x in (exclude_matches or ()) if x]
    for pid, cmd in rows:
        if "scripts/shadow_watchdog.py" in cmd:
            # Avoid matching launch commands embedded in watchdog args.
            continue
        if match not in cmd:
            continue
        if any(ex in cmd for ex in excludes):
            continue
        out.append((pid, cmd))
    return out


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


def _build_default_aggressive_modes_cmd(simulate: bool) -> str:
    base = f"{VENV_PY} {PARALLEL_AGGRESSIVE_SCRIPT}"
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


def _build_default_coinbase_futures_cmd() -> str:
    return (
        f"{VENV_PY} {SHADOW_LOOP_SCRIPT} "
        "--broker coinbase "
        "--profile crypto_futures "
        "--domain crypto "
        "--symbols BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD "
        "--context-symbols BTC-USD,ETH-USD,SOL-USD "
        "--interval-seconds 20"
    )


def _build_default_dividend_cmd(simulate: bool) -> str:
    base = f"{VENV_PY} {DIVIDEND_SHADOW_SCRIPT} --interval-seconds 60"
    if simulate:
        return base + " --simulate"
    return base


def _build_default_bond_cmd(simulate: bool) -> str:
    base = f"{VENV_PY} {BOND_SHADOW_SCRIPT} --interval-seconds 90"
    if simulate:
        return base + " --simulate"
    return base


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


def _age_seconds_from_mtime(path: Path, now_ts: float) -> float:
    try:
        return max(float(now_ts - path.stat().st_mtime), 0.0)
    except Exception:
        return 1e12


def _heartbeat_pid_from_filename(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("shadow_loop_") or not name.endswith(".json"):
        return None
    stem = name[:-5]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except Exception:
        return None


def _remove_with_marker(path: Path, *, dry_run: bool) -> tuple[list[str], list[dict]]:
    removed: list[str] = []
    errors: list[dict] = []
    candidates = [path, path.with_suffix(path.suffix + ".ok")]
    seen: set[str] = set()

    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists():
            continue
        if dry_run:
            removed.append(key)
            continue
        try:
            p.unlink()
            removed.append(key)
        except Exception as exc:
            errors.append({"path": key, "error": f"{type(exc).__name__}:{exc}"})

    return removed, errors


def _remove_marker(path: Path, *, dry_run: bool) -> tuple[bool, Optional[dict]]:
    if not path.exists():
        return False, None
    if dry_run:
        return True, None
    try:
        path.unlink()
        return True, None
    except Exception as exc:
        return False, {"path": str(path), "error": f"{type(exc).__name__}:{exc}"}


def _prune_stale_health_artifacts(
    *,
    live_pids: Set[int],
    dry_run: bool,
    shadow_loop_stale_seconds: int,
    data_ingress_stale_seconds: int,
    orphan_marker_stale_seconds: int,
    max_files_per_pass: int,
) -> dict:
    now_ts = time.time()
    max_files = max(int(max_files_per_pass), 1)

    summary: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "ran": True,
        "dry_run": bool(dry_run),
        "thresholds": {
            "shadow_loop_stale_seconds": int(max(shadow_loop_stale_seconds, 60)),
            "data_ingress_stale_seconds": int(max(data_ingress_stale_seconds, 60)),
            "orphan_marker_stale_seconds": int(max(orphan_marker_stale_seconds, 60)),
            "max_files_per_pass": int(max_files),
        },
        "shadow_loop": {
            "scanned": 0,
            "candidates": 0,
            "removed": 0,
            "errors": 0,
            "live_pid_skipped": 0,
        },
        "data_ingress": {
            "scanned": 0,
            "candidates": 0,
            "removed": 0,
            "errors": 0,
        },
        "orphan_markers": {
            "scanned": 0,
            "removed": 0,
            "errors": 0,
        },
        "removed_paths_sample": [],
        "error_sample": [],
    }

    removed_paths: list[str] = []
    error_rows: list[dict] = []

    shadow_candidates: list[Path] = []
    for p in sorted(HEALTH_DIR.glob("shadow_loop_*.json"), key=lambda x: x.stat().st_mtime if x.exists() else 0.0):
        summary["shadow_loop"]["scanned"] += 1
        age = _age_seconds_from_mtime(p, now_ts)
        if age < max(shadow_loop_stale_seconds, 60):
            continue
        pid = _heartbeat_pid_from_filename(p)
        if pid is not None and pid in live_pids:
            summary["shadow_loop"]["live_pid_skipped"] += 1
            continue
        shadow_candidates.append(p)
        if len(shadow_candidates) >= max_files:
            break

    summary["shadow_loop"]["candidates"] = len(shadow_candidates)
    for p in shadow_candidates:
        removed, errors = _remove_with_marker(p, dry_run=dry_run)
        removed_paths.extend(removed)
        error_rows.extend(errors)

    ingress_candidates: list[Path] = []
    for p in sorted(HEALTH_DIR.glob("data_ingress_latest_*.json"), key=lambda x: x.stat().st_mtime if x.exists() else 0.0):
        summary["data_ingress"]["scanned"] += 1
        age = _age_seconds_from_mtime(p, now_ts)
        if age < max(data_ingress_stale_seconds, 60):
            continue
        ingress_candidates.append(p)
        if len(ingress_candidates) >= max_files:
            break

    summary["data_ingress"]["candidates"] = len(ingress_candidates)
    for p in ingress_candidates:
        removed, errors = _remove_with_marker(p, dry_run=dry_run)
        removed_paths.extend(removed)
        error_rows.extend(errors)

    orphan_cutoff = max(orphan_marker_stale_seconds, 60)
    marker_patterns = ["shadow_loop_*.json.ok", "data_ingress_latest_*.json.ok"]
    marker_seen = 0
    marker_removed = 0
    marker_errors = 0
    for pattern in marker_patterns:
        for marker in HEALTH_DIR.glob(pattern):
            marker_seen += 1
            age = _age_seconds_from_mtime(marker, now_ts)
            if age < orphan_cutoff:
                continue
            target = marker.with_suffix("")
            if target.exists():
                continue
            ok, err = _remove_marker(marker, dry_run=dry_run)
            if ok:
                marker_removed += 1
                removed_paths.append(str(marker))
            else:
                marker_errors += 1
                if err is not None:
                    error_rows.append(err)
            if marker_removed >= max_files:
                break
        if marker_removed >= max_files:
            break

    summary["orphan_markers"]["scanned"] = marker_seen
    summary["orphan_markers"]["removed"] = marker_removed
    summary["orphan_markers"]["errors"] = marker_errors

    summary["shadow_loop"]["removed"] = sum(1 for p in removed_paths if "/shadow_loop_" in p)
    summary["data_ingress"]["removed"] = sum(1 for p in removed_paths if "/data_ingress_latest_" in p)
    summary["shadow_loop"]["errors"] = sum(1 for e in error_rows if "/shadow_loop_" in str(e.get("path", "")))
    summary["data_ingress"]["errors"] = sum(1 for e in error_rows if "/data_ingress_latest_" in str(e.get("path", "")))

    summary["removed_paths_sample"] = removed_paths[:20]
    summary["error_sample"] = error_rows[:20]
    summary["removed_total"] = len(removed_paths)
    summary["error_total"] = len(error_rows)

    _write_latest(HEALTH_PRUNE_LATEST, summary)
    return summary


def _status_payload(entries: list[dict], halt_recovery: Optional[dict] = None, health_prune: Optional[dict] = None) -> dict:
    payload = {
        "timestamp_utc": _now_iso(),
        "targets": entries,
    }
    if halt_recovery is not None:
        payload["global_halt_recovery"] = halt_recovery
    if health_prune is not None:
        payload["health_prune"] = health_prune
    return payload


def _run_iteration(
    targets: list[Target],
    max_restarts_per_window: int,
    restart_window_seconds: int,
    dry_run: bool,
    emit_json: bool,
    event_log_path: Optional[Path],
    auto_clear_global_halt: bool,
    auto_clear_global_halt_min_age_seconds: int,
    auto_clear_global_halt_allowed_reasons: Set[str],
    auto_clear_global_halt_require_paper_only: bool,
    run_health_prune: bool,
    health_prune_shadow_loop_stale_seconds: int,
    health_prune_data_ingress_stale_seconds: int,
    health_prune_orphan_marker_stale_seconds: int,
    health_prune_max_files_per_pass: int,
) -> int:
    halt_recovery = _auto_clear_global_halt(
        auto_clear_enabled=auto_clear_global_halt,
        min_age_seconds=auto_clear_global_halt_min_age_seconds,
        allowed_reasons=auto_clear_global_halt_allowed_reasons,
        require_paper_only=auto_clear_global_halt_require_paper_only,
        dry_run=dry_run,
    )

    rows = _scan_process_rows()
    now_ts = time.time()
    overall_rc = 0
    entries: list[dict] = []

    for target in targets:
        matches = _find_matching_rows(rows, target.match, target.exclude_matches)
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

    health_prune_summary: Optional[dict] = None
    if run_health_prune:
        live_pids = {pid for pid, _ in rows}
        health_prune_summary = _prune_stale_health_artifacts(
            live_pids=live_pids,
            dry_run=dry_run,
            shadow_loop_stale_seconds=max(int(health_prune_shadow_loop_stale_seconds), 60),
            data_ingress_stale_seconds=max(int(health_prune_data_ingress_stale_seconds), 60),
            orphan_marker_stale_seconds=max(int(health_prune_orphan_marker_stale_seconds), 60),
            max_files_per_pass=max(int(health_prune_max_files_per_pass), 1),
        )

    payload = _status_payload(entries, halt_recovery=halt_recovery, health_prune=health_prune_summary)

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
        if health_prune_summary is not None:
            print(
                " - health_prune: removed_total={removed} errors={errors} "
                "shadow_candidates={shadow} ingress_candidates={ingress}".format(
                    removed=health_prune_summary.get("removed_total", 0),
                    errors=health_prune_summary.get("error_total", 0),
                    shadow=(health_prune_summary.get("shadow_loop") or {}).get("candidates", 0),
                    ingress=(health_prune_summary.get("data_ingress") or {}).get("candidates", 0),
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
    parser.add_argument("--coinbase-futures-start-cmd", default=None)
    parser.add_argument("--aggressive-modes-start-cmd", default=None)
    parser.add_argument("--dividend-start-cmd", default=None)
    parser.add_argument("--bond-start-cmd", default=None)
    parser.add_argument("--watch-coinbase", action="store_true")
    parser.add_argument("--watch-coinbase-futures", action="store_true")
    parser.add_argument("--watch-aggressive-modes", action="store_true")
    parser.add_argument("--watch-dividend", action="store_true")
    parser.add_argument("--watch-bond", action="store_true")
    parser.add_argument("--coinbase-optional", action="store_true")
    parser.add_argument("--coinbase-futures-optional", action="store_true")
    parser.add_argument("--dividend-optional", action="store_true")
    parser.add_argument("--bond-optional", action="store_true")

    parser.add_argument("--schwab-heartbeat-stale-seconds", type=int, default=120)
    parser.add_argument("--coinbase-heartbeat-stale-seconds", type=int, default=180)
    parser.add_argument("--schwab-min-heartbeats", type=int, default=2)
    parser.add_argument("--coinbase-min-heartbeats", type=int, default=1)
    parser.add_argument("--aggressive-modes-heartbeat-stale-seconds", type=int, default=180)
    parser.add_argument("--aggressive-modes-min-heartbeats", type=int, default=2)
    parser.add_argument("--dividend-heartbeat-stale-seconds", type=int, default=240)
    parser.add_argument("--dividend-min-heartbeats", type=int, default=1)
    parser.add_argument("--bond-heartbeat-stale-seconds", type=int, default=240)
    parser.add_argument("--bond-min-heartbeats", type=int, default=1)

    parser.add_argument(
        "--auto-clear-global-halt",
        dest="auto_clear_global_halt",
        action="store_true",
        default=_env_flag("SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT", "0"),
        help="Auto-clear GLOBAL_TRADING_HALT when cooldown and safety conditions pass.",
    )
    parser.add_argument(
        "--no-auto-clear-global-halt",
        dest="auto_clear_global_halt",
        action="store_false",
        help="Disable automatic GLOBAL_TRADING_HALT clearing.",
    )
    parser.add_argument(
        "--auto-clear-global-halt-min-age-seconds",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS", "300")),
        help="Minimum halt flag age before auto-clear is allowed.",
    )
    parser.add_argument(
        "--auto-clear-global-halt-allowed-reasons",
        default=os.getenv(
            "SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS",
            "incident_auto_halt,global_risk_killswitch,repeated_hard_gates,softguard_api_circuit_opened",
        ),
        help="Comma-separated halt reasons that can be auto-cleared. Empty = allow any reason.",
    )
    parser.add_argument(
        "--auto-clear-global-halt-require-paper-only",
        dest="auto_clear_global_halt_require_paper_only",
        action="store_true",
        default=_env_flag("SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY", "1"),
        help="Require MARKET_DATA_ONLY=1 and ALLOW_ORDER_EXECUTION=0 for auto-clear.",
    )
    parser.add_argument(
        "--no-auto-clear-global-halt-require-paper-only",
        dest="auto_clear_global_halt_require_paper_only",
        action="store_false",
        help="Allow auto-clear even when paper-only guard is not active.",
    )

    parser.add_argument(
        "--health-prune-enabled",
        dest="health_prune_enabled",
        action="store_true",
        default=_env_flag("SHADOW_WATCHDOG_HEALTH_PRUNE_ENABLED", "1"),
        help="Automatically prune stale health latest/heartbeat artifacts.",
    )
    parser.add_argument(
        "--no-health-prune-enabled",
        dest="health_prune_enabled",
        action="store_false",
        help="Disable health artifact pruning.",
    )
    parser.add_argument(
        "--health-prune-every-seconds",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_HEALTH_PRUNE_EVERY_SECONDS", "300")),
        help="How often to run stale health pruning while watchdog is running.",
    )
    parser.add_argument(
        "--prune-shadow-loop-stale-seconds",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_PRUNE_SHADOW_LOOP_STALE_SECONDS", "1800")),
        help="Delete stale shadow_loop heartbeat files older than this when PID is no longer running.",
    )
    parser.add_argument(
        "--prune-data-ingress-stale-seconds",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_PRUNE_DATA_INGRESS_STALE_SECONDS", "3600")),
        help="Delete stale data_ingress_latest files older than this.",
    )
    parser.add_argument(
        "--prune-orphan-marker-stale-seconds",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_PRUNE_ORPHAN_MARKER_STALE_SECONDS", "1800")),
        help="Delete orphan *.ok markers older than this when the target json is gone.",
    )
    parser.add_argument(
        "--prune-max-files-per-pass",
        type=int,
        default=int(os.getenv("SHADOW_WATCHDOG_PRUNE_MAX_FILES_PER_PASS", "300")),
        help="Safety cap on number of stale health files pruned per pass.",
    )

    parser.add_argument(
        "--event-log-path",
        default=str(_event_log_path()),
        help="JSONL path for watchdog events (default: governance/watchdog/watchdog_events_YYYYMMDD.jsonl).",
    )
    parser.add_argument("--no-event-log", action="store_true")
    args = parser.parse_args()

    schwab_cmd = args.schwab_start_cmd or _build_default_schwab_cmd(simulate=args.simulate_schwab)
    coinbase_cmd = args.coinbase_start_cmd or _build_default_coinbase_cmd()
    coinbase_futures_cmd = args.coinbase_futures_start_cmd or _build_default_coinbase_futures_cmd()
    aggressive_modes_cmd = args.aggressive_modes_start_cmd or _build_default_aggressive_modes_cmd(simulate=args.simulate_schwab)
    dividend_cmd = args.dividend_start_cmd or _build_default_dividend_cmd(simulate=args.simulate_schwab)
    bond_cmd = args.bond_start_cmd or _build_default_bond_cmd(simulate=args.simulate_schwab)

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
                exclude_matches=("--profile crypto_futures",),
            )
        )

    if args.watch_coinbase_futures:
        targets.append(
            Target(
                name="coinbase_futures_shadow",
                match="scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures",
                start_cmd=coinbase_futures_cmd,
                required=not args.coinbase_futures_optional,
                heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*crypto_futures*_crypto_coinbase_*.json"),
                heartbeat_stale_seconds=max(args.coinbase_heartbeat_stale_seconds, 30),
                min_healthy_heartbeats=max(args.coinbase_min_heartbeats, 1),
            )
        )

    if args.watch_aggressive_modes:
        targets.append(
            Target(
                name="aggressive_modes_parallel",
                match="scripts/run_parallel_aggressive_modes.py",
                start_cmd=aggressive_modes_cmd,
                required=True,
                heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*aggressive*_equities_schwab_*.json"),
                heartbeat_stale_seconds=max(args.aggressive_modes_heartbeat_stale_seconds, 30),
                min_healthy_heartbeats=max(args.aggressive_modes_min_heartbeats, 1),
            )
        )

    if args.watch_dividend:
        targets.append(
            Target(
                name="dividend_shadow",
                match="scripts/run_dividend_shadow.py",
                start_cmd=dividend_cmd,
                required=not args.dividend_optional,
                heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*dividend*_equities_schwab_*.json"),
                heartbeat_stale_seconds=max(args.dividend_heartbeat_stale_seconds, 30),
                min_healthy_heartbeats=max(args.dividend_min_heartbeats, 1),
            )
        )

    if args.watch_bond:
        targets.append(
            Target(
                name="bond_shadow",
                match="scripts/run_bond_shadow.py",
                start_cmd=bond_cmd,
                required=not args.bond_optional,
                heartbeat_glob=str(PROJECT_ROOT / "governance" / "health" / "shadow_loop_*bond*_equities_schwab_*.json"),
                heartbeat_stale_seconds=max(args.bond_heartbeat_stale_seconds, 30),
                min_healthy_heartbeats=max(args.bond_min_heartbeats, 1),
            )
        )

    interval = max(args.interval_seconds, 5)
    max_restarts = max(args.max_restarts_per_window, 1)
    window_seconds = max(args.restart_window_seconds, 60)
    event_log_path = None if args.no_event_log else Path(args.event_log_path)

    allowed_halt_reasons = _parse_reason_set(args.auto_clear_global_halt_allowed_reasons)

    health_prune_enabled = bool(args.health_prune_enabled)
    health_prune_every_seconds = max(int(args.health_prune_every_seconds), 30)
    next_health_prune_ts = 0.0

    if args.once:
        return _run_iteration(
            targets=targets,
            max_restarts_per_window=max_restarts,
            restart_window_seconds=window_seconds,
            dry_run=args.dry_run,
            emit_json=args.json,
            event_log_path=event_log_path,
            auto_clear_global_halt=bool(args.auto_clear_global_halt),
            auto_clear_global_halt_min_age_seconds=max(int(args.auto_clear_global_halt_min_age_seconds), 0),
            auto_clear_global_halt_allowed_reasons=allowed_halt_reasons,
            auto_clear_global_halt_require_paper_only=bool(args.auto_clear_global_halt_require_paper_only),
            run_health_prune=health_prune_enabled,
            health_prune_shadow_loop_stale_seconds=max(int(args.prune_shadow_loop_stale_seconds), 60),
            health_prune_data_ingress_stale_seconds=max(int(args.prune_data_ingress_stale_seconds), 60),
            health_prune_orphan_marker_stale_seconds=max(int(args.prune_orphan_marker_stale_seconds), 60),
            health_prune_max_files_per_pass=max(int(args.prune_max_files_per_pass), 1),
        )

    while True:
        now_ts = time.time()
        run_health_prune = False
        if health_prune_enabled and now_ts >= next_health_prune_ts:
            run_health_prune = True
            next_health_prune_ts = now_ts + health_prune_every_seconds

        rc = _run_iteration(
            targets=targets,
            max_restarts_per_window=max_restarts,
            restart_window_seconds=window_seconds,
            dry_run=args.dry_run,
            emit_json=args.json,
            event_log_path=event_log_path,
            auto_clear_global_halt=bool(args.auto_clear_global_halt),
            auto_clear_global_halt_min_age_seconds=max(int(args.auto_clear_global_halt_min_age_seconds), 0),
            auto_clear_global_halt_allowed_reasons=allowed_halt_reasons,
            auto_clear_global_halt_require_paper_only=bool(args.auto_clear_global_halt_require_paper_only),
            run_health_prune=run_health_prune,
            health_prune_shadow_loop_stale_seconds=max(int(args.prune_shadow_loop_stale_seconds), 60),
            health_prune_data_ingress_stale_seconds=max(int(args.prune_data_ingress_stale_seconds), 60),
            health_prune_orphan_marker_stale_seconds=max(int(args.prune_orphan_marker_stale_seconds), 60),
            health_prune_max_files_per_pass=max(int(args.prune_max_files_per_pass), 1),
        )
        if rc != 0 and args.dry_run:
            return rc
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
