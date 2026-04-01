import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

VENV_PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_CMD_TIMEOUT_SEC = int(os.getenv("DAILY_AUTO_VERIFY_CMD_TIMEOUT_SEC", "90"))
DEFAULT_SLOW_CMD_TIMEOUT_SEC = int(os.getenv("DAILY_AUTO_VERIFY_SLOW_CMD_TIMEOUT_SEC", str(max(DEFAULT_CMD_TIMEOUT_SEC, 300))))
LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "daily_auto_verify.lock"
PROGRESS_PATH = PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_progress_latest.json"
STALE_PROGRESS_MAX_AGE_SECONDS = int(os.getenv("DAILY_AUTO_VERIFY_STALE_PROGRESS_MAX_AGE_SECONDS", "7200"))

DEFAULT_FRESHNESS_FILE_GROUPS = [
    [PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json"],
    [
        PROJECT_ROOT / "governance" / "health" / "jsonl_sql_ingestion_health_trading_latest.json",
        PROJECT_ROOT / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json",
        PROJECT_ROOT / "governance" / "health" / "jsonl_sql_ingestion_health_data_latest.json",
        PROJECT_ROOT / "governance" / "health" / "jsonl_sql_ingestion_health_governance_latest.json",
    ],
    [
        PROJECT_ROOT / "governance" / "health" / "sql_link_service_progress_latest.json",
        PROJECT_ROOT / "governance" / "health" / "sql_link_service_latest.json",
    ],
    [PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "paper_replay_drill_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "paper_reconciliation_slo_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "nightly_resilience_latest.json"],
    [PROJECT_ROOT / "governance" / "health" / "model_card_latest.json"],
    [PROJECT_ROOT / "governance" / "alerts" / "incident_auto_halt_latest.json"],
]


def _run(cmd: list[str], cwd: Path, *, timeout_sec: int = DEFAULT_CMD_TIMEOUT_SEC) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        timeout_msg = f"timeout_after_s={max(int(timeout_sec), 1)}"
        err = f"{err} {timeout_msg}".strip()
        return 124, out, err


def _db_check(db: Path, *, mode: str = "quick") -> dict:
    if not db.exists():
        return {"ok": False, "reason": "db_missing", "mode": mode}
    pragma = "PRAGMA integrity_check" if str(mode).lower() == "full" else "PRAGMA quick_check"
    try:
        conn = sqlite3.connect(str(db))
        row = conn.execute(pragma).fetchone()
        conn.close()
        ok = bool(row and str(row[0]).lower() == "ok")
        return {
            "ok": ok,
            "result": str(row[0]) if row else "none",
            "mode": "full" if pragma.endswith("integrity_check") else "quick",
            "pragma": pragma,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"integrity_exception:{exc}", "mode": mode, "pragma": pragma}


def _parse_iso_utc(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pid_is_running(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _artifact_time_utc(path: Path) -> datetime | None:
    payload = _load_json(path)
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "created_at", "ended_utc", "started_utc"):
        ts = _parse_iso_utc(str(payload.get(key, "")))
        if ts is not None:
            return ts
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _artifact_freshness_status(
    files: list[Path],
    max_age_minutes: float,
    *,
    fresh_if_newer_than: datetime | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    max_age = max(float(max_age_minutes), 0.1)
    rows: list[dict[str, Any]] = []
    stale: list[str] = []
    missing: list[str] = []

    for p in files:
        if not p.exists():
            missing.append(str(p))
            rows.append({"path": str(p), "exists": False, "age_minutes": None, "ok": False})
            continue

        ts = _artifact_time_utc(p)
        if ts is None:
            rows.append({"path": str(p), "exists": True, "age_minutes": None, "ok": False, "error": "timestamp_unavailable"})
            stale.append(str(p))
            continue

        age_minutes = max((now - ts).total_seconds() / 60.0, 0.0)
        refreshed_in_run = bool(fresh_if_newer_than is not None and ts >= fresh_if_newer_than)
        is_ok = refreshed_in_run or (age_minutes <= max_age)
        if not is_ok:
            stale.append(str(p))
        rows.append({
            "path": str(p),
            "exists": True,
            "timestamp_utc": ts.isoformat(),
            "age_minutes": round(age_minutes, 4),
            "refreshed_in_run": bool(refreshed_in_run),
            "ok": bool(is_ok),
        })

    return {
        "ok": len(stale) == 0 and len(missing) == 0,
        "max_age_minutes": max_age,
        "fresh_if_newer_than_utc": fresh_if_newer_than.isoformat() if fresh_if_newer_than is not None else "",
        "stale_files": stale,
        "missing_files": missing,
        "rows": rows,
    }


def _resolve_freshness_files(raw: str) -> list[Path]:
    text = str(raw or "").strip()
    if not text:
        out: list[Path] = []
        for group in DEFAULT_FRESHNESS_FILE_GROUPS:
            out.append(_pick_best_artifact(group))
        return out
    out: list[Path] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        out.append(p)
    if out:
        return out
    fallback: list[Path] = []
    for group in DEFAULT_FRESHNESS_FILE_GROUPS:
        fallback.append(_pick_best_artifact(group))
    return fallback


def _pick_best_artifact(paths: list[Path]) -> Path:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return paths[0]

    def _sort_key(path: Path) -> tuple[float, float]:
        ts = _artifact_time_utc(path)
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (float(ts.timestamp()) if ts is not None else mtime, mtime)

    existing.sort(key=_sort_key)
    return existing[-1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _build_payload(
    day: str,
    checks: dict[str, dict[str, Any]],
    *,
    started_at_utc: datetime,
    running: bool,
    current_check: str = "",
    note: str = "",
    pid: int | None = None,
) -> dict[str, Any]:
    failed = [k for k, v in checks.items() if not bool(v.get("ok", False))]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "day": day,
        "pid": int(pid if isinstance(pid, int) and pid > 0 else os.getpid()),
        "running": bool(running),
        "started_at_utc": started_at_utc.isoformat(),
        "elapsed_seconds": round(max(time.time() - started_at_utc.timestamp(), 0.0), 3),
        "current_check": current_check,
        "completed_checks": len(checks),
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "note": note,
        "lock_path": str(LOCK_PATH),
        "checks": checks,
    }


def _stale_progress_note(path: Path, *, max_age_seconds: int) -> str:
    payload = _load_json(path)
    if not payload:
        return ""
    if not bool(payload.get("running", False)):
        return ""
    ts = _parse_iso_utc(str(payload.get("timestamp_utc", "")))
    if ts is None:
        return "stale_progress_missing_timestamp"
    age_seconds = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    if age_seconds < max(float(max_age_seconds), 1.0):
        return ""
    return f"stale_progress_cleared age_seconds={round(age_seconds, 1)}"


def _active_progress_pid(path: Path, *, max_age_seconds: int) -> int | None:
    payload = _load_json(path)
    if not payload or not bool(payload.get("running", False)):
        return None
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except Exception:
        return None
    if pid <= 0 or not _pid_is_running(pid):
        return None
    ts = _parse_iso_utc(str(payload.get("timestamp_utc", "")))
    if ts is None:
        return None
    age_seconds = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    if age_seconds >= max(float(max_age_seconds), 1.0):
        return None
    return pid


def _recover_stale_progress(path: Path, latest_path: Path, *, max_age_seconds: int) -> str:
    payload = _load_json(path)
    if not payload or not bool(payload.get("running", False)):
        return ""

    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except Exception:
        pid = 0

    ts = _parse_iso_utc(str(payload.get("timestamp_utc", "")))
    age_seconds = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) if ts is not None else None
    pid_alive = _pid_is_running(pid) if pid > 0 else False
    is_stale = age_seconds is None or age_seconds >= max(float(max_age_seconds), 1.0)
    if pid_alive and not is_stale:
        return ""

    recovered = dict(payload)
    recovered["running"] = False
    recovered["ok"] = False
    failed_checks = list(recovered.get("failed_checks", []))
    if "incomplete_run_recovered" not in failed_checks:
        failed_checks.append("incomplete_run_recovered")
    recovered["failed_checks"] = failed_checks
    note_parts = [str(recovered.get("note", "")).strip(), "recovered_stale_progress"]
    if pid > 0:
        note_parts.append(f"pid={pid}")
    if age_seconds is not None:
        note_parts.append(f"age_seconds={round(age_seconds, 1)}")
    if recovered.get("current_check"):
        note_parts.append(f"current_check={recovered['current_check']}")
    recovered["note"] = " ".join(part for part in note_parts if part)
    recovered["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    latest_payload = _load_json(latest_path)
    latest_ts = _parse_iso_utc(str(latest_payload.get("timestamp_utc", ""))) if latest_payload else None
    if latest_ts is None or (ts is not None and ts >= latest_ts):
        _write_json(latest_path, recovered)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    return str(recovered["note"])


def _write_progress(
    day: str,
    checks: dict[str, dict[str, Any]],
    *,
    started_at_utc: datetime,
    current_check: str = "",
    note: str = "",
) -> None:
    _write_json(
        PROGRESS_PATH,
        _build_payload(
            day,
            checks,
            started_at_utc=started_at_utc,
            running=True,
            current_check=current_check,
            note=note,
        ),
    )


def _write_final(day: str, checks: dict[str, dict[str, Any]], *, started_at_utc: datetime) -> dict[str, Any]:
    payload = _build_payload(day, checks, started_at_utc=started_at_utc, running=False)
    out_dir = PROJECT_ROOT / "exports" / "sql_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_auto_verify_{day}.json"
    _write_json(out_path, payload)
    latest = PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"
    _write_json(latest, payload)
    return payload


def _refresh_runtime_dashboard() -> None:
    _run(
        [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "ops" / "runtime_gate_dashboard.py"), "--json"],
        PROJECT_ROOT,
        timeout_sec=min(DEFAULT_CMD_TIMEOUT_SEC, 30),
    )


def _acquire_lock(path: Path) -> tuple[int | None, Any | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.seek(0)
        raw = handle.read().strip()
        existing_pid = None
        try:
            existing_pid = int(raw) if raw else None
        except Exception:
            existing_pid = None
        handle.close()
        return existing_pid, None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return None, handle


def _run_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    cmd: list[str],
    *,
    ok_predicate,
    cwd: Path,
    started_at_utc: datetime,
    day: str,
    stdout_limit: int = 5000,
    timeout_sec: int = DEFAULT_CMD_TIMEOUT_SEC,
) -> dict[str, Any]:
    _write_progress(day, checks, started_at_utc=started_at_utc, current_check=name)
    rc, out, err = _run(cmd, cwd, timeout_sec=timeout_sec)
    result = {"ok": bool(ok_predicate(rc)), "rc": rc, "stdout": out[:stdout_limit], "stderr": err}
    checks[name] = result
    _write_progress(day, checks, started_at_utc=started_at_utc, current_check=name)
    return result


def _timeout_for_check(name: str, slow_timeout_sec: int) -> int:
    slow_names = {
        "daily_runtime_summary",
        "replay_preopen_sanity",
        "snapshot_coverage_sentinel",
        "guardrail_triprate_sentinel",
        "execution_queue_stress_bot",
        "state_snapshot_drill",
        "health_gates",
        "data_source_divergence_bot",
    }
    return slow_timeout_sec if name in slow_names else DEFAULT_CMD_TIMEOUT_SEC


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily auto-verify checks for runtime health.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3"))
    parser.add_argument("--db-check-mode", choices=["quick", "full"], default=os.getenv("DAILY_AUTO_VERIFY_DB_CHECK_MODE", "quick"))
    parser.add_argument(
        "--max-artifact-age-minutes",
        type=float,
        default=float(os.getenv("DAILY_AUTO_VERIFY_MAX_ARTIFACT_AGE_MINUTES", "20")),
    )
    parser.add_argument(
        "--freshness-files",
        default=os.getenv("DAILY_AUTO_VERIFY_FRESHNESS_FILES", ""),
        help="Comma-separated file list (absolute or project-relative) for freshness checks.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    day = args.day
    db_path = Path(args.db)
    latest_path = PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"
    started_at_utc = datetime.now(timezone.utc)
    recovered_progress_note = _recover_stale_progress(
        PROGRESS_PATH,
        latest_path,
        max_age_seconds=STALE_PROGRESS_MAX_AGE_SECONDS,
    )
    active_progress_pid = _active_progress_pid(PROGRESS_PATH, max_age_seconds=STALE_PROGRESS_MAX_AGE_SECONDS)
    if active_progress_pid is not None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "pid": os.getpid(),
            "running": False,
            "ok": True,
            "failed_checks": [],
            "note": f"already_running_progress pid={active_progress_pid}",
            "lock_path": str(LOCK_PATH),
            "checks": {},
        }
        _write_json(PROGRESS_PATH, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(payload["note"])
        return 0
    existing_pid, lock_handle = _acquire_lock(LOCK_PATH)
    if lock_handle is None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "pid": os.getpid(),
            "running": False,
            "ok": True,
            "failed_checks": [],
            "note": f"already_running pid={existing_pid}" if _pid_is_running(existing_pid) else "already_running_lock_held",
            "lock_path": str(LOCK_PATH),
            "checks": {},
        }
        _write_json(PROGRESS_PATH, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(payload["note"])
        return 0

    checks: dict[str, dict] = {}
    startup_note = (
        recovered_progress_note
        or _stale_progress_note(PROGRESS_PATH, max_age_seconds=STALE_PROGRESS_MAX_AGE_SECONDS)
        or "daily_auto_verify_started"
    )
    _write_progress(day, checks, started_at_utc=started_at_utc, current_check="starting", note=startup_note)
    slow_timeout_sec = max(int(DEFAULT_SLOW_CMD_TIMEOUT_SEC), int(DEFAULT_CMD_TIMEOUT_SEC))
    payload: dict[str, Any] | None = None
    try:
        # Ensure session readiness artifact exists before downstream drills consume it.
        session_ready_file = PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"
        rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "session_ready_check.py"), "--json"], PROJECT_ROOT)
        checks["session_ready_check"] = {
            "ok": (rc in {0, 1}) and session_ready_file.exists(),
            "ready_ok": rc == 0,
            "rc": rc,
            "stdout": out[:5000],
            "stderr": err,
            "session_ready_file": str(session_ready_file),
            "session_ready_exists": session_ready_file.exists(),
        }
        _write_progress(day, checks, started_at_utc=started_at_utc, current_check="session_ready_check")

        common_zero_checks = [
            ("resource_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "resource_guard.py")], 0),
            ("ingestion_backpressure", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "ingestion_backpressure_guard.py")], 0),
            ("daily_runtime_summary", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "daily_runtime_summary.py"), "--day", day, "--json"], 5000),
            ("replay_preopen_sanity", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_preopen_sanity_check.py"), "--hours", "24", "--json"], 5000),
            ("snapshot_coverage_sentinel", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "snapshot_coverage_sentinel.py"), "--json"], 5000),
            ("guardrail_triprate_sentinel", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "guardrail_triprate_sentinel.py"), "--json"], 5000),
            ("quarantine_pressure_bot", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "quarantine_pressure_bot.py"), "--json"], 5000),
            ("data_source_divergence_bot", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "data_source_divergence_bot.py"), "--json"], 5000),
            ("execution_queue_stress_bot", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "execution_queue_stress_bot.py"), "--json"], 5000),
            ("preopen_replay_drift_bot", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "preopen_replay_drift_bot.py"), "--json"], 5000),
            ("promotion_readiness_summary", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_readiness_summary.py"), "--json"], 5000),
            ("canary_diagnostics_loop", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "canary_diagnostics_loop.py"), "--json"], 5000),
            ("retire_persistent_losers", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "retire_persistent_losers.py"), "--json"], 5000),
            ("promotion_bottleneck_focus", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_bottleneck_focus.py"), "--json"], 5000),
            ("new_bot_graduation_gate", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "new_bot_graduation_gate.py"), "--json"], 5000),
            ("leak_overfit_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "leak_overfit_guard.py"), "--json"], 5000),
            ("weekly_gate_blocker_report", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "weekly_gate_blocker_report.py"), "--json"], 5000),
            ("replay_end_to_end_deterministic", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_end_to_end_deterministic.py"), "--json"], 5000),
            ("paper_replay_drill", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_replay_drill.py"), "--hours", "24", "--json"], 5000),
            ("replay_hash_registry_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_hash_registry_guard.py"), "--json"], 5000),
            ("live_reconciliation_slo_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "live_reconciliation_slo_guard.py"), "--json"], 5000),
            ("paper_reconciliation_slo_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_reconciliation_slo_guard.py"), "--json"], 5000),
            ("paper_execution_calibration_report", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_execution_calibration_report.py"), "--hours", "24", "--json"], 5000),
            ("nightly_resilience_check", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "nightly_resilience_check.py"), "--json"], 5000),
            ("export_model_card", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "export_model_card.py"), "--json"], 5000),
            ("promotion_quality_gate", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_quality_gate.py"), "--json"], 5000),
            ("incident_auto_halt", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "incident_auto_halt.py"), "--json"], 5000),
            ("sleeve_slo_guard", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "sleeve_slo_guard.py"), "--day", day, "--once", "--json"], 5000),
            ("sleeve_allocator", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "sleeve_allocator.py"), "--json"], 5000),
            ("portfolio_risk_ledger", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "portfolio_risk_ledger.py"), "--json"], 5000),
            ("execution_budgeter", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "execution_budgeter.py"), "--json"], 5000),
            ("distillation_plan", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "distill_new_bots.py"), "--json"], 5000),
            ("state_snapshot_drill", [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "daily_state_snapshot_drill.py"), "--json"], 5000),
        ]
        for name, cmd, stdout_limit in common_zero_checks:
            _run_check(
                checks,
                name,
                cmd,
                ok_predicate=lambda rc: rc == 0,
                cwd=PROJECT_ROOT,
                started_at_utc=started_at_utc,
                day=day,
                stdout_limit=stdout_limit,
                timeout_sec=_timeout_for_check(name, slow_timeout_sec),
            )

        _run_check(
            checks,
            "model_lifecycle_hygiene",
            [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "model_lifecycle_hygiene.py"), "--json"],
            ok_predicate=lambda rc: rc in {0, 2},
            cwd=PROJECT_ROOT,
            started_at_utc=started_at_utc,
            day=day,
        )
        _run_check(
            checks,
            "health_gates",
            [str(VENV_PY), str(PROJECT_ROOT / "scripts" / "health_gates.py")],
            ok_predicate=lambda rc: rc in {0, 2},
            cwd=PROJECT_ROOT,
            started_at_utc=started_at_utc,
            day=day,
            timeout_sec=_timeout_for_check("health_gates", slow_timeout_sec),
        )

        _write_progress(day, checks, started_at_utc=started_at_utc, current_check="db_integrity")
        checks["db_integrity"] = _db_check(db_path, mode=args.db_check_mode)
        _write_progress(day, checks, started_at_utc=started_at_utc, current_check="db_integrity")

        st = os.statvfs(str(PROJECT_ROOT))
        disk_free_gb = (st.f_bavail * st.f_frsize) / (1024.0 ** 3)
        checks["disk"] = {"ok": disk_free_gb >= 12.0, "disk_free_gb": round(disk_free_gb, 2)}
        _write_progress(day, checks, started_at_utc=started_at_utc, current_check="disk")

        freshness_files = _resolve_freshness_files(args.freshness_files)
        freshness = _artifact_freshness_status(
            freshness_files,
            max_age_minutes=float(args.max_artifact_age_minutes),
            fresh_if_newer_than=started_at_utc,
        )
        checks["artifact_freshness"] = freshness
        _write_progress(day, checks, started_at_utc=started_at_utc, current_check="artifact_freshness")

        payload = _write_final(day, checks, started_at_utc=started_at_utc)
        _refresh_runtime_dashboard()

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                f"daily_auto_verify_ok={payload['ok']} "
                f"failed_checks={','.join(payload['failed_checks']) if payload['failed_checks'] else 'none'} "
                f"elapsed_seconds={payload['elapsed_seconds']}"
            )
        return 0 if payload["ok"] else 2
    except Exception as exc:
        checks["unhandled_exception"] = {
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        payload = _write_final(day, checks, started_at_utc=started_at_utc)
        _refresh_runtime_dashboard()
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"daily_auto_verify_exception={type(exc).__name__}:{exc}")
        return 2
    finally:
        try:
            progress_note = _stale_progress_note(PROGRESS_PATH, max_age_seconds=STALE_PROGRESS_MAX_AGE_SECONDS)
            if payload is not None and progress_note:
                payload["note"] = progress_note
                _write_json(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json", payload)
        except Exception:
            pass
        try:
            PROGRESS_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if lock_handle is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
