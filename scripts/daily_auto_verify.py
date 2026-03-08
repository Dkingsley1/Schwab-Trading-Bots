import argparse
import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
DEFAULT_CMD_TIMEOUT_SEC = int(os.getenv("DAILY_AUTO_VERIFY_CMD_TIMEOUT_SEC", "180"))

DEFAULT_FRESHNESS_FILES = [
    PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json",
    PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json",
    PROJECT_ROOT / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json",
    PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json",
    PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json",
    PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json",
    PROJECT_ROOT / "governance" / "health" / "paper_replay_drill_latest.json",
    PROJECT_ROOT / "governance" / "health" / "paper_reconciliation_slo_latest.json",
    PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json",
    PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json",
    PROJECT_ROOT / "governance" / "health" / "nightly_resilience_latest.json",
    PROJECT_ROOT / "governance" / "health" / "model_card_latest.json",
    PROJECT_ROOT / "governance" / "alerts" / "incident_auto_halt_latest.json",
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


def _artifact_time_utc(path: Path) -> datetime | None:
    payload = _load_json(path)
    ts = _parse_iso_utc(str(payload.get("timestamp_utc", "")))
    if ts is not None:
        return ts
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _artifact_freshness_status(files: list[Path], max_age_minutes: float) -> dict[str, Any]:
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
        is_ok = age_minutes <= max_age
        if not is_ok:
            stale.append(str(p))
        rows.append({
            "path": str(p),
            "exists": True,
            "timestamp_utc": ts.isoformat(),
            "age_minutes": round(age_minutes, 4),
            "ok": bool(is_ok),
        })

    return {
        "ok": len(stale) == 0 and len(missing) == 0,
        "max_age_minutes": max_age,
        "stale_files": stale,
        "missing_files": missing,
        "rows": rows,
    }


def _resolve_freshness_files(raw: str) -> list[Path]:
    text = str(raw or "").strip()
    if not text:
        return list(DEFAULT_FRESHNESS_FILES)
    out: list[Path] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        out.append(p)
    return out if out else list(DEFAULT_FRESHNESS_FILES)


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

    checks: dict[str, dict] = {}

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

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "resource_guard.py")], PROJECT_ROOT)
    checks["resource_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out, "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "ingestion_backpressure_guard.py")], PROJECT_ROOT)
    checks["ingestion_backpressure"] = {"ok": rc == 0, "rc": rc, "stdout": out, "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "daily_runtime_summary.py"), "--day", day, "--json"], PROJECT_ROOT)
    checks["daily_runtime_summary"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_preopen_sanity_check.py"), "--hours", "24", "--json"], PROJECT_ROOT)
    checks["replay_preopen_sanity"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "snapshot_coverage_sentinel.py"), "--json"], PROJECT_ROOT)
    checks["snapshot_coverage_sentinel"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "guardrail_triprate_sentinel.py"), "--json"], PROJECT_ROOT)
    checks["guardrail_triprate_sentinel"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "quarantine_pressure_bot.py"), "--json"], PROJECT_ROOT)
    checks["quarantine_pressure_bot"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "data_source_divergence_bot.py"), "--json"], PROJECT_ROOT)
    checks["data_source_divergence_bot"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "execution_queue_stress_bot.py"), "--json"], PROJECT_ROOT)
    checks["execution_queue_stress_bot"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "preopen_replay_drift_bot.py"), "--json"], PROJECT_ROOT)
    checks["preopen_replay_drift_bot"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_readiness_summary.py"), "--json"], PROJECT_ROOT)
    checks["promotion_readiness_summary"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "canary_diagnostics_loop.py"), "--json"], PROJECT_ROOT)
    checks["canary_diagnostics_loop"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "retire_persistent_losers.py"), "--json"], PROJECT_ROOT)
    checks["retire_persistent_losers"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_bottleneck_focus.py"), "--json"], PROJECT_ROOT)
    checks["promotion_bottleneck_focus"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "new_bot_graduation_gate.py"), "--json"], PROJECT_ROOT)
    checks["new_bot_graduation_gate"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "leak_overfit_guard.py"), "--json"], PROJECT_ROOT)
    checks["leak_overfit_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "weekly_gate_blocker_report.py"), "--json"], PROJECT_ROOT)
    checks["weekly_gate_blocker_report"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "model_lifecycle_hygiene.py"), "--json"], PROJECT_ROOT)
    checks["model_lifecycle_hygiene"] = {"ok": rc in {0, 2}, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "health_gates.py")], PROJECT_ROOT)
    checks["health_gates"] = {"ok": rc in {0, 2}, "rc": rc, "stdout": out, "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_end_to_end_deterministic.py"), "--json"], PROJECT_ROOT)
    checks["replay_end_to_end_deterministic"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_replay_drill.py"), "--hours", "24", "--json"], PROJECT_ROOT)
    checks["paper_replay_drill"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "replay_hash_registry_guard.py"), "--json"], PROJECT_ROOT)
    checks["replay_hash_registry_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "live_reconciliation_slo_guard.py"), "--json"], PROJECT_ROOT)
    checks["live_reconciliation_slo_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_reconciliation_slo_guard.py"), "--json"], PROJECT_ROOT)
    checks["paper_reconciliation_slo_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "paper_execution_calibration_report.py"), "--hours", "24", "--json"], PROJECT_ROOT)
    checks["paper_execution_calibration_report"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "nightly_resilience_check.py"), "--json"], PROJECT_ROOT)
    checks["nightly_resilience_check"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "export_model_card.py"), "--json"], PROJECT_ROOT)
    checks["export_model_card"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "promotion_quality_gate.py"), "--json"], PROJECT_ROOT)
    checks["promotion_quality_gate"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "incident_auto_halt.py"), "--json"], PROJECT_ROOT)
    checks["incident_auto_halt"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "sleeve_slo_guard.py"), "--day", day, "--once", "--json"], PROJECT_ROOT)
    checks["sleeve_slo_guard"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "sleeve_allocator.py"), "--json"], PROJECT_ROOT)
    checks["sleeve_allocator"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "portfolio_risk_ledger.py"), "--json"], PROJECT_ROOT)
    checks["portfolio_risk_ledger"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "execution_budgeter.py"), "--json"], PROJECT_ROOT)
    checks["execution_budgeter"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "distill_new_bots.py"), "--json"], PROJECT_ROOT)
    checks["distillation_plan"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "daily_state_snapshot_drill.py"), "--json"], PROJECT_ROOT)
    checks["state_snapshot_drill"] = {"ok": rc == 0, "rc": rc, "stdout": out[:5000], "stderr": err}

    checks["db_integrity"] = _db_check(db_path, mode=args.db_check_mode)

    st = os.statvfs(str(PROJECT_ROOT))
    disk_free_gb = (st.f_bavail * st.f_frsize) / (1024.0 ** 3)
    checks["disk"] = {"ok": disk_free_gb >= 12.0, "disk_free_gb": round(disk_free_gb, 2)}

    freshness_files = _resolve_freshness_files(args.freshness_files)
    freshness = _artifact_freshness_status(freshness_files, max_age_minutes=float(args.max_artifact_age_minutes))
    checks["artifact_freshness"] = freshness

    failed = [k for k, v in checks.items() if not bool(v.get("ok", False))]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "day": day,
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
    }

    out_dir = PROJECT_ROOT / "exports" / "sql_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_auto_verify_{day}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    latest = PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"daily_auto_verify_ok={payload['ok']} failed_checks={','.join(failed) if failed else 'none'}")

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
