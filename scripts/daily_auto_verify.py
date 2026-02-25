import argparse
import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _db_check(db: Path) -> dict:
    if not db.exists():
        return {"ok": False, "reason": "db_missing"}
    try:
        conn = sqlite3.connect(str(db))
        row = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        ok = bool(row and str(row[0]).lower() == "ok")
        return {"ok": ok, "result": str(row[0]) if row else "none"}
    except Exception as exc:
        return {"ok": False, "reason": f"integrity_exception:{exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily auto-verify checks for runtime health.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    day = args.day
    db_path = Path(args.db)

    checks: dict[str, dict] = {}

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

    checks["db_integrity"] = _db_check(db_path)

    st = os.statvfs(str(PROJECT_ROOT))
    disk_free_gb = (st.f_bavail * st.f_frsize) / (1024.0 ** 3)
    checks["disk"] = {"ok": disk_free_gb >= 12.0, "disk_free_gb": round(disk_free_gb, 2)}

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
