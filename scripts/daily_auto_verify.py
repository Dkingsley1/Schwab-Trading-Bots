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

    rc, out, err = _run([str(VENV_PY), str(PROJECT_ROOT / "scripts" / "health_gates.py")], PROJECT_ROOT)
    checks["health_gates"] = {"ok": rc in {0, 2}, "rc": rc, "stdout": out, "stderr": err}

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
