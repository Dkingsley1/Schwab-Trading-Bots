import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _proc_count(match: str) -> int:
    p = subprocess.run(["/bin/ps", "-axo", "command"], capture_output=True, text=True, check=False)
    return sum(1 for line in (p.stdout or "").splitlines() if match in line)


def _profile_heartbeat_ok(profile: str, max_age_sec: float) -> tuple[bool, str]:
    hb_dir = PROJECT_ROOT / "governance" / "health"
    candidates = sorted(hb_dir.glob(f"shadow_loop_{profile}*.json"))
    if not candidates:
        return False, f"missing_pattern=shadow_loop_{profile}*.json"
    best_age = None
    for p in candidates:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            ts = str(payload.get("timestamp_utc", "")).replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
            age = max((datetime.now(timezone.utc) - dt).total_seconds(), 0.0)
            if best_age is None or age < best_age:
                best_age = age
        except Exception:
            continue
    if best_age is None:
        return False, "parse_error_all_candidates"
    return best_age <= max_age_sec, f"age_sec={best_age:.1f}"


def _latest_heartbeat_age_sec() -> float:
    hb_dir = PROJECT_ROOT / "governance" / "health"
    ages = []
    now = datetime.now(timezone.utc)
    for p in hb_dir.glob("shadow_loop_*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            ts = str(payload.get("timestamp_utc", "")).replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
            ages.append(max((now - dt).total_seconds(), 0.0))
        except Exception:
            continue
    return min(ages) if ages else 1e9


def _sql_writable() -> bool:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("CREATE TABLE IF NOT EXISTS readiness_probe(ts TEXT)")
        conn.execute("INSERT INTO readiness_probe(ts) VALUES (?)", (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Single PASS/FAIL readiness check.")
    parser.add_argument("--min-disk-gb", type=float, default=float(os.getenv("SESSION_READY_MIN_DISK_GB", "15.0")))
    parser.add_argument(
        "--heartbeat-max-age-sec",
        type=float,
        default=float(os.getenv("SESSION_READY_HEARTBEAT_MAX_AGE_SEC", "300.0")),
    )
    parser.add_argument("--expected-profiles", default=os.getenv("SESSION_READY_EXPECTED_PROFILES", "conservative,aggressive"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = []
    disk_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    checks.append({"name": "disk_headroom", "ok": disk_gb >= args.min_disk_gb, "details": f"disk_free_gb={disk_gb:.2f}"})

    sql_ok = _sql_writable()
    checks.append({"name": "sql_writable", "ok": sql_ok, "details": f"db_path={DB_PATH}"})

    launcher_count = _proc_count("scripts/run_parallel_shadows.py")
    checks.append({"name": "process_state", "ok": launcher_count <= 1, "details": f"parallel_launcher_count={launcher_count}"})

    age = _latest_heartbeat_age_sec()
    checks.append({"name": "heartbeat_freshness", "ok": age <= args.heartbeat_max_age_sec, "details": f"heartbeat_age_sec={age:.1f}"})

    halt_flag = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
    checks.append({"name": "global_halt_not_set", "ok": not halt_flag.exists(), "details": str(halt_flag)})

    expected_profiles = [x.strip() for x in str(args.expected_profiles).split(",") if x.strip()]
    for profile in expected_profiles:
        ok_prof, details = _profile_heartbeat_ok(profile, args.heartbeat_max_age_sec)
        checks.append({"name": f"profile_heartbeat_{profile}", "ok": ok_prof, "details": details})

    ok = all(c["ok"] for c in checks)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "checks": checks,
    }

    out = PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print("SESSION_READY PASS" if ok else "SESSION_READY FAIL")
        for c in checks:
            print(f" - {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['details']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
