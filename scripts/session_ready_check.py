import argparse
import json
import shlex
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_EXPECTED_PROFILES = ["conservative", "aggressive"]


def _command_invokes_target(command: str, match: str) -> bool:
    if (not command) or (not match):
        return False
    try:
        parts = shlex.split(command)
    except Exception:
        parts = str(command).split()
    if not parts:
        return False
    if parts[0].isdigit():
        parts = parts[1:]
    for part in parts[:3]:
        if part == match or part.endswith(f"/{match}"):
            return True
    return False


def _proc_count(match: str) -> int:
    commands: list[str] = []
    try:
        p = subprocess.run(["/bin/ps", "-axo", "command"], capture_output=True, text=True, check=False)
        commands = (p.stdout or "").splitlines()
    except Exception:
        commands = []

    if not commands:
        try:
            p = subprocess.run(["pgrep", "-af", match], capture_output=True, text=True, check=False)
            if p.returncode == 0:
                return sum(1 for line in (p.stdout or "").splitlines() if _command_invokes_target(line, match))
            return 0
        except Exception:
            return 0

    return sum(1 for line in commands if _command_invokes_target(line, match))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _profile_from_heartbeat_name(path: Path) -> str:
    name = path.name
    if not name.startswith("shadow_loop_") or not name.endswith(".json"):
        return ""
    core = name[len("shadow_loop_") : -5]
    parts = core.rsplit("_", 3)
    return parts[0].strip().lower() if len(parts) == 4 else core.strip().lower()


def _profile_from_runtime_checkpoint(path: Path) -> str:
    name = path.parent.name
    if not name.startswith("shadow_"):
        return ""
    profile = name[len("shadow_") :]
    for suffix in ("_equities", "_crypto"):
        if profile.endswith(suffix):
            profile = profile[: -len(suffix)]
            break
    return profile.strip().lower()


def _profile_activity_map() -> dict[str, datetime]:
    activity: dict[str, datetime] = {}
    hb_dir = PROJECT_ROOT / "governance" / "health"
    now = datetime.now(timezone.utc)

    for p in hb_dir.glob("shadow_loop_*.json"):
        profile = _profile_from_heartbeat_name(p)
        if not profile:
            continue
        ts = _parse_timestamp(_load_json(p).get("timestamp_utc"))
        if ts is None or ts > now:
            continue
        current = activity.get(profile)
        if current is None or ts > current:
            activity[profile] = ts

    for p in (PROJECT_ROOT / "governance").glob("shadow_*/runtime_checkpoint.json"):
        profile = _profile_from_runtime_checkpoint(p)
        if not profile:
            continue
        ts = _parse_timestamp(_load_json(p).get("timestamp_utc"))
        if ts is None or ts > now:
            continue
        current = activity.get(profile)
        if current is None or ts > current:
            activity[profile] = ts

    return activity


def _resolve_expected_profiles(raw: str, activity: dict[str, datetime], heartbeat_max_age_sec: float) -> list[str]:
    parts = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if parts and parts != ["auto"]:
        return parts

    now = datetime.now(timezone.utc)
    recent_window = max(float(heartbeat_max_age_sec) * 3.0, 900.0)
    recent = sorted(
        profile
        for profile, ts in activity.items()
        if max((now - ts).total_seconds(), 0.0) <= recent_window
    )
    return recent or list(DEFAULT_EXPECTED_PROFILES)


def _profile_heartbeat_ok(profile: str, max_age_sec: float) -> tuple[bool, str]:
    activity = _profile_activity_map()
    ts = activity.get(str(profile or "").strip().lower())
    if ts is None:
        return False, f"missing_profile={profile}"
    age = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    return age <= max_age_sec, f"age_sec={age:.1f}"


def _latest_heartbeat_age_sec() -> float:
    activity = _profile_activity_map()
    if not activity:
        return 1e9
    now = datetime.now(timezone.utc)
    ages = [max((now - ts).total_seconds(), 0.0) for ts in activity.values()]
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


def _halt_flag_detail() -> tuple[bool, str]:
    halt_flag = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
    if not halt_flag.exists():
        return False, str(halt_flag)
    payload = _load_json(halt_flag)
    reason = str(payload.get("reason") or "unknown")
    source = str(payload.get("source") or "")
    detail = str(halt_flag)
    if reason:
        detail += f" reason={reason}"
    if source:
        detail += f" source={source}"
    return True, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Single PASS/FAIL readiness check.")
    parser.add_argument("--min-disk-gb", type=float, default=float(__import__("os").getenv("SESSION_READY_MIN_DISK_GB", "15.0")))
    parser.add_argument(
        "--heartbeat-max-age-sec",
        type=float,
        default=float(__import__("os").getenv("SESSION_READY_HEARTBEAT_MAX_AGE_SEC", "300.0")),
    )
    parser.add_argument("--expected-profiles", default=__import__("os").getenv("SESSION_READY_EXPECTED_PROFILES", "auto"))
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

    halt_active, halt_detail = _halt_flag_detail()
    checks.append({"name": "global_halt_not_set", "ok": not halt_active, "details": halt_detail})

    activity = _profile_activity_map()
    expected_profiles = _resolve_expected_profiles(args.expected_profiles, activity, args.heartbeat_max_age_sec)
    for profile in expected_profiles:
        ok_prof, details = _profile_heartbeat_ok(profile, args.heartbeat_max_age_sec)
        checks.append({"name": f"profile_heartbeat_{profile}", "ok": ok_prof, "details": details})

    ok = all(c["ok"] for c in checks)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "expected_profiles": expected_profiles,
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
