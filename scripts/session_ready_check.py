#!/usr/bin/env python3
import argparse
import json
import os
import signal
import shlex
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_EXPECTED_PROFILES = ["conservative", "aggressive"]
MAX_HEALTH_JSON_BYTES = 1_000_000


class SessionReadyTimeout(RuntimeError):
    pass


def _session_timeout_seconds() -> int:
    try:
        return max(int(float(os.getenv("SESSION_READY_TIMEOUT_SECONDS", "20.0"))), 1)
    except Exception:
        return 20


def _timeout_payload(message: str) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "expected_profiles": [],
        "checks": [
            {
                "name": "session_ready_timeout",
                "ok": False,
                "details": message,
            }
        ],
    }


def _write_payload(payload: dict) -> None:
    out = PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _timeout_handler(signum: int, frame: object) -> None:
    raise SessionReadyTimeout(f"session_ready_check_timeout_seconds={_session_timeout_seconds()}")


def _proc_scan_timeout_sec() -> float:
    try:
        return max(float(os.getenv("SESSION_READY_PROC_SCAN_TIMEOUT_SECONDS", "4.0")), 0.5)
    except Exception:
        return 4.0


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
    timeout_sec = _proc_scan_timeout_sec()
    try:
        p = subprocess.run(
            ["/bin/ps", "-axo", "command"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        commands = (p.stdout or "").splitlines()
    except (subprocess.TimeoutExpired, Exception):
        commands = []

    if not commands:
        try:
            p = subprocess.run(
                ["pgrep", "-af", match],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            if p.returncode == 0:
                return sum(1 for line in (p.stdout or "").splitlines() if _command_invokes_target(line, match))
            return 0
        except (subprocess.TimeoutExpired, Exception):
            return 0

    return sum(1 for line in commands if _command_invokes_target(line, match))


def _load_json(path: Path) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > MAX_HEALTH_JSON_BYTES:
            return {}
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


def _runtime_checkpoint_paths() -> list[Path]:
    root = PROJECT_ROOT / "governance"
    paths: list[Path] = []
    try:
        candidates = list(root.iterdir())
    except Exception:
        return paths

    for shadow_dir in candidates:
        name = shadow_dir.name
        if not name.startswith("shadow_") or ".__external_symlink_backup_" in name:
            continue
        try:
            if shadow_dir.is_symlink() or not shadow_dir.is_dir():
                continue
            checkpoint = shadow_dir / "runtime_checkpoint.json"
            if checkpoint.is_symlink() or not checkpoint.is_file():
                continue
        except Exception:
            continue
        paths.append(checkpoint)
    return paths


def _profile_activity_details() -> dict[str, dict[str, object]]:
    activity: dict[str, dict[str, object]] = {}
    hb_dir = PROJECT_ROOT / "governance" / "health"
    now = datetime.now(timezone.utc)

    for p in hb_dir.glob("shadow_loop_*.json"):
        profile = _profile_from_heartbeat_name(p)
        if not profile:
            continue
        ts = _parse_timestamp(_load_json(p).get("timestamp_utc"))
        if ts is None or ts > now:
            continue
        current = activity.setdefault(profile, {"latest": None, "heartbeat": None, "checkpoint": None})
        heartbeat = current.get("heartbeat")
        if heartbeat is None or ts > heartbeat:
            current["heartbeat"] = ts
        latest = current.get("latest")
        if latest is None or ts > latest:
            current["latest"] = ts

    for p in _runtime_checkpoint_paths():
        profile = _profile_from_runtime_checkpoint(p)
        if not profile:
            continue
        ts = _parse_timestamp(_load_json(p).get("timestamp_utc"))
        if ts is None or ts > now:
            continue
        current = activity.setdefault(profile, {"latest": None, "heartbeat": None, "checkpoint": None})
        checkpoint = current.get("checkpoint")
        if checkpoint is None or ts > checkpoint:
            current["checkpoint"] = ts
        latest = current.get("latest")
        if latest is None or ts > latest:
            current["latest"] = ts

    return activity


def _profile_activity_map() -> dict[str, datetime]:
    return {
        profile: details["latest"]
        for profile, details in _profile_activity_details().items()
        if details.get("latest") is not None
    }


def _resolve_expected_profiles(
    raw: str,
    activity: dict[str, datetime],
    heartbeat_max_age_sec: float,
    *,
    activity_details: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    parts = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if parts and parts != ["auto"]:
        return parts

    now = datetime.now(timezone.utc)
    recent_window = max(float(heartbeat_max_age_sec), 1.0)
    checkpoint_window = recent_window
    details_map = activity_details if isinstance(activity_details, dict) else _profile_activity_details()
    recent = sorted(
        profile
        for profile, details in details_map.items()
        if (
            details.get("heartbeat") is not None
            and max((now - details["heartbeat"]).total_seconds(), 0.0) <= recent_window
        )
        or (
            details.get("heartbeat") is None
            and details.get("checkpoint") is not None
            and max((now - details["checkpoint"]).total_seconds(), 0.0) <= checkpoint_window
        )
    )
    return recent or list(DEFAULT_EXPECTED_PROFILES)


def _profile_heartbeat_ok(
    profile: str,
    max_age_sec: float,
    *,
    activity: dict[str, datetime] | None = None,
) -> tuple[bool, str]:
    activity_map = activity if isinstance(activity, dict) else _profile_activity_map()
    ts = activity_map.get(str(profile or "").strip().lower())
    if ts is None:
        return False, f"missing_profile={profile}"
    age = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    return age <= max_age_sec, f"age_sec={age:.1f}"


def _latest_heartbeat_age_sec(activity: dict[str, datetime] | None = None) -> float:
    activity_map = activity if isinstance(activity, dict) else _profile_activity_map()
    if not activity_map:
        return 1e9
    now = datetime.now(timezone.utc)
    ages = [max((now - ts).total_seconds(), 0.0) for ts in activity_map.values()]
    return min(ages) if ages else 1e9


def _all_sleeves_watchdog_readiness(max_age_sec: float) -> dict:
    payload = _load_json(PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json")
    status_rows = payload.get("status") if isinstance(payload.get("status"), list) else []
    row = next((item for item in status_rows if isinstance(item, dict) and item.get("name") == "all_sleeves"), None)
    if not isinstance(row, dict):
        return {"present": False, "ok": False, "supersedes_profile_heartbeats": False, "details": "all_sleeves_watchdog_missing"}

    launcher = row.get("launcher_artifact_health") if isinstance(row.get("launcher_artifact_health"), dict) else {}
    heartbeat_ok = bool(row.get("heartbeat_ok", False))
    launcher_ok = bool(launcher.get("ok", False))
    process_live = bool(row.get("process_live", False)) or _safe_positive_int(row.get("running")) > 0
    restart_skipped = str(row.get("restart_skipped") or "")
    reason = str(row.get("reason") or "")
    status = str(row.get("status") or "")
    managed_hold = bool(
        process_live
        and (
            status == "intentional_hold"
            or restart_skipped == "startup_not_ready"
            or reason == "process_fanout_guard_active"
        )
    )
    heartbeat_age = _safe_float(row.get("heartbeat_age_seconds"), 1e9)
    child_fanout = row.get("child_fanout") if isinstance(row.get("child_fanout"), dict) else {}
    child_fanout_ok = child_fanout.get("ok")
    ok = bool(heartbeat_ok or launcher_ok or managed_hold)
    details = (
        f"process_live={int(process_live)} heartbeat_ok={int(heartbeat_ok)} "
        f"launcher_ok={int(launcher_ok)} managed_hold={int(managed_hold)} "
        f"heartbeat_age_sec={heartbeat_age:.1f} child_fanout_ok={child_fanout_ok} "
        f"reason={reason or restart_skipped or status}"
    )
    return {
        "present": True,
        "ok": ok,
        "supersedes_profile_heartbeats": ok,
        "managed_hold": managed_hold,
        "heartbeat_age_seconds": heartbeat_age,
        "details": details,
        "max_age_seconds": float(max_age_sec),
    }


def _safe_positive_int(raw: object) -> int:
    try:
        return max(int(float(raw)), 0)
    except Exception:
        return 0


def _safe_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _sql_writable() -> bool:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DB_PATH.exists():
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0)
            conn.execute("PRAGMA schema_version").fetchone()
            conn.close()
        probe_path = DB_PATH.parent / ".session_ready_write_probe"
        probe_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return os.access(DB_PATH.parent, os.W_OK)
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

    activity_details = _profile_activity_details()
    activity = {
        profile: details["latest"]
        for profile, details in activity_details.items()
        if details.get("latest") is not None
    }

    all_sleeves_watchdog = _all_sleeves_watchdog_readiness(args.heartbeat_max_age_sec)
    profile_heartbeats_superseded = bool(all_sleeves_watchdog.get("supersedes_profile_heartbeats", False))
    if profile_heartbeats_superseded:
        checks.append(
            {
                "name": "heartbeat_freshness",
                "ok": True,
                "details": "superseded_by_all_sleeves_watchdog " + str(all_sleeves_watchdog.get("details") or ""),
            }
        )
    else:
        age = _latest_heartbeat_age_sec(activity)
        checks.append({"name": "heartbeat_freshness", "ok": age <= args.heartbeat_max_age_sec, "details": f"heartbeat_age_sec={age:.1f}"})

    halt_active, halt_detail = _halt_flag_detail()
    checks.append({"name": "global_halt_not_set", "ok": not halt_active, "details": halt_detail})

    if profile_heartbeats_superseded:
        expected_profiles = []
        checks.append(
            {
                "name": "all_sleeves_watchdog",
                "ok": bool(all_sleeves_watchdog.get("ok", False)),
                "details": str(all_sleeves_watchdog.get("details") or ""),
            }
        )
    else:
        expected_profiles = _resolve_expected_profiles(
            args.expected_profiles,
            activity,
            args.heartbeat_max_age_sec,
            activity_details=activity_details,
        )
    launcher_count = _proc_count("scripts/run_parallel_shadows.py")
    allowed_launchers = max(len(expected_profiles), 1)
    checks.append(
        {
            "name": "process_state",
            "ok": launcher_count <= allowed_launchers,
            "details": f"parallel_launcher_count={launcher_count} allowed={allowed_launchers}",
        }
    )
    for profile in expected_profiles:
        ok_prof, details = _profile_heartbeat_ok(profile, args.heartbeat_max_age_sec, activity=activity)
        checks.append({"name": f"profile_heartbeat_{profile}", "ok": ok_prof, "details": details})

    ok = all(c["ok"] for c in checks)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "expected_profiles": expected_profiles,
        "checks": checks,
    }

    _write_payload(payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print("SESSION_READY PASS" if ok else "SESSION_READY FAIL")
        for c in checks:
            print(f" - {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['details']}")
    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_session_timeout_seconds())
    try:
        raise SystemExit(main())
    except SessionReadyTimeout as exc:
        payload = _timeout_payload(str(exc))
        _write_payload(payload)
        if "--json" in sys.argv:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"SESSION_READY TIMEOUT: {exc}")
        raise SystemExit(2)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
