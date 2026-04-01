import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
STATE_FILE = PROJECT_ROOT / "governance" / "health" / "maintenance_strategy_reloader_latest.json"
SQL_WRITER_LABEL = f"gui/{os.getuid()}/com.dankingsley.ops.sql_link_writer"
WATCH_KEYS = ("SQL_LINK_SERVICE_", "RETENTION_", "SQLITE_", "BOT_LOGS_")
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "jsonl_link_archives"
MAINTENANCE_GLOBS = ("*.compact.sqlite3", "*.precompact.bak.sqlite3")
WATCH_FILES = [
    PROJECT_ROOT / "config" / ".env",
    PROJECT_ROOT / "config" / ".env.live",
    PROJECT_ROOT / "config" / ".env.storage_override",
    PROJECT_ROOT / "config" / ".env.maintenance",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _collect_settings() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in WATCH_FILES:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if (not line) or line.startswith("#") or ("=" not in line):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.startswith(WATCH_KEYS):
                continue
            rows.append(
                {
                    "file": _rel(path),
                    "key": key,
                    "value": value.strip(),
                }
            )
    rows.sort(key=lambda row: (row["key"], row["file"], row["value"]))
    return rows


def _fingerprint(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_previous_state() -> dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maintenance_blockers() -> list[str]:
    blockers: list[str] = []
    if not ARCHIVE_ROOT.exists():
        return blockers
    for pattern in MAINTENANCE_GLOBS:
        for path in sorted(ARCHIVE_ROOT.glob(pattern)):
            blockers.append(_rel(path))
    return blockers


def _env_overrides(rows: list[dict[str, str]]) -> dict[str, str]:
    env: dict[str, str] = {}
    for row in rows:
        key = str(row.get("key") or "").strip()
        value = str(row.get("value") or "")
        if key:
            env[key] = value
    return env


def _run(cmd: list[str], *, env_overrides: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return {
        "cmd": cmd,
        "rc": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
    }


def main() -> int:
    rows = _collect_settings()
    observed_fingerprint = _fingerprint(rows)
    previous_state = _load_previous_state()
    previous_applied = str(
        previous_state.get("applied_fingerprint")
        or previous_state.get("fingerprint")
        or ""
    )
    changed = bool(observed_fingerprint) and observed_fingerprint != previous_applied

    keys = {row["key"] for row in rows}
    env_overrides = _env_overrides(rows)
    actions: list[dict[str, object]] = []
    maintenance_blockers = _maintenance_blockers()
    deferred = bool(changed and maintenance_blockers)
    applied_fingerprint = previous_applied

    if changed and not deferred:
        if any(key.startswith("SQL_LINK_SERVICE_") for key in keys):
            actions.append(
                {
                    "name": "restart_sql_link_writer",
                    **_run(["launchctl", "kickstart", "-k", SQL_WRITER_LABEL]),
                }
            )
        if any(key.startswith(("RETENTION_", "BOT_LOGS_")) for key in keys):
            actions.append(
                {
                    "name": "apply_retention_policy",
                    **_run(
                        [
                            str(PY),
                            str(PROJECT_ROOT / "scripts" / "data_retention_policy.py"),
                            "--apply",
                            "--skip-sqlite-vacuum",
                            "--json",
                        ],
                        env_overrides=env_overrides,
                    ),
                }
            )
        if any(key.startswith("BOT_LOGS_") for key in keys):
            actions.append(
                {
                    "name": "refresh_storage_route",
                    **_run(
                        [
                            str(PY),
                            str(PROJECT_ROOT / "scripts" / "ops" / "storage_failback_sync.py"),
                            "--json",
                        ],
                        env_overrides=env_overrides,
                    ),
                }
            )
        if all(int(action.get("rc", 1)) == 0 for action in actions):
            applied_fingerprint = observed_fingerprint
    elif not changed:
        applied_fingerprint = observed_fingerprint

    payload = {
        "timestamp_utc": _now_utc(),
        "changed": bool(changed),
        "deferred": deferred,
        "maintenance_blockers": maintenance_blockers,
        "observed_fingerprint": observed_fingerprint,
        "applied_fingerprint": applied_fingerprint,
        "previous_applied_fingerprint": previous_applied,
        "fingerprint": applied_fingerprint,
        "watched_files": [_rel(path) for path in WATCH_FILES if path.exists()],
        "setting_count": int(len(rows)),
        "settings": rows,
        "actions": actions,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
