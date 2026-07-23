#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "sql_access_runtime_audit_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "config" / "runtime_profiles"
DEFAULT_PACKAGES = (
    "duckdb",
    "duckdb-engine",
    "SQLAlchemy",
    "polars",
    "pyarrow",
    "apsw",
    "adbc-driver-manager",
    "adbc-driver-sqlite",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _tail(text: str, n: int = 8) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _parse_version_lines(lines: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if (not line) or line.startswith("#") or ("==" not in line):
            continue
        package, version = line.split("==", 1)
        versions[_normalize_package_name(package)] = version.strip()
    return versions


def _load_lock_versions(lock_file: Path) -> dict[str, str]:
    if not lock_file.exists():
        return {}
    return _parse_version_lines(lock_file.read_text(encoding="utf-8").splitlines())


def _load_installed_versions(python_bin: Path) -> tuple[dict[str, str], dict[str, Any]]:
    rc, out, err = _run([str(python_bin), "-m", "pip", "list", "--format=freeze"])
    step = {
        "name": "installed_package_inventory",
        "ok": rc == 0,
        "rc": rc,
        "command": f"{python_bin} -m pip list --format=freeze",
        "accepted_rc": [0],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    return (_parse_version_lines(out.splitlines()) if rc == 0 else {}), step


def _step(name: str, cmd: list[str], accepted_rc: set[int] | None = None) -> dict[str, Any]:
    accepted = accepted_rc or {0}
    rc, out, err = _run(cmd)
    combined = f"{out}\n{err}".strip()
    hard_fail = any(
        marker in combined
        for marker in (
            "ModuleNotFoundError",
            "ImportError:",
            "No module named",
            "Traceback (most recent call last)",
        )
    )
    ok = (rc in accepted) and (not hard_fail)
    return {
        "name": name,
        "ok": ok,
        "rc": rc,
        "command": " ".join(shlex.quote(x) for x in cmd),
        "accepted_rc": sorted(accepted),
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }


def _package_rows(
    packages: tuple[str, ...],
    lock_versions: dict[str, str],
    installed_versions: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok = True
    for raw_name in packages:
        name = _normalize_package_name(raw_name)
        locked = lock_versions.get(name)
        installed = installed_versions.get(name)
        if locked and installed:
            status = "ok" if locked == installed else "version_mismatch"
        elif installed:
            status = "missing_lock"
        elif locked:
            status = "missing_runtime"
        else:
            status = "missing_both"
        ok = ok and (status == "ok")
        rows.append(
            {
                "package": name,
                "locked_version": locked,
                "installed_version": installed,
                "status": status,
            }
        )
    return rows, ok


def _recommendations(package_rows: list[dict[str, Any]], import_steps: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    if any(row.get("status") != "ok" for row in package_rows):
        recommendations.append("refresh_or_rebuild_sql_access_runtime_before_relying_on_it")
    step_map = {str(step.get("name") or ""): bool(step.get("ok", False)) for step in import_steps}
    if step_map.get("adbc_sqlite_smoke", False):
        recommendations.append("candidate_arrow_native_sqlite_reads_via_adbc")
    if step_map.get("sqlalchemy_duckdb_smoke", False):
        recommendations.append("candidate_duckdb_sqlalchemy_analytics_bridge")
    if not recommendations:
        recommendations.append("sql_access_runtime_ready")
    return recommendations


def _data_library_roles() -> dict[str, str]:
    return {
        "sqlite": "primary hot-path ingestion and bounded single-writer durability",
        "apsw": "low-level SQLite durability and contention-friendly maintenance tooling",
        "duckdb": "analytical read offload, summary scans, and mirror-side SQL",
        "adbc-driver-sqlite": "Arrow-native SQLite reads for lower-overhead analytical access",
        "pyarrow": "columnar interchange between SQLite, DuckDB, and downstream analytics",
        "polars": "fast in-memory transforms and report preparation outside the hot writer path",
        "pandas": "compatibility layer only; avoid on the hottest ingestion/reporting lanes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SQL access packages and smoke-test their runtime paths.")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    profile_dir = Path(args.profile_dir).expanduser()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lock_versions = _load_lock_versions(lock_file)
    inventory, inventory_step = _load_installed_versions(python_bin)
    package_rows, packages_ok = _package_rows(DEFAULT_PACKAGES, lock_versions, inventory)
    pip_check_step = _step("pip_check", [str(python_bin), "-m", "pip", "check"])
    import_steps = [
        _step(
            "duckdb_import",
            [str(python_bin), "-c", "import duckdb; print(duckdb.__version__)"],
        ),
        _step(
            "apsw_import",
            [str(python_bin), "-c", "import apsw; print(apsw.apswversion())"],
        ),
        _step(
            "sqlalchemy_duckdb_smoke",
            [
                str(python_bin),
                "-c",
                "from sqlalchemy import create_engine, text; engine = create_engine('duckdb:///:memory:'); "
                "conn = engine.connect(); print(conn.execute(text('select 1')).scalar()); conn.close(); engine.dispose()",
            ],
        ),
        _step(
            "adbc_sqlite_smoke",
            [
                str(python_bin),
                "-c",
                "import adbc_driver_sqlite.dbapi as adbcsqlite; conn = adbcsqlite.connect(); cur = conn.cursor(); "
                "cur.execute('select 1'); print(cur.fetchone()); cur.close(); conn.close()",
            ],
        ),
    ]

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(inventory_step["ok"] and pip_check_step["ok"] and packages_ok and all(step["ok"] for step in import_steps)),
        "python_bin": str(python_bin),
        "lock_file": str(lock_file),
        "profile_dir": str(profile_dir),
        "profile_files_present": {
            name: bool((profile_dir / f"{name}.lock.txt").exists())
            for name in ("live", "research", "media", "ops")
        },
        "inventory_step": inventory_step,
        "pip_check_step": pip_check_step,
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "import_steps": import_steps,
        "data_library_roles": _data_library_roles(),
        "recommendations": _recommendations(package_rows, import_steps),
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"sql_access_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
