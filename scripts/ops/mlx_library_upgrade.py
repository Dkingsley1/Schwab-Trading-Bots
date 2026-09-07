#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import (  # noqa: E402
    maintenance_hold_snapshot,
    maintenance_hold_token_authorized,
)

DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "mlx_library_upgrade_latest.json"
DEFAULT_BACKUP_DIR = (
    PROJECT_ROOT / "governance" / "health" / "dependency_upgrade_backups"
)
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
DEFAULT_TIMEOUT_SECONDS = 30 * 60
STACK_STOPPED_FLAG = PROJECT_ROOT / "governance" / "health" / "STACK_STOPPED.flag"
RUNTIME_PROCESS_MARKERS = (
    "scripts/shadow_watchdog.py",
    "scripts/ops/process_watchdog.py",
    "scripts/run_shadow_training_loop.py",
    "scripts/ops/sql_link_writer_service.py",
    "scripts/ops/sql_link_shard_manager.py",
)
MLX_PACKAGE_NAMES = {
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "mlx-data",
    "mlx-graphs",
    "mlx-cluster",
    "mlx-snn",
    "mlx-vision",
    "mlx-vlm",
    "mlx-whisper",
    "mlx-audio",
    "mlx-embeddings",
    "mlx-embedding-models",
    "esig",
    "roughpy",
    "pyrecombine",
    "parakeet-mlx",
}
QUANT_CAPABILITY_PACKAGES = {"cvxpy", "linearmodels", "river"}


def _normalize(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _lock_versions(lock_path: Path, *, scope: str = "mlx") -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        normalized = _normalize(package)
        if scope == "all" or normalized in MLX_PACKAGE_NAMES:
            versions[normalized] = version.strip()
    return versions


def _command_result(
    name: str,
    command: list[str],
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONNOUSERSITE", "1")
    merged_env.setdefault("PYTHONFAULTHANDLER", "1")
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_seconds), 1),
            start_new_session=True,
            env=merged_env,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stderr = f"{stderr}\nprocess_timeout_after={timeout_seconds}s".strip()
        timed_out = True
    signal_number = abs(rc) if rc < 0 else 0
    signal_name = ""
    if signal_number:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "name": name,
        "ok": rc == 0,
        "rc": rc,
        "timed_out": timed_out,
        "termination_signal": signal_name,
        "command": command,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": "\n".join(stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
        "stdout": stdout,
    }


def _installed_versions(python_bin: Path) -> tuple[dict[str, str], dict[str, Any]]:
    result = _command_result(
        "installed_inventory",
        [str(python_bin), "-m", "pip", "list", "--format=json"],
    )
    versions: dict[str, str] = {}
    if result["ok"]:
        try:
            rows = json.loads(str(result.get("stdout") or "[]"))
        except json.JSONDecodeError:
            result["ok"] = False
            result["stderr_tail"] = "installed inventory was not valid JSON"
            rows = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            package = _normalize(str(row.get("name") or ""))
            if package:
                versions[package] = str(row.get("version") or "")
    result.pop("stdout", None)
    return versions, result


def _runtime_processes(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    result = _command_result(
        "runtime_process_inventory", ["ps", "-axo", "pid=,command="]
    )
    if not result["ok"]:
        return [{"pid": 0, "command": "runtime process inventory failed"}]
    matches: list[dict[str, Any]] = []
    root_text = str(project_root)
    for raw in str(result.get("stdout") or "").splitlines():
        line = raw.strip()
        if not line or root_text not in line:
            continue
        if not any(marker in line for marker in RUNTIME_PROCESS_MARKERS):
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            pid = 0
        if pid != os.getpid():
            matches.append({"pid": pid, "command": command.strip()})
    return matches


def transaction_preflight(
    *,
    project_root: Path = PROJECT_ROOT,
    python_bin: Path = DEFAULT_PYTHON,
    lock_path: Path = DEFAULT_LOCK,
    maintenance_token: str = "",
    acknowledge_maintenance: bool = False,
    runtime_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hold = maintenance_hold_snapshot(project_root)
    token_authorized = maintenance_hold_token_authorized(hold, token=maintenance_token)
    stopped_flag = project_root / "governance" / "health" / STACK_STOPPED_FLAG.name
    active_processes = (
        _runtime_processes(project_root)
        if runtime_processes is None
        else runtime_processes
    )
    failures: list[str] = []
    if not acknowledge_maintenance:
        failures.append("maintenance_acknowledgement_required")
    if not bool(hold.get("active", False)):
        failures.append("active_maintenance_hold_required")
    if not token_authorized:
        failures.append("maintenance_token_required")
    if not stopped_flag.exists():
        failures.append("stack_stopped_flag_required")
    if active_processes:
        failures.append("runtime_processes_still_active")
    if not python_bin.exists():
        failures.append("target_python_missing")
    if not lock_path.exists():
        failures.append("lock_file_missing")
    return {
        "ok": not failures,
        "failures": failures,
        "maintenance_hold": {
            key: value for key, value in hold.items() if key not in {"token", "payload"}
        },
        "maintenance_token_authorized": token_authorized,
        "stack_stopped_flag": str(stopped_flag),
        "stack_stopped_flag_present": stopped_flag.exists(),
        "active_runtime_processes": active_processes,
    }


def build_payload(
    *,
    lock_path: Path = DEFAULT_LOCK,
    python_bin: Path = DEFAULT_PYTHON,
    scope: str = "mlx",
) -> dict[str, Any]:
    versions = _lock_versions(lock_path, scope=scope)
    install_args = [f"{name}=={version}" for name, version in sorted(versions.items())]
    if scope == "all":
        install_command = [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--upgrade-strategy",
            "only-if-needed",
            "-r",
            str(lock_path),
        ]
    else:
        install_command = [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "-U",
            *install_args,
        ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "ok": bool(install_args),
        "scope": scope,
        "apply_ran": False,
        "python_bin": str(python_bin),
        "lock_file": str(lock_path),
        "packages": [
            {"package": name, "version": version}
            for name, version in sorted(versions.items())
        ],
        "install_command": install_command,
        "transaction_guards": [
            "active_expiring_maintenance_hold",
            "matching_maintenance_token",
            "explicit_stack_stop",
            "runtime_process_absence",
            "pre_upgrade_environment_snapshot",
            "pip_dependency_check",
            "native_mlx_runtime_audit",
            "quant_capability_smoke",
            "pytest_validation",
            "automatic_dependency_rollback",
        ],
        "recommended_after_apply": [
            "./scripts/ops/opsctl.sh mlx-audit --json",
            "./scripts/ops/opsctl.sh quant-model-control --json",
            "./scripts/ops/opsctl.sh status",
        ],
    }


def write_payload(payload: dict[str, Any], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _snapshot_environment(
    python_bin: Path, backup_dir: Path
) -> tuple[Path | None, dict[str, Any]]:
    result = _command_result(
        "pre_upgrade_environment_snapshot",
        [
            str(python_bin),
            "-m",
            "pip",
            "freeze",
            "--all",
            "--exclude",
            "pip",
        ],
    )
    result_stdout = str(result.pop("stdout", "") or "")
    if not result["ok"]:
        return None, result
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir.mkdir(parents=True, exist_ok=True)
    snapshot = backup_dir / f"requirements_before_{timestamp}.txt"
    snapshot.write_text(result_stdout.rstrip() + "\n", encoding="utf-8")
    result["snapshot_file"] = str(snapshot)
    result["package_count"] = sum(
        1 for line in result_stdout.splitlines() if "==" in line
    )
    return snapshot, result


def _quant_capability_command(python_bin: Path) -> list[str]:
    code = (
        "import numpy as np; import cvxpy as cp; import river; import linearmodels; "
        "from river import drift; x=cp.Variable(2); "
        "p=cp.Problem(cp.Minimize(cp.sum_squares(x-np.array([0.25,0.75]))),"
        "[cp.sum(x)==1,x>=0]); p.solve(); d=drift.ADWIN(); "
        "[d.update(v) for v in ([0.0]*50+[1.0]*50)]; "
        "assert p.status in {'optimal','optimal_inaccurate'}; "
        "assert abs(float(sum(x.value))-1.0)<1e-6; "
        "print({'cvxpy':cp.__version__,'river':river.__version__,"
        "'linearmodels':linearmodels.__version__})"
    )
    return [str(python_bin), "-I", "-c", code]


def _validation_commands(
    python_bin: Path,
    lock_path: Path,
    *,
    full_test: bool,
    quant_capability: bool,
) -> list[tuple[str, list[str], int]]:
    commands: list[tuple[str, list[str], int]] = [
        ("pip_check", [str(python_bin), "-m", "pip", "check"], 5 * 60),
        (
            "mlx_runtime_audit",
            [
                str(python_bin),
                str(PROJECT_ROOT / "scripts" / "ops" / "mlx_runtime_audit.py"),
                "--python-bin",
                str(python_bin),
                "--lock-file",
                str(lock_path),
                "--out",
                str(
                    PROJECT_ROOT
                    / "governance"
                    / "health"
                    / "mlx_runtime_audit_latest.json"
                ),
                "--json",
            ],
            15 * 60,
        ),
    ]
    if quant_capability:
        commands.append(
            (
                "quant_capability_smoke",
                _quant_capability_command(python_bin),
                5 * 60,
            )
        )
    if full_test:
        test_args = [str(python_bin), "-m", "pytest", "-q"]
        test_timeout = 45 * 60
    else:
        test_args = [
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            "tests/test_mlx_runtime_guard.py",
            "tests/test_mlx_runtime_audit.py",
            "tests/test_mlx_library_upgrade.py",
            "tests/test_advanced_quant_models.py",
            "tests/test_monitoring_layer.py",
        ]
        test_timeout = 15 * 60
    commands.append(("pytest_validation", test_args, test_timeout))
    return commands


def _rollback_environment(
    *,
    python_bin: Path,
    snapshot: Path,
    before_versions: dict[str, str],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    restore = _command_result(
        "rollback_reinstall_snapshot",
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "-r",
            str(snapshot),
        ],
        timeout_seconds=45 * 60,
    )
    restore.pop("stdout", None)
    steps.append(restore)
    after_versions, inventory = _installed_versions(python_bin)
    steps.append(inventory)
    protected = {"pip", "setuptools", "wheel"}
    extras = sorted(set(after_versions) - set(before_versions) - protected)
    if extras:
        uninstall = _command_result(
            "rollback_remove_added_packages",
            [str(python_bin), "-m", "pip", "uninstall", "-y", *extras],
            timeout_seconds=15 * 60,
        )
        uninstall.pop("stdout", None)
        steps.append(uninstall)
    check = _command_result(
        "rollback_pip_check",
        [str(python_bin), "-m", "pip", "check"],
        timeout_seconds=5 * 60,
    )
    check.pop("stdout", None)
    steps.append(check)
    return {
        "attempted": True,
        "ok": all(bool(step.get("ok")) for step in steps),
        "removed_added_packages": extras,
        "steps": steps,
    }


def apply_transaction(
    payload: dict[str, Any],
    *,
    python_bin: Path,
    lock_path: Path,
    backup_dir: Path,
    maintenance_token: str,
    acknowledge_maintenance: bool,
    full_test: bool,
    rollback_on_failure: bool,
) -> dict[str, Any]:
    preflight = transaction_preflight(
        project_root=PROJECT_ROOT,
        python_bin=python_bin,
        lock_path=lock_path,
        maintenance_token=maintenance_token,
        acknowledge_maintenance=acknowledge_maintenance,
    )
    payload["preflight"] = preflight
    payload["apply_ran"] = True
    if not preflight["ok"]:
        payload["ok"] = False
        payload["transaction_status"] = "blocked_preflight"
        return payload

    before_versions, inventory = _installed_versions(python_bin)
    payload["before_inventory"] = inventory
    snapshot, snapshot_step = _snapshot_environment(python_bin, backup_dir)
    payload["snapshot"] = snapshot_step
    if not inventory["ok"] or snapshot is None:
        payload["ok"] = False
        payload["transaction_status"] = "blocked_snapshot"
        return payload

    install = _command_result(
        "install_locked_dependencies",
        list(payload["install_command"]),
        timeout_seconds=45 * 60,
    )
    install.pop("stdout", None)
    payload["install_result"] = install
    validation_steps: list[dict[str, Any]] = []
    if install["ok"]:
        locked_names = set(_lock_versions(lock_path, scope="all"))
        quant_capability = QUANT_CAPABILITY_PACKAGES.issubset(locked_names)
        for name, command, timeout_seconds in _validation_commands(
            python_bin,
            lock_path,
            full_test=full_test,
            quant_capability=quant_capability,
        ):
            step = _command_result(name, command, timeout_seconds=timeout_seconds)
            step.pop("stdout", None)
            validation_steps.append(step)
            if not step["ok"]:
                break
    payload["validation_steps"] = validation_steps
    transaction_ok = bool(
        install["ok"]
        and validation_steps
        and all(step["ok"] for step in validation_steps)
    )
    if transaction_ok:
        after_versions, after_inventory = _installed_versions(python_bin)
        payload["after_inventory"] = after_inventory
        payload["changed_packages"] = [
            {
                "package": package,
                "before": before_versions.get(package),
                "after": version,
            }
            for package, version in sorted(after_versions.items())
            if before_versions.get(package) != version
        ]
        payload["removed_packages"] = sorted(set(before_versions) - set(after_versions))
        payload["ok"] = bool(after_inventory["ok"])
        payload["transaction_status"] = (
            "validated" if payload["ok"] else "inventory_failed"
        )
        return payload

    payload["ok"] = False
    payload["transaction_status"] = "validation_failed"
    if rollback_on_failure:
        payload["rollback"] = _rollback_environment(
            python_bin=python_bin,
            snapshot=snapshot,
            before_versions=before_versions,
        )
        payload["transaction_status"] = (
            "rolled_back" if payload["rollback"]["ok"] else "rollback_failed"
        )
    else:
        payload["rollback"] = {"attempted": False, "ok": False, "steps": []}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or transactionally apply a pinned dependency upgrade bundle."
    )
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--scope", choices=("mlx", "all"), default="mlx")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack-maintenance", action="store_true")
    parser.add_argument("--maintenance-token", default="")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python_bin = _absolute_without_resolving_symlinks(Path(args.python_bin))
    lock_path = Path(args.lock_file).expanduser().resolve()
    out_file = Path(args.out_file).expanduser().resolve()
    backup_dir = Path(args.backup_dir).expanduser().resolve()
    try:
        payload = build_payload(
            lock_path=lock_path, python_bin=python_bin, scope=args.scope
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": 2,
            "ok": False,
            "scope": args.scope,
            "apply_ran": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.apply and payload.get("install_command"):
        payload = apply_transaction(
            payload,
            python_bin=python_bin,
            lock_path=lock_path,
            backup_dir=backup_dir,
            maintenance_token=str(args.maintenance_token or ""),
            acknowledge_maintenance=bool(args.ack_maintenance),
            full_test=bool(args.full_test),
            rollback_on_failure=not bool(args.no_rollback),
        )
    write_payload(payload, out_file)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"mlx_library_upgrade scope={payload.get('scope', args.scope)} "
            f"packages={len(payload.get('packages') or [])} "
            f"apply={int(bool(payload.get('apply_ran')))} "
            f"status={payload.get('transaction_status', 'planned')} out={out_file}"
        )
    return 0 if bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
