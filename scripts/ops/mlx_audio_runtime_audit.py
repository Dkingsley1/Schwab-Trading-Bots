#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    PROJECT_ROOT / "governance" / "health" / "mlx_audio_runtime_audit_latest.json"
)
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
DEFAULT_STEP_TIMEOUT_SECONDS = 60.0
DEFAULT_PACKAGES = (
    "mlx-audio",
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "transformers",
    "huggingface-hub",
    "miniaudio",
    "sounddevice",
)
HARD_FAILURE_MARKERS = (
    "ModuleNotFoundError",
    "ImportError:",
    "No module named",
    "Traceback (most recent call last)",
    "nanobind: critical error",
    "Fatal Python error",
    "Abort trap",
)
PROBE_STOP_FAILURE_KINDS = {"native_signal", "timeout"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _tail(text: str, n: int = 8) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _timeout_stream(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _run(
    cmd: list[str], *, timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=max(float(timeout_seconds), 1.0),
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_stream(exc.stdout)
        stderr = _timeout_stream(exc.stderr)
        return (
            124,
            stdout,
            f"{stderr}\nprocess_timeout_after={timeout_seconds:g}s".strip(),
        )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _termination_metadata(returncode: int) -> dict[str, Any]:
    if returncode < 0:
        signal_number = abs(int(returncode))
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
        return {
            "failure_kind": "native_signal",
            "termination_signal": signal_name,
            "termination_signal_number": signal_number,
        }
    if returncode == 124:
        return {"failure_kind": "timeout"}
    if returncode != 0:
        return {"failure_kind": "process_error"}
    return {"failure_kind": "none"}


def _isolated_code_command(python_bin: Path, code: str) -> list[str]:
    return [str(python_bin), "-I", "-c", code]


def _display_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _parse_version_lines(lines: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if (not line) or line.startswith("#") or ("==" not in line):
            continue
        package, version = line.split("==", 1)
        versions[_normalize_package_name(package)] = version.strip()
    return versions


def _step(
    name: str,
    cmd: list[str],
    accepted_rc: set[int] | None = None,
    *,
    command_display: str | None = None,
) -> dict[str, Any]:
    accepted = accepted_rc or {0}
    rc, out, err = _run(cmd)
    combined = f"{out}\n{err}".strip()
    hard_fail = any(marker in combined for marker in HARD_FAILURE_MARKERS)
    ok = (rc in accepted) and (not hard_fail)
    result = {
        "name": name,
        "ok": ok,
        "rc": rc,
        "command": command_display or _display_command(cmd),
        "accepted_rc": sorted(accepted),
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    result.update(_termination_metadata(rc))
    return result


def _blocked_step(name: str, blocked_by: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": False,
        "rc": 125,
        "command": "blocked",
        "accepted_rc": [0],
        "stdout_tail": "",
        "stderr_tail": f"native_probe_circuit_open:{blocked_by}",
        "failure_kind": "prerequisite_blocked",
        "blocked_by": blocked_by,
    }


def _run_probe_sequence(
    probes: list[tuple[str, list[str]]], *, prerequisite_ok: bool
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    blocked_by = "" if prerequisite_ok else "mlx_audio_runtime_snapshot"
    for name, cmd in probes:
        if blocked_by:
            results.append(_blocked_step(name, blocked_by))
            continue
        result = _step(name, cmd)
        results.append(result)
        if result.get("failure_kind") in PROBE_STOP_FAILURE_KINDS:
            blocked_by = name
    return results


def _load_installed_versions(python_bin: Path) -> tuple[dict[str, str], dict[str, Any]]:
    cmd = [str(python_bin), "-m", "pip", "list", "--format=freeze"]
    rc, out, err = _run(cmd)
    step = {
        "name": "installed_package_inventory",
        "ok": rc == 0,
        "rc": rc,
        "command": _display_command(cmd),
        "accepted_rc": [0],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    step.update(_termination_metadata(rc))
    return (_parse_version_lines(out.splitlines()) if rc == 0 else {}), step


def _package_rows(
    packages: tuple[str, ...], installed_versions: dict[str, str]
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok = True
    for raw_name in packages:
        name = _normalize_package_name(raw_name)
        installed = installed_versions.get(name)
        status = "ok" if installed else "missing_runtime"
        ok = ok and (status == "ok")
        rows.append(
            {
                "package": name,
                "installed_version": installed,
                "status": status,
            }
        )
    return rows, ok


def _runtime_snapshot_step(python_bin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    code = """
import json
import mlx.core as mx

payload = {
    "default_device": str(mx.default_device()),
    "compile_available": bool(hasattr(mx, "compile")),
    "metal_attr_available": bool(getattr(mx, "metal", None) is not None),
}
metal = getattr(mx, "metal", None)
if metal is not None and hasattr(metal, "is_available"):
    try:
        payload["metal_available"] = bool(metal.is_available())
    except Exception as exc:
        payload["metal_available"] = None
        payload["metal_error"] = repr(exc)
else:
    payload["metal_available"] = None

print(json.dumps(payload, ensure_ascii=True))
"""
    cmd = _isolated_code_command(python_bin, code)
    rc, out, err = _run(cmd)
    ok = rc == 0
    payload: dict[str, Any] = {}
    if ok:
        try:
            payload = json.loads(out.strip() or "{}")
        except json.JSONDecodeError:
            ok = False
    step = {
        "name": "mlx_audio_runtime_snapshot",
        "ok": ok,
        "rc": rc,
        "command": f"{python_bin} -I -c <mlx_audio_runtime_snapshot>",
        "accepted_rc": [0],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    step.update(_termination_metadata(rc))
    if not ok:
        payload = {
            "default_device": "",
            "compile_available": False,
            "metal_attr_available": False,
            "metal_available": None,
        }
    return payload, step


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the isolated mlx-audio runtime."
    )
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    inventory, inventory_step = _load_installed_versions(python_bin)
    package_rows, packages_ok = _package_rows(DEFAULT_PACKAGES, inventory)
    pip_check_step = _step("pip_check", [str(python_bin), "-m", "pip", "check"])
    runtime_payload, runtime_step = _runtime_snapshot_step(python_bin)
    import_steps = _run_probe_sequence(
        [
            (
                "mlx_core_import",
                _isolated_code_command(
                    python_bin, "import mlx.core as mx; print(mx.default_device())"
                ),
            ),
            (
                "mlx_audio_import",
                _isolated_code_command(
                    python_bin, "import mlx_audio; print(mlx_audio.__file__)"
                ),
            ),
            (
                "miniaudio_import",
                _isolated_code_command(
                    python_bin, "import miniaudio; print(miniaudio.__name__)"
                ),
            ),
        ],
        prerequisite_ok=bool(runtime_step["ok"]),
    )

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(
            inventory_step["ok"]
            and pip_check_step["ok"]
            and packages_ok
            and runtime_step["ok"]
            and all(step["ok"] for step in import_steps)
        ),
        "python_bin": str(python_bin),
        "inventory_step": inventory_step,
        "pip_check_step": pip_check_step,
        "runtime_step": runtime_step,
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "runtime": runtime_payload,
        "import_steps": import_steps,
        "process_isolation": {
            "isolated_python": True,
            "timeout_seconds": DEFAULT_STEP_TIMEOUT_SECONDS,
            "native_signal_blocks_remaining_probes": True,
        },
    }

    out_file.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"mlx_audio_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}"
        )
        print(f"default_device={runtime_payload.get('default_device') or 'unknown'}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
