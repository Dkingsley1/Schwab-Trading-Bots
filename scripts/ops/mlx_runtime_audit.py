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

try:
    from packaging.version import InvalidVersion, Version
except (
    Exception
):  # pragma: no cover - packaging is pinned, but keep the audit bootable.
    InvalidVersion = Exception
    Version = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "mlx_runtime_audit_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
DEFAULT_STEP_TIMEOUT_SECONDS = 60.0
COMPATIBILITY_EXCLUDED_PACKAGES = {
    "mlx-data": "no compatible distribution is available for the active Python 3.14 runtime",
    "mlx-graphs": "requires mlx-cluster and older shared dependency pins under latest MLX",
    "mlx-cluster": "native extension is not compatible with the latest MLX Metal device API",
}
DEFAULT_PACKAGES = (
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
    "transformers",
    "huggingface-hub",
    "scipy",
    "schwab-py",
    "duckdb",
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


def _run(
    cmd: list[str],
    *,
    timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS,
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


def _indicator_import_command(python_bin: Path) -> list[str]:
    project_root = repr(str(PROJECT_ROOT))
    return _isolated_code_command(
        python_bin,
        (
            f"import sys; sys.path.insert(0, {project_root}); "
            "import core.indicator_bot_common as mod; "
            "print(mod.__file__)"
        ),
    )


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


def _step(
    name: str, cmd: list[str], accepted_rc: set[int] | None = None
) -> dict[str, Any]:
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
            "nanobind: critical error",
            "Fatal Python error",
            "Abort trap",
        )
    )
    ok = (rc in accepted) and (not hard_fail)
    result = {
        "name": name,
        "ok": ok,
        "rc": rc,
        "command": " ".join(shlex.quote(x) for x in cmd),
        "accepted_rc": sorted(accepted),
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    result.update(_termination_metadata(rc))
    return result


def _skipped_step(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": True,
        "rc": 0,
        "command": "skipped",
        "accepted_rc": [0],
        "stdout_tail": f"compatibility_excluded: {reason}",
        "stderr_tail": "",
    }


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
    probe_specs: list[tuple[str, list[str], set[int] | None, str]],
    *,
    prerequisite_ok: bool,
    prerequisite_name: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    blocked_by = "" if prerequisite_ok else prerequisite_name
    for name, cmd, accepted_rc, skip_reason in probe_specs:
        if blocked_by:
            results.append(_blocked_step(name, blocked_by))
            continue
        if skip_reason:
            results.append(_skipped_step(name, skip_reason))
            continue
        result = _step(name, cmd, accepted_rc=accepted_rc)
        results.append(result)
        if result.get("failure_kind") in {"native_signal", "timeout"}:
            blocked_by = name
    return results


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
        if name in COMPATIBILITY_EXCLUDED_PACKAGES and not installed:
            status = "compatibility_excluded"
        elif locked and installed:
            status = (
                "ok"
                if locked == installed
                else _version_drift_status(locked, installed)
            )
        elif installed:
            status = "missing_lock"
        elif locked:
            status = "missing_runtime"
        else:
            status = "missing_both"
        ok = ok and (
            status in {"ok", "compatibility_excluded", "runtime_ahead_of_lock"}
        )
        row = {
            "package": name,
            "locked_version": locked,
            "installed_version": installed,
            "status": status,
        }
        if status == "compatibility_excluded":
            row["compatibility_exclusion_reason"] = COMPATIBILITY_EXCLUDED_PACKAGES[
                name
            ]
        rows.append(row)
    return rows, ok


def _version_drift_status(locked: str, installed: str) -> str:
    if Version is None:
        return "version_mismatch"
    try:
        locked_version = Version(str(locked))
        installed_version = Version(str(installed))
    except InvalidVersion:
        return "version_mismatch"
    if installed_version > locked_version:
        return "runtime_ahead_of_lock"
    if installed_version < locked_version:
        return "runtime_behind_lock"
    return "version_mismatch"


def _runtime_snapshot_step(python_bin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    code = """
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

payload = {
    "default_device": str(mx.default_device()),
    "compile_available": bool(hasattr(mx, "compile")),
    "nn_available": bool(nn is not None),
    "optimizers_available": bool(optim is not None),
    "metal_attr_available": bool(getattr(mx, "metal", None) is not None),
    "jit_env": os.getenv("MLX_METAL_JIT", "unset"),
    "float16_available": bool(hasattr(mx, "float16")),
    "bfloat16_available": bool(hasattr(mx, "bfloat16")),
}
metal = getattr(mx, "metal", None)
if metal is not None and hasattr(metal, "is_available"):
    try:
        payload["metal_available"] = bool(metal.is_available())
    except Exception as exc:  # pragma: no cover - exercised via subprocess
        payload["metal_available"] = None
        payload["metal_error"] = repr(exc)
else:
    payload["metal_available"] = None

if payload["compile_available"]:
    try:
        @mx.compile
        def compiled_add(x):
            return x + 1.0

        out = compiled_add(mx.array([1.0, 2.0], dtype=mx.float32))
        mx.eval(out)
        payload["compile_smoke_ok"] = bool(tuple(out.shape) == (2,))
    except Exception as exc:  # pragma: no cover - exercised via subprocess
        payload["compile_smoke_ok"] = False
        payload["compile_smoke_error"] = repr(exc)
else:
    payload["compile_smoke_ok"] = False
    payload["compile_smoke_error"] = "compile_unavailable"

print(json.dumps(payload, ensure_ascii=True))
"""
    rc, out, err = _run(_isolated_code_command(python_bin, code))
    ok = rc == 0
    payload: dict[str, Any] = {}
    if ok:
        try:
            payload = json.loads(out.strip() or "{}")
        except json.JSONDecodeError:
            ok = False
    step = {
        "name": "mlx_runtime_snapshot",
        "ok": ok,
        "rc": rc,
        "command": f"{python_bin} -c <mlx_runtime_snapshot>",
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
            "compile_smoke_ok": False,
            "compile_smoke_error": (_tail(err) or "snapshot_failed"),
            "jit_env": os.getenv("MLX_METAL_JIT", "unset"),
        }
    return payload, step


def _recommendations(
    package_rows: list[dict[str, Any]], runtime: dict[str, Any]
) -> list[str]:
    recommendations: list[str] = []
    for row in package_rows:
        package = str(row["package"])
        status = str(row["status"])
        if status == "runtime_ahead_of_lock":
            recommendations.append(
                f"align_lock:{package}:{row['locked_version']}->{row['installed_version']}"
            )
        elif status in {"runtime_behind_lock", "version_mismatch"}:
            recommendations.append(
                f"upgrade_runtime:{package}:{row['installed_version']}->{row['locked_version']}"
            )
        elif status == "missing_lock":
            recommendations.append(f"lock_missing:{package}")
        elif status == "missing_runtime":
            recommendations.append(f"runtime_missing:{package}")
        elif status == "missing_both":
            recommendations.append(f"unavailable:{package}")
        elif status == "compatibility_excluded":
            recommendations.append(f"compatibility_excluded:{package}")
    if runtime.get("compile_available") and not runtime.get("compile_smoke_ok"):
        recommendations.append("keep_mlx_compile_opt_in_until_compile_smoke_passes")
    elif (
        runtime.get("compile_available")
        and runtime.get("metal_available")
        and runtime.get("jit_env") != "1"
    ):
        recommendations.append("candidate_mlx_compile_direct_stable_after_smoke")
    if runtime.get("metal_available") and runtime.get("jit_env") != "1":
        recommendations.append("mlx_metal_jit_default_off")
    return recommendations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the MLX runtime, imports, and lock alignment."
    )
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser()
    lock_file = Path(args.lock_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lock_versions = _load_lock_versions(lock_file)
    installed_versions, inventory_step = _load_installed_versions(python_bin)
    package_rows, packages_ok = _package_rows(
        DEFAULT_PACKAGES, lock_versions, installed_versions
    )
    package_statuses = {str(row["package"]): str(row["status"]) for row in package_rows}
    runtime_payload, runtime_step = _runtime_snapshot_step(python_bin)

    def compatibility_reason(package: str) -> str:
        normalized = _normalize_package_name(package)
        if package_statuses.get(normalized) == "compatibility_excluded":
            return COMPATIBILITY_EXCLUDED_PACKAGES[normalized]
        return ""

    probe_specs: list[tuple[str, list[str], set[int] | None, str]] = [
        (
            "mlx_core_import",
            _isolated_code_command(
                python_bin, "import mlx.core as mx; print(mx.default_device())"
            ),
            None,
            "",
        ),
        (
            "mlx_nn_import",
            _isolated_code_command(
                python_bin, "import mlx.nn as nn; print(nn.__name__)"
            ),
            None,
            "",
        ),
        (
            "mlx_optimizers_import",
            _isolated_code_command(
                python_bin, "import mlx.optimizers as optim; print(optim.__name__)"
            ),
            None,
            "",
        ),
        (
            "mlx_lm_import",
            _isolated_code_command(python_bin, "import mlx_lm; print(mlx_lm.__name__)"),
            None,
            "",
        ),
        (
            "mlx_data_import",
            _isolated_code_command(
                python_bin, "import mlx.data as mxdata; print(mxdata.__file__)"
            ),
            None,
            compatibility_reason("mlx-data"),
        ),
        (
            "mlx_graphs_import",
            _isolated_code_command(
                python_bin, "import mlx_graphs; print(mlx_graphs.__name__)"
            ),
            None,
            compatibility_reason("mlx-graphs"),
        ),
        (
            "mlx_snn_import",
            _isolated_code_command(python_bin, "import mlxsnn; print(mlxsnn.__name__)"),
            None,
            "",
        ),
        (
            "mlx_vision_import",
            _isolated_code_command(
                python_bin, "import mlx_vision; print(mlx_vision.__name__)"
            ),
            None,
            "",
        ),
        (
            "mlx_vlm_import",
            _isolated_code_command(
                python_bin, "import mlx_vlm; print(mlx_vlm.__file__)"
            ),
            None,
            "",
        ),
        (
            "mlx_whisper_import",
            _isolated_code_command(
                python_bin, "import mlx_whisper; print(mlx_whisper.__name__)"
            ),
            None,
            "",
        ),
        (
            "mlx_audio_import",
            _isolated_code_command(
                python_bin, "import mlx_audio; print(mlx_audio.__name__)"
            ),
            {0, 1},
            "",
        ),
        (
            "mlx_embeddings_import",
            _isolated_code_command(
                python_bin, "import mlx_embeddings; print(mlx_embeddings.__name__)"
            ),
            {0, 1},
            "",
        ),
        (
            "mlx_embedding_models_import",
            _isolated_code_command(
                python_bin,
                "import mlx_embedding_models; print(mlx_embedding_models.__name__)",
            ),
            {0, 1},
            "",
        ),
        (
            "esig_import",
            _isolated_code_command(python_bin, "import esig; print(esig.__name__)"),
            None,
            "",
        ),
        (
            "roughpy_import",
            _isolated_code_command(
                python_bin, "import roughpy; print(roughpy.__name__)"
            ),
            None,
            "",
        ),
        (
            "parakeet_mlx_import",
            _isolated_code_command(
                python_bin, "import parakeet_mlx; print(parakeet_mlx.__name__)"
            ),
            {0, 1},
            "",
        ),
        (
            "indicator_bot_common_import",
            _indicator_import_command(python_bin),
            None,
            "",
        ),
    ]
    import_steps = _run_probe_sequence(
        probe_specs,
        prerequisite_ok=bool(runtime_step["ok"]),
        prerequisite_name="mlx_runtime_snapshot",
    )

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(
            inventory_step["ok"]
            and packages_ok
            and runtime_step["ok"]
            and all(step["ok"] for step in import_steps)
        ),
        "python_bin": str(python_bin),
        "lock_file": str(lock_file),
        "inventory_step": inventory_step,
        "runtime_step": runtime_step,
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "runtime": runtime_payload,
        "import_steps": import_steps,
        "recommendations": _recommendations(package_rows, runtime_payload),
        "process_isolation": {
            "isolated_python": True,
            "timeout_seconds": DEFAULT_STEP_TIMEOUT_SECONDS,
            "native_signal_blocks_promotion": True,
            "native_probe_circuit_breaker": True,
            "canonical_project_imports": True,
        },
    }

    out_file.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"mlx_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}")
        print(f"default_device={runtime_payload.get('default_device') or 'unknown'}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
