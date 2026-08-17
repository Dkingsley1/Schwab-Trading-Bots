#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - packaging is pinned, but keep the audit bootable.
    InvalidVersion = Exception
    Version = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "mlx_runtime_audit_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
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
            status = "ok" if locked == installed else _version_drift_status(locked, installed)
        elif installed:
            status = "missing_lock"
        elif locked:
            status = "missing_runtime"
        else:
            status = "missing_both"
        ok = ok and (status in {"ok", "compatibility_excluded", "runtime_ahead_of_lock"})
        row = {
            "package": name,
            "locked_version": locked,
            "installed_version": installed,
            "status": status,
        }
        if status == "compatibility_excluded":
            row["compatibility_exclusion_reason"] = COMPATIBILITY_EXCLUDED_PACKAGES[name]
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
    rc, out, err = _run([str(python_bin), "-c", code])
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


def _recommendations(package_rows: list[dict[str, Any]], runtime: dict[str, Any]) -> list[str]:
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
    elif runtime.get("compile_available") and runtime.get("metal_available") and runtime.get("jit_env") != "1":
        recommendations.append("candidate_mlx_compile_direct_stable_after_smoke")
    if runtime.get("metal_available") and runtime.get("jit_env") != "1":
        recommendations.append("mlx_metal_jit_default_off")
    return recommendations


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the MLX runtime, imports, and lock alignment.")
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
    package_rows, packages_ok = _package_rows(DEFAULT_PACKAGES, lock_versions, installed_versions)
    package_statuses = {str(row["package"]): str(row["status"]) for row in package_rows}
    runtime_payload, runtime_step = _runtime_snapshot_step(python_bin)

    def maybe_step(package: str, name: str, cmd: list[str], accepted_rc: set[int] | None = None) -> dict[str, Any]:
        if package_statuses.get(_normalize_package_name(package)) == "compatibility_excluded":
            return _skipped_step(name, COMPATIBILITY_EXCLUDED_PACKAGES[_normalize_package_name(package)])
        return _step(name, cmd, accepted_rc=accepted_rc)

    import_steps = [
        _step("mlx_core_import", [str(python_bin), "-c", "import mlx.core as mx; print(mx.default_device())"]),
        _step("mlx_nn_import", [str(python_bin), "-c", "import mlx.nn as nn; print(nn.__name__)"]),
        _step("mlx_optimizers_import", [str(python_bin), "-c", "import mlx.optimizers as optim; print(optim.__name__)"]),
        _step("mlx_lm_import", [str(python_bin), "-c", "import mlx_lm; print(mlx_lm.__name__)"]),
        maybe_step("mlx-data", "mlx_data_import", [str(python_bin), "-c", "import mlx.data as mxdata; print(mxdata.__file__)"]),
        maybe_step("mlx-graphs", "mlx_graphs_import", [str(python_bin), "-c", "import mlx_graphs; print(mlx_graphs.__name__)"]),
        _step("mlx_snn_import", [str(python_bin), "-c", "import mlxsnn; print(mlxsnn.__name__)"]),
        _step("mlx_vision_import", [str(python_bin), "-c", "import mlx_vision; print(mlx_vision.__name__)"]),
        _step("mlx_vlm_import", [str(python_bin), "-c", "import mlx_vlm; print(mlx_vlm.__file__)"]),
        _step("mlx_whisper_import", [str(python_bin), "-c", "import mlx_whisper; print(mlx_whisper.__name__)"]),
        _step("mlx_audio_import", [str(python_bin), "-c", "import mlx_audio; print(mlx_audio.__name__)"], accepted_rc={0, 1}),
        _step("mlx_embeddings_import", [str(python_bin), "-c", "import mlx_embeddings; print(mlx_embeddings.__name__)"], accepted_rc={0, 1}),
        _step("mlx_embedding_models_import", [str(python_bin), "-c", "import mlx_embedding_models; print(mlx_embedding_models.__name__)"], accepted_rc={0, 1}),
        _step("esig_import", [str(python_bin), "-c", "import esig; print(esig.__name__)"]),
        _step("roughpy_import", [str(python_bin), "-c", "import roughpy; print(roughpy.__name__)"]),
        _step("parakeet_mlx_import", [str(python_bin), "-c", "import parakeet_mlx; print(parakeet_mlx.__name__)"], accepted_rc={0, 1}),
        _step(
            "indicator_bot_common_import",
            [
                str(python_bin),
                "-c",
                (
                    "import sys; "
                    "sys.path.insert(0, 'core'); "
                    "import indicator_bot_common as mod; "
                    "print(mod.__file__)"
                ),
            ],
        ),
    ]

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(inventory_step["ok"] and packages_ok and runtime_step["ok"] and all(step["ok"] for step in import_steps)),
        "python_bin": str(python_bin),
        "lock_file": str(lock_file),
        "inventory_step": inventory_step,
        "runtime_step": runtime_step,
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "runtime": runtime_payload,
        "import_steps": import_steps,
        "recommendations": _recommendations(package_rows, runtime_payload),
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"mlx_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}")
        print(f"default_device={runtime_payload.get('default_device') or 'unknown'}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
