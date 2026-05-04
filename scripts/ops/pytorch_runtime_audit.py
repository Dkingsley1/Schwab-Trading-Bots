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
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "pytorch_runtime_audit_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv312" / "bin" / "python"
DEFAULT_PACKAGES = ("torch",)


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


def _runtime_snapshot_step(python_bin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    code = """
import json

import torch

mps_backend = getattr(torch.backends, "mps", None)
mps_built = bool(mps_backend is not None and getattr(mps_backend, "is_built", lambda: False)())
mps_available = bool(mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)())
device = "mps" if mps_available else "cpu"
payload = {
    "torch_version": str(torch.__version__),
    "mps_built": mps_built,
    "mps_available": mps_available,
    "cuda_available": bool(torch.cuda.is_available()),
    "compile_available": bool(hasattr(torch, "compile")),
    "selected_device": device,
}

try:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(device)
    x = torch.randn(128, 32, device=device)
    y = model(x)
    loss = (y ** 2).mean()
    loss.backward()
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    payload["tensor_smoke_ok"] = True
    payload["tensor_smoke_loss"] = float(loss.detach().cpu().item())
except Exception as exc:
    payload["tensor_smoke_ok"] = False
    payload["tensor_smoke_error"] = repr(exc)

if payload["compile_available"]:
    try:
        torch.manual_seed(11)
        compiled_model = torch.compile(
            torch.nn.Sequential(
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            ).to(device)
        )
        x2 = torch.randn(64, 16, device=device)
        y2 = compiled_model(x2)
        loss2 = (y2 ** 2).mean()
        loss2.backward()
        if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        payload["compile_smoke_ok"] = True
        payload["compile_smoke_loss"] = float(loss2.detach().cpu().item())
    except Exception as exc:
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
        "name": "pytorch_runtime_snapshot",
        "ok": ok,
        "rc": rc,
        "command": f"{python_bin} -c <pytorch_runtime_snapshot>",
        "accepted_rc": [0],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    if not ok:
        payload = {
            "torch_version": "",
            "mps_built": False,
            "mps_available": False,
            "cuda_available": False,
            "compile_available": False,
            "selected_device": "unknown",
            "tensor_smoke_ok": False,
            "tensor_smoke_error": (_tail(err) or "snapshot_failed"),
            "compile_smoke_ok": False,
            "compile_smoke_error": "snapshot_failed",
        }
    return payload, step


def _recommendations(package_rows: list[dict[str, Any]], runtime: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    for row in package_rows:
        package = str(row["package"])
        status = str(row["status"])
        if status == "version_mismatch":
            recommendations.append(f"align_lock:{package}:{row['locked_version']}->{row['installed_version']}")
        elif status == "missing_lock":
            recommendations.append(f"lock_missing:{package}")
        elif status == "missing_runtime":
            recommendations.append(f"runtime_missing:{package}")
        elif status == "missing_both":
            recommendations.append(f"unavailable:{package}")

    if not runtime.get("mps_built"):
        recommendations.append("install_pytorch_build_with_mps_support")
    elif not runtime.get("mps_available"):
        recommendations.append("keep_pytorch_cpu_shadow_only_until_mps_available")
    elif runtime.get("tensor_smoke_ok"):
        recommendations.append("pytorch_runtime_available_for_manual_offline_replay_only")

    if runtime.get("compile_available") and not runtime.get("compile_smoke_ok"):
        recommendations.append("keep_torch_compile_off_for_canary")
    if runtime.get("selected_device") == "mps":
        recommendations.append("keep_mlx_default_live_backend_on_apple_silicon")
    recommendations.append("keep_pytorch_replay_canary_disabled_during_live_mlx_collection")
    return recommendations


def _pip_check_effectively_ok(step: dict[str, Any]) -> bool:
    if bool(step.get("ok")):
        return True
    combined = "\n".join([str(step.get("stdout_tail") or ""), str(step.get("stderr_tail") or "")])
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not lines:
        return False
    tolerated = (
        "mlx-graphs 0.0.9 has requirement fsspec==2024.2.0",
        "mlx-graphs 0.0.9 has requirement requests==2.31.0",
        "mlx-graphs 0.0.9 has requirement tqdm==4.66.1",
    )
    return all(any(line.startswith(prefix) for prefix in tolerated) for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the PyTorch runtime on the main project environment.")
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
    pip_check_step = _step("pip_check", [str(python_bin), "-m", "pip", "check"])
    pip_check_ok = _pip_check_effectively_ok(pip_check_step)
    runtime_payload, runtime_step = _runtime_snapshot_step(python_bin)
    import_steps = [
        _step("torch_import", [str(python_bin), "-c", "import torch; print(torch.__version__)"]),
        _step(
            "torch_mps_probe",
            [
                str(python_bin),
                "-c",
                (
                    "import torch; "
                    "mps = getattr(torch.backends, 'mps', None); "
                    "print(getattr(mps, 'is_built', lambda: False)()); "
                    "print(getattr(mps, 'is_available', lambda: False)())"
                ),
            ],
        ),
    ]
    recommendations = _recommendations(package_rows, runtime_payload)
    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(
            inventory_step["ok"]
            and pip_check_ok
            and packages_ok
            and all(step["ok"] for step in import_steps)
            and runtime_step["ok"]
        ),
        "python_bin": str(python_bin),
        "lock_file": str(lock_file),
        "inventory_step": inventory_step,
        "pip_check_step": pip_check_step,
        "pip_check_effectively_ok": bool(pip_check_ok),
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "runtime": runtime_payload,
        "runtime_step": runtime_step,
        "import_steps": import_steps,
        "recommendations": recommendations,
        "notes": [
            "PyTorch canary here is a runtime sidecar check and does not reroute the MLX trading brain.",
            "PyTorch replay is disabled by default; use MLX for live collection and only run PyTorch checks intentionally.",
            "mlx-graphs pins older requests/fsspec/tqdm metadata, so the audit tolerates those exact optional-package conflicts while preserving the newer ingestion-safe versions.",
        ],
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"pytorch_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
