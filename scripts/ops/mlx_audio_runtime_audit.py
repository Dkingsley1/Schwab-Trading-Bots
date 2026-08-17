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
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "mlx_audio_runtime_audit_latest.json"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
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


def _package_rows(packages: tuple[str, ...], installed_versions: dict[str, str]) -> tuple[list[dict[str, Any]], bool]:
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
    rc, out, err = _run([str(python_bin), "-c", code])
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
        "command": f"{python_bin} -c <mlx_audio_runtime_snapshot>",
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
        }
    return payload, step


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the isolated mlx-audio runtime.")
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
    import_steps = [
        _step("mlx_audio_import", [str(python_bin), "-c", "import mlx_audio; print(mlx_audio.__file__)"]),
        _step("mlx_core_import", [str(python_bin), "-c", "import mlx.core as mx; print(mx.default_device())"]),
        _step("miniaudio_import", [str(python_bin), "-c", "import miniaudio; print(miniaudio.__name__)"]),
    ]

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(inventory_step["ok"] and pip_check_step["ok"] and packages_ok and runtime_step["ok"] and all(step["ok"] for step in import_steps)),
        "python_bin": str(python_bin),
        "inventory_step": inventory_step,
        "pip_check_step": pip_check_step,
        "runtime_step": runtime_step,
        "critical_packages_ok": bool(packages_ok),
        "package_rows": package_rows,
        "runtime": runtime_payload,
        "import_steps": import_steps,
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"mlx_audio_runtime_audit ok={str(payload['ok']).lower()} python={python_bin}")
        print(f"default_device={runtime_payload.get('default_device') or 'unknown'}")
        print(f"report={out_file}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
