#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "mlx_library_upgrade_latest.json"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv312" / "bin" / "python"
MLX_PACKAGE_NAMES = {
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "mlx-data",
    "mlx-vlm",
    "mlx-whisper",
    "mlx-audio",
    "mlx-embeddings",
    "mlx-embedding-models",
    "parakeet-mlx",
}


def _normalize(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _lock_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        normalized = _normalize(package)
        if normalized in MLX_PACKAGE_NAMES:
            versions[normalized] = version.strip()
    return versions


def build_payload(*, lock_path: Path = DEFAULT_LOCK, python_bin: Path = DEFAULT_PYTHON) -> dict[str, Any]:
    versions = _lock_versions(lock_path)
    install_args = [f"{name}=={version}" for name, version in sorted(versions.items())]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": bool(install_args),
        "apply_ran": False,
        "python_bin": str(python_bin),
        "lock_file": str(lock_path),
        "packages": [{"package": name, "version": version} for name, version in sorted(versions.items())],
        "install_command": [str(python_bin), "-m", "pip", "install", "-U", *install_args],
        "recommended_after_apply": [
            "./scripts/ops/opsctl.sh mlx-audit --json",
            "./scripts/ops/opsctl.sh quant-model-control --json",
        ],
    }


def write_payload(payload: dict[str, Any], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or apply the pinned MLX library upgrade bundle.")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser()
    lock_path = Path(args.lock_file).expanduser()
    out_file = Path(args.out_file).expanduser()
    payload = build_payload(lock_path=lock_path, python_bin=python_bin)
    if args.apply and payload["install_command"]:
        proc = subprocess.run(
            payload["install_command"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        payload["apply_ran"] = True
        payload["install_result"] = {
            "rc": int(proc.returncode),
            "ok": proc.returncode == 0,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        }
        payload["ok"] = bool(proc.returncode == 0)
    write_payload(payload, out_file)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"mlx_library_upgrade packages={len(payload.get('packages') or [])} apply={int(bool(payload.get('apply_ran')))} out={out_file}")
    return 0 if bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
