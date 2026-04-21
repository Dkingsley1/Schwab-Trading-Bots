#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml_backend_contract import detect_installed_backends, resolve_backend_contract

DEFAULT_OVERRIDE = PROJECT_ROOT / "config" / ".env.access_mode_override"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "runtime_access_mode_latest.json"

NATIVE_MODE = "native"
PORTABLE_MODE = "portable"
SUPPORTED_MODES = {NATIVE_MODE, PORTABLE_MODE}
NATIVE_DEFAULT_BACKEND = "native_default"
PORTABLE_DEFAULT_BACKEND = "portable_auto"
SUPPORTED_ML_BACKENDS = {
    NATIVE_DEFAULT_BACKEND,
    PORTABLE_DEFAULT_BACKEND,
    "mlx",
    "onnx",
    "pytorch",
    "tensorflow",
    "jax",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_mode(raw: str | None) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"local", "default", "native"}:
        return NATIVE_MODE
    if mode in {"portable", "export", "accessible"}:
        return PORTABLE_MODE
    return NATIVE_MODE


def _default_ml_backend_for_mode(mode: str) -> str:
    normalized = _normalize_mode(mode)
    return PORTABLE_DEFAULT_BACKEND if normalized == PORTABLE_MODE else NATIVE_DEFAULT_BACKEND


def _normalize_ml_backend(raw: str | None, *, mode: str | None = None) -> str:
    backend = str(raw or "").strip().lower().replace("-", "_")
    normalized_mode = _normalize_mode(mode)
    if backend in {"", "auto"}:
        return _default_ml_backend_for_mode(normalized_mode)
    aliases = {
        "portable": PORTABLE_DEFAULT_BACKEND,
        "portable_default": PORTABLE_DEFAULT_BACKEND,
        "native": NATIVE_DEFAULT_BACKEND,
        "native_default": NATIVE_DEFAULT_BACKEND,
        "torch": "pytorch",
        "tf": "tensorflow",
    }
    backend = aliases.get(backend, backend)
    if backend not in SUPPORTED_ML_BACKENDS:
        return _default_ml_backend_for_mode(normalized_mode)
    return backend


def override_lines_for_mode(mode: str, ml_backend: str | None = None) -> list[str]:
    normalized = _normalize_mode(mode)
    backend = _normalize_ml_backend(ml_backend, mode=normalized)
    if normalized == PORTABLE_MODE:
        return [
            "# Auto-managed by scripts/ops/runtime_access_mode.py",
            "BOT_RUNTIME_ACCESS_MODE=portable",
            "BOT_LOGS_PREFER_EXTERNAL=0",
            "BOT_PORTABLE_RUNTIME=1",
            "BOT_SQL_ACCESS_PORTABLE=1",
            f"BOT_ML_BACKEND={backend}",
            "BOT_ML_RUNTIME_OPTIONAL=1",
            "BOT_MLX_OPTIONAL=1",
            "RUNTIME_TRAIN_USE_SNAPSHOT=1",
            "RUNTIME_TRAIN_PREFER_SQLITE=1",
            "RUNTIME_TRAIN_SNAPSHOT_PREFER_SQLITE=1",
            "MLX_METAL_JIT=0",
        ]
    return [
        "# Auto-managed by scripts/ops/runtime_access_mode.py",
        "BOT_RUNTIME_ACCESS_MODE=native",
    ]


def _write_override(path: Path, mode: str) -> bool:
    normalized = _normalize_mode(mode)
    if normalized == NATIVE_MODE:
        if path.exists():
            path.unlink()
            return True
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(override_lines_for_mode(normalized, _default_ml_backend_for_mode(normalized))) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == payload:
        return False
    path.write_text(payload, encoding="utf-8")
    return True


def _write_override_with_backend(path: Path, mode: str, ml_backend: str | None) -> bool:
    normalized = _normalize_mode(mode)
    if normalized == NATIVE_MODE:
        if path.exists():
            path.unlink()
            return True
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(override_lines_for_mode(normalized, ml_backend)) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == payload:
        return False
    path.write_text(payload, encoding="utf-8")
    return True


def _parse_override_mode(path: Path) -> str:
    if not path.exists():
        return NATIVE_MODE
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "BOT_RUNTIME_ACCESS_MODE":
            return _normalize_mode(value)
    return PORTABLE_MODE


def _parse_override_ml_backend(path: Path) -> str:
    if not path.exists():
        return NATIVE_DEFAULT_BACKEND
    mode = _parse_override_mode(path)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "BOT_ML_BACKEND":
            return _normalize_ml_backend(value, mode=mode)
    return _default_ml_backend_for_mode(mode)


def _effective_mode(path: Path) -> tuple[str, str]:
    if path.exists():
        return _parse_override_mode(path), "override"
    return _normalize_mode(os.environ.get("BOT_RUNTIME_ACCESS_MODE")), "environment"


def _effective_settings(path: Path) -> tuple[str, str, str]:
    if path.exists():
        mode = _parse_override_mode(path)
        backend = _parse_override_ml_backend(path)
        return mode, backend, "override"
    mode = _normalize_mode(os.environ.get("BOT_RUNTIME_ACCESS_MODE"))
    backend = _normalize_ml_backend(os.environ.get("BOT_ML_BACKEND"), mode=mode)
    return mode, backend, "environment"


def build_payload(
    mode: str,
    override_path: Path,
    *,
    ml_backend: str | None = None,
    changed: bool,
    action: str,
    mode_source: str = "override",
) -> dict[str, Any]:
    normalized = _normalize_mode(mode)
    backend = _normalize_ml_backend(ml_backend, mode=normalized)
    portable = normalized == PORTABLE_MODE
    payload = {
        "timestamp_utc": _now_utc(),
        "ok": True,
        "action": action,
        "mode": normalized,
        "ml_backend": backend,
        "mode_source": mode_source,
        "portable_enabled": portable,
        "changed": bool(changed),
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "detected_backends": detect_installed_backends(),
        "backend_contract": resolve_backend_contract(backend, mode=normalized),
        "runtime_flags": {
            "BOT_RUNTIME_ACCESS_MODE": normalized,
            "BOT_LOGS_PREFER_EXTERNAL": "0" if portable else "native_default",
            "BOT_PORTABLE_RUNTIME": "1" if portable else "0",
            "BOT_SQL_ACCESS_PORTABLE": "1" if portable else "0",
            "BOT_ML_BACKEND": backend,
            "BOT_ML_RUNTIME_OPTIONAL": "1" if portable else "0",
            "BOT_MLX_OPTIONAL": "1" if portable else "0",
            "RUNTIME_TRAIN_USE_SNAPSHOT": "1" if portable else "native_default",
            "RUNTIME_TRAIN_PREFER_SQLITE": "1" if portable else "native_default",
            "RUNTIME_TRAIN_SNAPSHOT_PREFER_SQLITE": "1" if portable else "native_default",
            "MLX_METAL_JIT": "0" if portable else "native_default",
        },
        "notes": [
            "portable mode keeps storage project-local and steers runtime training toward the portable snapshot and SQLite paths",
            "portable mode now publishes a generic ML backend contract so future non-MLX runtimes can opt in cleanly",
            "native mode leaves your current machine behavior unchanged unless other overrides are present",
            "portable mode does not rewrite existing MLX-only training codepaths by itself",
            "backend_contract describes which roles are live-trading capable versus observation-only sidecars",
        ],
    }
    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Toggle native vs portable runtime access mode.")
    parser.add_argument("action", choices=("set", "status"))
    parser.add_argument("mode", nargs="?", default=None, help="native|portable when using set")
    parser.add_argument("--ml-backend", default=None, help="portable_auto|native_default|mlx|onnx|pytorch|tensorflow|jax")
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    override_path = Path(args.override_file).expanduser()
    out_file = Path(args.out_file).expanduser()

    if args.action == "status":
        mode, ml_backend, mode_source = _effective_settings(override_path)
        payload = build_payload(
            mode,
            override_path,
            ml_backend=ml_backend,
            changed=False,
            action="status",
            mode_source=mode_source,
        )
    else:
        requested = _normalize_mode(args.mode)
        requested_backend = _normalize_ml_backend(args.ml_backend, mode=requested)
        changed = _write_override_with_backend(override_path, requested, requested_backend)
        mode, ml_backend, mode_source = _effective_settings(override_path)
        payload = build_payload(
            mode,
            override_path,
            ml_backend=ml_backend,
            changed=changed,
            action="set",
            mode_source=mode_source,
        )

    _write_payload(out_file, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_access_mode "
            f"mode={payload['mode']} "
            f"portable_enabled={int(bool(payload['portable_enabled']))} "
            f"changed={int(bool(payload['changed']))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
