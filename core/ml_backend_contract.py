from __future__ import annotations

import importlib.util
import platform
from typing import Any, Dict


NATIVE_MODE = "native"
PORTABLE_MODE = "portable"
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

_PACKAGE_CHECKS = {
    "mlx": ("mlx.core", "mlx"),
    "onnx": ("onnxruntime",),
    "pytorch": ("torch",),
    "tensorflow": ("tensorflow",),
    "jax": ("jax",),
}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False
    except Exception:
        return False


def _normalize_mode(raw: str | None) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"portable", "export", "accessible"}:
        return PORTABLE_MODE
    return NATIVE_MODE


def _default_backend_for_mode(mode: str) -> str:
    return PORTABLE_DEFAULT_BACKEND if _normalize_mode(mode) == PORTABLE_MODE else NATIVE_DEFAULT_BACKEND


def normalize_backend(raw: str | None, *, mode: str | None = None) -> str:
    backend = str(raw or "").strip().lower().replace("-", "_")
    normalized_mode = _normalize_mode(mode)
    if backend in {"", "auto"}:
        return _default_backend_for_mode(normalized_mode)
    aliases = {
        "portable": PORTABLE_DEFAULT_BACKEND,
        "portable_default": PORTABLE_DEFAULT_BACKEND,
        "native": NATIVE_DEFAULT_BACKEND,
        "torch": "pytorch",
        "tf": "tensorflow",
    }
    backend = aliases.get(backend, backend)
    if backend not in SUPPORTED_ML_BACKENDS:
        return _default_backend_for_mode(normalized_mode)
    return backend


def detect_installed_backends() -> Dict[str, bool]:
    detected: Dict[str, bool] = {}
    for backend, module_names in _PACKAGE_CHECKS.items():
        detected[backend] = any(_module_available(module_name) for module_name in module_names)
    return detected


def _portable_auto_backend(installed: Dict[str, bool]) -> str:
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    preferred = ["pytorch", "onnx", "jax", "tensorflow"] if is_apple_silicon else ["onnx", "pytorch", "tensorflow", "jax"]
    for backend in preferred:
        if bool(installed.get(backend)):
            return backend
    return "unavailable"


def resolve_backend_contract(
    raw_backend: str | None,
    *,
    mode: str | None = None,
    installed: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    installed_map = dict(installed or detect_installed_backends())
    requested_backend = normalize_backend(raw_backend, mode=mode)
    normalized_mode = _normalize_mode(mode)

    if requested_backend == NATIVE_DEFAULT_BACKEND:
        effective_backend = "mlx"
    elif requested_backend == PORTABLE_DEFAULT_BACKEND:
        effective_backend = _portable_auto_backend(installed_map)
    else:
        effective_backend = requested_backend

    package_available = bool(installed_map.get(effective_backend, False)) if effective_backend in installed_map else False
    live_trading_supported = effective_backend == "mlx" and package_available
    runtime_training_supported = live_trading_supported
    shadow_replay_supported = effective_backend in {"mlx", "onnx", "pytorch", "tensorflow", "jax"} and package_available
    sidecar_canary_supported = effective_backend in {"onnx", "pytorch", "tensorflow", "jax"} and package_available
    observation_only = shadow_replay_supported and not live_trading_supported

    roles_supported = []
    if live_trading_supported:
        roles_supported.append("live_trading")
    if runtime_training_supported:
        roles_supported.append("runtime_training")
    if shadow_replay_supported:
        roles_supported.append("shadow_replay")
    if sidecar_canary_supported:
        roles_supported.append("sidecar_canary")

    implementation_state = {
        "mlx": "live_trading_brain",
        "onnx": "portable_inference_sidecar",
        "pytorch": "shadow_replay_sidecar",
        "tensorflow": "shadow_replay_sidecar",
        "jax": "shadow_replay_sidecar",
        "unavailable": "no_supported_runtime_detected",
    }.get(effective_backend, "unknown")

    notes = []
    if requested_backend == NATIVE_DEFAULT_BACKEND:
        notes.append("native_default resolves to MLX for the current live trading brain")
    if requested_backend == PORTABLE_DEFAULT_BACKEND:
        notes.append("portable_auto resolves to the first installed non-MLX backend suitable for shadow or replay roles")
    if observation_only:
        notes.append("this backend is observation-only in the current codebase and does not own the live TradingBrain path")
    if effective_backend == "mlx":
        notes.append("MLX remains the Apple Silicon-optimized live backend")
    elif effective_backend == "unavailable":
        notes.append("no supported optional backend is currently installed for this contract")

    return {
        "mode": normalized_mode,
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "package_available": package_available,
        "live_trading_supported": live_trading_supported,
        "runtime_training_supported": runtime_training_supported,
        "shadow_replay_supported": shadow_replay_supported,
        "sidecar_canary_supported": sidecar_canary_supported,
        "observation_only": observation_only,
        "implementation_state": implementation_state,
        "roles_supported": roles_supported,
        "installed_backends": installed_map,
        "apple_silicon_optimized": effective_backend == "mlx",
        "notes": notes,
    }
