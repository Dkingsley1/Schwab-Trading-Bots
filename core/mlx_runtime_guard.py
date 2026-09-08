"""Process-wide MLX import boundary.

MLX can leave native nanobind registrations behind when initialization fails
after the extension starts loading. Retrying that import in the same process
can abort Python, so every core module must share this single attempt.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any


class MLXUnavailableError(RuntimeError):
    """Raised when MLX cannot be used safely for the lifetime of this process."""


class _UnavailableMLXModule(types.ModuleType):
    def __init__(self, name: str, reason: BaseException) -> None:
        super().__init__(name)
        self.__dict__["_unavailable_reason"] = reason

    def __getattr__(self, name: str) -> Any:
        reason = self.__dict__["_unavailable_reason"]
        raise MLXUnavailableError(
            f"{self.__name__}.{name} is unavailable because MLX initialization "
            f"failed earlier in this process: {reason!r}"
        ) from reason


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _quarantine_failed_import(reason: BaseException) -> None:
    package = sys.modules.get("mlx")
    if package is None:
        package = types.ModuleType("mlx")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["mlx"] = package

    replacements = {
        "mlx.core": _UnavailableMLXModule("mlx.core", reason),
        "mlx.nn": _UnavailableMLXModule("mlx.nn", reason),
        "mlx.optimizers": _UnavailableMLXModule("mlx.optimizers", reason),
    }
    replacements["mlx.nn"].Module = object
    for qualified_name, module in replacements.items():
        sys.modules[qualified_name] = module
        setattr(package, qualified_name.rsplit(".", 1)[1], module)


def _load_once() -> tuple[Any | None, Any | None, Any | None, BaseException | None]:
    if _truthy_env("BOT_MLX_DISABLE"):
        error = MLXUnavailableError("MLX disabled for this process by BOT_MLX_DISABLE")
        _quarantine_failed_import(error)
        return None, None, None, error

    try:
        mx = importlib.import_module("mlx.core")
        nn = importlib.import_module("mlx.nn")
        optim = importlib.import_module("mlx.optimizers")
    except Exception as exc:
        _quarantine_failed_import(exc)
        return None, None, None, exc
    return mx, nn, optim, None


_MX, _NN, _OPTIM, _IMPORT_ERROR = _load_once()


def mlx_modules() -> tuple[Any | None, Any | None, Any | None, BaseException | None]:
    return _MX, _NN, _OPTIM, _IMPORT_ERROR


def mlx_available() -> bool:
    return _MX is not None and _NN is not None and _OPTIM is not None


def require_mlx() -> tuple[Any, Any, Any]:
    if not mlx_available():
        error = _IMPORT_ERROR or MLXUnavailableError("MLX runtime unavailable")
        raise MLXUnavailableError(
            f"MLX is unavailable for the lifetime of this process: {error!r}"
        ) from error
    return _MX, _NN, _OPTIM
