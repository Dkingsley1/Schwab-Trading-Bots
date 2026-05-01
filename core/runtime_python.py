from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

AUTO_RUNTIME_LANES = {"auto", "shadow", "shadow_auto", "native", "mlx", "local"}
PY312_RUNTIME_LANES = {"production", "py312", "python312"}
PY314_RUNTIME_LANES = {"shadow314", "py314", "canary314", "python314"}


def runtime_lane() -> str:
    lane = str(
        os.getenv("BOT_RUNTIME_LANE")
        or os.getenv("BOT_PYTHON_RUNTIME")
        or "auto"
    ).strip().lower()
    return lane or "auto"


def runtime_version() -> str | None:
    explicit = str(os.getenv("BOT_PYTHON_VERSION", "")).strip()
    if explicit:
        return explicit
    lane = runtime_lane()
    if lane in PY314_RUNTIME_LANES:
        return "3.14"
    if lane in PY312_RUNTIME_LANES:
        return "3.12"
    return None


def training_lane() -> str:
    lane = str(
        os.getenv("BOT_TRAINING_RUNTIME_LANE")
        or os.getenv("BOT_TRAINING_PYTHON_RUNTIME")
        or "training"
    ).strip().lower()
    return lane or "training"


def training_version() -> str:
    explicit = str(os.getenv("BOT_TRAINING_PYTHON_VERSION", "")).strip()
    if explicit:
        return explicit
    lane = training_lane()
    if lane in PY314_RUNTIME_LANES:
        return "3.14"
    return "3.12"


def _candidate_paths(root: Path, version: str) -> list[Path]:
    candidates: list[Path] = []
    if str(version).startswith("3.14"):
        candidates.extend(
            [
                root / ".venv314" / "bin" / "python",
                root / ".venv313" / "bin" / "python",
                root / ".venv312" / "bin" / "python",
            ]
        )
    else:
        candidates.extend(
            [
                root / ".venv312" / "bin" / "python",
                root / ".venv314" / "bin" / "python",
                root / ".venv313" / "bin" / "python",
            ]
        )
    return candidates


@lru_cache(maxsize=32)
def _python_supports_module(path_text: str, module_name: str) -> bool:
    path = Path(path_text)
    if not path.exists():
        return False
    try:
        proc = subprocess.run(
            [
                str(path),
                "-c",
                "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                module_name,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception:
        return False
    return int(proc.returncode) == 0


def _runtime_prefers_mlx(root: Path) -> bool:
    override = str(os.getenv("BOT_PREFER_MLX_RUNTIME", "")).strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    for path in _candidate_paths(root, "3.12"):
        if _python_supports_module(str(path), "mlx"):
            return True
    return False


def resolve_runtime_python(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()

    explicit = str(os.getenv("BOT_PYTHON_BIN", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (root / path).resolve()

    version = runtime_version()
    if version:
        candidates = _candidate_paths(root, version)
    elif _runtime_prefers_mlx(root):
        candidates = _candidate_paths(root, "3.12")
    else:
        candidates = _candidate_paths(root, "3.14")

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_training_python(project_root: str | Path, *, require_mlx: bool = True) -> Path:
    root = Path(project_root).expanduser().resolve()

    explicit = str(os.getenv("BOT_TRAINING_PYTHON_BIN", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (root / path).resolve()

    candidates = _candidate_paths(root, training_version())
    if require_mlx:
        for path in candidates:
            if path.exists() and _python_supports_module(str(path), "mlx"):
                return path
    for path in candidates:
        if path.exists():
            return path
    return resolve_runtime_python(root)


def resolve_runtime_pip(project_root: str | Path) -> Path:
    py = resolve_runtime_python(project_root)
    return py.parent / "pip"
