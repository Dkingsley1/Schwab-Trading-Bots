from __future__ import annotations

import os
from pathlib import Path


def runtime_lane() -> str:
    lane = str(
        os.getenv("BOT_RUNTIME_LANE")
        or os.getenv("BOT_PYTHON_RUNTIME")
        or "production"
    ).strip().lower()
    return lane or "production"


def runtime_version() -> str:
    explicit = str(os.getenv("BOT_PYTHON_VERSION", "")).strip()
    if explicit:
        return explicit
    lane = runtime_lane()
    if lane in {"shadow314", "py314", "canary314", "python314"}:
        return "3.14"
    return "3.12"


def resolve_runtime_python(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()

    explicit = str(os.getenv("BOT_PYTHON_BIN", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (root / path).resolve()

    version = runtime_version()
    candidates = []
    if version.startswith("3.14"):
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
            ]
        )

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_runtime_pip(project_root: str | Path) -> Path:
    py = resolve_runtime_python(project_root)
    return py.parent / "pip"
