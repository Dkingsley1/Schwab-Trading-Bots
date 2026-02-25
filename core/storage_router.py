from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTERNAL_MOUNT = "/Volumes/BOT_LOGS"
DEFAULT_EXTERNAL_PROJECT = "schwab_trading_bot"
DEFAULT_LOCAL_FALLBACK = "local_fallback_storage"
DEFAULT_LINK_DIRS: tuple[str, ...] = (
    "logs",
    "decision_explanations",
    "decisions",
    "governance",
    "exports",
    "data",
    "models",
)


@dataclass(frozen=True)
class StorageRoutingResult:
    mode: str
    active_root: Path
    switched_links: tuple[str, ...]
    passthrough_paths: tuple[str, ...]


def _resolve_link_target(link_path: Path) -> Path | None:
    try:
        raw = os.readlink(link_path)
    except OSError:
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = (link_path.parent / target)
    return target.resolve(strict=False)


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(path, os.W_OK)


def _external_project_root() -> Path:
    configured = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", DEFAULT_EXTERNAL_MOUNT)).expanduser()
    project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", DEFAULT_EXTERNAL_PROJECT).strip() or DEFAULT_EXTERNAL_PROJECT
    return mount_root / project_dir


def route_runtime_storage(project_root: str | Path, link_dirs: Iterable[str] = DEFAULT_LINK_DIRS) -> StorageRoutingResult:
    root = Path(project_root).resolve()
    external_root = _external_project_root()
    local_root = Path(
        os.getenv(
            "BOT_LOGS_LOCAL_FALLBACK_ROOT",
            str(root / DEFAULT_LOCAL_FALLBACK),
        )
    ).expanduser()

    prefer_external = os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1").strip().lower() not in {"0", "false", "no", "off"}
    external_ready = prefer_external and _is_writable_directory(external_root)
    active_root = external_root if external_ready else local_root
    mode = "external" if external_ready else "local_fallback"

    if not _is_writable_directory(active_root):
        raise RuntimeError(f"active storage root is not writable: {active_root}")

    switched: list[str] = []
    passthrough: list[str] = []

    for rel_name in link_dirs:
        name = str(rel_name).strip().strip("/")
        if not name:
            continue
        path_in_repo = root / name
        target = active_root / name
        target.mkdir(parents=True, exist_ok=True)

        if path_in_repo.is_symlink():
            current_target = _resolve_link_target(path_in_repo)
            desired_target = target.resolve(strict=False)
            if current_target != desired_target:
                path_in_repo.unlink()
                path_in_repo.symlink_to(target)
                switched.append(name)
            continue

        if path_in_repo.exists():
            passthrough.append(name)
            continue

        path_in_repo.symlink_to(target)
        switched.append(name)

    os.environ["BOT_LOGS_ACTIVE_MODE"] = mode
    os.environ["BOT_LOGS_ACTIVE_ROOT"] = str(active_root)
    return StorageRoutingResult(
        mode=mode,
        active_root=active_root,
        switched_links=tuple(sorted(switched)),
        passthrough_paths=tuple(sorted(passthrough)),
    )


def describe_storage_routing(result: StorageRoutingResult) -> str:
    switched = ",".join(result.switched_links) if result.switched_links else "none"
    passthrough = ",".join(result.passthrough_paths) if result.passthrough_paths else "none"
    return (
        f"[StorageRoute] mode={result.mode} active_root={result.active_root} "
        f"switched={switched} passthrough={passthrough}"
    )
