from __future__ import annotations

import os
import shutil
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
    autosync_copied_files: int = 0
    autosync_copy_errors: int = 0
    autosync_pruned_files: int = 0


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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _auto_sync_local_to_external(
    local_root: Path,
    external_root: Path,
    link_dirs: Iterable[str],
    *,
    prune_local: bool,
    max_copy_files: int,
) -> tuple[int, int, int]:
    copied = 0
    errors = 0
    pruned = 0

    if max_copy_files <= 0:
        return 0, 0, 0

    try:
        same_root = local_root.resolve(strict=False) == external_root.resolve(strict=False)
    except Exception:
        same_root = False
    if same_root:
        return 0, 0, 0

    for rel_name in link_dirs:
        name = str(rel_name).strip().strip("/")
        if not name:
            continue

        src_dir = local_root / name
        dst_dir = external_root / name
        if not src_dir.exists() or not src_dir.is_dir():
            continue

        for root, _, files in os.walk(src_dir):
            root_path = Path(root)
            rel_dir = root_path.relative_to(src_dir)
            dst_base = dst_dir / rel_dir
            try:
                dst_base.mkdir(parents=True, exist_ok=True)
            except Exception:
                errors += len(files)
                continue

            for fname in files:
                if copied >= max_copy_files:
                    return copied, errors, pruned

                src_file = root_path / fname
                dst_file = dst_base / fname

                if dst_file.exists():
                    continue

                try:
                    shutil.copy2(src_file, dst_file)
                    copied += 1
                    if prune_local:
                        try:
                            src_file.unlink()
                            pruned += 1
                        except Exception:
                            pass
                except Exception:
                    errors += 1

    return copied, errors, pruned


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
    autosync_copied = 0
    autosync_errors = 0
    autosync_pruned = 0

    link_dirs_tuple = tuple(link_dirs)

    if mode == "external" and _env_flag("BOT_LOGS_AUTO_SYNC_ON_RECONNECT", "1"):
        prune_local = _env_flag("BOT_LOGS_AUTO_SYNC_PRUNE_LOCAL", "1")
        max_copy_files = max(int(os.getenv("BOT_LOGS_AUTO_SYNC_MAX_FILES", "50000") or 50000), 1)
        autosync_copied, autosync_errors, autosync_pruned = _auto_sync_local_to_external(
            local_root=local_root,
            external_root=external_root,
            link_dirs=link_dirs_tuple,
            prune_local=prune_local,
            max_copy_files=max_copy_files,
        )

    for rel_name in link_dirs_tuple:
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
    os.environ["BOT_LOGS_AUTOSYNC_COPIED_FILES"] = str(autosync_copied)
    os.environ["BOT_LOGS_AUTOSYNC_COPY_ERRORS"] = str(autosync_errors)
    os.environ["BOT_LOGS_AUTOSYNC_PRUNED_FILES"] = str(autosync_pruned)
    return StorageRoutingResult(
        mode=mode,
        active_root=active_root,
        switched_links=tuple(sorted(switched)),
        passthrough_paths=tuple(sorted(passthrough)),
        autosync_copied_files=int(autosync_copied),
        autosync_copy_errors=int(autosync_errors),
        autosync_pruned_files=int(autosync_pruned),
    )


def describe_storage_routing(result: StorageRoutingResult) -> str:
    switched = ",".join(result.switched_links) if result.switched_links else "none"
    passthrough = ",".join(result.passthrough_paths) if result.passthrough_paths else "none"
    autosync = (
        f"copied={result.autosync_copied_files} "
        f"errors={result.autosync_copy_errors} "
        f"pruned={result.autosync_pruned_files}"
    )
    return (
        f"[StorageRoute] mode={result.mode} active_root={result.active_root} "
        f"switched={switched} passthrough={passthrough} autosync={autosync}"
    )
