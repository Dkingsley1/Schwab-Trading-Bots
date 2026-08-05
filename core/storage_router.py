from __future__ import annotations

import json
import os
import filecmp
import shutil
import time
from fnmatch import fnmatchcase
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.runtime_maintenance import maintenance_hold_snapshot
from core.storage_mounts import resolve_external_storage_paths

DEFAULT_EXTERNAL_MOUNT = "/Volumes/BOT_LOGS"
DEFAULT_EXTERNAL_PROJECT = "schwab_trading_bot"
DEFAULT_LOCAL_FALLBACK = "local_fallback_storage"
DEFAULT_AUTO_SYNC_MIN_FREE_GB = 150.0
DEFAULT_LINK_DIRS: tuple[str, ...] = (
    "logs",
    "decision_explanations",
    "decisions",
    "governance",
    "exports",
    "data",
    "models",
)
NESTED_SQLITE_ROUTE_RELS: tuple[str, ...] = (
    "data/jsonl_link.sqlite3",
    "data/jsonl_link.sqlite3-wal",
    "data/jsonl_link.sqlite3-shm",
    "data/bot_channel_queue.sqlite3",
    "data/bot_channel_queue.sqlite3-wal",
    "data/bot_channel_queue.sqlite3-shm",
    "data/snapshot_context.sqlite3",
    "data/snapshot_context.sqlite3-wal",
    "data/snapshot_context.sqlite3-shm",
)
NESTED_LOCAL_ROUTE_ROOTS: tuple[str, ...] = ("governance", "data")
NESTED_LOCAL_ROUTE_SKIP_MARKERS: tuple[str, ...] = (
    ".__external_symlink_backup_",
    ".symlink_disabled_",
    ".route_symlink_disabled_",
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
    autosync_error_details: tuple[str, ...] = ()
    autosync_skip_details: tuple[str, ...] = ()
    autosync_skipped_reason: str = ""
    autosync_free_bytes: int | None = None
    autosync_min_free_bytes: int = 0
    split_brain_conflicts: int = 0
    ops_event_recorded: bool = False
    ops_event_error: str = ""


def _normalized_path_no_io(path: Path | str) -> Path:
    text = os.path.abspath(str(Path(path).expanduser()))
    if text == "/var" or text.startswith("/var/") or text == "/tmp" or text.startswith("/tmp/"):
        text = f"/private{text}"
    return Path(text)


def _resolve_link_target(link_path: Path) -> Path | None:
    try:
        raw = os.readlink(link_path)
    except OSError:
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = (link_path.parent / target)
    return _normalized_path_no_io(target)


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(path, os.W_OK)


def _external_project_root() -> Path:
    _, external_root = resolve_external_storage_paths()
    return external_root


def _configured_external_project_root_no_io() -> Path:
    configured = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", DEFAULT_EXTERNAL_MOUNT)).expanduser()
    project_dir = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", DEFAULT_EXTERNAL_PROJECT) or DEFAULT_EXTERNAL_PROJECT).strip()
    return mount_root / project_dir


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _external_min_free_bytes() -> int:
    raw_bytes = os.getenv("BOT_LOGS_EXTERNAL_MIN_FREE_BYTES", "").strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0

    raw_gb = os.getenv("BOT_LOGS_EXTERNAL_MIN_FREE_GB", "").strip()
    if raw_gb:
        try:
            return max(int(float(raw_gb) * (1024 ** 3)), 0)
        except Exception:
            return 0

    return 0


def _auto_sync_min_free_bytes() -> int:
    raw_bytes = os.getenv("BOT_LOGS_AUTO_SYNC_MIN_FREE_BYTES", "").strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0

    raw_gb = os.getenv("BOT_LOGS_AUTO_SYNC_MIN_FREE_GB", "").strip()
    if raw_gb:
        try:
            return max(int(float(raw_gb) * (1024 ** 3)), 0)
        except Exception:
            return 0

    return int(DEFAULT_AUTO_SYNC_MIN_FREE_GB * (1024 ** 3))


def _available_free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(path).free)
    except Exception:
        return None


def _external_root_ready(path: Path) -> bool:
    if not _is_writable_directory(path):
        return False

    min_free_bytes = _external_min_free_bytes()
    if min_free_bytes <= 0:
        return True

    free_bytes = _available_free_bytes(path)
    if free_bytes is None:
        return False

    return free_bytes >= min_free_bytes


def _external_root_ready_read_only(path: Path) -> bool:
    try:
        if not path.is_dir() or not os.access(path, os.W_OK):
            return False
    except OSError:
        return False

    min_free_bytes = _external_min_free_bytes()
    if min_free_bytes <= 0:
        return True
    free_bytes = _available_free_bytes(path)
    return bool(free_bytes is not None and free_bytes >= min_free_bytes)


def _auto_sync_pressure_gate(external_root: Path) -> tuple[bool, str, int | None, int]:
    min_free_bytes = _auto_sync_min_free_bytes()
    free_bytes = _available_free_bytes(external_root)
    if min_free_bytes <= 0:
        return True, "", free_bytes, 0
    if free_bytes is None:
        return False, "autosync_skipped_external_free_space_unknown", free_bytes, min_free_bytes
    if free_bytes < min_free_bytes:
        return (
            False,
            f"autosync_skipped_external_low_space:free_bytes={free_bytes}:min_free_bytes={min_free_bytes}",
            free_bytes,
            min_free_bytes,
        )
    return True, "", free_bytes, min_free_bytes


def _failback_skip_patterns() -> tuple[str, ...]:
    raw = os.getenv("BOT_LOGS_FAILBACK_SKIP_PATHS", "").strip()
    patterns = [token.strip().replace("\\", "/").lstrip("./") for token in raw.split(",") if token.strip()]
    if not patterns:
        patterns = [
            "data/jsonl_link.sqlite3",
            "data/jsonl_link.sqlite3-wal",
            "data/jsonl_link.sqlite3-shm",
            "data/bot_channel_queue.sqlite3",
            "data/bot_channel_queue.sqlite3-wal",
            "data/bot_channel_queue.sqlite3-shm",
            "data/snapshot_context.sqlite3",
            "data/snapshot_context.sqlite3-wal",
            "data/snapshot_context.sqlite3-shm",
        ]
    return tuple(patterns)


def _should_skip_failback_rel(rel_path: str) -> bool:
    rel_norm = str(rel_path).replace("\\", "/").lstrip("./")
    if not rel_norm:
        return False
    for pattern in _failback_skip_patterns():
        if fnmatchcase(rel_norm, pattern):
            return True
    return False


def _record_autosync_error(details: list[str], rel_path: str, exc: Exception) -> None:
    if len(details) >= max(int(os.getenv("BOT_LOGS_AUTO_SYNC_ERROR_DETAIL_LIMIT", "8") or 8), 1):
        return
    rel = str(rel_path or "").replace("\\", "/").lstrip("./") or "unknown"
    details.append(f"{rel}:{exc.__class__.__name__}:{exc}")


def _record_autosync_skip(details: list[str], value: str) -> None:
    limit = _bounded_env_int("BOT_LOGS_AUTO_SYNC_SKIP_DETAIL_LIMIT", 100)
    if limit <= 0 or len(details) >= limit:
        return
    details.append(str(value))


def _bounded_env_int(name: str, default: int) -> int:
    try:
        return max(int(float(os.getenv(name, str(default)) or default)), 0)
    except Exception:
        return max(int(default), 0)


def _atomic_copy2(src: Path, dst: Path) -> None:
    before = src.stat()
    tmp = dst.with_name(f"{dst.name}.tmp.autosync.{os.getpid()}")
    try:
        tmp.unlink(missing_ok=True)
        shutil.copy2(src, tmp)
        after = src.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError("source_changed_during_autosync")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _auto_sync_local_to_external(
    local_root: Path,
    external_root: Path,
    link_dirs: Iterable[str],
    *,
    prune_local: bool,
    max_copy_files: int,
    max_file_bytes: int = 0,
    max_total_bytes: int = 0,
    min_file_age_seconds: float = 0.0,
    skip_details: list[str] | None = None,
) -> tuple[int, int, int, list[str]]:
    copied = 0
    errors = 0
    pruned = 0
    error_details: list[str] = []
    skipped = skip_details if skip_details is not None else []
    copied_bytes = 0

    if max_copy_files <= 0:
        return 0, 0, 0, error_details

    try:
        same_root = local_root.resolve(strict=False) == external_root.resolve(strict=False)
    except Exception:
        same_root = False
    if same_root:
        return 0, 0, 0, error_details

    for rel_name in link_dirs:
        name = str(rel_name).strip().strip("/")
        if not name:
            continue

        src_dir = local_root / name
        dst_dir = external_root / name
        if not src_dir.exists() or not src_dir.is_dir():
            continue
        if src_dir.is_symlink():
            _record_autosync_skip(skipped, f"{name}:source_route_is_symlink")
            continue

        for root, _, files in os.walk(src_dir):
            root_path = Path(root)
            rel_dir = root_path.relative_to(src_dir)
            dst_base = dst_dir / rel_dir
            try:
                dst_base.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                errors += len(files)
                _record_autosync_error(error_details, str(dst_base.relative_to(external_root)), exc)
                continue

            for fname in files:
                if copied >= max_copy_files:
                    return copied, errors, pruned, error_details

                src_file = root_path / fname
                dst_file = dst_base / fname
                rel_file = str(src_file.relative_to(local_root))

                if _should_skip_failback_rel(rel_file):
                    continue
                try:
                    src_stat = src_file.stat()
                except FileNotFoundError:
                    continue
                if src_file.is_symlink():
                    _record_autosync_skip(skipped, f"{rel_file}:source_file_is_symlink")
                    continue
                if max_file_bytes > 0 and src_stat.st_size > max_file_bytes:
                    _record_autosync_skip(
                        skipped,
                        f"{rel_file}:file_bytes={src_stat.st_size}>max_file_bytes={max_file_bytes}",
                    )
                    continue
                age_seconds = max(time.time() - src_stat.st_mtime, 0.0)
                if min_file_age_seconds > 0 and age_seconds < min_file_age_seconds:
                    _record_autosync_skip(
                        skipped,
                        f"{rel_file}:file_age_seconds={age_seconds:.3f}<min_file_age_seconds={min_file_age_seconds:.3f}"
                    )
                    continue
                if max_total_bytes > 0 and copied_bytes + src_stat.st_size > max_total_bytes:
                    _record_autosync_skip(
                        skipped,
                        f"{rel_file}:copy_budget_bytes={copied_bytes + src_stat.st_size}>max_total_bytes={max_total_bytes}"
                    )
                    continue

                if dst_file.exists():
                    try:
                        dst_stat = dst_file.stat()
                        same_size = src_stat.st_size == dst_stat.st_size
                        same_content = same_size and filecmp.cmp(src_file, dst_file, shallow=False)

                        if same_content:
                            if prune_local:
                                src_file.unlink()
                                pruned += 1
                            continue

                        if copied >= max_copy_files:
                            return copied, errors, pruned, error_details

                        # Preserve local-only deltas under a suffix instead of dropping them.
                        conflict = dst_file.with_name(f"{dst_file.name}.local_fallback")
                        seq = 1
                        while conflict.exists():
                            conflict = dst_file.with_name(f"{dst_file.name}.local_fallback.{seq}")
                            seq += 1

                        _atomic_copy2(src_file, conflict)
                        copied += 1
                        copied_bytes += src_stat.st_size
                        if prune_local:
                            src_file.unlink()
                            pruned += 1
                    except FileNotFoundError:
                        # The local fallback file may disappear between os.walk() and copy/stat.
                        # Treat that as already reconciled rather than a hard autosync error.
                        continue
                    except Exception as exc:
                        errors += 1
                        _record_autosync_error(error_details, rel_file, exc)
                    continue

                try:
                    _atomic_copy2(src_file, dst_file)
                    copied += 1
                    copied_bytes += src_stat.st_size
                    if prune_local:
                        try:
                            src_file.unlink()
                            pruned += 1
                        except Exception:
                            pass
                except FileNotFoundError:
                    # A concurrently removed local fallback file does not need failback action.
                    continue
                except Exception as exc:
                    errors += 1
                    _record_autosync_error(error_details, rel_file, exc)

    return copied, errors, pruned, error_details

def _scan_tree_signature(base: Path, link_dirs: Iterable[str], *, max_files: int) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    if max_files <= 0:
        return out

    for rel_name in link_dirs:
        name = str(rel_name).strip().strip("/")
        if not name:
            continue
        root_dir = base / name
        if not root_dir.exists() or not root_dir.is_dir():
            continue
        for root, _, files in os.walk(root_dir):
            root_path = Path(root)
            for fname in files:
                if len(out) >= max_files:
                    return out
                if ".local_fallback" in fname:
                    continue
                fp = root_path / fname
                try:
                    st = fp.stat()
                except Exception:
                    continue
                rel = str(fp.relative_to(base))
                if _should_skip_failback_rel(rel):
                    continue
                out[rel] = (int(st.st_size), int(st.st_mtime))
    return out


def _split_brain_conflicts(local_root: Path, external_root: Path, link_dirs: Iterable[str], *, max_files: int) -> int:
    local_sig = _scan_tree_signature(local_root, link_dirs, max_files=max_files)
    if not local_sig:
        return 0
    external_sig = _scan_tree_signature(external_root, link_dirs, max_files=max_files)
    if not external_sig:
        return 0

    conflicts = 0
    for rel, local_meta in local_sig.items():
        ext_meta = external_sig.get(rel)
        if not ext_meta:
            continue
        if local_meta != ext_meta:
            conflicts += 1
    return int(conflicts)


def _record_route_event_safe(project_root: Path, result: StorageRoutingResult) -> tuple[bool, str]:
    try:
        from scripts import ops_data_plane
    except Exception:
        return False, "ops_data_plane_import_failed"
    try:
        with ops_data_plane.connect(project_root) as conn:
            ops_data_plane.record_storage_route_event(
                conn,
                project_root=project_root,
                mode=result.mode,
                active_root=result.active_root,
                switched_links=list(result.switched_links),
                passthrough_paths=list(result.passthrough_paths),
                autosync_copied_files=result.autosync_copied_files,
                autosync_copy_errors=result.autosync_copy_errors,
                autosync_pruned_files=result.autosync_pruned_files,
                split_brain_conflicts=result.split_brain_conflicts,
                metadata={
                    "autosync_error_details": list(result.autosync_error_details),
                    "autosync_skip_details": list(result.autosync_skip_details),
                    "autosync_skipped_reason": result.autosync_skipped_reason,
                    "autosync_free_bytes": result.autosync_free_bytes,
                    "autosync_min_free_bytes": result.autosync_min_free_bytes,
                },
            )
    except Exception:
        return False, "storage_route_event_record_failed"
    return True, ""


def _links_route_to_root(project_root: Path, link_dirs: Iterable[str], target_root: Path) -> bool:
    desired_root = target_root.resolve(strict=False)
    saw_symlink = False

    for rel_name in link_dirs:
        name = str(rel_name).strip().strip('/')
        if not name:
            continue
        path_in_repo = project_root / name
        if not path_in_repo.is_symlink():
            return False
        current_target = _resolve_link_target(path_in_repo)
        if current_target != (desired_root / name):
            return False
        saw_symlink = True

    return saw_symlink


def _reconcile_nested_sqlite_routes(
    project_root: Path,
    active_root: Path,
) -> tuple[list[str], list[str]]:
    switched: list[str] = []
    skipped: list[str] = []
    for rel in NESTED_SQLITE_ROUTE_RELS:
        repo_path = project_root / rel
        target = active_root / rel
        primary_target = active_root / rel.removesuffix("-wal").removesuffix("-shm")
        target_available = bool(target.exists() or primary_target.exists())
        if not target_available:
            skipped.append(f"nested_sqlite_skipped:{rel}")
            continue
        if not repo_path.is_symlink() and repo_path.exists():
            skipped.append(f"nested_sqlite_passthrough:{rel}")
            continue
        current_target = _resolve_link_target(repo_path) if repo_path.is_symlink() else None
        desired_target = target.resolve(strict=False)
        if current_target == desired_target:
            continue
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.is_symlink() or repo_path.exists():
            repo_path.unlink()
        repo_path.symlink_to(target)
        switched.append(rel)
    return switched, skipped


def _reconcile_nested_local_routes(project_root: Path, local_root: Path) -> tuple[list[str], list[str]]:
    switched: list[str] = []
    skipped: list[str] = []
    for root_name in NESTED_LOCAL_ROUTE_ROOTS:
        scan_root = project_root / root_name
        if not scan_root.is_dir() or scan_root.is_symlink():
            continue
        for current_root, dir_names, file_names in os.walk(scan_root, followlinks=False):
            current = Path(current_root)
            symlink_dirs = [name for name in dir_names if (current / name).is_symlink()]
            dir_names[:] = [name for name in dir_names if name not in symlink_dirs]
            for name in [*symlink_dirs, *file_names]:
                repo_path = current / name
                if not repo_path.is_symlink():
                    continue
                rel = repo_path.relative_to(project_root)
                rel_text = rel.as_posix()
                if any(marker in rel_text for marker in NESTED_LOCAL_ROUTE_SKIP_MARKERS):
                    skipped.append(f"nested_local_route_backup_skipped:{rel_text}")
                    continue
                target = local_root / rel
                if name in symlink_dirs:
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists():
                    skipped.append(f"nested_local_route_target_missing:{rel_text}")
                    continue
                if _resolve_link_target(repo_path) == target.resolve(strict=False):
                    continue
                repo_path.unlink()
                repo_path.symlink_to(target)
                switched.append(rel_text)
    return switched, skipped


def inspect_runtime_storage(
    project_root: str | Path,
    link_dirs: Iterable[str] = DEFAULT_LINK_DIRS,
) -> StorageRoutingResult:
    """Resolve the effective route without copying data, changing links, or recording events."""
    root = Path(project_root).resolve()
    local_root = Path(
        os.getenv(
            "BOT_LOGS_LOCAL_FALLBACK_ROOT",
            str(root / DEFAULT_LOCAL_FALLBACK),
        )
    ).expanduser()
    external_root = _configured_external_project_root_no_io()

    active_mode = str(os.getenv("BOT_LOGS_ACTIVE_MODE", "") or "").strip().lower()
    active_root_text = str(os.getenv("BOT_LOGS_ACTIVE_ROOT", "") or "").strip()
    if active_root_text and active_mode in {"external", "local_fallback", "local_fallback_split_brain"}:
        active_root = Path(active_root_text).expanduser()
        if active_root.exists():
            return StorageRoutingResult(
                mode=active_mode,
                active_root=active_root,
                switched_links=(),
                passthrough_paths=(),
                autosync_skipped_reason="inspection_only_no_route_mutation",
            )

    local_link_count = 0
    external_link_count = 0
    local_prefix = str(_normalized_path_no_io(local_root))
    external_prefix = str(_normalized_path_no_io(external_root))
    for rel_name in link_dirs:
        target = _resolve_link_target(root / str(rel_name).strip().strip("/"))
        if target is None:
            continue
        target_text = str(target)
        if target_text == local_prefix or target_text.startswith(f"{local_prefix}/"):
            local_link_count += 1
        elif target_text == external_prefix or target_text.startswith(f"{external_prefix}/"):
            external_link_count += 1

    prefer_external = os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1").strip().lower() not in {"0", "false", "no", "off"}
    external_ready = bool(prefer_external and _external_root_ready_read_only(external_root))
    if local_link_count > external_link_count:
        mode = "local_fallback"
        active_root = local_root
    elif external_link_count > local_link_count and external_ready:
        mode = "external"
        active_root = external_root
    else:
        mode = "external" if external_ready else "local_fallback"
        active_root = external_root if external_ready else local_root
    return StorageRoutingResult(
        mode=mode,
        active_root=active_root,
        switched_links=(),
        passthrough_paths=(),
        autosync_skipped_reason="inspection_only_no_route_mutation",
    )


def route_runtime_storage(
    project_root: str | Path,
    link_dirs: Iterable[str] = DEFAULT_LINK_DIRS,
    *,
    allow_autosync: bool = False,
) -> StorageRoutingResult:
    root = Path(project_root).resolve()
    maintenance_hold = maintenance_hold_snapshot(root)
    if bool(maintenance_hold.get("active", False)) and not _env_flag(
        "BOT_STORAGE_ROUTE_ALLOW_DURING_MAINTENANCE",
        "0",
    ):
        raise RuntimeError("runtime_maintenance_hold_blocks_storage_route_mutation")
    local_root = Path(
        os.getenv(
            "BOT_LOGS_LOCAL_FALLBACK_ROOT",
            str(root / DEFAULT_LOCAL_FALLBACK),
        )
    ).expanduser()

    prefer_external = os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1").strip().lower() not in {"0", "false", "no", "off"}
    external_root = _external_project_root() if prefer_external else _configured_external_project_root_no_io()
    external_ready = prefer_external and _external_root_ready(external_root)
    active_root = external_root if external_ready else local_root
    mode = "external" if external_ready else "local_fallback"

    switched: list[str] = []
    passthrough: list[str] = []
    autosync_copied = 0
    autosync_errors = 0
    autosync_pruned = 0
    autosync_error_details: list[str] = []
    autosync_skip_details: list[str] = []
    autosync_skipped_reason = ""
    autosync_free_bytes: int | None = None
    autosync_min_free_bytes = 0
    split_brain_conflicts = 0

    link_dirs_tuple = tuple(link_dirs)

    autosync_enabled = bool(allow_autosync and _env_flag("BOT_LOGS_AUTO_SYNC_ON_RECONNECT", "1"))
    if mode == "external" and autosync_enabled:
        allowed, skipped_reason, autosync_free_bytes, autosync_min_free_bytes = _auto_sync_pressure_gate(external_root)
        if allowed:
            prune_local = _env_flag("BOT_LOGS_AUTO_SYNC_PRUNE_LOCAL", "1")
            max_copy_files = max(int(os.getenv("BOT_LOGS_AUTO_SYNC_MAX_FILES", "50000") or 50000), 1)
            autosync_copied, autosync_errors, autosync_pruned, autosync_error_details = _auto_sync_local_to_external(
                local_root=local_root,
                external_root=external_root,
                link_dirs=link_dirs_tuple,
                prune_local=prune_local,
                max_copy_files=max_copy_files,
                max_file_bytes=_bounded_env_int("BOT_LOGS_AUTO_SYNC_MAX_FILE_BYTES", 1024 ** 3),
                max_total_bytes=_bounded_env_int("BOT_LOGS_AUTO_SYNC_MAX_TOTAL_BYTES", 4 * (1024 ** 3)),
                min_file_age_seconds=float(
                    _bounded_env_int("BOT_LOGS_AUTO_SYNC_MIN_FILE_AGE_SECONDS", 300)
                ),
                skip_details=autosync_skip_details,
            )
        else:
            autosync_skipped_reason = skipped_reason
            autosync_error_details = [skipped_reason] if skipped_reason else []
    elif mode == "external" and _env_flag("BOT_LOGS_AUTO_SYNC_ON_RECONNECT", "1"):
        autosync_skipped_reason = "autosync_requires_explicit_failback_owner"

    if mode == "external" and _env_flag("BOT_LOGS_BLOCK_SPLIT_BRAIN", "1"):
        scan_max_files = max(int(os.getenv("BOT_LOGS_SPLIT_BRAIN_SCAN_MAX_FILES", "5000") or 5000), 100)
        split_brain_conflicts = _split_brain_conflicts(
            local_root=local_root,
            external_root=external_root,
            link_dirs=link_dirs_tuple,
            max_files=scan_max_files,
        )
        if split_brain_conflicts > 0:
            links_already_external = _links_route_to_root(root, link_dirs_tuple, external_root)
            if not links_already_external:
                mode = "local_fallback_split_brain"
                active_root = local_root

    if not _is_writable_directory(active_root):
        raise RuntimeError(f"active storage root is not writable: {active_root}")

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

    nested_switched, nested_skipped = _reconcile_nested_sqlite_routes(root, active_root)
    switched.extend(nested_switched)
    passthrough.extend(nested_skipped)
    if mode.startswith("local_fallback"):
        local_switched, local_skipped = _reconcile_nested_local_routes(root, local_root)
        switched.extend(local_switched)
        passthrough.extend(local_skipped)

    os.environ["BOT_LOGS_ACTIVE_MODE"] = mode
    os.environ["BOT_LOGS_ACTIVE_ROOT"] = str(active_root)
    os.environ["BOT_LOGS_AUTOSYNC_COPIED_FILES"] = str(autosync_copied)
    os.environ["BOT_LOGS_AUTOSYNC_COPY_ERRORS"] = str(autosync_errors)
    os.environ["BOT_LOGS_AUTOSYNC_PRUNED_FILES"] = str(autosync_pruned)
    os.environ["BOT_LOGS_AUTOSYNC_ERROR_DETAILS"] = json.dumps(autosync_error_details, ensure_ascii=True)
    os.environ["BOT_LOGS_AUTOSYNC_SKIP_DETAILS"] = json.dumps(autosync_skip_details, ensure_ascii=True)
    os.environ["BOT_LOGS_AUTOSYNC_SKIPPED_REASON"] = str(autosync_skipped_reason)
    os.environ["BOT_LOGS_AUTOSYNC_FREE_BYTES"] = "" if autosync_free_bytes is None else str(autosync_free_bytes)
    os.environ["BOT_LOGS_AUTOSYNC_MIN_FREE_BYTES"] = str(autosync_min_free_bytes)
    os.environ["BOT_LOGS_SPLIT_BRAIN_CONFLICTS"] = str(split_brain_conflicts)
    provisional_result = StorageRoutingResult(
        mode=mode,
        active_root=active_root,
        switched_links=tuple(sorted(switched)),
        passthrough_paths=tuple(sorted(passthrough)),
        autosync_copied_files=int(autosync_copied),
        autosync_copy_errors=int(autosync_errors),
        autosync_pruned_files=int(autosync_pruned),
        autosync_error_details=tuple(autosync_error_details),
        autosync_skip_details=tuple(autosync_skip_details),
        autosync_skipped_reason=str(autosync_skipped_reason),
        autosync_free_bytes=autosync_free_bytes,
        autosync_min_free_bytes=int(autosync_min_free_bytes),
        split_brain_conflicts=int(split_brain_conflicts),
    )
    ops_event_recorded, ops_event_error = _record_route_event_safe(root, provisional_result)
    result = StorageRoutingResult(
        mode=provisional_result.mode,
        active_root=provisional_result.active_root,
        switched_links=provisional_result.switched_links,
        passthrough_paths=provisional_result.passthrough_paths,
        autosync_copied_files=provisional_result.autosync_copied_files,
        autosync_copy_errors=provisional_result.autosync_copy_errors,
        autosync_pruned_files=provisional_result.autosync_pruned_files,
        autosync_error_details=provisional_result.autosync_error_details,
        autosync_skip_details=provisional_result.autosync_skip_details,
        autosync_skipped_reason=provisional_result.autosync_skipped_reason,
        autosync_free_bytes=provisional_result.autosync_free_bytes,
        autosync_min_free_bytes=provisional_result.autosync_min_free_bytes,
        split_brain_conflicts=provisional_result.split_brain_conflicts,
        ops_event_recorded=bool(ops_event_recorded),
        ops_event_error=str(ops_event_error or ""),
    )
    return result


def describe_storage_routing(result: StorageRoutingResult) -> str:
    switched = ",".join(result.switched_links) if result.switched_links else "none"
    passthrough = ",".join(result.passthrough_paths) if result.passthrough_paths else "none"
    autosync = (
        f"copied={result.autosync_copied_files} "
        f"errors={result.autosync_copy_errors} "
        f"pruned={result.autosync_pruned_files} "
        f"skipped={result.autosync_skipped_reason or 'none'}"
    )
    return (
        f"[StorageRoute] mode={result.mode} active_root={result.active_root} "
        f"switched={switched} passthrough={passthrough} autosync={autosync} "
        f"split_brain_conflicts={result.split_brain_conflicts} "
        f"ops_event_recorded={str(result.ops_event_recorded).lower()}"
    )
