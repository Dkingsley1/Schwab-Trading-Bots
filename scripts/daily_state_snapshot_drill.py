import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path
from core.local_storage_reserve import DEFAULT_TARGET_FREE_GB
from scripts.ops.long_runtime_common import write_payload

DEFAULT_TARGETS = [
    PROJECT_ROOT / "master_bot_registry.json",
    PROJECT_ROOT / "data" / "jsonl_link.sqlite3",
    PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json",
    PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json",
    PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json",
]


def _local_fallback_equivalent(path: Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    local_fallback_root = project_root / "local_fallback_storage"
    try:
        rel = candidate.relative_to(project_root)
    except ValueError:
        return candidate
    if rel.parts and rel.parts[0] == local_fallback_root.name:
        return candidate
    return local_fallback_root / rel


def _routed_or_local_fallback_path(path: Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    observation = inspect_storage_path(candidate)
    if observation["status"] == "missing" and observation["symlinks"]:
        return _local_fallback_equivalent(candidate, project_root=project_root)
    return candidate


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prune_old_runs(
    out_root: Path, keep_runs: int, *, current_run: Path, current_verified: bool
) -> int:
    if not current_verified:
        return 0
    keep_n = max(int(keep_runs), 1)
    dirs = []
    for path in out_root.iterdir():
        if path == current_run or not re.fullmatch(
            r"\d{8}_\d{6}(?:_\d{6})?", path.name
        ):
            continue
        if path.is_symlink() or not path.is_dir():
            continue
        manifest = path / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("run_dir") == str(path)
            and payload.get("ok") is True
        ):
            dirs.append(path)
    dirs.sort(key=lambda p: p.name, reverse=True)
    removed = 0
    for p in dirs[keep_n - 1 :]:
        shutil.rmtree(p)
        removed += 1
    return removed


def _capacity_preflight(out_root: Path, copy_count: int, max_copy_bytes: int) -> dict:
    observation = inspect_storage_path(out_root)
    try:
        reserve_gb = float(os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", str(DEFAULT_TARGET_FREE_GB)))
        if not math.isfinite(reserve_gb) or reserve_gb < 0:
            raise ValueError("invalid_storage_reserve")
    except ValueError:
        return {"known": False, "sufficient": False, "error": "invalid_storage_reserve"}
    reserve = int(max(reserve_gb, DEFAULT_TARGET_FREE_GB) * 2**30)
    required = 2 * copy_count * max_copy_bytes + reserve
    payload = {
        "known": False,
        "sufficient": False,
        "copy_count": copy_count,
        "required_free_bytes": required,
        "reserve_bytes": reserve,
        "allocation_scope": "two_capped_copies_per_target_plus_configured_reserve_at_least_64_gib",
        "reservation_scope": "cooperating_storage_maintenance_lock_owners_only",
    }
    if observation["status"] not in {"present", "missing"}:
        return {**payload, "error": "output_route_unavailable"}
    probe = Path(str(observation["resolved_path"]))
    try:
        while not probe.exists():
            probe = probe.parent
        free = int(shutil.disk_usage(probe).free)
        return {
            **payload,
            "known": True,
            "free_bytes": free,
            "sufficient": free >= required,
        }
    except (OSError, ValueError, OverflowError) as exc:
        return {**payload, "error": type(exc).__name__}


def _copy_bounded(src: Path, dst: Path, *, max_bytes: int) -> None:
    deadline = time.monotonic() + 30.0
    copied = 0
    with src.open("rb") as source, dst.open("xb") as target:
        os.fchmod(target.fileno(), 0o600)
        while chunk := source.read(1024 * 1024):
            copied += len(chunk)
            if copied > max_bytes:
                raise RuntimeError("file_snapshot_exceeds_copy_budget")
            if time.monotonic() >= deadline:
                raise TimeoutError("file_snapshot_deadline_exceeded")
            target.write(chunk)


def _backup_sqlite(src: Path, dst: Path, *, max_bytes: int) -> None:
    deadline = time.monotonic() + 30.0
    with closing(
        sqlite3.connect(src.as_uri() + "?mode=ro", uri=True, timeout=5.0)
    ) as source:
        page_size = int(source.execute("PRAGMA page_size").fetchone()[0])

        def progress(status: int, remaining: int, total: int) -> None:
            if total * page_size > max_bytes:
                raise RuntimeError("sqlite_snapshot_exceeds_copy_budget")
            if time.monotonic() > deadline:
                raise TimeoutError("sqlite_snapshot_deadline_exceeded")

        if (
            int(source.execute("PRAGMA page_count").fetchone()[0]) * page_size
            > max_bytes
        ):
            raise RuntimeError("sqlite_snapshot_exceeds_copy_budget")
        with closing(sqlite3.connect(dst)) as snapshot:
            source.backup(snapshot, pages=256, progress=progress, sleep=0.05)


def _verify_sqlite(path: Path) -> bool:
    deadline = time.monotonic() + 30.0
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        return conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]


def _write_json_atomic(path: Path, payload: dict) -> None:
    if inspect_storage_path(path)["status"] not in {"present", "missing"}:
        raise ValueError("snapshot_publication_route_unavailable")
    write_payload(path, payload)


def _latest_write_verified(latest: Path, expected_timestamp: str) -> bool:
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("timestamp_utc") or "") == str(expected_timestamp)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily state snapshot + restore drill."
    )
    parser.add_argument(
        "--out-root", default=str(PROJECT_ROOT / "exports" / "state_snapshot_drills")
    )
    parser.add_argument(
        "--targets", nargs="*", default=[str(p) for p in DEFAULT_TARGETS]
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--keep-runs", type=int, default=int(os.getenv("SNAPSHOT_DRILLS_KEEP", "5"))
    )
    parser.add_argument(
        "--max-copy-bytes",
        type=int,
        default=int(
            os.getenv("SNAPSHOT_DRILL_MAX_COPY_BYTES", str(2 * 1024 * 1024 * 1024))
        ),
        help="Files larger than this receive metadata observations only, not verified recovery credit.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().absolute()
    lock_path = PROJECT_ROOT / "governance" / "locks" / "storage_maintenance.lock"
    for output in (
        out_root, out_root / "latest.json", lock_path,
        PROJECT_ROOT / "governance/watchdog/state_snapshot_drill_events.jsonl",
    ):
        if inspect_storage_path(output)["status"] not in {"present", "missing"}:
            parser.error("snapshot output route is protected or unavailable")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lane:
        try:
            fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "overall_status": "deferred",
                        "reason": "storage_maintenance_lock_busy",
                    }
                )
            )
            return 2
        return _run_drill(args, out_root)


def _run_drill(args: argparse.Namespace, out_root: Path) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, mode=0o700)
    snap_dir = run_dir / "snapshot"
    restore_dir = run_dir / "restore_probe"
    snap_dir.mkdir(parents=True, exist_ok=True)
    restore_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    missing: list[str] = []
    max_copy_bytes = max(int(args.max_copy_bytes), 0)
    copy_count = 0
    for raw in args.targets:
        source = _routed_or_local_fallback_path(
            Path(raw).expanduser().absolute(), project_root=PROJECT_ROOT
        )
        route = inspect_storage_path(source)
        size = route.get("size_bytes")
        if route["status"] == "present" and size is not None and size <= max_copy_bytes:
            copy_count += 1
    capacity = _capacity_preflight(out_root, copy_count, max_copy_bytes)

    for target_raw in args.targets:
        requested_src = Path(target_raw).expanduser().absolute()
        src = _routed_or_local_fallback_path(requested_src, project_root=PROJECT_ROOT)
        observation = inspect_storage_path(src)
        if observation["status"] != "present" or observation.get("size_bytes") is None:
            missing.append(str(requested_src))
            continue

        src = Path(str(observation["resolved_path"]))
        try:
            rel_name = src.relative_to(PROJECT_ROOT)
        except ValueError:
            namespace = hashlib.sha256(str(src.parent).encode()).hexdigest()[:16]
            rel_name = Path("external") / namespace / src.name
        size_bytes = src.stat().st_size
        copy_mode = "full_copy_restore"
        snap_path: Path | None = None
        restore_path: Path | None = None
        src_hash = ""
        snap_hash = ""
        restore_hash = ""
        error = ""
        metadata = {}
        hash_scope = "source_file"
        sqlite_verified = False

        try:
            if size_bytes <= max_copy_bytes:
                if not capacity["sufficient"]:
                    raise RuntimeError("insufficient_snapshot_capacity")
                remaining_capacity = _capacity_preflight(out_root, 1, max_copy_bytes)
                if not remaining_capacity["sufficient"]:
                    raise RuntimeError("snapshot_capacity_changed")
                snap_path = snap_dir / rel_name
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                before = src.stat()
                with src.open("rb") as handle:
                    is_sqlite = handle.read(16) == b"SQLite format 3\x00"
                is_sqlite = is_sqlite or src.suffix.lower() in {
                    ".sqlite",
                    ".sqlite3",
                    ".db",
                }
                if is_sqlite:
                    copy_mode = "online_sqlite_backup_restore"
                    hash_scope = "consistent_sqlite_snapshot"
                    _backup_sqlite(src, snap_path, max_bytes=max_copy_bytes)
                else:
                    src_hash = _sha256(src)
                    _copy_bounded(src, snap_path, max_bytes=max_copy_bytes)
                    after = src.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise RuntimeError("source_changed_during_snapshot")

                restore_path = restore_dir / rel_name
                restore_path.parent.mkdir(parents=True, exist_ok=True)
                _copy_bounded(snap_path, restore_path, max_bytes=max_copy_bytes)

                snap_hash = _sha256(snap_path)
                restore_hash = _sha256(restore_path)
                if is_sqlite:
                    sqlite_verified = _verify_sqlite(restore_path)
                    ok = snap_hash == restore_hash and sqlite_verified
                else:
                    ok = src_hash == snap_hash == restore_hash
            else:
                copy_mode = "metadata_only_large_file"
                stat = src.stat()
                metadata = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                hash_scope = "not_measured"
                error = "copy_budget_exceeded_restore_not_verified"
                ok = False
        except Exception as exc:
            ok = False
            error = str(exc)

        manifest_rows.append(
            {
                "source": str(src),
                "requested_source": str(requested_src),
                "effective_source": str(src),
                "snapshot": str(snap_path) if snap_path is not None else "",
                "restored": str(restore_path) if restore_path is not None else "",
                "size_bytes": size_bytes,
                "sha256": src_hash,
                "snapshot_sha256": snap_hash,
                "restore_sha256": restore_hash,
                "hash_scope": hash_scope,
                "metadata_observation": metadata,
                "sqlite_integrity_verified": sqlite_verified,
                "copy_mode": copy_mode,
                "max_copy_bytes": max_copy_bytes,
                "restore_ok": ok,
                "restore_verified": ok,
                "error": error,
            }
        )

    all_ok = (
        bool(manifest_rows)
        and not missing
        and all(bool(r.get("restore_ok")) for r in manifest_rows)
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ok": all_ok,
        "evidence_scope": "selected_files_copy_restore_not_full_platform_recovery",
        "full_platform_restore_verified": False,
        "files_restore_verified": sum(
            bool(r["restore_verified"]) for r in manifest_rows
        ),
        "files_checked": len(manifest_rows),
        "missing_files": missing,
        "capacity_preflight": capacity,
        "rows": manifest_rows,
    }

    manifest_path = run_dir / "manifest.json"
    latest = out_root / "latest.json"
    payload["manifest_file"] = str(manifest_path)
    payload["latest_file"] = str(latest)

    payload["retention"] = {
        "keep_runs": max(int(args.keep_runs), 1),
        "pruned_runs": 0,
    }
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(latest, payload)
    payload["latest_write_verified"] = _latest_write_verified(
        latest, str(payload["timestamp_utc"])
    )
    payload["ok"] = bool(payload["ok"] and payload["latest_write_verified"])
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(latest, payload)
    payload["retention"]["pruned_runs"] = _prune_old_runs(
        out_root, args.keep_runs, current_run=run_dir, current_verified=payload["ok"]
    )
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(latest, payload)

    events = (
        PROJECT_ROOT / "governance" / "watchdog" / "state_snapshot_drill_events.jsonl"
    )
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"state_snapshot_drill_ok={all_ok} files_checked={len(manifest_rows)} missing={len(missing)}"
        )

    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
