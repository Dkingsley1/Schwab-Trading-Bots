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


def _default_out_root() -> str:
    return os.getenv(
        "SNAPSHOT_DRILL_OUT_ROOT",
        str(PROJECT_ROOT / "exports" / "state_snapshot_drills"),
    )


def _default_publish_latest() -> str:
    return os.getenv(
        "SNAPSHOT_DRILL_PUBLISH_LATEST",
        str(PROJECT_ROOT / "exports" / "state_snapshot_drills" / "latest.json"),
    )


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


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


def _sha256(path: Path, guard=None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if guard is not None:
                guard.check()
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
        reserve_gb = float(
            os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", str(DEFAULT_TARGET_FREE_GB))
        )
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


def _copy_bounded(src: Path, dst: Path, *, max_bytes: int, guard=None) -> None:
    deadline = time.monotonic() + 30.0
    copied = 0
    with src.open("rb") as source, dst.open("xb") as target:
        os.fchmod(target.fileno(), 0o600)
        while chunk := source.read(1024 * 1024):
            if guard is not None:
                guard.check()
            copied += len(chunk)
            if copied > max_bytes:
                raise RuntimeError("file_snapshot_exceeds_copy_budget")
            if guard is None and time.monotonic() >= deadline:
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
    parser.add_argument("--out-root", default=_default_out_root())
    parser.add_argument(
        "--publish-latest",
        default=_default_publish_latest(),
        help="Optional latest.json publication path when the drill scratch root differs from the standard exports route.",
    )
    parser.add_argument(
        "--targets", nargs="*", default=[str(p) for p in DEFAULT_TARGETS]
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--recover-latest-verified", action="store_true",
                        help="Recheck retained archive hashes and republish existing unexpired restore proof, preserving its original timestamp.")
    parser.add_argument(
        "--compact-sqlite",
        action="store_true",
        help="Create bounded VACUUM INTO logical snapshots without copying freelist pages or changing the source.",
    )
    parser.add_argument(
        "--clone-restore",
        action="store_true",
        help="Require a same-volume APFS copy-on-write restore; never fall back to an unreserved physical copy.",
    )
    parser.add_argument("--operation-seconds", type=int, default=900)
    parser.add_argument(
        "--operator-approved-recovery",
        action="store_true",
        help="One explicitly approved bounded attempt through the support pause, with fresh hard resource admission.",
    )
    parser.add_argument(
        "--capacity-only",
        action="store_true",
        help="Print the measured compact-mode capacity plan without copying or publishing readiness.",
    )
    parser.add_argument(
        "--resume-run",
        default="",
        help="Finish one owned archive-cap failure from retained hash-verified copies; requires explicit recovery approval.",
    )
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
    parser.add_argument(
        "--allow-large-metadata-only",
        action="store_true",
        default=_env_truthy("SNAPSHOT_DRILL_ALLOW_LARGE_METADATA_ONLY"),
        help="Accept observed metadata for oversized targets while keeping restore verification counts separate.",
    )
    from scripts.ops.state_snapshot_capacity import policy_arguments

    args = parser.parse_args([*policy_arguments(PROJECT_ROOT), *sys.argv[1:]])
    if (args.clone_restore or args.capacity_only) and not args.compact_sqlite:
        parser.error("--clone-restore and --capacity-only require --compact-sqlite")
    if not 30 <= args.operation_seconds <= 1800:
        parser.error("--operation-seconds must be between 30 and 1800")
    if args.operator_approved_recovery and not args.compact_sqlite:
        parser.error("operator-approved recovery requires compact mode")
    if args.resume_run and (not args.operator_approved_recovery or args.capacity_only):
        parser.error(
            "--resume-run requires --operator-approved-recovery and cannot be a capacity-only check"
        )
    if args.recover_latest_verified and (args.resume_run or args.capacity_only or not args.compact_sqlite):
        parser.error("--recover-latest-verified requires compact mode and cannot be combined with resume or capacity-only")

    out_root = Path(args.out_root).expanduser().absolute()
    publish_latest = Path(args.publish_latest).expanduser().absolute()
    lock_path = PROJECT_ROOT / "governance" / "locks" / "storage_maintenance.lock"
    for output in (
        out_root,
        out_root / "latest.json",
        publish_latest,
        lock_path,
        PROJECT_ROOT / "governance/watchdog/state_snapshot_drill_events.jsonl",
    ):
        if inspect_storage_path(output)["status"] not in {"present", "missing"}:
            parser.error("snapshot output route is protected or unavailable")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if args.capacity_only:
        plans, errors, capacity = _compact_plan(args, out_root)
        print(
            json.dumps(
                {"capacity_preflight": capacity, "targets": plans, "errors": errors}
            )
        )
        return 0 if capacity.get("sufficient") and not errors else 2
    if args.compact_sqlite:
        from scripts.ops.support_maintenance_gate import (
            support_maintenance_freeze_contract,
        )

        freeze = support_maintenance_freeze_contract(
            PROJECT_ROOT, "state_snapshot_drill"
        )
        if args.operator_approved_recovery:
            from scripts.ops.approved_storage_recovery import resources_admitted

            freeze = {
                "active": not resources_admitted(PROJECT_ROOT),
                "reason": "approved_recovery_hard_resource_admission_not_ready",
            }
        if freeze["active"]:
            attempt = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "overall_status": "deferred",
                "reason": freeze["reason"],
                "previous_restore_evidence_unchanged": True,
            }
            _write_json_atomic(
                PROJECT_ROOT
                / "governance/health/state_snapshot_drill_attempt_latest.json",
                attempt,
            )
            print(json.dumps(attempt))
            return 2
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
        if args.resume_run:
            return _resume_archive(args, out_root, publish_latest=publish_latest)
        if args.recover_latest_verified:
            return _recover_latest_verified(args, out_root, publish_latest)
        return _run_drill(args, out_root, publish_latest=publish_latest)


def _recover_latest_verified(args, out_root, publish_latest):
    import re
    from scripts.ops import state_snapshot_capacity as compact

    stage = "archive_root"
    active_path = out_root
    skipped = []
    try:
        root = compact.allowed(out_root, stage=stage)
        entries = []
        for entry in root.iterdir():
            entries.append(entry)
            if len(entries) > 200:
                raise ValueError("restore_manifest_discovery_budget_exceeded")
        required = set(map(str, args.targets))
        candidates = []
        for entry in entries:
            if not re.fullmatch(r"\d{8}_\d{6}_\d{6}", entry.name):
                continue
            stage = "candidate_manifest"
            active_path = entry / "manifest.json"
            try:
                manifest = compact.allowed(active_path, stage=stage, within=root)
            except compact.SnapshotRouteError as exc:
                if exc.diagnostic["route_status"] != "missing":
                    raise
                skipped.append({**exc.diagnostic, "reason": "missing_manifest"})
                continue
            if not manifest.is_relative_to(root) or manifest.stat().st_size > 2 * 1024**2:
                raise ValueError("restore_manifest_route_or_size_invalid")
            try:
                payload = json.loads(manifest.read_text())
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                skipped.append({"path": str(active_path), "stage": stage,
                                "reason": f"incomplete_manifest:{type(exc).__name__}"})
                continue
            if not compact.complete_restore_evidence(payload):
                skipped.append({"path": str(manifest), "stage": stage, "reason": "incomplete_restore_evidence"})
                continue
            try:
                produced = datetime.fromisoformat(payload["timestamp_utc"].replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError, AttributeError):
                skipped.append({"path": str(manifest), "stage": stage, "reason": "invalid_producer_time"})
                continue
            if produced.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - produced).total_seconds() <= 168 * 3600:
                skipped.append({"path": str(manifest), "stage": stage, "reason": "producer_time_outside_window"})
                continue
            rows = payload.get("rows", [])
            if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
                    or len(rows) != payload["files_checked"]
                    or {str(row.get("requested_source", row.get("source"))) for row in rows} != required):
                skipped.append({"path": str(manifest), "stage": stage, "reason": "incomplete_target_scope"})
                continue
            candidates.append((produced, manifest, payload))
        stage = "candidate_selection"
        active_path = root
        if not candidates:
            raise ValueError("no_complete_unexpired_restore_manifest")
        _, manifest, payload = max(candidates, key=lambda item: item[0])
        stage = "archive_admission"
        reserve, _ = compact.reserve_bytes(PROJECT_ROOT, root)
        guard = compact.CopyGuard(PROJECT_ROOT, root, reserve,
                                  min(args.operation_seconds, 180),
                                  operator_approved=bool(getattr(args, "operator_approved_recovery", False)))
        total = 0
        for row in payload["rows"]:
            stage = "selected_archive"
            proof = row.get("archive_proof", {})
            active_path = manifest
            if not isinstance(proof, dict) or not isinstance(proof.get("archive_path"), str) or not proof["archive_path"]:
                raise ValueError("retained_archive_receipt_invalid")
            active_path = Path(proof["archive_path"])
            archive = compact.allowed(active_path, stage=stage, within=manifest.parent)
            if (not archive.is_relative_to(manifest.parent) or str(archive) != row.get("snapshot")
                    or row.get("restore_verified") is not True or row.get("error")
                    or proof.get("verification") != "full_decoded_bytes_match_verified_sqlite_or_file_restore"
                    or not re.fullmatch(r"[a-f0-9]{64}", str(proof.get("archive_sha256", "")))
                    or not re.fullmatch(r"[a-f0-9]{64}", str(proof.get("decoded_sha256", "")))
                    or proof["decoded_sha256"] != row.get("snapshot_sha256")
                    or proof["decoded_sha256"] != row.get("restore_sha256")
                    or archive.stat().st_size != proof.get("archive_bytes")):
                raise ValueError("retained_archive_receipt_invalid")
            total += archive.stat().st_size
            if total > 32 * 1024**3:
                raise ValueError("retained_archive_verification_budget_exceeded")
            guard.check()
            if _sha256(archive, guard) != proof["archive_sha256"]:
                raise ValueError("retained_archive_digest_mismatch")
        guard.check()
        payload = {**payload, "receipt_republished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "receipt_recovery_scope": "retained_archive_hashes_match_original_restore_proof",
                   "original_manifest_file": str(manifest),
                   "skipped_restore_candidates": skipped}
        stage = "receipt_publication"
        for path in (root / "latest.json", publish_latest):
            active_path = path
            _write_json_atomic(path, payload)
        if not all(_latest_write_verified(path, payload["timestamp_utc"])
                   for path in (root / "latest.json", publish_latest)):
            raise OSError("restore_receipt_publication_failed")
        print(json.dumps(payload))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        attempt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": False,
                   "overall_status": "blocked", "reason": f"restore_receipt_recovery_failed:{exc}",
                   "failure_context": getattr(exc, "diagnostic", {
                       "stage": stage, "path": str(active_path),
                       "route_status": "not_checked" if stage == "receipt_publication" else "present",
                       "error_type": type(exc).__name__}),
                   "skipped_restore_candidates": skipped,
                   "previous_restore_evidence_unchanged": stage != "receipt_publication"}
        _write_json_atomic(PROJECT_ROOT / "governance/health/state_snapshot_drill_attempt_latest.json", attempt)
        print(json.dumps(attempt))
        return 2


def _resume_archive(args, out_root, *, publish_latest):
    import gzip
    import re
    from scripts.ops import state_snapshot_capacity as compact

    # Only a single completed copy/restore whose archive cap failed is resumable.
    # Recheck every retained proof before releasing either verified temporary copy.
    try:
        root = compact.allowed(out_root)
        run = compact.allowed(Path(args.resume_run).expanduser().absolute())
        if run.parent != root or not re.fullmatch(r"\d{8}_\d{6}_\d{6}", run.name):
            raise ValueError("resume_requires_owned_run_directory")
        manifest = run / "manifest.json"
        if compact.allowed(manifest) != manifest:
            raise ValueError("resume_manifest_alias_not_allowed")
        raw = manifest.read_bytes()
        payload = json.loads(raw)
        rows = payload.get("rows", [])
        if (
            payload.get("run_dir") != str(run)
            or payload.get("missing_files")
            or payload.get("files_checked") != len(rows)
            or not rows
            or payload.get("full_platform_restore_verified") is not False
        ):
            raise ValueError("invalid_resume_manifest")
        failed = [row for row in rows if row.get("restore_verified") is not True]
        if (
            len(failed) != 1
            or failed[0].get("error") != "compressed_snapshot_archive_budget_exceeded"
        ):
            raise ValueError("resume_requires_one_archive_cap_failure")
        reserve, scope = compact.reserve_bytes(PROJECT_ROOT, root)
        archive_cap = 4 * compact.GIB
        free = shutil.disk_usage(root).free
        if free < reserve + archive_cap:
            raise RuntimeError("insufficient_resume_archive_capacity")
        guard = compact.CopyGuard(
            PROJECT_ROOT, root, reserve, args.operation_seconds, operator_approved=True
        )
        guard.check()

        def owned(raw_path, subdir):
            path = Path(raw_path)
            resolved = compact.allowed(path)
            if (
                path != resolved
                or not resolved.is_relative_to(run / subdir)
                or not path.is_file()
            ):
                raise ValueError("resume_file_outside_owned_run")
            return path

        for row in rows:
            if row in failed:
                continue
            proof = row.get("archive_proof", {})
            archive = owned(row["snapshot"], "snapshot")
            if (
                str(archive) != proof.get("archive_path")
                or archive.stat().st_size != proof.get("archive_bytes")
                or _sha256(archive, guard) != proof.get("archive_sha256")
            ):
                raise ValueError("resume_retained_archive_hash_mismatch")
            digest, size = hashlib.sha256(), 0
            with gzip.open(archive, "rb") as source:
                for block in iter(lambda: source.read(1024**2), b""):
                    guard.check()
                    size += len(block)
                    if size > int(proof.get("decoded_bytes", 0)):
                        raise ValueError("resume_retained_archive_size_mismatch")
                    digest.update(block)
            if (
                size != proof.get("decoded_bytes")
                or digest.hexdigest() != proof.get("decoded_sha256")
                or digest.hexdigest() != row.get("snapshot_sha256")
                or digest.hexdigest() != row.get("restore_sha256")
            ):
                raise ValueError("resume_retained_archive_restore_mismatch")
        row = failed[0]
        snapshot = owned(row["snapshot"], "snapshot")
        restored = owned(row["restored"], "restore_probe")
        for path in (snapshot, restored):
            for suffix in ("-wal", "-shm", "-journal"):
                if inspect_storage_path(str(path) + suffix)["status"] != "missing":
                    raise ValueError("resume_copy_has_unexpected_sqlite_sidecar")
        if (
            snapshot == restored
            or snapshot.stat().st_ino == restored.stat().st_ino
            or row.get("copy_mode") != "compact_sqlite_logical_snapshot_restore"
            or row.get("sqlite_integrity_verified") is not True
        ):
            raise ValueError("resume_requires_distinct_verified_sqlite_copies")
        a, b = compact.verify_snapshot(snapshot, restored, guard, True)
        if a != row.get("snapshot_sha256") or b != row.get("restore_sha256"):
            raise ValueError("resume_snapshot_identity_changed")
        archive = snapshot.with_name(snapshot.name + f".resume-{time.time_ns()}.gz")
        _write_json_atomic(
            run / f"manifest.before-resume-{time.time_ns()}.json", payload
        )
        proof = compact.seal_compressed_snapshot(
            snapshot, restored, a, guard, archive_cap, archive_path=archive
        )
        row.update(
            snapshot=str(archive),
            snapshot_bytes=proof["archive_bytes"],
            archive_proof=proof,
            restore_verified=True,
            restore_ok=True,
            restore_probe_retained=False,
            error="",
        )
        row["capacity_plan"]["archive_limit_bytes"] = archive_cap
        payload.update(
            ok=True,
            files_restore_verified=len(rows),
            resume_verified_at_utc=datetime.now(timezone.utc).isoformat(),
            resume_from_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            worker_budget=guard.pace.snapshot(),
            resume_capacity={
                "free_bytes": free,
                "archive_limit_bytes": archive_cap,
                "reserve_bytes": reserve,
                "reserve_scope": scope,
            },
            latest_write_verified=False,
            published_latest_write_verified=False,
        )
        # Keep the original snapshot timestamp: resealing is not newer source data.
        for target in (manifest, root / "latest.json", publish_latest):
            _write_json_atomic(target, payload)
        payload["latest_write_verified"] = _latest_write_verified(
            root / "latest.json", payload["timestamp_utc"]
        )
        payload["published_latest_write_verified"] = _latest_write_verified(
            publish_latest, payload["timestamp_utc"]
        )
        payload["ok"] = (
            payload["latest_write_verified"]
            and payload["published_latest_write_verified"]
        )
        for target in (manifest, root / "latest.json", publish_latest):
            _write_json_atomic(target, payload)
        print(json.dumps(payload))
        return 0 if payload["ok"] else 2
    except Exception as exc:
        attempt = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "overall_status": "deferred",
            "reason": str(exc),
            "resume_run": str(args.resume_run),
        }
        _write_json_atomic(
            PROJECT_ROOT / "governance/health/state_snapshot_drill_attempt_latest.json",
            attempt,
        )
        print(json.dumps(attempt))
        return 2


def _compact_plan(args, out_root):
    from scripts.ops import state_snapshot_capacity as compact

    plans, errors = {}, {}
    for raw in args.targets:
        try:
            source = _routed_or_local_fallback_path(
                Path(raw).expanduser().absolute(), project_root=PROJECT_ROOT
            )
            plans[raw] = compact.plan_target(source, args.max_copy_bytes)
        except (ValueError, OSError, sqlite3.Error) as exc:
            errors[raw] = str(exc)
    try:
        capacity = compact.capacity(
            PROJECT_ROOT, out_root, list(plans.values()), args.clone_restore
        )
    except (ValueError, OSError) as exc:
        capacity = {"known": False, "sufficient": False, "error": str(exc)}
    if errors:
        capacity["sufficient"] = False
    return plans, errors, capacity


def _run_drill(
    args: argparse.Namespace, out_root: Path, *, publish_latest: Path
) -> int:
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
    compact_mode = bool(getattr(args, "compact_sqlite", False))
    if compact_mode:
        from scripts.ops import state_snapshot_capacity as compact

        plans, plan_errors, capacity = _compact_plan(args, out_root)
        guard = compact.CopyGuard(
            PROJECT_ROOT,
            out_root,
            capacity.get("reserve_bytes", 64 * 1024**3),
            args.operation_seconds,
            operator_approved=getattr(args, "operator_approved_recovery", False),
        )
        if capacity.get("sufficient") and args.clone_restore:
            try:
                compact.probe_clone(out_root)
            except (OSError, RuntimeError, ValueError) as exc:
                capacity.update(sufficient=False, error=str(exc))
    else:
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
        plan = plans.get(target_raw) if compact_mode else None
        archive_proof = {}

        try:
            if compact_mode:
                if target_raw in plan_errors:
                    raise RuntimeError(plan_errors[target_raw])
                if not capacity["sufficient"]:
                    raise RuntimeError("insufficient_snapshot_capacity")
                guard.check()
                snap_path = snap_dir / rel_name
                restore_path = restore_dir / rel_name
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                restore_path.parent.mkdir(parents=True, exist_ok=True)
                before = src.stat()
                if plan["sqlite"]:
                    copy_mode = "compact_sqlite_logical_snapshot_restore"
                    hash_scope = (
                        "consistent_compact_sqlite_snapshot_not_source_file_bytes"
                    )
                    compact.compact_snapshot(src, snap_path, plan, guard)
                else:
                    src_hash = _sha256(src, guard)
                    _copy_bounded(
                        src,
                        snap_path,
                        max_bytes=plan["output_limit_bytes"],
                        guard=guard,
                    )
                    after = src.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise RuntimeError("source_changed_during_snapshot")
                if args.clone_restore:
                    compact.clone_restore(snap_path, restore_path)
                else:
                    _copy_bounded(
                        snap_path,
                        restore_path,
                        max_bytes=plan["output_limit_bytes"],
                        guard=guard,
                    )
                snap_hash, restore_hash = compact.verify_snapshot(
                    snap_path, restore_path, guard, plan["sqlite"]
                )
                sqlite_verified = bool(plan["sqlite"])
                ok = bool(sqlite_verified or src_hash == snap_hash == restore_hash)
                guard.check()
                if not ok:
                    raise RuntimeError("source_snapshot_hash_mismatch")
                archive_proof = compact.seal_compressed_snapshot(
                    snap_path,
                    restore_path,
                    snap_hash,
                    guard,
                    plan["archive_limit_bytes"],
                )
                snap_path = Path(archive_proof["archive_path"])
            elif size_bytes <= max_copy_bytes:
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
                "capacity_plan": plan,
                "restore_storage_mode": (
                    "apfs_copy_on_write_shared_extents"
                    if compact_mode and args.clone_restore
                    else "physical_copy"
                ),
                "independent_physical_media": False,
                "snapshot_bytes": (
                    snap_path.stat().st_size
                    if snap_path is not None and snap_path.exists()
                    else None
                ),
                "archive_proof": archive_proof,
                "restore_probe_retained": bool(
                    restore_path is not None and restore_path.exists()
                ),
            }
        )

    accepted_large_metadata_only = sum(
        bool(
            args.allow_large_metadata_only
            and r.get("copy_mode") == "metadata_only_large_file"
            and r.get("metadata_observation")
            and r.get("error") == "copy_budget_exceeded_restore_not_verified"
        )
        for r in manifest_rows
    )
    all_ok = (
        bool(manifest_rows)
        and not missing
        and all(
            bool(r.get("restore_ok"))
            or bool(
                args.allow_large_metadata_only
                and r.get("copy_mode") == "metadata_only_large_file"
                and r.get("metadata_observation")
                and r.get("error") == "copy_budget_exceeded_restore_not_verified"
            )
            for r in manifest_rows
        )
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ok": all_ok,
        "evidence_scope": "selected_files_copy_restore_not_full_platform_recovery",
        "full_platform_restore_verified": False,
        "large_file_restore_required": not bool(args.allow_large_metadata_only),
        "accepted_metadata_only_large_files": accepted_large_metadata_only,
        "files_restore_verified": sum(
            bool(r["restore_verified"]) for r in manifest_rows
        ),
        "files_checked": len(manifest_rows),
        "missing_files": missing,
        "capacity_preflight": capacity,
        "rows": manifest_rows,
        "worker_budget": guard.pace.snapshot() if compact_mode else None,
    }

    manifest_path = run_dir / "manifest.json"
    latest = out_root / "latest.json"
    payload["manifest_file"] = str(manifest_path)
    payload["latest_file"] = str(latest)
    payload["published_latest_file"] = str(publish_latest)

    payload["retention"] = {
        "keep_runs": max(int(args.keep_runs), 1),
        "pruned_runs": 0,
    }
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(PROJECT_ROOT / "governance/health/state_snapshot_drill_attempt_latest.json", payload)
    if not (all_ok and payload["files_restore_verified"] == payload["files_checked"]
            and not accepted_large_metadata_only):
        from scripts.ops.state_snapshot_capacity import complete_restore_evidence

        try:
            previous = json.loads(publish_latest.read_text())
        except (OSError, ValueError):
            previous = {}
        if complete_restore_evidence(previous):
            payload["previous_restore_evidence_unchanged"] = True
            _write_json_atomic(manifest_path, payload)
            _write_json_atomic(PROJECT_ROOT / "governance/health/state_snapshot_drill_attempt_latest.json", payload)
            print(json.dumps(payload))
            return 2
    _write_json_atomic(latest, payload)
    if publish_latest != latest:
        _write_json_atomic(publish_latest, payload)
    latest_verified = _latest_write_verified(latest, str(payload["timestamp_utc"]))
    published_latest_verified = _latest_write_verified(
        publish_latest, str(payload["timestamp_utc"])
    )
    payload["latest_write_verified"] = latest_verified
    payload["published_latest_write_verified"] = published_latest_verified
    payload["ok"] = bool(
        payload["ok"] and latest_verified and published_latest_verified
    )
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(latest, payload)
    if publish_latest != latest:
        _write_json_atomic(publish_latest, payload)
    payload["retention"]["pruned_runs"] = _prune_old_runs(
        out_root, args.keep_runs, current_run=run_dir, current_verified=payload["ok"]
    )
    _write_json_atomic(manifest_path, payload)
    _write_json_atomic(latest, payload)
    if publish_latest != latest:
        _write_json_atomic(publish_latest, payload)

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
