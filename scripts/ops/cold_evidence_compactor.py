"""Bounded, lossless compression of inactive local quarantine logs."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import zlib
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.background_work_budget import WorkBudget, background_policy
from core.workload_admission import POLICIES, current_lease
from core.runtime_maintenance import maintenance_hold_snapshot
from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
from scripts.ops.soak_self_healing_control import _storage_memory_observation

GIB = 1024**3
CHUNK = 1024**2
ROOT_REL = Path("local_fallback_storage/quarantine/bot_logs_cleanup/cold_archive")
NAME = re.compile(r".+\.(?:jsonl|log)(?:\.local_fallback(?:\.\d+)?)?$")
MIN_AGE_SECONDS = 7 * 86400
RECOVERY_CPU_POLICY = {
    "scope": "single_paced_compression_worker_only",
    "minimum_logical_cpus": 8,
    "local_free_gib_below": 125,
    "foreground_cpu_percent_below": 150,
    "system_cpu_percent_below": 200,
    "combined_cpu_percent_below": 300,
    "maximum_host_saturation_score": 70,
    "maximum_load_per_logical_cpu": 0.85,
    "source_max_age_seconds": 90,
    "cpu_fraction_of_one_core": 0.25,
}


class Deferred(RuntimeError):
    """An observed resource or ownership hold, not failed data verification."""


def allowed(path, *, missing=False):
    route = inspect_storage_path(path)
    if route.get("status") not in ({"present", "missing"} if missing else {"present"}):
        raise RuntimeError("protected_or_unavailable_route")
    if route.get("symlinks"):
        raise RuntimeError("symlink_route_not_eligible")
    return Path(path)


def identity(path):
    allowed(path)
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise RuntimeError("source_not_regular")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def database_header(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        header = stream.read(16)
    return header == b"SQLite format 3\x00" or header[:4] in (
        b"\x37\x7f\x06\x82",
        b"\x37\x7f\x06\x83",
    )


def sync_dir(path):
    fd = os.open(allowed(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def idle(path):
    allowed(path)
    p = subprocess.run(
        ["/usr/sbin/lsof", "-t", "--", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 1 or p.stdout.strip() or p.stderr.strip():
        raise Deferred("source_open_or_handle_probe_failed")


def fresh(payload, seconds=90):
    try:
        stamp = datetime.fromisoformat(
            str(payload["timestamp_utc"]).replace("Z", "+00:00")
        )
        return (
            stamp.tzinfo is not None
            and 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= seconds
        )
    except (KeyError, TypeError, ValueError):
        return False


def as_dict(value):
    return value if isinstance(value, dict) else {}


def _disk_recovery_cpu_relief(root, memory, runtime, cpu_count):
    if cpu_count < RECOVERY_CPU_POLICY["minimum_logical_cpus"]:
        return False
    if (
        runtime.get("input_evidence_ready") is not True
        or as_dict(runtime.get("runtime_measurement_evidence")).get("ready") is not True
        or not fresh(
            {"timestamp_utc": runtime.get("source_timestamp_utc")},
            seconds=RECOVERY_CPU_POLICY["source_max_age_seconds"],
        )
        or runtime.get("protective_hold") is not False
    ):
        return False
    creative = as_dict(memory.get("creative_session"))
    if (
        creative.get("active") is not False
        or creative.get("cooldown_active") is not False
    ):
        return False
    fluidity = as_dict(runtime.get("mac_fluidity_contract"))
    if fluidity.get("support_pause_recommended") is not False or fluidity.get(
        "fluidity_band"
    ) not in {"smooth", "guarded_smooth"}:
        return False
    score = runtime.get("host_saturation_score")
    if (
        type(score) not in (int, float)
        or not 0 <= score <= RECOVERY_CPU_POLICY["maximum_host_saturation_score"]
    ):
        return False
    try:
        free_gib = shutil.disk_usage(root).free / GIB
    except OSError:
        return False
    return 16 <= free_gib < RECOVERY_CPU_POLICY["local_free_gib_below"]


def admission(root):
    allowed(root)
    for flag in (
        root / "OPERATOR_STOP.flag",
        root / "governance/health/OPERATOR_STOP.flag",
        root / "RUNTIME_MAINTENANCE_HOLD.flag",
    ):
        if allowed(flag, missing=True).exists():
            return False, "operator_or_maintenance_hold"
    if os.getenv("RUNTIME_MAINTENANCE_HOLD") == "1" or maintenance_hold_snapshot(
        root
    ).get("active", True):
        return False, "operator_or_maintenance_hold"
    health = root / "governance/health"
    memory = load_json(
        allowed(health / "memory_efficiency_control_latest.json", missing=True)
    )
    memory = as_dict(memory)
    observed = as_dict(memory.get("storage_recovery_memory_observation"))
    if (
        memory.get("input_evidence_ready") is not True
        or observed.get("input_evidence_ready") is not True
        or not fresh(observed)
    ):
        return False, "memory_source_evidence_not_ready"
    if not _storage_memory_observation({"rc": 0, "parsed": memory}).get(
        "admission_ready"
    ):
        return False, "memory_admission_not_ready"
    runtime = load_json(
        allowed(health / "runtime_throttle_control_latest.json", missing=True)
    )
    runtime = as_dict(runtime)
    if not fresh(runtime):
        return False, "runtime_observation_stale_or_missing"
    thermal = as_dict(as_dict(runtime.get("runtime_snapshot")).get("thermal"))
    if thermal.get("measurement_available") is not True or any(
        thermal.get(k) is not False
        for k in (
            "thermal_warning_active",
            "performance_warning_active",
            "cpu_power_warning_active",
        )
    ):
        return False, "thermal_observation_not_clear"
    measures = as_dict(
        as_dict(runtime.get("mac_fluidity_contract")).get("measurements")
    )
    cpu_count = max(os.cpu_count() or 1, 1)
    workload_lease = as_dict(runtime.get("workload_admission"))
    if (
        current_lease(workload_lease, "storage_recovery")
        and workload_lease.get("cpu_capacity_percent") == cpu_count * 100
        and runtime.get("protective_hold") is False
        and as_dict(runtime.get("adaptive_safety_limits")).get("active") is False
    ):
        if (
            not 0
            <= os.getloadavg()[1] / cpu_count
            <= POLICIES["storage_recovery"]["load"]
        ):
            return False, "host_load_above_recovery_budget"
        return True, "bounded_workload_storage_recovery_admitted"
    recovery_relief = _disk_recovery_cpu_relief(root, memory, runtime, cpu_count)
    limits = {
        "foreground_app_cpu_percent": (
            RECOVERY_CPU_POLICY["foreground_cpu_percent_below"]
            if recovery_relief
            else 90
        ),
        "macos_system_cpu_percent": (
            RECOVERY_CPU_POLICY["system_cpu_percent_below"] if recovery_relief else 90
        ),
    }
    for key in ("foreground_app_cpu_percent", "macos_system_cpu_percent"):
        value = measures.get(key)
        if type(value) not in (int, float) or not 0 <= value < limits[key]:
            return False, "foreground_or_system_pressure"
    if (
        recovery_relief
        and sum(measures[key] for key in limits)
        >= RECOVERY_CPU_POLICY["combined_cpu_percent_below"]
    ):
        return False, "combined_foreground_system_pressure"
    load_limit = (
        RECOVERY_CPU_POLICY["maximum_load_per_logical_cpu"] if recovery_relief else 0.62
    )
    if not 0 <= os.getloadavg()[1] / cpu_count <= load_limit:
        return False, "host_load_above_recovery_budget"
    return True, (
        "bounded_disk_recovery_relaxed_cpu_admitted"
        if recovery_relief
        else "bounded_cold_log_recovery_admitted"
    )


class Guard:
    def __init__(self, root, seconds):
        self.root = root
        reserve = float(os.getenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", "16"))
        if not math.isfinite(reserve) or reserve < 0:
            raise ValueError("invalid_emergency_reserve")
        self.reserve = int(max(reserve, 16) * GIB)
        self.deadline = time.monotonic() + min(max(seconds, 1), 1800)
        self.pace = WorkBudget()
        self.last_probe = 0.0
        self.last_progress = 0.0
        self.lock_anchor = None
        self.last_admission = ""

    def check(self):
        self.pace.tick()
        now = time.monotonic()
        if now >= self.deadline:
            raise Deferred("compaction_deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss >= 256 * 1024**2:
            raise Deferred("resident_memory_budget_reached")
        if now - self.last_probe >= 1:
            if self.lock_anchor is not None:
                path, expected = self.lock_anchor
                try:
                    actual = identity(path)[:2]
                except (OSError, RuntimeError) as exc:
                    raise Deferred("storage_maintenance_lock_anchor_lost") from exc
                if actual != expected:
                    raise Deferred("storage_maintenance_lock_anchor_changed")
            allowed(self.root)
            ready, reason = admission(self.root)
            self.last_admission = reason
            if not ready:
                raise Deferred(reason)
            if reason == "bounded_workload_storage_recovery_admitted":
                self.deadline = min(
                    self.deadline, now + POLICIES["storage_recovery"]["seconds"]
                )
            if shutil.disk_usage(self.root).free < self.reserve:
                raise Deferred("emergency_reserve_reached")
            self.last_probe = now

    def progress(self, path, phase, count, size):
        now = time.monotonic()
        if now - self.last_progress >= 25:
            self.last_progress = now
            print(
                json.dumps(
                    {
                        "event": "cold_evidence_progress",
                        "source": str(path.relative_to(self.root)),
                        "phase": phase,
                        "processed_gib": round(count / GIB, 3),
                        "source_gib": round(size / GIB, 3),
                    }
                ),
                flush=True,
            )


def inventory(root):
    archive = allowed(root / ROOT_REL, missing=True)
    if not archive.exists():
        return []
    rows = []
    started = time.monotonic()
    count = 0
    device = root.stat().st_dev
    for base, dirs, files in os.walk(archive, followlinks=False):
        allowed(Path(base))
        dirs[:] = [
            n for n in dirs if not n.startswith(".") and not Path(base, n).is_symlink()
        ]
        for name in files:
            count += 1
            if count > 20000 or time.monotonic() - started > 30:
                raise RuntimeError("inventory_budget_exceeded")
            if name.startswith(".") or not NAME.fullmatch(name):
                continue
            path = Path(base, name)
            if path.is_symlink():
                continue
            info = identity(path)
            if info[0] != device or info[2] < 1024**2:
                continue
            if time.time() - info[3] / 1e9 < MIN_AGE_SECONDS:
                continue
            if database_header(path):
                continue
            rows.append((path, info))
    return sorted(rows, key=lambda row: (-row[1][2], str(row[0])))


def receipt(root, payload):
    folder = allowed(root / "governance/storage_recovery", missing=True)
    folder.mkdir(parents=True, exist_ok=True)
    path = allowed(folder / "cold_evidence_compression.jsonl", missing=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(
            json.dumps({"timestamp_utc": iso_now(), **payload}, sort_keys=True) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())
    sync_dir(folder)


def digest_stream(stream, guard, path, size, phase):
    digest, count = hashlib.sha256(), 0
    while data := stream.read(CHUNK):
        guard.check()
        count += len(data)
        if count > size:
            raise RuntimeError("source_or_restored_stream_grew")
        digest.update(data)
        guard.progress(path, phase, count, size)
    return digest.hexdigest(), count


def compact_one(root, path, expected, guard, *, codec="zstd", compression_level=1):
    from compression import zstd

    if codec not in {"zstd", "gzip"}:
        raise ValueError("unsupported_cold_archive_codec")
    guard.check()
    before = identity(path)
    if before != expected:
        raise RuntimeError("source_changed_since_inventory")
    if database_header(path):
        raise RuntimeError("database_content_not_eligible")
    idle(path)
    target = allowed(
        Path(str(path) + (".gz" if codec == "gzip" else ".zst")), missing=True
    )
    temporary = None
    try:
        if target.exists():
            target_before = identity(target)
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as src:
                source_hash, count = digest_stream(
                    src, guard, path, before[2], "hash_existing"
                )
        else:
            if shutil.disk_usage(root).free < before[2] * 1.01 + guard.reserve:
                raise RuntimeError("insufficient_worst_case_scratch_reserve")
            fd, name = tempfile.mkstemp(
                prefix=".cold_compact_", suffix=".tmp", dir=path.parent
            )
            temporary = Path(name)
            with os.fdopen(fd, "wb") as dst, os.fdopen(
                os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb"
            ) as src:
                opened = os.fstat(src.fileno())
                if (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                ) != before:
                    raise RuntimeError("source_changed_before_open")
                compressor = (
                    zlib.compressobj(max(1, min(int(compression_level), 9)), wbits=31)
                    if codec == "gzip"
                    else zstd.ZstdCompressor(
                        options={
                            zstd.CompressionParameter.compression_level: 1,
                            zstd.CompressionParameter.nb_workers: 0,
                            zstd.CompressionParameter.checksum_flag: 1,
                        }
                    )
                )
                digest, count = hashlib.sha256(), 0
                while data := src.read(CHUNK):
                    guard.check()
                    count += len(data)
                    if count > before[2]:
                        raise RuntimeError("source_grew_during_compression")
                    digest.update(data)
                    dst.write(compressor.compress(data))
                    guard.progress(path, "compress", count, before[2])
                dst.write(compressor.flush())
                dst.flush()
                os.fsync(dst.fileno())
                source_hash = digest.hexdigest()
            target_before = None
        verify_path = temporary or target
        with os.fdopen(os.open(verify_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as raw:
            with (
                gzip.GzipFile(fileobj=raw, mode="rb")
                if codec == "gzip"
                else zstd.ZstdFile(raw, "rb")
            ) as restored:
                restored_hash, restored_size = digest_stream(
                    restored, guard, path, before[2], "verify"
                )
        if (restored_hash, restored_size) != (
            source_hash,
            before[2],
        ) or count != before[2]:
            raise RuntimeError("full_restore_sha256_mismatch")
        if identity(path) != before:
            raise RuntimeError("source_changed_during_compression")
        if target_before is not None and identity(target) != target_before:
            raise RuntimeError("existing_archive_changed")
        if identity(verify_path)[2] >= before[2]:
            raise RuntimeError("no_positive_space_saving")
        idle(path)
        if temporary is not None:
            os.link(temporary, target, follow_symlinks=False)
            sync_dir(path.parent)
        target_id = identity(target)
        proof = {
            "source": str(path.relative_to(root)),
            "compressed": str(target.relative_to(root)),
            "source_identity": before,
            "source_bytes": before[2],
            "compressed_bytes": target_id[2],
            "sha256_uncompressed": restored_hash,
            "verified_restored_bytes": restored_size,
            "saved_bytes": before[2] - target_id[2],
            "codec": codec,
            "verification": f"full_{codec}_restore_sha256_stable_idle_source",
            "resource_admission": getattr(guard, "last_admission", ""),
        }
        receipt(root, {"event": "verified_before_release", **proof})
        guard.check()
        idle(path)
        if identity(path) != before or identity(target) != target_id:
            raise RuntimeError("source_or_archive_changed_before_release")
        path.unlink()
        sync_dir(path.parent)
        receipt(root, {"event": "original_replaced", **proof})
        print(
            json.dumps(
                {
                    "event": "cold_evidence_completed",
                    "source": proof["source"],
                    "saved_gib": round(proof["saved_bytes"] / GIB, 3),
                }
            ),
            flush=True,
        )
        return proof
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_payload(
    root=PROJECT_ROOT, *, apply=False, target_free_gb=64, max_files=256, seconds=900
):
    root = allowed(Path(root))
    if not math.isfinite(target_free_gb) or not 16 <= target_free_gb <= 256:
        raise ValueError("invalid_free_space_target")
    before = shutil.disk_usage(root).free
    rows = inventory(root) if before < target_free_gb * GIB else []
    records, errors = [], []
    reason = (
        "headroom_sufficient"
        if before >= target_free_gb * GIB
        else "no_eligible_cold_logs" if not rows else "planned"
    )
    if apply and rows:
        lock_path = allowed(
            root / "governance/locks/storage_maintenance.lock", missing=True
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(
            os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+"
        ) as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {
                    "timestamp_utc": iso_now(),
                    "ok": True,
                    "assessment_complete": True,
                    "target_ready": False,
                    "overall_status": "deferred",
                    "reason": "storage_maintenance_lock_busy",
                    "completed_count": 0,
                    "saved_bytes": 0,
                }
            background_policy()
            guard = Guard(root, seconds)
            held = os.fstat(lock.fileno())
            guard.lock_anchor = (lock_path, (held.st_dev, held.st_ino))
            pending = list(rows)
            reason = "eligible_files_exhausted"
            while pending and len(records) + len(errors) < max(1, min(max_files, 1024)):
                try:
                    guard.check()
                except (RuntimeError, TimeoutError) as exc:
                    reason = str(exc)
                    break
                free = shutil.disk_usage(root).free
                if free >= target_free_gb * GIB:
                    reason = "target_reached"
                    break
                fitting = next(
                    (
                        row
                        for row in pending
                        if row[1][2] * 1.01 + guard.reserve <= free
                    ),
                    None,
                )
                if fitting is None:
                    reason = "insufficient_scratch_for_remaining_files"
                    break
                pending.remove(fitting)
                path, info = fitting
                try:
                    records.append(compact_one(root, path, info, guard))
                except Deferred as exc:
                    reason = str(exc)
                    break
                except Exception as exc:
                    errors.append(
                        {
                            "source": str(path.relative_to(root)),
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    reason = "file_deferred_or_failed"
                    break
            if pending and len(records) + len(errors) >= max(1, min(max_files, 1024)):
                reason = "file_budget_reached"
    after = shutil.disk_usage(root).free
    status = (
        "ready"
        if after >= target_free_gb * GIB
        else "planned" if not apply else "deferred"
    )
    return {
        "timestamp_utc": iso_now(),
        "ok": not errors,
        "assessment_complete": True,
        "target_ready": after >= target_free_gb * GIB,
        "overall_status": status,
        "apply": apply,
        "reason": reason,
        "candidate_count": len(rows),
        "candidate_bytes": sum(i[2] for _, i in rows),
        "completed_count": len(records),
        "saved_bytes": sum(r["saved_bytes"] for r in records),
        "local_free_before_gib": round(before / GIB, 3),
        "local_free_after_gib": round(after / GIB, 3),
        "target_free_gib": target_free_gb,
        "records": records,
        "errors": errors,
        "policy": {
            "scope": str(ROOT_REL),
            "min_age_days": 7,
            "workers": 1,
            "cpu_fraction_of_one_core": 0.25,
            "disk_recovery_cpu_policy": dict(RECOVERY_CPU_POLICY),
            "emergency_reserve_min_gib": 16,
            "database_mutation_allowed": False,
            "protected_volumes_allowed": False,
            "automatic_promotion_allowed": False,
            "live_execution_allowed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target-free-gb", type=float, default=64)
    parser.add_argument("--max-files", type=int, default=256)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    payload = build_payload(
        args.project_root,
        apply=args.apply,
        target_free_gb=args.target_free_gb,
        max_files=args.max_files,
        seconds=args.seconds,
    )
    out = (
        args.out_file
        or args.project_root / "governance/health/cold_evidence_compactor_latest.json"
    )
    write_payload(allowed(out, missing=True), payload)
    print(json.dumps(payload), flush=True)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
