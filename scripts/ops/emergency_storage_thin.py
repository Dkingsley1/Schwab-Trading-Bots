#!/usr/bin/env python3
"""One paced, verified compression wave for a disk-only recovery deadlock."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.background_work_budget import WorkBudget
from scripts.ops import raw_training_compaction_intelligence as raw
from scripts.ops.approved_storage_recovery import recovery_hold_active
from scripts.ops.self_healing_gap_audit import age, read
from scripts.ops.system_power import local, write
from scripts.resource_guard import _memory_pressure_state

GIB = 1024**3
STATE = "governance/runtime/emergency_storage_thin.json"
OUT = "governance/health/emergency_storage_thin_latest.json"
DISK_ONLY = {
    "storage_not_ready_for_raw_compaction",
    "storage_pressure_above_raw_compaction_ceiling",
    "local_free_space_below_raw_compaction_reserve",
    "storage_efficiency_contract_apply_not_allowed_now",
}


def fresh(payload, now, seconds):
    hours, state = age(payload, now)
    return hours is not None and state != "future" and 0 <= hours * 3600 <= seconds


def resources_clear(root, now):
    runtime, _ = read(root, "governance/health/runtime_throttle_control_latest.json")
    snapshot, _ = read(root, "governance/health/resource_guard_latest.json")
    try:
        if not fresh(runtime, now, 120) or not fresh(snapshot, now, 120):
            return False
        if snapshot.get("input_evidence_ready") is not True:
            return False
        # Admit only the disk component of memory pressure, never actual RAM/swap pressure.
        state, reasons, _ = _memory_pressure_state(snapshot)
        if state != "green" and not (
            reasons
            and all(r.startswith("local_disk_swap_headroom_gb:") for r in reasons)
        ):
            return False
        limits = (
            (snapshot["memory_available_pct"], 50, 100),
            (snapshot["memory_free_pct"], 8, 100),
            (snapshot["load1_per_core"], 0, 1.2),
            (runtime["host_saturation_score"], 0, 60),
        )
        if any(
            not math.isfinite(float(v)) or not lo <= float(v) <= hi
            for v, lo, hi in limits
        ):
            return False
        if runtime.get("compute_pressure_level") not in {"normal", "elevated"}:
            return False
        fluidity = runtime["mac_fluidity_contract"]
        if fluidity.get("fluidity_band") not in {
            "silky",
            "comfortable",
            "guarded_smooth",
        }:
            return False
        thermal = runtime["runtime_snapshot"]["thermal"]
        return all(
            thermal.get(k) is False
            for k in (
                "thermal_warning_active",
                "performance_warning_active",
                "cpu_power_warning_active",
            )
        )
    except (KeyError, ValueError, TypeError):
        return False


def admission(root, now):
    free = shutil.disk_usage(root).free
    blockers = []
    if not 16 * GIB < free < 32 * GIB:
        blockers.append("outside_disk_only_emergency_band_16_to_32_gib")
    if any(
        local(root, "governance/health/" + flag).exists()
        for flag in ("SYSTEM_POWER_OFF.flag", "GLOBAL_TRADING_HALT.flag")
    ) or recovery_hold_active(root):
        blockers.append("operator_or_maintenance_hold")
    if not resources_clear(root, now):
        blockers.append("resource_evidence_not_admitted")
    # Use the actual recovery owner's preconditions, not a reported command string.
    report, _ = read(
        root,
        "governance/health/storage_backpressure_autopilot_latest.json",
        max_bytes=16 * 1024**2,
    )
    preview = report.get("previews", {}).get("raw_training_compaction", {})
    reasons = preview.get("blockers", [])
    if not fresh(report, now, 900) or not reasons or not set(reasons) <= DISK_ONLY:
        blockers.append("disk_only_repair_deadlock_not_proven")
    live = preview.get("raw_live", {})
    for key, ceiling in (
        ("total_pending_lines", 15000),
        ("core_pending_lines", 10000),
        ("oldest_pending_age_seconds", 900),
    ):
        value = live.get(key)
        if (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= ceiling
        ):
            blockers.append("active_ingestion_backpressure")
            break
    return sorted(set(blockers)), free


def candidates(root, now):
    manifest, _ = read(
        root, "governance/health/raw_training_compaction_intelligence_latest.json"
    )
    if not fresh(manifest, now, 900):
        return []
    selected, used, seen = [], 0, set()
    for item in manifest.get("top_compaction_candidates", [])[:25]:
        path = Path(str(item.get("path", "")))
        if (
            not path.is_absolute()
            or not path.is_relative_to(root)
            or path.suffix != ".jsonl"
        ):
            continue
        if "local_fallback_storage" in path.relative_to(root).parts:
            continue
        if raw._date_token_matches_current_day(
            path, now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        ):
            continue
        try:
            path = local(root, str(path.relative_to(root)))
            local(root, str(raw._compressed_sibling(path).relative_to(root)))
            identity = raw._file_identity(path)
            if (
                path in seen
                or identity[0] != root.stat().st_dev
                or identity[2] > 3 * GIB
            ):
                continue
            seen.add(path)
            row = raw._classify_row(
                path,
                scan_root=root,
                now_ts=now.timestamp(),
                today=now.strftime("%Y-%m-%d"),
                min_age_hours=24,
                sample_bytes=0,
            )
            if row["compression_candidate"]:
                selected.append(row)
        except (OSError, ValueError):
            continue
    result = []
    for row in sorted(selected, key=lambda r: (-r["age_hours"], -r["size_bytes"])):
        if len(result) < 2 and used + row["size_bytes"] <= 3 * GIB:
            result.append(row)
            used += row["size_bytes"]
    return result


class ThinGuard:
    def __init__(self, root):
        self.root = self.output = root
        self.reserve = 16 * GIB
        self.deadline = time.monotonic() + 90
        self.pace = WorkBudget()
        self.probed = 0.0

    def check(self):
        self.pace.tick()
        now = time.monotonic()
        if now >= self.deadline:
            raise TimeoutError("emergency_thin_deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        if rss >= 256 * 1024**2:
            raise RuntimeError("emergency_thin_memory_cap")
        if now - self.probed >= 1:
            blockers, _ = admission(self.root, datetime.now(timezone.utc))
            if blockers:
                raise RuntimeError(
                    "emergency_thin_admission_withdrawn:" + ",".join(blockers)
                )
            self.probed = now


def run(root, *, apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    blockers, free = admission(root, now)
    rows = candidates(root, now) if not blockers else []
    result = {
        "timestamp_utc": now.isoformat(),
        "ok": True,
        "state": "deferred",
        "blockers": blockers,
        "free_gib": round(free / GIB, 3),
        "caps": {
            "files": 2,
            "source_gib": 3,
            "attempts_per_hour": 2,
            "workers": 1,
            "seconds": 90,
        },
        "candidates": rows,
        "records": [],
        "live_execution_authority": False,
    }
    if blockers:
        return result
    if not rows:
        result["blockers"] = ["no_eligible_local_closed_raw_sources"]
        return result
    result["state"] = "planned"
    if not apply:
        return result
    lock = local(root, "governance/locks/storage_maintenance.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.update(state="deferred", blockers=["storage_maintenance_busy"])
            return result
        prior, error = read(root, STATE)
        if error not in {"", "missing_or_nonlocal"} or (
            error and local(root, STATE).exists()
        ):
            result.update(state="deferred", blockers=["retry_history_unreadable"])
            return result
        attempts = prior.get("attempts", [])
        if not isinstance(attempts, list) or any(
            not isinstance(t, (float, int)) or not math.isfinite(t) for t in attempts
        ):
            result.update(state="deferred", blockers=["retry_history_invalid"])
            return result
        recent = [t for t in attempts if now.timestamp() - t <= 3600]
        if len(recent) >= 2 or any(t > now.timestamp() for t in recent):
            result.update(state="deferred", blockers=["hourly_attempt_budget"])
            return result
        # Reserve before compression: crashes and failed attempts consume the same budget.
        write(root, STATE, {"attempts": recent + [now.timestamp()]})
        guard = ThinGuard(root)
        for row in rows:
            record = raw._compress_and_clear(
                Path(row["path"]),
                Path(row["compressed_path"]),
                compress_level=1,
                keep_raw=False,
                guard=guard,
            )
            result["records"].append(record)
            if record.get("status") != "ok":
                break
        result.update(
            state=(
                "completed"
                if all(r.get("status") == "ok" for r in result["records"])
                else "incomplete"
            ),
            ok=all(r.get("status") == "ok" for r in result["records"]),
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, apply=args.apply)
    write(ROOT, OUT, result)
    print(json.dumps(result, indent=None if args.json else 2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
