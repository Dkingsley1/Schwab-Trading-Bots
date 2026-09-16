"""Narrow admission for explicitly approved, bounded storage recovery commands."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex

OWNERS = (
    "scripts/ops/raw_training_compaction_intelligence.py",
    "scripts/daily_state_snapshot_drill.py",
)


def recovery_hold_active(root):
    from core.runtime_maintenance import maintenance_hold_snapshot

    # Legacy root-level flags remain conservative; the owned health hold expires.
    return bool(
        os.environ.get("RUNTIME_MAINTENANCE_HOLD") == "1"
        or any((parent / "OPERATOR_STOP.flag").exists()
               for parent in (root, root / "governance/health"))
        or (root / "RUNTIME_MAINTENANCE_HOLD.flag").exists()
        or maintenance_hold_snapshot(root).get("active", True)
    )


def _fresh(payload):
    measured = datetime.fromisoformat(
        str(payload["timestamp_utc"]).replace("Z", "+00:00")
    )
    return (
        measured.tzinfo is not None
        and 0 <= (datetime.now(timezone.utc) - measured).total_seconds() <= 120
    )


def resources_admitted(root, runtime=None):
    """The operator exception covers the support latch, never hard resource stops."""
    try:
        if recovery_hold_active(root):
            return False
        if runtime is None:
            runtime = json.loads(
                (
                    root / "governance/health/runtime_throttle_control_latest.json"
                ).read_text()
            )
        resource = json.loads(
            (root / "governance/health/resource_guard_latest.json").read_text()
        )
        if not _fresh(runtime) or not _fresh(resource):
            return False
        if runtime.get("throttle_profile") not in {"observe", "soft_cap"}:
            return False
        if runtime.get("compute_pressure_level") not in {"normal", "elevated"}:
            return False
        if not 0 <= float(runtime.get("host_saturation_score", 100)) <= 60:
            return False
        fluidity = runtime.get("mac_fluidity_contract", {})
        if fluidity.get("fluidity_band") not in {
            "silky",
            "comfortable",
            "guarded_smooth",
        }:
            return False
        measurements = fluidity.get("measurements", {})
        for key in ("foreground_app_cpu_percent", "macos_system_cpu_percent"):
            if not 0 <= float(measurements.get(key, 100)) < 90:
                return False
        if runtime.get("memory_pressure_level") not in {"normal", "elevated"}:
            return False
        thermal = runtime.get("runtime_snapshot", {}).get("thermal", {})
        if any(
            thermal.get(key) is not False
            for key in (
                "thermal_warning_active",
                "performance_warning_active",
                "cpu_power_warning_active",
            )
        ):
            return False
        if resource.get("memory_pressure_state") != "green":
            return False
        if not 0 <= float(resource.get("load1_per_core", 100)) <= 1.2:
            return False
        from scripts.resource_guard import evaluate_refresh_job

        admitted, _, _ = evaluate_refresh_job(resource)
        return bool(admitted)
    except (KeyError, OSError, ValueError, TypeError, AttributeError):
        return False


def process_exempt(root, row, runtime):
    try:
        command = shlex.split(str(row.get("command") or ""))
        if len(command) < 3 or command[1] not in {
            str(root / owner) for owner in OWNERS
        }:
            return False
        if "--operator-approved-recovery" not in command[2:]:
            return False
        if not 0 <= float(row.get("cpu_percent", 100)) <= 35:
            return False
        elapsed = str(row.get("elapsed") or "")
        parts = [int(part) for part in elapsed.split(":")]
        if len(parts) not in {2, 3} or any(part < 0 for part in parts):
            return False
        seconds = sum(part * 60**i for i, part in enumerate(reversed(parts)))
        return 0 <= seconds < 1800 and resources_admitted(root, runtime)
    except (ValueError, TypeError):
        return False
