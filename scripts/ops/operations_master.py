"""Bounded dispatch adapter used by the existing master infrastructure supervisor."""

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import time

from core.operations_master import ACTIONS, build_directions
from core.runtime_maintenance import maintenance_hold_snapshot
from core.storage_router import inspect_storage_path
from core.workload_admission import current_lease
from scripts.ops.long_runtime_common import write_payload


def local(root, name):
    path = Path(root) / "governance/health" / name
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_operations_master_route")
    return path


def read_input(root, name):
    path = local(root, name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024**2:
            raise ValueError("bounded_operations_input_required")
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("operations_input_size_exceeded")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("operations_input_object_required")
    return payload


def stamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source_timezone_required")
    return parsed


def controls(root, *, now=None):
    now = now or datetime.now(timezone.utc)
    result = {"fresh": False, "operator_hold": True, "source_observations": {}}
    try:
        hold = any(
            local(root, name).exists()
            for name in (
                "SYSTEM_POWER_OFF.flag",
                "OPERATOR_STOP.flag",
                "GLOBAL_TRADING_HALT.flag",
            )
        )
        local(root, "RUNTIME_MAINTENANCE_HOLD.flag")
        hold |= maintenance_hold_snapshot(root, now_utc=now).get("active", True)
        result["operator_hold"] = hold
    except (ValueError, OSError, TypeError):
        return result
    try:
        payloads = {}
        for key, name in (
            ("runtime", "runtime_throttle_control_latest.json"),
            ("storage", "local_storage_reserve_guard_latest.json"),
        ):
            payload = read_input(root, name)
            age = (now - stamp(payload.get("timestamp_utc"))).total_seconds()
            result["source_observations"][key] = {
                "timestamp_utc": payload.get("timestamp_utc"),
                "age_seconds": age,
            }
            if not 0 <= age <= 180:
                raise ValueError("stale_operations_control")
            payloads[key] = payload
        reserve = payloads["storage"].get("local_storage_reserve", {})
        # Only the reserve owner can clear its pressure condition.
        pressure = reserve.get("pressure_active")
        reserve_age = (now - stamp(reserve.get("timestamp_utc"))).total_seconds()
        if (
            type(pressure) is not bool
            or reserve.get("disk", {}).get("known") is not True
            or not 0 <= reserve_age <= 180
        ):
            raise ValueError("storage_pressure_observation_required")
        result.update(
            fresh=True,
            storage_pressure=pressure,
            maintenance_admitted=current_lease(
                payloads["runtime"].get("workload_admission"), "maintenance", now=now
            ),
            storage_recovery_admitted=current_lease(
                payloads["runtime"].get("workload_admission"),
                "storage_recovery",
                now=now,
            ),
        )
    except (ValueError, OSError, TypeError, AttributeError):
        result["fresh"] = False
    try:
        provider = read_input(root, "provider_access_guard_schwab_latest.json")
        age = (now - stamp(provider.get("timestamp_utc"))).total_seconds()
        deadline = stamp(provider.get("cooldown_until_utc"))
        result["provider_cooldown"] = (
            provider.get("provider") == "schwab"
            and provider.get("state") == "cooldown"
            and 0 <= age <= 1800
            and deadline > now
        )
    except (ValueError, OSError, TypeError):
        result["provider_cooldown"] = None
    return result


def directions(root, checks, *, now=None):
    return build_directions(checks, controls(root, now=now), now=now)


def dispatch(root, plan, *, runner, timeout_sec=150):
    """Persist attempt admission before each child; a crash cannot erase cooldown."""
    root = Path(root)
    attempts, deferred = [], []
    start = time.monotonic()
    budget = min(max(float(timeout_sec), 1), 150)
    try:
        lock_path = local(root, "operations_master.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        with os.fdopen(fd, "a+") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("regular_operations_lock_required")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return [], [{"reason": "operations_master_busy"}]
            state_path = local(root, "operations_master_dispatch_state.json")
            state = read_input(root, state_path.name) if state_path.exists() else {}
            recent = state.get("last_attempt_epoch", {})
            if not isinstance(recent, dict):
                raise ValueError("invalid_operations_dispatch_state")
            history = state.get("recent_dispatch_history", [])
            if not isinstance(history, list):
                raise ValueError("invalid_operations_dispatch_history")
            for row in plan["directives"]:
                group = row["subgroup"]
                if row.get("directive") != "delegate":
                    continue
                expected, child_timeout = ACTIONS.get(group, ((), 0))
                if not expected or row.get("command") != list(expected):
                    deferred.append(
                        {"subgroup": group, "reason": "command_not_allowlisted"}
                    )
                    continue
                now = time.time()
                previous = float(recent.get(group, 0))
                remaining = budget - (time.monotonic() - start)
                reason = ""
                if (
                    not math.isfinite(previous)
                    or previous > now
                    or now - previous < 600
                ):
                    reason = "owner_cooldown_or_invalid_clock"
                elif len(attempts) >= 2 or remaining < child_timeout:
                    reason = "shared_dispatch_budget"
                else:
                    # Re-evaluate admission; a serialized plan is not a permission token.
                    fresh_plan = build_directions([], controls(root))
                    current = next(
                        item
                        for item in fresh_plan["directives"]
                        if item["subgroup"] == group
                    )
                    if current["blocked_by"]:
                        reason = "current_owner_admission_not_ready"
                if reason:
                    deferred.append({"subgroup": group, "reason": reason})
                    continue
                recent[group] = now
                receipt = {
                    "subgroup": group,
                    "owner": row.get("owner"),
                    "issues_before": list(row.get("issues") or [])[:32],
                    "command": list(expected),
                    "started_epoch": now,
                    "phase": "started",
                    "completion_credit": False,
                }
                history = [*history[-63:], receipt]
                write_payload(
                    state_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "last_attempt_epoch": recent,
                        "recent_dispatch_history": history,
                    },
                )
                command = [str(root / "scripts/ops/opsctl.sh"), *expected]
                try:
                    result = runner(command, cwd=root, timeout_sec=child_timeout)
                except (OSError, ValueError) as exc:
                    result = {
                        "cmd": command,
                        "rc": 127,
                        "timed_out": False,
                        "error_type": type(exc).__name__,
                    }
                attempts.append(
                    {**result, "subgroup": group, "completion_credit": False}
                )
                receipt.update(
                    phase="finished",
                    rc=result.get("rc"),
                    timed_out=bool(result.get("timed_out", False)),
                )
                write_payload(
                    state_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "last_attempt_epoch": recent,
                        "recent_dispatch_history": history,
                    },
                )
            return attempts, deferred
    except (ValueError, OSError, TypeError):
        return attempts, deferred + [
            {"reason": "unsafe_or_invalid_operations_dispatch_state"}
        ]
