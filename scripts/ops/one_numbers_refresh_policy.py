"""Schedule guarded refreshes from measurement evidence, never file mtime."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def refresh_policy(
    summary_path: Path,
    token_path: Path,
    *,
    target_interval_seconds: int,
    breaker_max_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    interval = max(int(target_interval_seconds), 1)
    # Leave half the consumer's freshness budget for launch cadence and build time.
    if breaker_max_age_seconds > 0:
        interval = min(interval, max(int(breaker_max_age_seconds) // 2, 1))
    measured = None
    status = "measurement_missing_or_invalid"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = payload.get("data_quality_session_local_timestamp")
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            if parsed.timestamp() > now.timestamp() + 2.0:
                status = "measurement_in_future"
            else:
                measured = parsed
                status = "measurement_available"
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        auth_epoch = token_path.stat().st_mtime
    except OSError:
        auth_epoch = None
    age = max(int(now.timestamp() - measured.timestamp()), 0) if measured else None
    auth_due = bool(
        auth_epoch is not None
        and (measured is None or measured.timestamp() + 2.0 < auth_epoch)
    )
    due = age is None or age >= interval or auth_due
    return {
        "measurement_status": status,
        "measurement_age_seconds": age,
        "auth_epoch_refresh_required": auth_due,
        "target_interval_seconds": interval,
        "refresh_due": due,
        "resource_admission_required": True,
        "maintenance_admission_required": True,
    }


def bounded_refresh_admitted(project_root: Path, payload: dict[str, Any]) -> bool:
    """Admit overdue risk evidence only with fresh, bounded host headroom."""
    try:
        now = datetime.now(timezone.utc)
        measured = datetime.fromisoformat(str(payload.get("timestamp_utc", "")).replace("Z", "+00:00"))
        if measured.tzinfo is None or not 0 <= (now - measured).total_seconds() <= 120:
            return False
        if payload.get("throttle_profile") != "soft_cap" or payload.get("memory_pressure_level") != "normal":
            return False
        if payload.get("compute_pressure_level") not in {"normal", "elevated"}:
            return False
        if not 0 < float(payload.get("host_saturation_score", 100)) <= 60:
            return False
        if (payload.get("mac_fluidity_contract") or {}).get("support_pause_recommended") is not False:
            return False
        thermal = (payload.get("runtime_snapshot") or {}).get("thermal") or {}
        if any(thermal.get(key) is not False for key in ("thermal_warning_active", "performance_warning_active", "cpu_power_warning_active")):
            return False
        resource = json.loads((project_root / "governance/health/resource_guard_latest.json").read_text())
        measured = datetime.fromisoformat(str(resource.get("timestamp_utc", "")).replace("Z", "+00:00"))
        if measured.tzinfo is None or not 0 <= (now - measured).total_seconds() <= 120:
            return False
        if resource.get("memory_pressure_state") != "green" or not 0 <= float(resource.get("load1_per_core", 100)) <= 1.5:
            return False
        from scripts.resource_guard import evaluate_refresh_job

        admitted, _, _ = evaluate_refresh_job(resource)
        if not admitted:
            return False
        return bool(refresh_policy(
            project_root / "exports/one_numbers/one_numbers_summary.json",
            project_root / "token.json",
            target_interval_seconds=int(os.getenv("ONE_NUMBERS_REFRESH_INTERVAL_SECONDS", "300")),
            breaker_max_age_seconds=int(os.getenv("ONE_NUMBERS_BREAKER_MAX_AGE_SECONDS", "600")),
        )["refresh_due"])
    except (OSError, TypeError, ValueError, AttributeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--target-interval", type=int, required=True)
    parser.add_argument("--breaker-max-age", type=int, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    policy = refresh_policy(
        args.summary,
        args.token,
        target_interval_seconds=args.target_interval,
        breaker_max_age_seconds=args.breaker_max_age,
    )
    if args.shell:
        age = policy["measurement_age_seconds"]
        print(
            age if age is not None else 999999999,
            int(policy["auth_epoch_refresh_required"]),
            policy["target_interval_seconds"],
            policy["measurement_status"],
        )
    else:
        print(json.dumps(policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
