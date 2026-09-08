"""Schedule guarded refreshes from measurement evidence, never file mtime."""

from __future__ import annotations

import argparse
import json
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
