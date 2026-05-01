#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "auth_lease_manager_latest.json"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    min_lease_seconds: int = 1200,
    critical_lease_seconds: int = 600,
    max_guard_age_minutes: float = 60.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    token_guard_path = health_root / "premarket_token_guard_latest.json"
    broker_readiness_path = health_root / "broker_readiness_latest.json"
    token_guard = load_json(token_guard_path)
    broker_readiness = load_json(broker_readiness_path)
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")

    token_after = token_guard.get("token_after") if isinstance(token_guard.get("token_after"), dict) else {}
    token_before = token_guard.get("token_before") if isinstance(token_guard.get("token_before"), dict) else {}
    expires_in_seconds = float(
        token_after.get("expires_in_seconds", token_before.get("expires_in_seconds", 0.0)) or 0.0
    )
    guard_age_minutes = payload_age_minutes(token_guard, token_guard_path)
    network_ok = bool((token_guard.get("network") or {}).get("ok", False))
    auth_ok = bool((token_guard.get("auth") or {}).get("ok", True))
    broker_ready = bool(broker_readiness.get("ready_for_open", broker_readiness.get("auth_ok", False)))
    configured_for_refresh = bool(token_before.get("exists", False) or token_after.get("exists", False))

    lease_state = "healthy"
    if expires_in_seconds < float(critical_lease_seconds) or not network_ok or not auth_ok:
        lease_state = "critical"
    elif expires_in_seconds < float(min_lease_seconds) or (
        guard_age_minutes is not None and float(guard_age_minutes) > float(max_guard_age_minutes)
    ):
        lease_state = "warning"

    overall_status = "ready"
    if lease_state == "critical" or not configured_for_refresh:
        overall_status = "blocked"
    elif lease_state == "warning" or not broker_ready:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "./scripts/ops/opsctl.sh token-refresh --json" if lease_state != "healthy" else "",
            "keep a browser-assisted auth fallback ready if lease refresh stops extending expiry" if not auth_ok or expires_in_seconds < float(min_lease_seconds) else "",
            "page the operator out-of-band when lease time drops under the critical floor" if expires_in_seconds < float(critical_lease_seconds) else "",
            "treat missing or stale token guard data as a runtime degradation, not a silent success" if guard_age_minutes is None or float(guard_age_minutes or 0.0) > float(max_guard_age_minutes) else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "lease_state": lease_state,
        "lease_budget": {
            "expires_in_seconds": round(float(expires_in_seconds), 3),
            "min_lease_seconds": int(min_lease_seconds),
            "critical_lease_seconds": int(critical_lease_seconds),
            "guard_age_minutes": round(float(guard_age_minutes), 4) if guard_age_minutes is not None else None,
        },
        "broker_state": {
            "broker_ready": bool(broker_ready),
            "network_ok": bool(network_ok),
            "auth_ok": bool(auth_ok),
            "configured_for_refresh": bool(configured_for_refresh),
            "restart_storms": len(process_watchdog.get("restart_storms") or []),
        },
        "fallback_ladder": [
            "silent_refresh",
            "interactive_token_refresh",
            "browser_auth_fallback",
            "operator_page_and_runtime_degrade",
        ],
        "infra_bots": ["auth_lease_manager", "premarket_token_guard", "live_readiness_smoke"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track broker/auth lease health for multi-week runtime windows.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--min-lease-seconds", type=int, default=int(os.getenv("SCHWAB_AUTH_LEASE_MIN_SECONDS", "1200")))
    parser.add_argument("--critical-lease-seconds", type=int, default=int(os.getenv("SCHWAB_AUTH_LEASE_CRITICAL_SECONDS", "600")))
    parser.add_argument("--max-guard-age-minutes", type=float, default=60.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        min_lease_seconds=int(args.min_lease_seconds),
        critical_lease_seconds=int(args.critical_lease_seconds),
        max_guard_age_minutes=float(args.max_guard_age_minutes),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "auth_lease_manager "
            f"overall_status={payload.get('overall_status', '')} "
            f"lease_state={payload.get('lease_state', '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
