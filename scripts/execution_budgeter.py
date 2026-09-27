"""Build bounded execution budgets from fresh owner observations, never orders."""

import argparse
import fcntl
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.long_runtime_common import (
    evidence_freshness,
    governor_observation_contract,
    write_payload,
)
from scripts.sleeve_slo_guard import local_path, read_local_json, refresh_current


def build_payload(root, *, allocator=None, risk=None, slo=None, now=None):
    now = now or datetime.now(timezone.utc)
    paths = {
        "allocator": allocator
        or root / "governance/allocator/sleeve_allocator_latest.json",
        "risk": risk or root / "governance/risk/portfolio_risk_latest.json",
        "slo": slo or root / "governance/watchdog/sleeve_slo_latest.json",
    }
    inputs = {name: read_local_json(root, path) for name, path in paths.items()}
    sources, blockers = {}, []
    for name, data in inputs.items():
        freshness = evidence_freshness(
            data, now=now, max_age_minutes=6 if name == "slo" else 120
        )
        reasons = [] if freshness["fresh"] else [freshness["status"]]
        health = data.get("input_freshness") or {}
        if health.get("sources_ready") is False:
            reasons.append("upstream_sources_not_ready")
        if name == "risk" and (
            data.get("ok") is not True or data.get("overall_status") != "ready"
        ):
            reasons.append("upstream_risk_not_ready")
        if name == "slo":
            observed = evidence_freshness(
                {"timestamp_utc": data.get("source_observed_at_utc")},
                now=now,
                max_age_minutes=6,
            )
            if not observed["fresh"] or health.get("sources_ready") is not True:
                reasons.append("watchdog_observation_not_current")
            if not isinstance(data.get("alerts"), list) or not data.get("targets"):
                reasons.append("watchdog_coverage_missing")
        sources[name] = {
            "ready": not reasons,
            "timestamp_utc": data.get("timestamp_utc"),
            "age_minutes": freshness.get("age_minutes"),
            "blockers": reasons,
            "sha256": hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest(),
        }
        blockers.extend(f"{name}:{reason}" for reason in reasons)
    alloc, risk_data, slo_data = (inputs[name] for name in ("allocator", "risk", "slo"))
    weights = alloc.get("target_weights")
    caps = (risk_data.get("limits") or {}).get("sleeve_exposure_caps")
    risk_level = risk_data.get("risk_level")

    def valid_numbers(values):
        return (
            isinstance(values, dict)
            and bool(values)
            and all(
                isinstance(k, str)
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and 0 <= v <= 1
                for k, v in values.items()
            )
        )

    if (
        not valid_numbers(weights)
        or not valid_numbers(caps)
        or set(weights) - set(caps)
    ):
        blockers.append("allocation_or_exposure_contract_invalid")
    if risk_level not in {"low", "medium", "high"}:
        blockers.append("risk_level_unknown")
    alert_count = len(slo_data.get("alerts") or [])
    mult = {"high": 0.55, "medium": 0.75, "low": 1.0}.get(risk_level, 0.0)
    if alert_count:
        mult *= 0.85
    base_actions = {
        "core": 80,
        "aggressive": 160,
        "dividend": 30,
        "bond": 30,
        "crypto": 120,
    }
    per_sleeve = {}
    if not blockers:
        for sleeve, weight in weights.items():
            cap = float(caps[sleeve])
            actions = (
                max(
                    4,
                    int(round(base_actions.get(sleeve, 40) * mult * max(weight, 0.05))),
                )
                if weight > 0 and cap > 0
                else 0
            )
            per_sleeve[sleeve] = {
                "target_weight": round(weight, 6),
                "exposure_cap": round(cap, 6),
                "max_actions_per_hour": actions,
                "max_open_orders": max(2, actions // 8) if actions else 0,
            }
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        **governor_observation_contract(
            {
                name: (data, 360 if name == "slo" else 7200)
                for name, data in inputs.items()
            },
            now=now,
        ),
        "ok": not blockers,
        "overall_status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "input_freshness": {"sources_ready": not blockers, "sources": sources},
        "risk_level": risk_level,
        "slo_alert_count": alert_count,
        "global": {
            "max_total_actions_per_hour": sum(
                v["max_actions_per_hour"] for v in per_sleeve.values()
            ),
            "max_total_open_orders": sum(
                v["max_open_orders"] for v in per_sleeve.values()
            ),
            "multiplier": round(mult, 4) if not blockers else 0.0,
        },
        "sleeves": per_sleeve,
        "live_execution_authority": False,
    }


def refresh(root, *, allocator=None, risk=None, slo=None, out=None, refresh_slo=False):
    out = local_path(root, out or root / "governance/risk/execution_budget_latest.json")
    lock = local_path(root, out.with_suffix(".lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if refresh_slo:
            if slo is not None:
                raise ValueError("custom_slo_cannot_request_canonical_refresh")
            refresh_current(root)
        payload = build_payload(root, allocator=allocator, risk=risk, slo=slo)
        write_payload(out, payload)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        events = local_path(
            root, root / "governance/risk" / f"execution_budget_events_{day}.jsonl"
        )
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
        return payload


def main():
    parser = argparse.ArgumentParser(
        description="Build execution budgets from fresh allocator, risk and watchdog evidence."
    )
    parser.add_argument("--allocator", type=Path)
    parser.add_argument("--risk", type=Path)
    parser.add_argument("--slo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--refresh-slo", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = refresh(
        PROJECT_ROOT,
        allocator=args.allocator,
        risk=args.risk,
        slo=args.slo,
        out=args.out,
        refresh_slo=args.refresh_slo,
    )
    print(
        json.dumps(payload)
        if args.json
        else f"execution_budget_ok={payload['ok']} total_actions={payload['global']['max_total_actions_per_hour']}"
    )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
