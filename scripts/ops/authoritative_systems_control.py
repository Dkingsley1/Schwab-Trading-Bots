#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.accountability import safe_write_json_atomic
from core.authoritative_systems import load_registry, validate_registry
from core.brokers.capability_contract import all_adapter_conformance
from core.causal_attribution import build_execution_trace, verify_execution_trace
from core.event_time_control import EventTimeGuard, EventTimePolicy
from core.execution_scenarios import run_execution_scenarios
from core.paper_live_equivalence import compare_pair
from scripts.strategy_validity_control import build_payload as build_validity_payload


DEFAULT_CONFIG = Path("config/authoritative_systems_v1.json")
DEFAULT_OUT = Path("governance/health/authoritative_systems_control_latest.json")


def _event_time_probe() -> dict[str, Any]:
    guard = EventTimeGuard(
        EventTimePolicy(allowed_lateness_seconds=30.0, max_future_skew_seconds=5.0)
    )
    first = guard.ingest(
        stream_id="probe",
        event_id="event-1",
        event_time_utc="2026-08-21T12:00:30+00:00",
        observed_at_utc="2026-08-21T12:00:31+00:00",
        payload={"price": 100.0},
    )
    within = guard.ingest(
        stream_id="probe",
        event_id="event-2",
        event_time_utc="2026-08-21T12:00:10+00:00",
        observed_at_utc="2026-08-21T12:00:32+00:00",
        payload={"price": 99.0},
    )
    late = guard.ingest(
        stream_id="probe",
        event_id="event-3",
        event_time_utc="2026-08-21T11:59:00+00:00",
        observed_at_utc="2026-08-21T12:00:33+00:00",
        payload={"price": 98.0},
    )
    restored = EventTimeGuard.restore(guard.snapshot())
    return {
        "ok": bool(
            first["accepted"]
            and within["disposition"] == "out_of_order_within_bound"
            and late["reason"] == "event_arrived_after_watermark"
            and restored.stream_status("probe") == guard.stream_status("probe")
        ),
        "first": first,
        "within_bound": within,
        "late": late,
    }


def _trace_probe() -> dict[str, Any]:
    intent = {
        "message_id": "trace-probe",
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 1.0,
        "model_score": 0.7,
        "threshold": 0.6,
        "features": {"expected_edge_bps": 8.0, "spread_bps": 1.0},
        "metadata": {"source_broker": "schwab", "source_profile": "default"},
    }
    result = {
        "status": "PAPER_EXECUTED",
        "paper_order": {"filled_quantity": 1.0, "fee_bps": 0.2, "slippage_bps": 0.8},
    }
    trace = build_execution_trace(
        intent=intent,
        result=result,
        gateway={"allow_execute": True, "reasons": []},
        mode="paper",
    )
    verification = verify_execution_trace(trace)
    return {
        "ok": bool(
            verification["ok"] and trace["attribution"]["no_fabricated_defaults"]
        ),
        "verification": verification,
        "trace_id": trace["trace_context"]["trace_id"],
        "stage_count": trace["stage_count"],
    }


def _equivalence_probe() -> dict[str, Any]:
    base = {
        "trace_context": {"trace_id": "trace_equivalence_probe"},
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 1.0,
        "asset_type": "EQUITY",
        "strategy": "probe",
        "metadata": {"production_candidate_id": "candidate-probe"},
    }
    comparison = compare_pair(
        {**base, "target_mode": "paper", "latency_ms": 10.0},
        {
            **base,
            "target_mode": "live",
            "latency_ms": 50.0,
            "broker_order_id": "broker-probe",
        },
    )
    return comparison


def build_payload(
    project_root: Path = PROJECT_ROOT, *, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    config_file = (
        config_path if config_path.is_absolute() else project_root / config_path
    )
    registry = load_registry(config_file)
    registry_report = validate_registry(registry, project_root=project_root)
    broker_report = all_adapter_conformance()
    scenario_report = run_execution_scenarios()
    validity_report = build_validity_payload(project_root)
    event_report = _event_time_probe()
    trace_report = _trace_probe()
    equivalence_report = _equivalence_probe()
    scenario_by_name = {
        str(row.get("scenario")): bool(row.get("ok", False))
        for row in scenario_report.get("scenarios") or []
    }
    controls = {
        "broker_capability_conformance": bool(broker_report["ok"]),
        "order_state_idempotency": all(
            scenario_by_name.get(name, False)
            for name in (
                "normal_fill",
                "submit_disconnect",
                "duplicate_intent",
                "progressive_partial_fill",
                "cancel_fill_race",
            )
        ),
        "point_in_time_validity": bool(validity_report["ok"]),
        "event_time_watermarks": bool(event_report["ok"]),
        "causal_attribution": bool(trace_report["ok"]),
        "paper_live_equivalence": bool(equivalence_report["ok"]),
        "execution_fault_simulation": bool(scenario_report["ok"]),
        "end_to_end_traceability": bool(
            trace_report["ok"] and trace_report["stage_count"] == 8
        ),
    }
    implementation_ready = bool(registry_report["ok"] and all(controls.values()))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": implementation_ready,
        "overall_status": "ready" if implementation_ready else "blocked",
        "implementation_status": "implementation_ready"
        if implementation_ready
        else "implementation_blocked",
        "grade": "A+" if implementation_ready else "F",
        "grade_scope": "local structural implementation only",
        "reference_count": registry_report["reference_count"],
        "control_count": len(controls),
        "ready_control_count": sum(1 for ready in controls.values() if ready),
        "controls": controls,
        "registry_validation": registry_report,
        "broker_conformance": broker_report,
        "strategy_validity": validity_report,
        "event_time": event_report,
        "paper_live_equivalence": equivalence_report,
        "execution_scenarios": scenario_report,
        "causal_trace": trace_report,
        "soak_acceptance": {
            "classification": "additive_production_hardening",
            "reset_soak_clock": False,
            "preserve_prior_runtime_segments": True,
            "new_behavior_requires_post_change_observation": True,
        },
        "evidence_semantics": {
            "implementation_ready_is_not_profitability_evidence": True,
            "implementation_ready_is_not_live_promotion_ready": True,
            "observed_paper_live_pairs_still_required": True,
            "candidate_bound_forward_post_cost_runtime_still_required": True,
        },
        "live_execution_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the 20-system, eight-control production hardening contract."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(root, config_path=Path(args.config).expanduser())
    out = Path(args.out_file).expanduser()
    if not out.is_absolute():
        out = root / out
    safe_write_json_atomic(
        str(out),
        payload,
        project_root=str(root),
        source="authoritative_systems_control",
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"authoritative_systems status={payload['overall_status']} "
            f"references={payload['reference_count']} controls={payload['ready_control_count']}/{payload['control_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
