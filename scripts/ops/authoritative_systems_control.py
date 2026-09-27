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
from core.archive_snapshot_control import (
    SnapshotCatalog,
    build_manifest,
    build_snapshot,
    plan_compaction,
)
from core.authoritative_systems import (
    EXPECTED_CONTROL_COUNT,
    EXPECTED_REFERENCE_COUNT,
    load_registry,
    validate_registry,
)
from core.brokers.capability_contract import all_adapter_conformance
from core.build_provenance import build_local_provenance, verify_local_provenance
from core.causal_attribution import build_execution_trace, verify_execution_trace
from core.data_quality_checkpoints import run_checkpoint
from core.event_time_control import EventTimeGuard, EventTimePolicy
from core.exchange_sequence_control import ExchangeSequenceGuard
from core.execution_scenarios import run_execution_scenarios
from core.formal_safety_invariants import bounded_verify_order_safety, inspect_tla_spec
from core.independent_risk_oracle import build_risk_request, reconcile_risk_results
from core.institutional_research_extensions import (
    load_policy as load_institutional_extension_policy,
)
from core.institutional_research_extensions import (
    structural_probe as institutional_extension_probe,
)
from core.paper_live_equivalence import compare_pair
from core.portfolio_advisory import build_multi_period_advisory
from core.research_data_platform import load_policy as load_research_data_policy
from core.research_data_platform import structural_probe as research_data_platform_probe
from core.trade_lifecycle import TradeLifecycle
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


def _exchange_sequence_probe() -> dict[str, Any]:
    guard = ExchangeSequenceGuard()
    first = guard.ingest(
        channel="probe", session_id="A", sequence_number=100, payload={"price": 100.0}
    )
    contiguous = guard.ingest(
        channel="probe", session_id="A", sequence_number=101, payload={"price": 100.1}
    )
    duplicate = guard.ingest(
        channel="probe", session_id="A", sequence_number=101, payload={"price": 100.1}
    )
    gap = guard.ingest(
        channel="probe", session_id="A", sequence_number=103, payload={"price": 100.3}
    )
    recovered = guard.ingest(
        channel="probe", session_id="A", sequence_number=102, payload={"price": 100.2}
    )
    reset = guard.ingest(
        channel="probe",
        session_id="B",
        sequence_number=1,
        payload={"session": "B"},
        session_reset=True,
    )
    restored = ExchangeSequenceGuard.restore(guard.snapshot())
    return {
        "ok": bool(
            first["accepted"]
            and contiguous["accepted"]
            and duplicate["disposition"] == "duplicate_idempotent"
            and gap.get("missing_range") == [102, 102]
            and recovered["accepted"]
            and reset["disposition"] == "session_reset"
            and restored.snapshot() == guard.snapshot()
        ),
        "gap": gap,
        "duplicate": duplicate,
        "native_itch_ouch_connected": False,
        "execution_authority": False,
    }


def _archive_snapshot_probe() -> dict[str, Any]:
    entries = [
        {"path": "a.jsonl", "sha256": "a" * 64, "size_bytes": 100, "row_count": 10},
        {"path": "b.jsonl", "sha256": "b" * 64, "size_bytes": 100, "row_count": 10},
    ]
    manifest = build_manifest(entries)
    first = build_snapshot(
        manifest,
        schema_id="probe-v1",
        committed_at_utc="2026-08-21T12:00:00+00:00",
    )
    catalog = SnapshotCatalog()
    first_commit = catalog.commit(first, expected_parent_snapshot_id=None)
    second = build_snapshot(
        manifest,
        schema_id="probe-v1",
        parent_snapshot_id=first["snapshot_id"],
        committed_at_utc="2026-08-21T12:01:00+00:00",
    )
    stale_commit = catalog.commit(second, expected_parent_snapshot_id=None)
    second_commit = catalog.commit(
        second, expected_parent_snapshot_id=first["snapshot_id"]
    )
    compaction = plan_compaction(entries, target_bytes=500)
    return {
        "ok": bool(
            first_commit["committed"]
            and not stale_commit["committed"]
            and second_commit["committed"]
            and compaction["group_count"] == 1
            and compaction["source_delete_authority"] is False
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "snapshot_count": len(catalog.snapshots),
        "stale_commit_rejected": not stale_commit["committed"],
        "compaction": compaction,
        "production_archive_migrated": False,
    }


def _formal_safety_probe(project_root: Path) -> dict[str, Any]:
    bounded = bounded_verify_order_safety()
    specification = inspect_tla_spec(project_root / "formal" / "OrderSafety.tla")
    return {
        "ok": bool(bounded["ok"] and specification["ok"]),
        "bounded_model": bounded,
        "specification": specification,
        "formal_proof_ready": False,
    }


def _build_provenance_probe(project_root: Path) -> dict[str, Any]:
    statement = build_local_provenance(
        project_root=project_root,
        subject_paths=["core/build_provenance.py"],
        material_paths=["config/authoritative_systems_v1.json"],
        git_commit="untrusted-local-structural-probe",
    )
    verification = verify_local_provenance(statement, project_root=project_root)
    return {
        "ok": bool(
            verification["ok"]
            and not verification["signed"]
            and not verification["promotion_evidence_eligible"]
        ),
        "verification": verification,
        "statement_type": statement["_type"],
        "predicate_type": statement["predicateType"],
    }


def _trade_lifecycle_probe() -> dict[str, Any]:
    economics = "a" * 64
    lifecycle = TradeLifecycle("trade-probe", "SPY", economics)

    def event(event_id: str, event_type: str) -> dict[str, str]:
        return {
            "event_id": event_id,
            "event_type": event_type,
            "trade_id": "trade-probe",
            "product_id": "SPY",
            "economics_sha256": economics,
        }

    receipts = [
        lifecycle.apply(event("e1", "execution")),
        lifecycle.apply(event("e2", "confirmation")),
        lifecycle.apply(event("e3", "allocation")),
        lifecycle.apply(event("e4", "settlement")),
    ]
    chain = lifecycle.verify_chain()
    return {
        "ok": bool(
            all(row["accepted"] for row in receipts)
            and chain["ok"]
            and chain["state"] == "SETTLED"
        ),
        "chain": chain,
        "external_cdm_interoperability_observed": False,
        "execution_authority": False,
    }


def _risk_oracle_probe() -> dict[str, Any]:
    request = build_risk_request(
        candidate_id="candidate-probe",
        product_id="SPY",
        valuation_time_utc="2026-08-21T12:00:00+00:00",
        measures=["pv", "delta"],
    )
    primary = {
        "provider_id": "local-probe",
        "model_id": "local-model",
        "request_sha256": request["request_sha256"],
        "values": {"pv": 100.0, "delta": 0.5},
    }
    oracle = {
        "provider_id": "synthetic-external-probe",
        "model_id": "synthetic-oracle-model",
        "request_sha256": request["request_sha256"],
        "values": {"pv": 100.001, "delta": 0.5001},
        "signed_attestation_id": "synthetic-only",
    }
    report = reconcile_risk_results(
        primary,
        oracle,
        absolute_tolerance={"pv": 0.01, "delta": 0.001},
        synthetic_probe=True,
    )
    return {
        "ok": bool(report["ok"] and not report["evidence_eligible"]),
        "reconciliation": report,
        "real_external_observation_ready": False,
    }


def _portfolio_advisory_probe() -> dict[str, Any]:
    names = ("dividend", "bond", "fx", "volatility")
    sleeves = [
        {
            "sleeve_id": name,
            "candidate_id": "candidate-probe",
            "qualified": True,
            "independent_fills": 35,
            "expected_return_bps": edge,
            "cost_bps": 1.0,
        }
        for name, edge in zip(names, (8.0, 6.0, 7.0, 5.0), strict=True)
    ]
    covariance = {
        left: {right: (1.0 if left == right else 0.1) for right in names}
        for left in names
    }
    report = build_multi_period_advisory(
        candidate_id="candidate-probe",
        sleeves=sleeves,
        covariance=covariance,
        current_weights={},
    )
    return {
        "ok": bool(
            report["ok"]
            and report["advisory_only"]
            and not report["execution_authority"]
        ),
        "advisory": report,
        "candidate_runtime_evidence_ready": False,
    }


def _data_quality_probe() -> dict[str, Any]:
    expectations = {
        "required_columns": ["event_id", "event_time_utc", "price"],
        "types": {
            "event_id": "string",
            "event_time_utc": "timestamp",
            "price": "number",
        },
        "ranges": {"price": {"min": 0.01}},
        "unique_columns": ["event_id"],
        "monotonic_columns": ["event_time_utc"],
        "freshness": {"column": "event_time_utc", "max_age_seconds": 120},
        "failure_action": "block",
    }
    valid = run_checkpoint(
        checkpoint_id="probe-valid",
        rows=[
            {
                "event_id": "1",
                "event_time_utc": "2026-08-21T12:00:00+00:00",
                "price": 100.0,
            }
        ],
        expectations=expectations,
        observed_at_utc="2026-08-21T12:00:30+00:00",
    )
    invalid = run_checkpoint(
        checkpoint_id="probe-invalid",
        rows=[
            {
                "event_id": "1",
                "event_time_utc": "2026-08-21T11:00:00+00:00",
                "price": -1.0,
            }
        ],
        expectations=expectations,
        observed_at_utc="2026-08-21T12:00:30+00:00",
    )
    return {
        "ok": bool(
            valid["ok"] and not invalid["ok"] and invalid["status"] == "blocked"
        ),
        "valid_checkpoint": valid,
        "invalid_checkpoint": invalid,
        "great_expectations_runtime_installed": False,
    }


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
    exchange_sequence_report = _exchange_sequence_probe()
    archive_snapshot_report = _archive_snapshot_probe()
    formal_safety_report = _formal_safety_probe(project_root)
    build_provenance_report = _build_provenance_probe(project_root)
    trade_lifecycle_report = _trade_lifecycle_probe()
    risk_oracle_report = _risk_oracle_probe()
    portfolio_advisory_report = _portfolio_advisory_probe()
    data_quality_report = _data_quality_probe()
    research_data_platform_report = research_data_platform_probe(
        load_research_data_policy(project_root / "config/research_data_platform_v1.json")
    )
    institutional_extension_report = institutional_extension_probe(
        load_institutional_extension_policy(
            project_root / "config/institutional_research_extensions_v1.json"
        ),
        project_root=project_root,
    )
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
        "exchange_sequence_integrity": bool(exchange_sequence_report["ok"]),
        "atomic_archive_snapshots": bool(archive_snapshot_report["ok"]),
        "formal_safety_specification": bool(formal_safety_report["ok"]),
        "build_provenance_attestation": bool(build_provenance_report["ok"]),
        "canonical_trade_lifecycle": bool(trade_lifecycle_report["ok"]),
        "independent_pricing_risk_oracle": bool(risk_oracle_report["ok"]),
        "constrained_portfolio_advisory": bool(portfolio_advisory_report["ok"]),
        "declarative_data_quality": bool(data_quality_report["ok"]),
        "research_data_platform_contract": bool(research_data_platform_report["ok"]),
        "institutional_research_extensions_contract": bool(
            institutional_extension_report["ok"]
        ),
    }
    implementation_ready = bool(registry_report["ok"] and all(controls.values()))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": implementation_ready,
        "overall_status": "ready" if implementation_ready else "blocked",
        "implementation_status": (
            "implementation_ready" if implementation_ready else "implementation_blocked"
        ),
        "grade": "A+" if implementation_ready else "F",
        "grade_scope": "local structural implementation only",
        "reference_count": registry_report["reference_count"],
        "reference_target": EXPECTED_REFERENCE_COUNT,
        "control_count": len(controls),
        "control_target": EXPECTED_CONTROL_COUNT,
        "ready_control_count": sum(1 for ready in controls.values() if ready),
        "controls": controls,
        "registry_validation": registry_report,
        "broker_conformance": broker_report,
        "strategy_validity": validity_report,
        "event_time": event_report,
        "paper_live_equivalence": equivalence_report,
        "execution_scenarios": scenario_report,
        "causal_trace": trace_report,
        "deep_influence_probes": {
            "exchange_sequence_integrity": exchange_sequence_report,
            "atomic_archive_snapshots": archive_snapshot_report,
            "formal_safety_specification": formal_safety_report,
            "build_provenance_attestation": build_provenance_report,
            "canonical_trade_lifecycle": trade_lifecycle_report,
            "independent_pricing_risk_oracle": risk_oracle_report,
            "constrained_portfolio_advisory": portfolio_advisory_report,
            "declarative_data_quality": data_quality_report,
            "research_data_platform_contract": research_data_platform_report,
            "institutional_research_extensions_contract": institutional_extension_report,
        },
        "external_evidence": {
            "grade_scope": "external or candidate-bound observations not supplied by structural probes",
            "ready_count": 0,
            "item_count": 10,
            "items": {
                "native_exchange_protocol_observation": False,
                "production_archive_snapshot_observation": False,
                "tlc_model_check_and_independent_review": False,
                "signed_trusted_builder_attestation": False,
                "external_cdm_interoperability_observation": False,
                "signed_independent_risk_oracle_observation": False,
                "candidate_bound_portfolio_advisory_observation": False,
                "great_expectations_runtime_validation": False,
                "research_data_platform_candidate_bound_evidence": False,
                "institutional_research_extensions_candidate_bound_evidence": False,
            },
            "paper_impact": "none",
            "does_not_reduce_structural_grade": True,
        },
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
            "synthetic_probes_are_not_external_evidence": True,
            "unsigned_local_provenance_is_not_trusted_attestation": True,
            "portfolio_advisory_has_no_order_authority": True,
            "public_firm_influence_is_not_proprietary_replication": True,
            "institutional_extension_structure_is_not_candidate_evidence": True,
        },
        "live_execution_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the 39-reference, 18-control production hardening contract."
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
