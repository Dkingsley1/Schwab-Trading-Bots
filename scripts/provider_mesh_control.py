#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "provider_mesh_latest.json"
CAPABILITY_CONFIG_PATH = PROJECT_ROOT / "config" / "collector_capability_catalog_v1.json"
CAPABILITY_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "collector_capability_control_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _collector_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _parse_iso_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _twelve_data_cooldown(payload: dict[str, Any]) -> dict[str, Any]:
    active = False
    cooldown_until = _parse_iso_ts(payload.get("cooldown_until_utc"))
    now = datetime.now(timezone.utc)
    if cooldown_until is not None and cooldown_until > now:
        active = True
    return {
        "active": bool(active),
        "kind": str(payload.get("kind") or ""),
        "symbol": str(payload.get("symbol") or ""),
        "cooldown_until_utc": payload.get("cooldown_until_utc"),
        "remaining_seconds": max(int((cooldown_until - now).total_seconds()), 0) if cooldown_until is not None and active else 0,
        "failure_count": int(payload.get("failure_count", 0) or 0),
    }


def _group_status(*, total: int, contract_ok: int, snapshot_ready: int, degraded_ok: bool = False) -> str:
    if total <= 0:
        return "missing"
    if contract_ok >= total and snapshot_ready >= total:
        return "ready"
    if degraded_ok and snapshot_ready >= total:
        return "degraded"
    if contract_ok <= 0 and snapshot_ready <= 0:
        return "blocked"
    return "degraded"


def _optional_mesh_is_advisory(
    *,
    optional_total: int,
    optional_snapshot_ready: int,
    soft_failure_count: int,
    required_snapshot_ready: int,
) -> bool:
    return bool(
        optional_total > 0
        and soft_failure_count > 0
        and optional_snapshot_ready > 0
        and required_snapshot_ready > 0
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    collector_contracts = _load_json(health_root / "collector_contracts_latest.json")
    source_verification = _load_json(health_root / "source_verification_latest.json")
    fx_guard = _load_json(health_root / "fx_twelve_data_guard_latest.json")
    capability_configured = (project_root / "config" / CAPABILITY_CONFIG_PATH.name).is_file()
    capability_health = _load_json(health_root / CAPABILITY_HEALTH_PATH.name) if capability_configured else {}

    rows = _collector_rows(collector_contracts)
    required_rows = [row for row in rows if bool(row.get("required", False))]
    evidence_rows = [row for row in rows if str(row.get("collector_class") or "") == "evidence_accrual"]
    optional_rows = [
        row
        for row in rows
        if not bool(row.get("required", False)) and str(row.get("collector_class") or "") != "evidence_accrual"
    ]

    required_contract_ok = sum(1 for row in required_rows if bool(row.get("contract_ok", False)))
    required_snapshot_ready = sum(
        1
        for row in required_rows
        if bool(row.get("payload_present", False)) and int(row.get("payload_size_bytes", 0) or 0) > 0
    )
    optional_contract_ok = sum(1 for row in optional_rows if bool(row.get("contract_ok", False)))
    optional_snapshot_ready = sum(
        1
        for row in optional_rows
        if bool(row.get("payload_present", False)) and int(row.get("payload_size_bytes", 0) or 0) > 0
    )
    organic_readiness = (
        collector_contracts.get("organic_readiness")
        if isinstance(collector_contracts.get("organic_readiness"), dict)
        else {}
    )
    organic_ready_count = int(organic_readiness.get("ready_collector_count", 0) or 0)
    organic_collector_count = int(organic_readiness.get("collector_count", 0) or 0)
    organic_status = str(organic_readiness.get("status") or "missing")

    source_overall = source_verification.get("overall") if isinstance(source_verification.get("overall"), dict) else {}
    source_counts = source_overall.get("counts") if isinstance(source_overall.get("counts"), dict) else {}
    all_verified = bool(source_overall.get("all_verified", False))
    all_cross_verified = bool(source_overall.get("all_cross_verified", False))
    source_runtime_contract = (
        source_verification.get("source_runtime_contract")
        if isinstance(source_verification.get("source_runtime_contract"), dict)
        else {}
    )
    decision_critical_sources_ready = bool(
        source_runtime_contract.get("decision_critical_sources_ready", all_verified)
    )
    decision_critical_blockers = list(source_runtime_contract.get("decision_critical_blockers") or [])
    decision_context_debt = list(source_runtime_contract.get("decision_context_debt") or [])
    optional_enrichment_debt = list(source_runtime_contract.get("optional_enrichment_debt") or [])

    cooldown = _twelve_data_cooldown(fx_guard)

    required_status = _group_status(
        total=len(required_rows),
        contract_ok=required_contract_ok,
        snapshot_ready=required_snapshot_ready,
    )
    verification_status = (
        "ready"
        if decision_critical_sources_ready
        else ("degraded" if bool(source_overall) else "missing")
    )
    verification_depth_status = "cross_verified" if all_cross_verified else "single_source_verified"
    soft_failure_count = int(collector_contracts.get("soft_failure_count", 0) or 0)
    required_failures = list(collector_contracts.get("required_failures") or [])
    required_ready = bool(required_status == "ready" and not required_failures)
    quota_status = "ready"
    if cooldown["active"]:
        quota_status = "advisory" if required_ready else ("degraded" if required_snapshot_ready > 0 else "blocked")
    elif soft_failure_count > 0:
        quota_status = "advisory"
    optional_advisory = _optional_mesh_is_advisory(
        optional_total=len(optional_rows),
        optional_snapshot_ready=optional_snapshot_ready,
        soft_failure_count=soft_failure_count,
        required_snapshot_ready=required_snapshot_ready,
    )
    capability_summary = (
        capability_health.get("summary") if isinstance(capability_health.get("summary"), dict) else {}
    )
    capability_authority = (
        capability_health.get("authority_contract")
        if isinstance(capability_health.get("authority_contract"), dict)
        else {}
    )
    capability_routing = (
        capability_health.get("ingestion_routing_contract")
        if isinstance(capability_health.get("ingestion_routing_contract"), dict)
        else {}
    )
    capability_ingestion_authority = (
        capability_health.get("ingestion_authority_contract")
        if isinstance(capability_health.get("ingestion_authority_contract"), dict)
        else {}
    )
    capability_transport = (
        capability_routing.get("transport_contract")
        if isinstance(capability_routing.get("transport_contract"), dict)
        else {}
    )
    capability_routing_v2_ready = bool(
        int(capability_health.get("schema_version", 1) or 1) < 2
        or (
            capability_routing.get("policy_id")
            and capability_routing.get("decision_stage") == "02_data_qualification"
            and int(capability_routing.get("runtime_route_count", 0) or 0) > 0
            and capability_routing.get("routing_artifact_receipt_sha256")
            and capability_transport
            and all(bool(value) for value in capability_transport.values())
            and not any(
                bool(value) for value in capability_ingestion_authority.values()
            )
        )
    )
    capability_structural_ready = bool(
        not capability_configured
        or (
            capability_health
            and capability_health.get("ok") is True
            and bool((capability_health.get("current_collector_mapping") or {}).get("complete", False))
            and float(capability_summary.get("bot_binding_coverage_ratio", 0.0) or 0.0) >= 1.0
            and not any(bool(value) for value in capability_authority.values())
            and capability_routing_v2_ready
        )
    )
    capability_paper_soak_ready = bool(
        not capability_configured
        or (capability_structural_ready and capability_health.get("paper_soak_ready") is True)
    )
    paper_context_ready = bool(
        required_ready and decision_critical_sources_ready and capability_paper_soak_ready
    )

    overall_status = "ready"
    if capability_configured and not capability_structural_ready:
        overall_status = "blocked"
    elif required_status == "blocked":
        overall_status = "blocked"
    elif required_status in {"degraded", "missing"} or required_failures or not capability_paper_soak_ready:
        overall_status = "degraded"
    elif verification_status in {"degraded", "missing"} and not optional_advisory:
        overall_status = "degraded"

    recommended_actions = _ordered_unique(
        [
            "treat provider cooldowns as mesh-level state and serve last-good snapshots until the provider recovers" if cooldown["active"] else "",
            "raise required collector snapshot coverage so required lanes keep a usable last-good state during provider outages" if required_snapshot_ready < len(required_rows) else "",
            "repair required collector failures before trusting live context-driven decisions" if collector_contracts.get("required_failures") else "",
            "cross-verify more sources to raise optional verification depth from ready to A+"
            if all_verified and not all_cross_verified and bool(source_overall)
            else "",
            "keep optional collectors on a degraded path instead of letting them block the required context mesh" if soft_failure_count > 0 else "",
            "continue bounded point-in-time, lineage, and candidate-fill collection until every organic evidence target is met"
            if organic_collector_count > 0 and organic_status != "ready"
            else "",
            "refresh collector capability routing after collector contracts and bot hierarchy change"
            if capability_configured and not capability_structural_ready
            else "",
            "repair required collector failures before the capability router can clear guarded paper-soak readiness"
            if capability_configured and capability_structural_ready and not capability_paper_soak_ready
            else "",
        ]
    )

    average_quality_score = float(collector_contracts.get("average_quality_score", 0.0) or 0.0)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "summary": {
            "collector_count": len(rows),
            "required_collectors": len(required_rows),
            "optional_collectors": len(optional_rows),
            "evidence_collectors": len(evidence_rows),
            "required_contract_ok": required_contract_ok,
            "required_snapshot_ready": required_snapshot_ready,
            "optional_contract_ok": optional_contract_ok,
            "optional_snapshot_ready": optional_snapshot_ready,
            "average_quality_score": round(average_quality_score, 6),
            "soft_failure_count": soft_failure_count,
            "required_failure_count": int(collector_contracts.get("required_failure_count", 0) or 0),
            "organic_readiness_score": float(organic_readiness.get("score", 0.0) or 0.0),
            "organic_ready_collectors": organic_ready_count,
            "organic_collector_count": organic_collector_count,
            "capability_plane_count": int(capability_summary.get("plane_count", 0) or 0),
            "capability_count": int(capability_summary.get("capability_count", 0) or 0),
            "capability_bot_binding_count": int(capability_summary.get("bot_binding_count", 0) or 0),
            "capability_subscription_profile_count": int(
                capability_summary.get("subscription_profile_count", 0) or 0
            ),
            "capability_runtime_route_count": int(
                capability_routing.get("runtime_route_count", 0) or 0
            ),
            "capability_runtime_paper_ready_route_count": int(
                capability_routing.get("runtime_paper_ready_route_count", 0) or 0
            ),
            "capability_runtime_live_ready_route_count": int(
                capability_routing.get("runtime_live_ready_route_count", 0) or 0
            ),
        },
        "continuity_contract": {
            "ready": paper_context_ready,
            "required_context_usable": required_ready,
            "decision_critical_sources_ready": decision_critical_sources_ready,
            "collector_capability_paper_soak_ready": capability_paper_soak_ready,
            "serving_last_good_during_cooldown": bool(cooldown["active"] and required_ready),
            "cooldown_isolated_from_required_context": bool(not cooldown["active"] or required_ready),
            "policy": "provider cooldowns stay advisory only while every required contract and snapshot remains usable",
        },
        "provider_groups": {
            "required_context": {
                "status": required_status,
                "summary": f"contract_ok={required_contract_ok}/{len(required_rows)} snapshot_ready={required_snapshot_ready}/{len(required_rows)}",
                "collectors": [str(row.get("name") or "") for row in required_rows],
            },
            "optional_context": {
                "status": (
                    "advisory"
                    if optional_advisory
                    else _group_status(
                        total=len(optional_rows),
                        contract_ok=optional_contract_ok,
                        snapshot_ready=optional_snapshot_ready,
                        degraded_ok=True,
                    )
                ),
                "summary": f"contract_ok={optional_contract_ok}/{len(optional_rows)} snapshot_ready={optional_snapshot_ready}/{len(optional_rows)}",
                "collectors": [str(row.get("name") or "") for row in optional_rows],
            },
            "organic_evidence_accrual": {
                "status": organic_status,
                "summary": f"ready={organic_ready_count}/{organic_collector_count} score={float(organic_readiness.get('score', 0.0) or 0.0):.3f}",
                "collectors": [str(row.get("name") or "") for row in evidence_rows],
                "pending_collectors": list(organic_readiness.get("pending_collectors") or []),
                "blocks_paper_soak": False,
                "blocks_live_promotion_until_ready": organic_status != "ready",
            },
            "verification_mesh": {
                "status": "advisory" if verification_status == "degraded" and optional_advisory else verification_status,
                "depth_status": verification_depth_status,
                "summary": (
                    f"cross_verified={int(source_counts.get('cross_verified', 0) or 0)} "
                    f"single_verified={int(source_counts.get('single_verified', source_counts.get('single_source_verified', 0)) or 0)} "
                    f"unverified={int(source_counts.get('single_unverified', source_counts.get('single_source_unverified', 0)) or 0)}"
                ),
                "all_verified": all_verified,
                "all_cross_verified": all_cross_verified,
                "decision_critical_sources_ready": decision_critical_sources_ready,
                "decision_critical_blockers": decision_critical_blockers,
                "decision_context_debt": decision_context_debt,
                "optional_enrichment_debt": optional_enrichment_debt,
                "context_debt_blocks_guarded_paper_soak": False,
            },
            "quota_limited_providers": {
                "status": quota_status,
                "summary": (
                    f"cooldowns_active={int(cooldown['active'])} "
                    f"soft_failures={soft_failure_count}"
                ),
                "active_cooldowns": [cooldown] if cooldown["active"] else [],
            },
            "collector_capability_routing": {
                "status": (
                    "legacy_not_configured"
                    if not capability_configured
                    else (
                        "ready_with_coverage_debt"
                        if capability_structural_ready and capability_paper_soak_ready
                        else ("degraded" if capability_structural_ready else "blocked")
                    )
                ),
                "configured": capability_configured,
                "structural_ready": capability_structural_ready,
                "paper_soak_ready": capability_paper_soak_ready,
                "live_promotion_ready": bool(capability_health.get("live_promotion_ready", False)),
                "planes": int(capability_summary.get("plane_count", 0) or 0),
                "capabilities": int(capability_summary.get("capability_count", 0) or 0),
                "bot_bindings": int(capability_summary.get("bot_binding_count", 0) or 0),
                "subscription_profiles": int(capability_summary.get("subscription_profile_count", 0) or 0),
                "ingestion_route_profiles": int(
                    capability_summary.get("ingestion_route_profile_count", 0)
                    or 0
                ),
                "routing_policy": str(capability_routing.get("policy_id") or ""),
                "decision_stage": str(
                    capability_routing.get("decision_stage") or ""
                ),
                "decision_families": int(
                    capability_routing.get("decision_family_count", 0) or 0
                ),
                "runtime_routes": int(
                    capability_routing.get("runtime_route_count", 0) or 0
                ),
                "runtime_paper_ready_routes": int(
                    capability_routing.get("runtime_paper_ready_route_count", 0)
                    or 0
                ),
                "runtime_live_ready_routes": int(
                    capability_routing.get("runtime_live_ready_route_count", 0)
                    or 0
                ),
                "average_route_quality": float(
                    capability_routing.get("average_profile_route_quality", 0.0)
                    or 0.0
                ),
                "independent_redundancy_ratio": float(
                    capability_summary.get(
                        "required_capability_independent_redundancy_ratio", 0.0
                    )
                    or 0.0
                ),
                "transport_contract_complete": bool(
                    capability_transport
                    and all(bool(value) for value in capability_transport.values())
                ),
                "routing_contract_ready": capability_routing_v2_ready,
                "routing_receipt_sha256": str(
                    capability_routing.get("routing_artifact_receipt_sha256")
                    or ""
                ),
                "unsupported_capabilities_are_live_promotion_debt": True,
                "blocks_healthy_guarded_paper_soak": bool(
                    capability_configured and not capability_paper_soak_ready
                ),
            },
        },
        "mesh_contracts": [
            {
                "name": str(row.get("name") or ""),
                "required": bool(row.get("required", False)),
                "safe_to_degrade": bool(row.get("safe_to_degrade", False)),
                "contract_ok": bool(row.get("contract_ok", False)),
                "payload_present": bool(row.get("payload_present", False)),
                "payload_size_bytes": int(row.get("payload_size_bytes", 0) or 0),
                "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                "collector_class": str(row.get("collector_class") or "core_context"),
                "evidence_domains": list(row.get("evidence_domains") or []),
                "organic_readiness": dict(row.get("organic_readiness") or {}),
                "authority_contract": dict(row.get("authority_contract") or {}),
            }
            for row in rows
        ],
        "organic_readiness": organic_readiness,
        "collector_expansion_contract": dict(collector_contracts.get("collector_expansion_contract") or {}),
        "authority_contract": {
            "observation_only": True,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
            "organic_readiness_may_block_live_promotion": True,
            "organic_readiness_may_not_block_healthy_paper_collection": True,
            "capability_router_changes_runtime_decisions": False,
            "capability_router_launches_collectors": False,
            "capability_router_paper_execution_authority": False,
            "capability_router_live_execution_authority": False,
        },
        "cooldowns": [cooldown] if cooldown["active"] else [],
        "required_failures": required_failures,
        "soft_failures": collector_contracts.get("soft_failures", []),
        "advisories": _ordered_unique(
            [
                "optional_context_soft_failures" if optional_advisory and soft_failure_count > 0 else "",
                "verification_depth_soft_debt" if verification_status == "degraded" and optional_advisory else "",
                "decision_context_source_debt" if decision_context_debt else "",
                "provider_cooldown_serving_last_good" if cooldown["active"] and required_ready else "",
                "organic_evidence_still_accumulating" if organic_collector_count > 0 and organic_status != "ready" else "",
                "capability_coverage_debt_is_live_promotion_only"
                if capability_configured
                and capability_structural_ready
                and bool((capability_health.get("coverage_debt") or {}).get("gap_count", 0))
                else "",
            ]
        ),
        "recommended_actions": recommended_actions,
        "top_actions": recommended_actions[:4],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish provider-mesh readiness across required collectors, verification, and cooldown state.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    _atomic_write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "provider_mesh "
            f"overall_status={payload.get('overall_status', '')} "
            f"required_contract_ok={int(((payload.get('summary') or {}).get('required_contract_ok', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
