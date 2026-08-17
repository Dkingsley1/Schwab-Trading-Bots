import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.provider_mesh_control as src  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_provider_mesh_control_tracks_required_collectors_and_cooldowns(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.82,
            "required_failure_count": 0,
            "soft_failure_count": 1,
            "required_failures": [],
            "soft_failures": ["fx_market_context"],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.91,
                },
                {
                    "name": "fx_market_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 256,
                    "quality_score": 0.66,
                },
                {
                    "name": "sec_edgar_context",
                    "required": False,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.84,
                },
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": True,
                "all_cross_verified": False,
                "counts": {
                    "cross_verified": 2,
                    "single_verified": 1,
                    "single_unverified": 0,
                },
            }
        },
    )
    _write_json(
        health / "fx_twelve_data_guard_latest.json",
        {
            "kind": "daily_quota",
            "symbol": "EURUSD",
            "cooldown_until_utc": "2099-01-01T00:05:00+00:00",
            "failure_count": 3,
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["provider_groups"]["required_context"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["depth_status"] == "single_source_verified"
    assert payload["provider_groups"]["quota_limited_providers"]["status"] == "advisory"
    assert payload["continuity_contract"]["serving_last_good_during_cooldown"] is True
    assert "provider_cooldown_serving_last_good" in payload["advisories"]
    assert payload["cooldowns"][0]["active"] is True
    assert "treat provider cooldowns as mesh-level state and serve last-good snapshots until the provider recovers" in payload["recommended_actions"]


def test_provider_mesh_consumes_configured_capability_routing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(project_root / "config" / "collector_capability_catalog_v1.json", {"schema_version": 1})
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failure_count": 0,
            "soft_failure_count": 0,
            "required_failures": [],
            "soft_failures": [],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                }
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {"overall": {"all_verified": True, "all_cross_verified": True, "counts": {}}},
    )
    _write_json(
        health / "collector_capability_control_latest.json",
        {
            "ok": True,
            "paper_soak_ready": True,
            "live_promotion_ready": False,
            "summary": {
                "plane_count": 25,
                "capability_count": 257,
                "bot_binding_count": 100,
                "bot_binding_coverage_ratio": 1.0,
                "subscription_profile_count": 12,
            },
            "current_collector_mapping": {"complete": True},
            "coverage_debt": {"gap_count": 10},
            "authority_contract": {"paper_execution_authority": False, "live_execution_authority": False},
        },
    )

    payload = src.build_payload(project_root)

    group = payload["provider_groups"]["collector_capability_routing"]
    assert payload["overall_status"] == "ready"
    assert group["status"] == "ready_with_coverage_debt"
    assert group["paper_soak_ready"] is True
    assert group["live_promotion_ready"] is False
    assert group["bot_bindings"] == 100
    assert "capability_coverage_debt_is_live_promotion_only" in payload["advisories"]


def test_provider_mesh_keeps_guarded_paper_context_ready_with_context_only_source_debt(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failure_count": 0,
            "soft_failure_count": 0,
            "required_failures": [],
            "soft_failures": [],
            "rows": [
                {
                    "name": "market_micro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                }
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": False,
                "all_cross_verified": False,
                "counts": {"single_source_verified": 2, "single_source_unverified": 1},
            },
            "source_runtime_contract": {
                "decision_critical_sources_ready": True,
                "decision_critical_blockers": [],
                "decision_context_debt": ["schwab_symbol_news"],
                "optional_enrichment_debt": [],
            },
        },
    )

    payload = src.build_payload(project_root)

    verification = payload["provider_groups"]["verification_mesh"]
    assert payload["overall_status"] == "ready"
    assert payload["continuity_contract"]["ready"] is True
    assert verification["status"] == "ready"
    assert verification["all_verified"] is False
    assert verification["decision_context_debt"] == ["schwab_symbol_news"]
    assert verification["context_debt_blocks_guarded_paper_soak"] is False
    assert "decision_context_source_debt" in payload["advisories"]


def test_provider_mesh_cooldown_still_degrades_when_required_snapshot_is_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failure_count": 1,
            "soft_failure_count": 0,
            "required_failures": ["fx_market_context"],
            "soft_failures": [],
            "rows": [
                {
                    "name": "fx_market_context",
                    "required": True,
                    "contract_ok": False,
                    "payload_present": False,
                    "payload_size_bytes": 0,
                }
            ],
        },
    )
    _write_json(health / "source_verification_latest.json", {"overall": {"all_verified": True}})
    _write_json(
        health / "fx_twelve_data_guard_latest.json",
        {"kind": "auth", "symbol": "EURUSD", "cooldown_until_utc": "2099-01-01T00:05:00+00:00"},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["continuity_contract"]["ready"] is False
    assert payload["provider_groups"]["quota_limited_providers"]["status"] == "blocked"


def test_provider_mesh_control_ready_when_all_sources_verified_without_cooldowns(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.92,
            "required_failure_count": 0,
            "soft_failure_count": 0,
            "required_failures": [],
            "soft_failures": [],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.97,
                },
                {
                    "name": "market_micro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 256,
                    "quality_score": 0.91,
                },
                {
                    "name": "sec_edgar_context",
                    "required": False,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.84,
                },
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": True,
                "all_cross_verified": False,
                "counts": {
                    "cross_verified": 2,
                    "single_source_verified": 1,
                    "single_source_unverified": 0,
                },
            }
        },
    )
    _write_json(health / "fx_twelve_data_guard_latest.json", {})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["depth_status"] == "single_source_verified"
    assert "single_verified=1" in payload["provider_groups"]["verification_mesh"]["summary"]
    assert "cross-verify more sources to raise optional verification depth from ready to A+" in payload["recommended_actions"]


def test_provider_mesh_control_keeps_required_mesh_ready_with_optional_soft_debt(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.81,
            "required_failure_count": 0,
            "soft_failure_count": 2,
            "required_failures": [],
            "soft_failures": ["sec_edgar_context", "options_flow_context"],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.97,
                },
                {
                    "name": "market_micro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 256,
                    "quality_score": 0.91,
                },
                {
                    "name": "sec_edgar_context",
                    "required": False,
                    "safe_to_degrade": True,
                    "contract_ok": False,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.52,
                },
                {
                    "name": "options_flow_context",
                    "required": False,
                    "safe_to_degrade": True,
                    "contract_ok": False,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.48,
                },
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": False,
                "all_cross_verified": False,
                "counts": {
                    "cross_verified": 2,
                    "single_source_verified": 2,
                    "single_source_unverified": 2,
                },
            }
        },
    )
    _write_json(health / "fx_twelve_data_guard_latest.json", {})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["provider_groups"]["required_context"]["status"] == "ready"
    assert payload["provider_groups"]["optional_context"]["status"] == "advisory"
    assert payload["provider_groups"]["verification_mesh"]["status"] == "advisory"
    assert payload["provider_groups"]["quota_limited_providers"]["status"] == "advisory"
    assert "optional_context_soft_failures" in payload["advisories"]


def test_provider_mesh_surfaces_organic_evidence_without_blocking_paper_context(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    pending = {
        "name": "candidate_fill_replay",
        "required": False,
        "collector_class": "evidence_accrual",
        "contract_ok": True,
        "payload_present": True,
        "payload_size_bytes": 128,
        "quality_score": 0.9,
        "evidence_domains": ["profitability_research", "promotion_release"],
        "organic_readiness": {"required": True, "ready": False, "status": "accumulating", "progress": 0.25},
        "authority_contract": {"live_execution_authority": False},
    }
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.9,
            "required_failure_count": 0,
            "soft_failure_count": 0,
            "required_failures": [],
            "soft_failures": [],
            "organic_readiness": {
                "status": "accumulating",
                "score": 25.0,
                "ready_collector_count": 0,
                "collector_count": 1,
                "pending_collectors": [{"name": "candidate_fill_replay"}],
            },
            "collector_expansion_contract": {"configured_added_collectors": 9},
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.97,
                },
                pending,
            ],
        },
    )
    _write_json(health / "source_verification_latest.json", {"overall": {"all_verified": True}})
    _write_json(health / "fx_twelve_data_guard_latest.json", {})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["provider_groups"]["organic_evidence_accrual"]["status"] == "accumulating"
    assert payload["provider_groups"]["organic_evidence_accrual"]["blocks_paper_soak"] is False
    assert payload["provider_groups"]["organic_evidence_accrual"]["blocks_live_promotion_until_ready"] is True
    assert "organic_evidence_still_accumulating" in payload["advisories"]
    assert payload["authority_contract"]["live_execution_authority"] is False
