from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.ops import investor_readiness_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _config(project_root: Path) -> Path:
    payload = json.loads(src.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    path = project_root / "config" / "investor_readiness_v1.json"
    _write_json(path, payload)
    return path


def _seed_sources(project_root: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["source_artifacts"]
    payloads = {
        "paper_performance": {
            "ok": True,
            "profitability_evidence_window": {"candidate_id": "candidate-1", "candidate_cutoff_utc": "2026-01-01T00:00:00+00:00"},
            "post_cost_expectancy": {
                "available": False,
                "evidence_sufficient": False,
                "positive_clustered_lower_confidence_bound_95": False,
                "promotion_blockers": ["no_post_cost_observations"],
            },
            "accounting_views": {
                "candidate_forward_flow": {"sample_count": 0, "observed_days": 0, "post_cost_pnl_delta_total": 0.0},
                "current_day_flow": {"sample_count": 0, "post_cost_pnl_delta_total": 0.0},
                "lifetime_flow": {"sample_count": 999, "post_cost_pnl_delta_total": 1234.0},
                "active_book_snapshot": {"ending_net_pnl_total": -50.0, "candidate_grade_eligible": False},
            },
            "sleeve_latest": {"lifetime_winner": {"net_pnl": 999999.0}},
        },
        "profitability_independent_validator": {
            "implementation_ready": True,
            "evidence_ready": False,
            "blockers": ["candidate_bound_post_cost_rows_pending"],
            "risk_of_ruin": {"available": False, "passes": False, "blockers": ["minimum_independent_days_pending"]},
        },
        "profitability_evidence_firewall": {
            "control_implementation_ready": True,
            "allocation_proposal": {"ready": False, "qualified_sleeves": [], "qualified_sleeve_count": 0},
        },
        "multiple_testing_guard": {
            "contract_present": True,
            "statistical_evidence_ready": False,
            "statistical_evidence_blockers": ["actual_sleeve_p_values_pending"],
            "family_size": 10,
        },
        "portfolio_capacity_curve": {"summary": {"curve_count": 0, "allocator_ready": False}, "curves": []},
        "live_transition_integrity": {
            "overall_status": "ready_locked",
            "current_canary_stage": {
                "automatic_scaling_allowed": False,
                "operator_release_required_for_each_stage": True,
                "stages": [{"stage": "micro", "max_weight": 0.0025, "required_clean_windows": 1}],
            },
            "release_interlock": {"checks": {"drawdown_limit_breached": False}},
            "live_execution_authority": False,
        },
        "live_canary_control": {"overall_status": "blocked", "supervised_canary_ready": False, "blocking_reasons": ["evidence_pending"]},
        "continuous_soak_integrity": {
            "clean_720_hours_complete": False,
            "main_soak_elapsed_hours": 318.0,
            "main_soak_elapsed_days": 13.25,
            "main_soak_progress_percent": 44.167,
            "main_soak_includes_pre_reset_time": True,
            "main_soak_count_is_promotion_credit": False,
            "clean_window_elapsed_hours": 12.0,
            "observed_window_elapsed_hours": 14.5,
            "historical_soak_evidence": {
                "historical_segmented_wall_clock_hours": 318.0,
                "historical_segmented_wall_clock_days": 13.25,
                "segment_count": 53,
                "counts_toward_current_clean_720_hours": False,
            },
        },
        "production_resilience": {"implementation_ready": True, "paper_soak_ready": False, "paper_blockers": ["runtime_pending"]},
        "commercial_readiness": {"overall_status": "ready", "commercial_product_mode": "personal_only"},
        "immutable_experiment_ledger": {
            "append_only_ready": True,
            "latest_signature_ready": True,
            "latest_exact_replay_ready": False,
            "ledger_row_count": 4,
        },
        "independent_fill_acquisition": {"overall_status": "waiting_for_source", "candidate_eligible_ledger_records": 0},
        "broker_shared_truth": {"broker": "schwab", "fetched": True},
        "live_order_ledger": {"overall_status": "ready", "ok": True, "live_execution_authority": False},
        "system_role_contract": {"overall_status": "ready", "ok": True},
        "independent_runtime_monitor": {"overall_status": "degraded", "local_monitor_ready": True, "production_monitor_ready": False},
    }
    for source_id, payload in payloads.items():
        _write_json(project_root / sources[source_id], payload)


def _payload(tmp_path: Path) -> tuple[Path, Path, dict]:
    project_root = tmp_path / "project"
    config_path = _config(project_root)
    _seed_sources(project_root, config_path)
    return project_root, config_path, src.build_payload(project_root, config_path=config_path)


def _by_id(payload: dict) -> dict[str, dict]:
    return {row["control_id"]: row for row in payload["controls"]}


def test_manifest_defines_exactly_twenty_unique_fail_closed_controls() -> None:
    config = json.loads(src.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    ids = [row["control_id"] for row in config["controls"]]

    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert config["safety_contract"]["live_execution_authority"] is False
    assert config["safety_contract"]["automatic_allocation"] is False
    assert config["safety_contract"]["automatic_canary_scaling"] is False
    assert config["safety_contract"]["profitability_guaranteed"] is False


def test_evidence_debt_and_external_reviews_never_become_fake_implementation_failures(tmp_path: Path) -> None:
    _, _, payload = _payload(tmp_path)
    controls = _by_id(payload)

    assert payload["control_count"] == 20
    assert payload["status_counts"][src.STATUS_IMPLEMENTATION_GAP] == 0
    assert controls["i02_net_of_all_costs"]["status"] == src.STATUS_EVIDENCE_PENDING
    assert controls["i10_commercial_defensibility"]["status"] == src.STATUS_EXTERNAL_REQUIRED
    assert controls["i08_bounded_automation"]["status"] == src.STATUS_READY
    assert payload["evidence_facet_counts"]["implementation_ready"] == 20
    assert payload["evidence_facet_counts"]["external_evidence_pending"] >= 4
    assert payload["readiness_percentage_published"] is False
    assert payload["readiness_percentage"] is None
    assert payload["safety_contract"]["live_execution_authority"] is False
    soak = controls["r04_complete_soak_before_canary"]
    assert soak["status"] == src.STATUS_EVIDENCE_PENDING
    assert soak["evidence"]["main_soak_elapsed_hours"] == 318.0
    assert soak["evidence"]["main_soak_progress_percent"] == 44.167
    assert soak["evidence"]["main_soak_includes_pre_reset_time"] is True
    assert soak["evidence"]["main_soak_count_is_promotion_credit"] is False
    assert soak["evidence"]["historical_segmented_wall_clock_hours"] == 318.0
    assert soak["evidence"]["historical_segment_count"] == 53
    assert soak["evidence"]["historical_counts_toward_clean_720_hours"] is False


def test_shortlist_uses_only_firewall_qualified_sleeves_and_caps_at_three(tmp_path: Path) -> None:
    project_root, config_path, _ = _payload(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    firewall_path = project_root / config["source_artifacts"]["profitability_evidence_firewall"]
    firewall = json.loads(firewall_path.read_text(encoding="utf-8"))
    firewall["allocation_proposal"] = {
        "ready": True,
        "qualified_sleeves": ["dividend", "bond", "pairs", "not_selected_fourth"],
        "qualified_sleeve_count": 4,
    }
    _write_json(firewall_path, firewall)

    payload = src.build_payload(project_root, config_path=config_path)
    shortlisted = [row["sleeve"] for row in payload["shortlisted_sleeves"]]

    assert shortlisted == ["dividend", "bond", "pairs"]
    assert "lifetime_winner" not in shortlisted
    assert _by_id(payload)["r01_shortlist_strong_sleeves"]["status"] == src.STATUS_READY


def test_external_attestation_requires_real_document_and_matching_hash(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    document = project_root / "reviews" / "accounting.pdf"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_bytes(b"independent review")
    digest = hashlib.sha256(document.read_bytes()).hexdigest()
    attestation_path = project_root / "governance" / "investor" / "evidence" / "review.json"
    contract = {
        "ready_statuses": ["approved", "complete"],
        "independent_must_be_true": True,
        "required_fields": ["provider", "signed_by", "signed_at_utc", "document_path", "document_sha256"],
    }
    _write_json(
        attestation_path,
        {
            "status": "approved",
            "independent": True,
            "provider": "Outside CPA LLC",
            "signed_by": "Reviewer Name",
            "signed_at_utc": "2026-08-18T12:00:00+00:00",
            "document_path": "reviews/accounting.pdf",
            "document_sha256": digest,
        },
    )

    valid = src.validate_external_attestation(project_root, attestation_path, contract)
    assert valid["valid"] is True
    assert valid["document_sha256_matches"] is True

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["document_sha256"] = "0" * 64
    _write_json(attestation_path, attestation)
    invalid = src.validate_external_attestation(project_root, attestation_path, contract)
    assert invalid["valid"] is False
    assert "attested_document_hash_mismatch" in invalid["blockers"]


def test_tear_sheet_labels_paper_truth_and_keeps_accounting_scopes_separate(tmp_path: Path) -> None:
    project_root, config_path, payload = _payload(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paper = json.loads((project_root / config["source_artifacts"]["paper_performance"]).read_text(encoding="utf-8"))

    rendered = src.render_tear_sheet(payload, paper)

    assert "PAPER / HYPOTHETICAL - NOT LIVE PERFORMANCE" in rendered
    assert "Candidate-Forward Promotion Scope" in rendered
    assert "Separate Non-Promotion Scopes" in rendered
    assert "Lifetime history" in rendered
    assert "promise of future results" in rendered


def test_two_hundred_dollar_canary_is_validation_only_and_deposits_do_not_auto_scale(tmp_path: Path) -> None:
    _, _, payload = _payload(tmp_path)
    contract = payload["capital_scaling_contract"]

    assert contract["initial_canary_budget_usd"] == 200
    assert contract["initial_canary_purpose"] == "execution_validation_not_income_target"
    assert contract["future_deposits_may_increase_account_equity"] is True
    assert contract["account_growth_alone_may_increase_weight"] is False
    assert contract["automatic_scaling_allowed"] is False
    assert _by_id(payload)["r06_predetermined_scaling_gates"]["status"] == src.STATUS_READY
