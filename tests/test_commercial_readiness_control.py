from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import commercial_readiness_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_use_mode_ready(project_root: Path) -> None:
    _write_json(
        project_root / "governance" / "health" / "use_mode_compliance_guard_latest.json",
        {
            "overall_status": "ready",
            "use_mode": "personal",
            "personal_use": {"grade": "A+", "perfect_personal_use_ready": True},
            "commercial_use": {
                "commercial_use_intent_detected": False,
                "commercial_clearance_status": "not_requested_personal_mode",
                "blockers": [],
            },
            "authority_boundaries": {
                "does_not_enable_live_execution": True,
                "live_execution_authority": False,
                "customer_funds_allowed": False,
                "customer_order_execution_allowed": False,
            },
        },
    )


def _seed_security_ready(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    _write_json(health / "security_audit_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "secret_scan_latest.json", {"timestamp_utc": now, "overall_status": "ready", "findings_count": 0})
    _write_json(health / "telemetry_redaction_canary_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})


def _seed_commercial_evidence(project_root: Path) -> None:
    for rel in (
        "governance/commercial/evidence/business_mode_statement.md",
        "governance/commercial/evidence/review_approvals.md",
        "governance/commercial/evidence/performance_methodology.md",
        "governance/commercial/evidence/marketing_claim_register.md",
        "governance/commercial/evidence/customer_funds_custody_attestation.md",
        "governance/commercial/evidence/security_privacy_program.md",
        "governance/commercial/evidence/incident_response_plan.md",
        "governance/commercial/evidence/audit_log_retention_policy.md",
        "governance/commercial/evidence/service_provider_register.md",
        "governance/commercial/evidence/commercial_release_approval.md",
    ):
        _write_text(project_root / rel)


def test_commercial_readiness_defaults_to_personal_only_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_use_mode_ready(project_root)

    payload = src.build_payload(project_root, env={})

    assert payload["overall_status"] == "ready"
    assert payload["commercial_product_mode"] == "personal_only"
    assert payload["commercial_intent"] is False
    assert payload["commercial_release_ready"] is False
    assert payload["section_count"] == 7
    assert payload["seven_section_contract"]["commercial_use_modes"] is True
    assert payload["authority_boundaries"]["live_execution_authority"] is False


def test_commercial_readiness_blocks_paid_signals_without_reviews_or_claim_register(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_use_mode_ready(project_root)

    payload = src.build_payload(
        project_root,
        env={"COMMERCIAL_PRODUCT_MODE": "paid_signals_newsletter", "PAID_SIGNALS_ENABLED": "1"},
    )

    assert payload["overall_status"] == "blocked"
    assert payload["commercial_intent"] is True
    assert "registration_review_gates:investment_adviser_review_not_approved" in payload["blockers"]
    assert "marketing_claim_control:marketing_review_not_approved" in payload["blockers"]
    assert "marketing_claim_control:marketing_claim_register_missing" in payload["blockers"]
    assert payload["authority_boundaries"]["live_execution_authority"] is False


def test_commercial_readiness_hard_blocks_customer_funds_without_program_evidence(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_use_mode_ready(project_root)

    payload = src.build_payload(
        project_root,
        env={
            "CUSTOMER_FUNDS_ENABLED": "1",
            "COMMERCIAL_LEGAL_REVIEW_APPROVED": "1",
            "COMMERCIAL_COMPLIANCE_REVIEW_APPROVED": "1",
            "PRIVACY_SECURITY_REVIEW_APPROVED": "1",
            "TERMS_DISCLOSURE_REVIEW_APPROVED": "1",
        },
    )

    assert payload["commercial_product_mode"] == "pooled_customer_funds_model"
    assert payload["overall_status"] == "blocked"
    assert "customer_funds_hard_blocks:customer_funds_program_registered_missing" in payload["blockers"]
    assert "commercial_product_mode_hard_blocked" in payload["blockers"]
    assert payload["authority_boundaries"]["customer_funds_allowed"] is False
    assert payload["authority_boundaries"]["custody_allowed"] is False


def test_commercial_readiness_can_clear_paid_analytics_with_full_packet(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_use_mode_ready(project_root)
    _seed_security_ready(project_root)
    _seed_commercial_evidence(project_root)
    _write_text(project_root / "governance" / "commercial" / "evidence" / "claim_001.md")
    _write_json(
        project_root / "governance" / "commercial" / "marketing_claims.json",
        {
            "claims": [
                {
                    "claim": "Paper-mode dashboard summarizes controlled profitability only.",
                    "approved": True,
                    "evidence_artifact": "governance/commercial/evidence/claim_001.md",
                    "uses_gross_performance": True,
                    "gross_net_disclosure": True,
                    "uses_backtest_or_hypothetical": True,
                    "hypothetical_or_backtest_label": True,
                    "mentions_profitability": True,
                    "raw_vs_controlled_label": True,
                }
            ]
        },
    )

    payload = src.build_payload(
        project_root,
        env={
            "COMMERCIAL_PRODUCT_MODE": "paid_analytics_reporting",
            "COMMERCIAL_USE_ENABLED": "1",
            "COMMERCIAL_LEGAL_REVIEW_APPROVED": "1",
            "COMMERCIAL_COMPLIANCE_REVIEW_APPROVED": "1",
            "MARKETING_REVIEW_APPROVED": "1",
            "PRIVACY_SECURITY_REVIEW_APPROVED": "1",
            "TERMS_DISCLOSURE_REVIEW_APPROVED": "1",
            "PERFORMANCE_MARKETING_ENABLED": "1",
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["commercial_release_ready"] is True
    assert payload["ready_section_count"] == payload["section_count"]
    assert payload["blockers"] == []
