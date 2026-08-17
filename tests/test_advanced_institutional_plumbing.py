import pytest

from core.advanced_institutional_plumbing import (
    proof_quantum_backend_readiness_score,
    provider_readiness_score,
    repo_lending_pressure_score,
    securitized_prepayment_oas_score,
    tape_quality_score,
    tranche_waterfall_proxy,
    xva_margin_proxy,
)


def test_xva_margin_proxy_keeps_unsecured_exposure_and_scores_stress() -> None:
    result = xva_margin_proxy(
        expected_exposure=1_000_000,
        collateral=600_000,
        cva_spread_bps=120,
        dva_spread_bps=35,
        funding_spread_bps=80,
        margin_rate=0.08,
        wrong_way_risk=0.5,
    )

    assert result.residual_unsecured_exposure == 400_000
    assert result.cva > result.dva
    assert 0.0 < result.stress_score < 1.0


def test_tranche_waterfall_proxy_allocates_loss_to_attachment_band() -> None:
    result = tranche_waterfall_proxy(
        notional=10_000_000,
        default_rate=0.18,
        recovery_rate=0.35,
        attachment=0.03,
        detachment=0.07,
    )

    assert result.portfolio_loss > 0
    assert result.tranche_loss_rate == 1.0
    assert result.tranche_loss == pytest.approx(400_000)


def test_institutional_scores_are_bounded() -> None:
    assert 0.0 <= securitized_prepayment_oas_score(mortgage_rate=0.06, coupon_rate=0.075, prepayment_speed=18, duration_years=6) <= 1.0
    assert 0.0 <= repo_lending_pressure_score(repo_rate=0.055, sofr_rate=0.052, borrow_fee_rate=0.08, short_interest_ratio=0.35, fail_to_deliver_ratio=0.04) <= 1.0
    assert 0.0 <= tape_quality_score(opra_nbbo_alignment=0.98, taq_sip_latency_ms=80, dedupe_quality=0.99, depth_integrity=0.92, off_exchange_share=0.38) <= 1.0
    assert 0.0 <= provider_readiness_score(credentials_ok=True, entitlement_coverage=0.8, freshness_seconds=120, rate_limit_remaining_ratio=0.7) <= 1.0
    assert proof_quantum_backend_readiness_score(
        zkp_prover_ok=True,
        formal_invariants_pass=True,
        quantum_backend_available=False,
        fallback_path_ok=True,
    ) == pytest.approx(0.85)
