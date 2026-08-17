from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def bps_to_decimal(value: Any) -> float:
    return _num(value) / 10_000.0


@dataclass(frozen=True)
class XvaMarginProxy:
    expected_exposure: float
    collateral: float
    cva: float
    dva: float
    fva: float
    capital_margin_addon: float
    residual_unsecured_exposure: float
    stress_score: float


def xva_margin_proxy(
    *,
    expected_exposure: Any,
    collateral: Any = 0.0,
    cva_spread_bps: Any = 0.0,
    dva_spread_bps: Any = 0.0,
    funding_spread_bps: Any = 0.0,
    margin_rate: Any = 0.0,
    wrong_way_risk: Any = 0.0,
) -> XvaMarginProxy:
    exposure = max(_num(expected_exposure), 0.0)
    posted_collateral = max(_num(collateral), 0.0)
    unsecured = max(exposure - posted_collateral, 0.0)
    cva = unsecured * bps_to_decimal(cva_spread_bps) * (1.0 + _clamp(_num(wrong_way_risk)))
    dva = unsecured * bps_to_decimal(dva_spread_bps)
    fva = unsecured * bps_to_decimal(funding_spread_bps)
    margin_addon = exposure * _clamp(_num(margin_rate))
    stress_score = _clamp((cva + fva + margin_addon) / max(exposure, 1.0))
    return XvaMarginProxy(
        expected_exposure=exposure,
        collateral=posted_collateral,
        cva=cva,
        dva=dva,
        fva=fva,
        capital_margin_addon=margin_addon,
        residual_unsecured_exposure=unsecured,
        stress_score=stress_score,
    )


@dataclass(frozen=True)
class TrancheWaterfallProxy:
    portfolio_loss: float
    tranche_loss: float
    tranche_loss_rate: float
    attachment: float
    detachment: float


def tranche_waterfall_proxy(
    *,
    notional: Any,
    default_rate: Any,
    recovery_rate: Any,
    attachment: Any,
    detachment: Any,
) -> TrancheWaterfallProxy:
    base_notional = max(_num(notional), 0.0)
    defaults = _clamp(_num(default_rate))
    recovery = _clamp(_num(recovery_rate))
    attach = _clamp(_num(attachment))
    detach = _clamp(_num(detachment), attach, 1.0)
    loss_rate = defaults * (1.0 - recovery)
    tranche_width = max(detach - attach, 1e-9)
    tranche_loss_rate = _clamp((loss_rate - attach) / tranche_width)
    portfolio_loss = base_notional * loss_rate
    tranche_loss = base_notional * tranche_width * tranche_loss_rate
    return TrancheWaterfallProxy(
        portfolio_loss=portfolio_loss,
        tranche_loss=tranche_loss,
        tranche_loss_rate=tranche_loss_rate,
        attachment=attach,
        detachment=detach,
    )


def securitized_prepayment_oas_score(
    *,
    mortgage_rate: Any,
    coupon_rate: Any,
    prepayment_speed: Any,
    duration_years: Any,
) -> float:
    refi_incentive = max(_num(coupon_rate) - _num(mortgage_rate), 0.0)
    speed = _clamp(_num(prepayment_speed) / 40.0)
    duration = _clamp(_num(duration_years) / 12.0)
    return _clamp(0.45 * refi_incentive + 0.35 * speed + 0.20 * duration)


def repo_lending_pressure_score(
    *,
    repo_rate: Any,
    sofr_rate: Any,
    borrow_fee_rate: Any,
    short_interest_ratio: Any,
    fail_to_deliver_ratio: Any,
) -> float:
    funding_spread = max(_num(repo_rate) - _num(sofr_rate), 0.0)
    borrow_fee = max(_num(borrow_fee_rate), 0.0)
    short_interest = _clamp(_num(short_interest_ratio))
    ftd = _clamp(_num(fail_to_deliver_ratio))
    return _clamp(2.0 * funding_spread + 1.5 * borrow_fee + 0.35 * short_interest + 0.25 * ftd)


def tape_quality_score(
    *,
    opra_nbbo_alignment: Any,
    taq_sip_latency_ms: Any,
    dedupe_quality: Any,
    depth_integrity: Any,
    off_exchange_share: Any,
) -> float:
    latency_penalty = _clamp(_num(taq_sip_latency_ms) / 1000.0)
    hidden_liquidity_penalty = _clamp(max(_num(off_exchange_share) - 0.45, 0.0) / 0.55)
    return _clamp(
        0.30 * _clamp(_num(opra_nbbo_alignment))
        + 0.25 * _clamp(_num(dedupe_quality))
        + 0.25 * _clamp(_num(depth_integrity))
        + 0.20 * (1.0 - latency_penalty)
        - 0.10 * hidden_liquidity_penalty
    )


def provider_readiness_score(
    *,
    credentials_ok: Any,
    entitlement_coverage: Any,
    freshness_seconds: Any,
    rate_limit_remaining_ratio: Any,
) -> float:
    credential = 1.0 if bool(credentials_ok) else 0.0
    freshness = 1.0 - _clamp(_num(freshness_seconds) / 3600.0)
    return _clamp(
        0.35 * credential
        + 0.30 * _clamp(_num(entitlement_coverage))
        + 0.20 * freshness
        + 0.15 * _clamp(_num(rate_limit_remaining_ratio))
    )


def proof_quantum_backend_readiness_score(
    *,
    zkp_prover_ok: Any,
    formal_invariants_pass: Any,
    quantum_backend_available: Any,
    fallback_path_ok: Any,
) -> float:
    return _clamp(
        0.30 * (1.0 if bool(zkp_prover_ok) else 0.0)
        + 0.35 * (1.0 if bool(formal_invariants_pass) else 0.0)
        + 0.15 * (1.0 if bool(quantum_backend_available) else 0.0)
        + 0.20 * (1.0 if bool(fallback_path_ok) else 0.0)
    )
