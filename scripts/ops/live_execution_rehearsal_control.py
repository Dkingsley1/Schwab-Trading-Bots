#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.live_execution_envelope import (
        broker_operation_retry_contract,
        build_live_execution_envelope,
        file_sha256,
        verify_live_execution_envelope,
    )
    from core.live_order_ledger import ALLOWED_TRANSITIONS
    from core.order_intent import build_order_intent_evidence
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from core.live_execution_envelope import (
        broker_operation_retry_contract,
        build_live_execution_envelope,
        file_sha256,
        verify_live_execution_envelope,
    )
    from core.live_order_ledger import ALLOWED_TRANSITIONS
    from core.order_intent import build_order_intent_evidence
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "live_execution_rehearsal_control_latest.json"
)


def _truthy(value: Any, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _control(
    control_id: str, title: str, ready: bool, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "title": title,
        "implemented": bool(ready),
        "status": "ready" if ready else "blocked",
        "evidence": evidence,
    }


def _policy(project_root: Path) -> tuple[dict[str, Any], Path]:
    path = project_root / "config" / "production_readiness_control_v1.json"
    payload = load_json(path)
    policy = (
        payload.get("live_execution_risk_firewall") if isinstance(payload, dict) else {}
    )
    return (policy if isinstance(policy, dict) else {}), path


def _intent(
    now: datetime, *, quote_time: datetime | None = None, spread_bps: float = 5.0
) -> dict[str, Any]:
    observed = quote_time or now
    return build_order_intent_evidence(
        decision_id="live-execution-rehearsal-intent",
        symbol="SPY",
        action="BUY",
        quantity=1.0,
        strategy="validate_only_rehearsal",
        asset_type="EQUITY",
        limit_price=100.0,
        quote_snapshot={
            "timestamp_utc": observed.isoformat(),
            "last_price": 100.0,
            "bid_price": 99.98,
            "ask_price": 100.02,
            "spread_bps": float(spread_bps),
            "quote_age_ms": 0.0,
            "source_provider": "synthetic_rehearsal_only",
            "snapshot_id": "validate-only-snapshot",
        },
        expected_fill={
            "expected_fill_price": 100.0,
            "expected_slippage_bps": 2.0,
            "partial_fill_ratio": 1.0,
            "paper_execution_status": "rehearsal",
        },
        risk_decision={
            "ok": True,
            "gate": "validate_only_pretrade",
            "reason": "synthetic_rehearsal_only",
            "details": {"order_notional": 100.0, "reference_price": 100.0},
        },
    )


def _order_request() -> dict[str, Any]:
    return {
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 1.0,
        "asset_type": "EQUITY",
        "limit_price": 100.0,
        "account_reference": "synthetic-account-reference-never-sent",
        "order_spec": {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "price": "100.00",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": "BUY",
                    "quantity": 1.0,
                    "instrument": {"symbol": "SPY", "assetType": "EQUITY"},
                }
            ],
        },
    }


def _envelope(
    *,
    now: datetime,
    policy_sha256: str,
    intent: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    ttl_seconds: float = 15.0,
) -> dict[str, Any]:
    return build_live_execution_envelope(
        intent_evidence=intent or _intent(now),
        order_request=_order_request(),
        candidate_id="pc-validate-only-rehearsal",
        broker="schwab",
        account_reference="synthetic-account-reference-never-sent",
        account_snapshot_evidence=(
            snapshot
            if snapshot is not None
            else {
                "broker_position_snapshot_sha256": "a" * 64,
                "broker_position_snapshot_captured_at_utc": now.isoformat(),
                "broker_position_snapshot_quantity": 0.0,
            }
        ),
        policy_sha256=policy_sha256,
        ttl_seconds=ttl_seconds,
        created_at_utc=now,
    )


def _verify(
    envelope: dict[str, Any],
    *,
    now: datetime,
    policy: dict[str, Any],
    policy_sha256: str,
    expected_candidate_id: str = "pc-validate-only-rehearsal",
    expected_account_reference: str = "synthetic-account-reference-never-sent",
) -> dict[str, Any]:
    return verify_live_execution_envelope(
        envelope,
        expected_candidate_id=expected_candidate_id,
        expected_account_reference=expected_account_reference,
        expected_policy_sha256=policy_sha256,
        now_utc=now,
        max_quote_age_seconds=max(
            float(policy.get("max_quote_age_seconds") or 15.0), 0.001
        ),
        max_account_snapshot_age_seconds=max(
            float(policy.get("max_account_snapshot_age_seconds") or 30.0), 0.001
        ),
        max_spread_bps=max(float(policy.get("max_spread_bps") or 75.0), 0.001),
        max_future_skew_seconds=max(
            float(policy.get("max_future_clock_skew_seconds") or 2.0), 0.0
        ),
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    policy, policy_path = _policy(root)
    now = datetime.now(timezone.utc)
    policy_digest = file_sha256(policy_path)
    ttl_seconds = max(
        float(policy.get("live_execution_envelope_ttl_seconds") or 15.0), 0.001
    )

    positive_envelope = _envelope(
        now=now,
        policy_sha256=policy_digest,
        ttl_seconds=ttl_seconds,
    )
    positive = _verify(
        positive_envelope,
        now=now + timedelta(seconds=1),
        policy=policy,
        policy_sha256=policy_digest,
    )

    tampered_envelope = copy.deepcopy(positive_envelope)
    tampered_envelope["broker_order_request"]["quantity"] = 2.0
    tampered = _verify(
        tampered_envelope,
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )

    stale = _verify(
        _envelope(
            now=now,
            policy_sha256=policy_digest,
            intent=_intent(now, quote_time=now - timedelta(seconds=60)),
            ttl_seconds=ttl_seconds,
        ),
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )
    future = _verify(
        _envelope(
            now=now,
            policy_sha256=policy_digest,
            intent=_intent(now, quote_time=now + timedelta(seconds=30)),
            ttl_seconds=ttl_seconds,
        ),
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )
    wide_spread = _verify(
        _envelope(
            now=now,
            policy_sha256=policy_digest,
            intent=_intent(
                now,
                spread_bps=max(
                    float(policy.get("max_spread_bps") or 75.0) + 25.0, 100.0
                ),
            ),
            ttl_seconds=ttl_seconds,
        ),
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )
    expired = _verify(
        positive_envelope,
        now=now + timedelta(seconds=ttl_seconds + 1.0),
        policy=policy,
        policy_sha256=policy_digest,
    )
    candidate_drift = _verify(
        positive_envelope,
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
        expected_candidate_id="pc-different-candidate",
    )
    account_drift = _verify(
        positive_envelope,
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
        expected_account_reference="different-account-reference",
    )
    missing_snapshot = _verify(
        _envelope(
            now=now,
            policy_sha256=policy_digest,
            snapshot={},
            ttl_seconds=ttl_seconds,
        ),
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )
    stale_snapshot = _verify(
        _envelope(
            now=now,
            policy_sha256=policy_digest,
            snapshot={
                "broker_position_snapshot_sha256": "a" * 64,
                "broker_position_snapshot_captured_at_utc": (
                    now - timedelta(seconds=60)
                ).isoformat(),
                "broker_position_snapshot_quantity": 0.0,
            },
            ttl_seconds=ttl_seconds,
        ),
        now=now,
        policy=policy,
        policy_sha256=policy_digest,
    )
    policy_drift = _verify(
        positive_envelope,
        now=now,
        policy=policy,
        policy_sha256="f" * 64,
    )

    mutating_retry = {
        operation: broker_operation_retry_contract(operation, 4)
        for operation in ("place_order", "replace_order", "cancel_order")
    }
    read_retry = {
        operation: broker_operation_retry_contract(operation, 4)
        for operation in ("get_order", "get_quote", "get_accounts_snapshot")
    }
    state_machine_ready = bool(
        "submit_unknown" in ALLOWED_TRANSITIONS
        and "cancel_unknown" in ALLOWED_TRANSITIONS
        and "submit_unknown" in ALLOWED_TRANSITIONS.get("submitting", set())
        and "cancel_unknown" in ALLOWED_TRANSITIONS.get("cancel_pending", set())
    )

    negative_scenarios = {
        "tampered_payload": {
            "passed": not tampered.get("ok", True)
            and "intent_to_broker_payload_parity_failed"
            in tampered.get("blockers", []),
            "blockers": tampered.get("blockers", []),
        },
        "stale_quote": {
            "passed": not stale.get("ok", True)
            and "quote_is_stale" in stale.get("blockers", []),
            "blockers": stale.get("blockers", []),
        },
        "future_quote": {
            "passed": not future.get("ok", True)
            and "quote_timestamp_in_future" in future.get("blockers", []),
            "blockers": future.get("blockers", []),
        },
        "wide_spread": {
            "passed": not wide_spread.get("ok", True)
            and "spread_exceeds_cap" in wide_spread.get("blockers", []),
            "blockers": wide_spread.get("blockers", []),
        },
        "expired_envelope": {
            "passed": not expired.get("ok", True)
            and "execution_envelope_expired" in expired.get("blockers", []),
            "blockers": expired.get("blockers", []),
        },
        "candidate_drift": {
            "passed": not candidate_drift.get("ok", True)
            and "candidate_id_mismatch" in candidate_drift.get("blockers", []),
            "blockers": candidate_drift.get("blockers", []),
        },
        "account_drift": {
            "passed": not account_drift.get("ok", True)
            and "account_reference_hash_mismatch" in account_drift.get("blockers", []),
            "blockers": account_drift.get("blockers", []),
        },
        "missing_snapshot": {
            "passed": not missing_snapshot.get("ok", True)
            and "account_snapshot_evidence_missing"
            in missing_snapshot.get("blockers", []),
            "blockers": missing_snapshot.get("blockers", []),
        },
        "stale_account_snapshot": {
            "passed": not stale_snapshot.get("ok", True)
            and "account_snapshot_is_stale" in stale_snapshot.get("blockers", []),
            "blockers": stale_snapshot.get("blockers", []),
        },
        "policy_drift": {
            "passed": not policy_drift.get("ok", True)
            and "policy_sha256_mismatch" in policy_drift.get("blockers", []),
            "blockers": policy_drift.get("blockers", []),
        },
    }
    negative_suite_ready = all(
        bool(row.get("passed", False)) for row in negative_scenarios.values()
    )
    raw_envelope = json.dumps(positive_envelope, ensure_ascii=True, sort_keys=True)
    account_redacted = "synthetic-account-reference-never-sent" not in raw_envelope

    controls = [
        _control(
            "01_immutable_intent",
            "Mode-invariant intent evidence verifies before release",
            bool(positive.get("intent_verification", {}).get("ok", False)),
            positive.get("intent_verification", {}),
        ),
        _control(
            "02_sealed_broker_payload",
            "The exact broker payload is hash-bound to the intent",
            bool(
                positive.get("ok", False)
                and all(positive.get("parity_fields", {}).values())
            ),
            {
                "parity_fields": positive.get("parity_fields", {}),
                "envelope_sha256": positive.get("envelope_sha256", ""),
            },
        ),
        _control(
            "03_account_reference_redaction",
            "Raw account references never enter the release envelope",
            account_redacted,
            {"raw_account_reference_present": not account_redacted},
        ),
        _control(
            "04_candidate_binding",
            "Candidate identity drift invalidates the envelope",
            bool(negative_scenarios["candidate_drift"]["passed"]),
            negative_scenarios["candidate_drift"],
        ),
        _control(
            "05_account_snapshot_binding",
            "Fresh account-position evidence is required and hash-bound",
            bool(
                negative_scenarios["account_drift"]["passed"]
                and negative_scenarios["missing_snapshot"]["passed"]
                and negative_scenarios["stale_account_snapshot"]["passed"]
            ),
            {
                "account_drift": negative_scenarios["account_drift"],
                "missing_snapshot": negative_scenarios["missing_snapshot"],
                "stale_account_snapshot": negative_scenarios["stale_account_snapshot"],
            },
        ),
        _control(
            "06_quote_freshness",
            "Stale and future-skewed quotes fail closed",
            bool(
                negative_scenarios["stale_quote"]["passed"]
                and negative_scenarios["future_quote"]["passed"]
            ),
            {
                "stale_quote": negative_scenarios["stale_quote"],
                "future_quote": negative_scenarios["future_quote"],
            },
        ),
        _control(
            "07_spread_collar",
            "Spread evidence is mandatory and capped",
            bool(negative_scenarios["wide_spread"]["passed"]),
            negative_scenarios["wide_spread"],
        ),
        _control(
            "08_short_lived_release",
            "Expired release envelopes cannot submit",
            bool(negative_scenarios["expired_envelope"]["passed"]),
            negative_scenarios["expired_envelope"],
        ),
        _control(
            "09_payload_tamper_detection",
            "Post-approval payload mutation is detected",
            bool(
                negative_scenarios["tampered_payload"]["passed"]
                and negative_scenarios["policy_drift"]["passed"]
            ),
            {
                "tampered_payload": negative_scenarios["tampered_payload"],
                "policy_drift": negative_scenarios["policy_drift"],
            },
        ),
        _control(
            "10_one_shot_mutations",
            "Submit, replace, and cancel never retry after possible dispatch",
            all(
                row["max_attempts"] == 1 and not row["retry_after_dispatch_allowed"]
                for row in mutating_retry.values()
            ),
            mutating_retry,
        ),
        _control(
            "11_bounded_read_retries",
            "Read-only broker operations retain bounded recovery",
            all(
                row["max_attempts"] == 4 and row["retry_after_dispatch_allowed"]
                for row in read_retry.values()
            ),
            read_retry,
        ),
        _control(
            "12_unknown_outcome_states",
            "Ambiguous submit and cancel outcomes require reconciliation",
            state_machine_ready,
            {
                "submitting": sorted(ALLOWED_TRANSITIONS.get("submitting", set())),
                "cancel_pending": sorted(
                    ALLOWED_TRANSITIONS.get("cancel_pending", set())
                ),
            },
        ),
        _control(
            "13_fail_closed_policy",
            "The canonical firewall requires seals and forbids mutation retries",
            bool(
                policy.get("require_sealed_live_execution_envelope") is True
                and policy.get("mutating_broker_retries_after_dispatch_allowed")
                is False
                and policy.get("allow_live_order_replace") is False
            ),
            {
                "require_sealed_live_execution_envelope": policy.get(
                    "require_sealed_live_execution_envelope"
                ),
                "mutating_broker_retries_after_dispatch_allowed": policy.get(
                    "mutating_broker_retries_after_dispatch_allowed"
                ),
                "allow_live_order_replace": policy.get("allow_live_order_replace"),
            },
        ),
        _control(
            "14_validate_only_negative_suite",
            "Ten deterministic failures are rejected without broker access",
            negative_suite_ready,
            {
                "scenario_count": len(negative_scenarios),
                "passed_count": sum(
                    1 for row in negative_scenarios.values() if row.get("passed")
                ),
                "scenarios": negative_scenarios,
            },
        ),
    ]

    implemented_count = sum(1 for row in controls if row.get("implemented", False))
    control_ready = implemented_count == len(controls)
    market_data_only = _truthy(os.getenv("MARKET_DATA_ONLY", "1"), True)
    order_execution_enabled = _truthy(os.getenv("ALLOW_ORDER_EXECUTION", "0"), False)
    paper_lock_present = (
        root / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
    ).exists()
    live_locked = bool(
        market_data_only and not order_execution_enabled and paper_lock_present
    )
    ledger = load_json(
        root / "governance" / "health" / "live_order_ledger_control_latest.json"
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": control_ready,
        "overall_status": "ready_locked" if control_ready else "blocked",
        "control_grade": "A+" if control_ready else "F",
        "control_score": round(100.0 * implemented_count / max(len(controls), 1), 3),
        "implemented_control_count": implemented_count,
        "control_count": len(controls),
        "controls": controls,
        "negative_scenarios": negative_scenarios,
        "runtime_lock_observation": {
            "live_locked": live_locked,
            "market_data_only": market_data_only,
            "allow_order_execution": order_execution_enabled,
            "paper_trade_lock_present": paper_lock_present,
            "observation_is_not_release_authority": True,
        },
        "ledger_observation": {
            "ok": bool(ledger.get("ok", False)),
            "overall_status": str(ledger.get("overall_status") or "missing"),
            "unresolved_count": int(ledger.get("unresolved_count", 0) or 0),
        },
        "source_influences": [
            {
                "source": "SEC Market Access Rule 15c3-5",
                "official_url": "https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access",
                "adopted_pattern": "pre-order financial and regulatory controls plus immediate post-trade reporting",
            },
            {
                "source": "FINRA Regulatory Notice 15-09",
                "official_url": "https://www.finra.org/industry/notices/15-09",
                "adopted_pattern": "segregated testing, pilot deployment, change review, monitoring, disable controls, and reconciliation",
            },
            {
                "source": "FIX Order State Changes",
                "official_url": "https://www.fixtrading.org/online-specification/order-state-changes/",
                "adopted_pattern": "explicit pending, acknowledged, partial, terminal, and cancel state semantics",
            },
            {
                "source": "Nasdaq OUCH 5.0",
                "official_url": "https://nasdaqtrader.com/content/technicalsupport/specifications/TradingProducts/Ouch5.0.pdf",
                "adopted_pattern": "distinct accepted, replaced, canceled, executed, pending-cancel, and rejected outcomes",
            },
            {
                "source": "AWS Making retries safe with idempotent APIs",
                "official_url": "https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/",
                "adopted_pattern": "stable intent identity, parameter equivalence, and no unsafe retries for non-idempotent mutations",
            },
        ],
        "authority": {
            "validate_only": True,
            "network_access": False,
            "broker_client_created": False,
            "paper_order_authority": False,
            "live_execution_authority": False,
            "live_orders_must_remain_disabled": True,
        },
        "evidence_debt": [
            "real broker acknowledgements and fills are intentionally absent",
            "real account-position and buying-power reconciliation remains external runtime evidence",
            "operator release and live canary gates remain independently blocked",
            "an A+ control grade is implementation evidence, not profitability or live-money readiness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a no-network, no-order rehearsal of the sealed live execution path."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out_path = args.out_file if args.out_file.is_absolute() else root / args.out_file
    payload = build_payload(root)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_execution_rehearsal_control "
            f"status={payload['overall_status']} grade={payload['control_grade']} "
            f"controls={payload['implemented_control_count']}/{payload['control_count']} "
            "authority=validate_only"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
