import hashlib
from datetime import datetime, timedelta, timezone

from core.live_execution_envelope import (
    broker_operation_retry_contract,
    build_live_execution_envelope,
    verify_live_execution_envelope,
)
from core.order_intent import build_order_intent_evidence, canonical_payload_sha256

NOW = datetime(2026, 8, 24, 16, 0, 0, tzinfo=timezone.utc)


def _intent(
    *,
    quote_time: datetime = NOW,
    spread_bps: float = 5.0,
    risk_ok: bool = True,
    source_provider: str = "",
) -> dict:
    return build_order_intent_evidence(
        decision_id="decision-1",
        symbol="SPY",
        action="BUY",
        quantity=1.0,
        strategy="canary",
        asset_type="EQUITY",
        limit_price=100.0,
        quote_snapshot={
            "timestamp_utc": quote_time.isoformat(),
            "last_price": 100.0,
            "bid_price": 99.98,
            "ask_price": 100.02,
            "spread_bps": spread_bps,
            "quote_age_ms": 0.0,
            "source_provider": source_provider,
        },
        expected_fill={"expected_fill_price": 100.0, "partial_fill_ratio": 1.0},
        risk_decision={"ok": risk_ok, "gate": "pre_trade", "reason": "ok", "details": {}},
    )


def _request() -> dict:
    return {
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 1.0,
        "asset_type": "EQUITY",
        "limit_price": 100.0,
        "account_reference": "account-secret",
        "order_spec": {
            "orderType": "LIMIT",
            "price": "100.00",
            "orderLegCollection": [
                {
                    "instruction": "BUY",
                    "quantity": 1.0,
                    "instrument": {"symbol": "SPY", "assetType": "EQUITY"},
                }
            ],
        },
    }


def _envelope(*, intent: dict | None = None) -> dict:
    return build_live_execution_envelope(
        intent_evidence=intent or _intent(),
        order_request=_request(),
        candidate_id="pc-candidate-g100",
        broker="schwab",
        account_reference="account-secret",
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": "a" * 64,
            "broker_position_snapshot_captured_at_utc": NOW.isoformat(),
            "broker_position_snapshot_quantity": 0.0,
        },
        policy_sha256="b" * 64,
        ttl_seconds=15.0,
        created_at_utc=NOW,
    )


def test_live_execution_envelope_is_redacted_stable_and_valid() -> None:
    first = _envelope()
    second = _envelope()

    assert first["envelope_sha256"] == second["envelope_sha256"]
    assert first["client_order_id"] == second["client_order_id"]
    assert "account-secret" not in str(first)
    result = verify_live_execution_envelope(
        first,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW + timedelta(seconds=1),
    )
    assert result["ok"] is True
    assert result["parity_fields"] == {
        "symbol": True,
        "action": True,
        "asset_type": True,
        "quantity": True,
        "limit_price": True,
    }


def test_live_execution_envelope_rejects_payload_tampering() -> None:
    envelope = _envelope()
    envelope["broker_order_request"]["quantity"] = 2.0

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW,
    )

    assert result["ok"] is False
    assert "broker_order_request_sha256_mismatch" in result["blockers"]
    assert "intent_to_broker_payload_parity_failed" in result["blockers"]
    assert "execution_envelope_sha256_mismatch" in result["blockers"]


def test_live_execution_envelope_rejects_stale_quote_and_expiry() -> None:
    envelope = _envelope(intent=_intent(quote_time=NOW - timedelta(seconds=30)))

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW + timedelta(seconds=16),
        max_quote_age_seconds=15.0,
    )

    assert result["ok"] is False
    assert "quote_is_stale" in result["blockers"]
    assert "execution_envelope_expired" in result["blockers"]


def test_live_execution_envelope_rejects_candidate_and_account_drift() -> None:
    result = verify_live_execution_envelope(
        _envelope(),
        expected_candidate_id="different-candidate",
        expected_account_reference="different-account",
        now_utc=NOW,
    )

    assert result["ok"] is False
    assert "candidate_id_mismatch" in result["blockers"]
    assert "account_reference_hash_mismatch" in result["blockers"]


def _reseal(envelope: dict) -> None:
    immutable = {
        key: envelope.get(key)
        for key in (
            "schema_version",
            "created_at_utc",
            "expires_at_utc",
            "candidate_id",
            "broker",
            "client_order_id",
            "intent_evidence",
            "broker_order_request",
            "account_snapshot_evidence",
            "component_hashes",
            "authority",
        )
    }
    envelope["envelope_sha256"] = canonical_payload_sha256(immutable)


def test_live_execution_envelope_rejects_policy_and_account_request_drift() -> None:
    envelope = _envelope()
    envelope["broker_order_request"]["account_reference_sha256"] = "c" * 64
    envelope["component_hashes"]["broker_order_request_sha256"] = (
        canonical_payload_sha256(envelope["broker_order_request"])
    )
    _reseal(envelope)

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        expected_policy_sha256="d" * 64,
        now_utc=NOW,
    )

    assert result["ok"] is False
    assert "broker_order_account_reference_hash_mismatch" in result["blockers"]
    assert "policy_sha256_mismatch" in result["blockers"]


def test_live_execution_envelope_rejects_stale_account_snapshot() -> None:
    envelope = _envelope()
    envelope["account_snapshot_evidence"][
        "broker_position_snapshot_captured_at_utc"
    ] = (NOW - timedelta(seconds=60)).isoformat()
    envelope["component_hashes"]["account_snapshot_sha256"] = canonical_payload_sha256(
        envelope["account_snapshot_evidence"]
    )
    _reseal(envelope)

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW,
        max_account_snapshot_age_seconds=30.0,
    )

    assert result["ok"] is False
    assert "account_snapshot_is_stale" in result["blockers"]


def test_live_execution_envelope_requires_approved_risk_and_quote_provenance() -> None:
    result = verify_live_execution_envelope(
        _envelope(intent=_intent(risk_ok=False)),
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW,
        require_affirmative_risk_decision=True,
        require_quote_provenance=True,
        allowed_quote_providers=("schwab",),
    )

    assert result["ok"] is False
    assert "order_intent_risk_decision_not_approved" in result["blockers"]
    assert "quote_source_provider_missing" in result["blockers"]


def test_live_execution_envelope_binds_ready_canary_preflight_receipt() -> None:
    account_hash = hashlib.sha256(b"account-secret").hexdigest()
    envelope = build_live_execution_envelope(
        intent_evidence=_intent(source_provider="schwab"),
        order_request=_request(),
        candidate_id="pc-candidate-g100",
        broker="schwab",
        account_reference="account-secret",
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": "a" * 64,
            "broker_position_snapshot_captured_at_utc": NOW.isoformat(),
            "live_canary_preflight_receipt": {
                "ready": True,
                "receipt_sha256": "c" * 64,
                "account_policy_key": "schwab_cash_account_1",
                "account_reference_sha256": account_hash,
            },
        },
        policy_sha256="b" * 64,
        created_at_utc=NOW,
    )

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW,
        require_affirmative_risk_decision=True,
        require_quote_provenance=True,
        allowed_quote_providers=("schwab",),
        require_canary_preflight_receipt=True,
        expected_account_policy_key="schwab_cash_account_1",
    )

    assert result["ok"] is True


def test_live_execution_envelope_rejects_wrong_preflight_account() -> None:
    envelope = build_live_execution_envelope(
        intent_evidence=_intent(source_provider="schwab"),
        order_request=_request(),
        candidate_id="pc-candidate-g100",
        broker="schwab",
        account_reference="account-secret",
        account_snapshot_evidence={
            "broker_position_snapshot_sha256": "a" * 64,
            "broker_position_snapshot_captured_at_utc": NOW.isoformat(),
            "live_canary_preflight_receipt": {
                "ready": True,
                "receipt_sha256": "c" * 64,
                "account_policy_key": "schwab_cash_account_1",
                "account_reference_sha256": "d" * 64,
            },
        },
        policy_sha256="b" * 64,
        created_at_utc=NOW,
    )

    result = verify_live_execution_envelope(
        envelope,
        expected_candidate_id="pc-candidate-g100",
        expected_account_reference="account-secret",
        now_utc=NOW,
        require_canary_preflight_receipt=True,
        expected_account_policy_key="schwab_cash_account_1",
    )

    assert result["ok"] is False
    assert "live_canary_preflight_account_reference_mismatch" in result["blockers"]


def test_mutating_broker_operations_are_one_shot_after_dispatch() -> None:
    for operation in ("place_order", "replace_order", "cancel_order"):
        contract = broker_operation_retry_contract(operation, 4)
        assert contract["mutating"] is True
        assert contract["max_attempts"] == 1
        assert contract["retry_after_dispatch_allowed"] is False
        assert contract["ambiguous_failure_requires_reconciliation"] is True

    read_contract = broker_operation_retry_contract("get_order", 4)
    assert read_contract["mutating"] is False
    assert read_contract["max_attempts"] == 4
    assert read_contract["retry_after_dispatch_allowed"] is True
