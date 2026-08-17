from core.order_intent import build_order_intent_evidence, verify_order_intent_evidence


def _sample() -> dict:
    return build_order_intent_evidence(
        decision_id="decision-1",
        symbol="spy",
        action="buy",
        quantity=1.0,
        strategy="alpha",
        quote_snapshot={"last_price": 100.0, "snapshot_id": "quote-1"},
        expected_fill={"expected_fill_price": 100.01, "partial_fill_ratio": 1.0},
        risk_decision={"ok": True, "gate": "pre_trade", "reason": "ok", "details": {"order_notional": 100.0}},
    )


def test_order_intent_is_deterministic_and_mode_invariant() -> None:
    paper = _sample()
    live = _sample()

    assert paper["intent_sha256"] == live["intent_sha256"]
    assert paper["mode_excluded_from_hash"] is True
    assert paper["adapter_excluded_from_hash"] is True
    assert verify_order_intent_evidence(paper)["ok"] is True


def test_order_intent_detects_component_tampering() -> None:
    evidence = _sample()
    evidence["quote_snapshot"]["last_price"] = 101.0

    verification = verify_order_intent_evidence(evidence)

    assert verification["ok"] is False
    assert "quote_snapshot_sha256" in verification["errors"]
    assert "intent_sha256" in verification["errors"]
