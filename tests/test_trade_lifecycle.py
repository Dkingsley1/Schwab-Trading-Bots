from core.trade_lifecycle import TradeLifecycle


def _event(event_id: str, event_type: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "trade_id": "trade-1",
        "product_id": "SPY",
        "economics_sha256": "a" * 64,
    }


def test_trade_lifecycle_is_idempotent_hash_chained_and_legally_ordered() -> None:
    lifecycle = TradeLifecycle("trade-1", "SPY", "a" * 64)

    assert lifecycle.apply(_event("e1", "execution"))["accepted"] is True
    assert (
        lifecycle.apply(_event("e1", "execution"))["disposition"]
        == "duplicate_idempotent"
    )
    assert lifecycle.apply(_event("e2", "confirmation"))["accepted"] is True
    assert lifecycle.apply(_event("e3", "allocation"))["accepted"] is True
    assert lifecycle.apply(_event("e4", "settlement"))["accepted"] is True
    assert lifecycle.state == "SETTLED"
    assert lifecycle.verify_chain()["ok"] is True
    assert lifecycle.apply(_event("e5", "cancellation"))["accepted"] is False


def test_trade_lifecycle_rejects_mutated_economic_terms() -> None:
    lifecycle = TradeLifecycle("trade-1", "SPY", "a" * 64)
    event = _event("e1", "execution")
    event["economics_sha256"] = "b" * 64

    assert lifecycle.apply(event)["disposition"] == "economic_terms_mutated"
