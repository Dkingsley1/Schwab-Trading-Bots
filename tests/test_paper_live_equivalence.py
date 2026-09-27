from core.paper_live_equivalence import compare_pair, compare_record_sets


def _intent(mode: str, quantity: float = 1.0) -> dict:
    return {
        "trace_context": {"trace_id": "trace-one"},
        "target_mode": mode,
        "symbol": "SPY",
        "action": "BUY",
        "quantity": quantity,
        "asset_type": "EQUITY",
        "strategy": "trend",
        "latency_ms": 10 if mode == "paper" else 50,
        "broker_order_id": "" if mode == "paper" else "broker-one",
    }


def test_mode_specific_execution_fields_do_not_create_false_mismatch() -> None:
    assert compare_pair(_intent("paper"), _intent("live"))["ok"] is True


def test_quantity_or_action_drift_is_a_semantic_mismatch() -> None:
    comparison = compare_pair(_intent("paper"), _intent("live", quantity=2.0))

    assert comparison["ok"] is False
    assert comparison["differences"] == ["quantity"]


def test_missing_live_shadow_samples_are_evidence_debt_not_paper_failure() -> None:
    report = compare_record_sets([_intent("paper")], [])

    assert report["ok"] is True
    assert report["empirical_ready"] is False
    assert report["status"] == "awaiting_live_shadow_samples"
    assert report["unpaired_paper_count"] == 1
    assert report["missing_live_count"] == 0


def test_orphan_live_shadow_intent_is_a_failure() -> None:
    report = compare_record_sets([], [_intent("live")])

    assert report["ok"] is False
    assert report["status"] == "orphan_live_shadow_intent"
    assert report["missing_paper_count"] == 1
