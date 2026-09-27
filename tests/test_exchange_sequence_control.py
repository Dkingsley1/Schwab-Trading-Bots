from core.exchange_sequence_control import ExchangeSequenceGuard


def test_exchange_sequence_guard_detects_duplicates_gaps_and_explicit_resets() -> None:
    guard = ExchangeSequenceGuard()

    assert (
        guard.ingest(
            channel="equities", session_id="A", sequence_number=100, payload={"px": 100}
        )["disposition"]
        == "session_initialized"
    )
    assert (
        guard.ingest(
            channel="equities", session_id="A", sequence_number=101, payload={"px": 101}
        )["disposition"]
        == "contiguous"
    )
    assert (
        guard.ingest(
            channel="equities", session_id="A", sequence_number=101, payload={"px": 101}
        )["disposition"]
        == "duplicate_idempotent"
    )

    gap = guard.ingest(
        channel="equities", session_id="A", sequence_number=103, payload={"px": 103}
    )
    assert gap["accepted"] is False
    assert gap["missing_range"] == [102, 102]
    assert (
        guard.ingest(
            channel="equities", session_id="A", sequence_number=102, payload={"px": 102}
        )["accepted"]
        is True
    )
    assert (
        guard.ingest(
            channel="equities",
            session_id="B",
            sequence_number=1,
            payload={"reset": True},
            session_reset=True,
        )["disposition"]
        == "session_reset"
    )

    restored = ExchangeSequenceGuard.restore(guard.snapshot())
    assert restored.snapshot() == guard.snapshot()
