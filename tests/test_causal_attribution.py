from copy import deepcopy

from core.causal_attribution import build_execution_trace, verify_execution_trace


def _trace() -> dict:
    return build_execution_trace(
        intent={
            "message_id": "decision-1",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 2.0,
            "features": {"expected_edge_bps": 9.0},
            "metadata": {"source_broker": "schwab"},
        },
        result={
            "status": "PAPER_EXECUTED",
            "paper_order": {
                "filled_quantity": 2.0,
                "fee_bps": 0.2,
                "slippage_bps": 0.7,
            },
        },
        gateway={"allow_execute": True, "reasons": []},
        mode="paper",
    )


def test_trace_has_all_eight_hash_linked_stages_and_nonfabricated_attribution() -> None:
    trace = _trace()

    assert verify_execution_trace(trace)["ok"] is True
    assert [row["stage"] for row in trace["stages"]] == [
        "source",
        "feature",
        "signal",
        "sizing",
        "risk",
        "execution",
        "cost",
        "outcome",
    ]
    assert trace["attribution"]["values"]["realized_net_bps"] is None
    assert "realized_net_bps" in trace["attribution"]["missing_fields"]
    assert trace["attribution"]["no_fabricated_defaults"] is True


def test_trace_tampering_is_detected() -> None:
    trace = deepcopy(_trace())
    trace["stages"][3]["payload_sha256"] = "tampered"

    assert verify_execution_trace(trace)["ok"] is False
