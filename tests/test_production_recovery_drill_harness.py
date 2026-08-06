from scripts.ops import production_recovery_drill_harness as src


def test_isolated_recovery_harness_proves_all_required_failure_modes() -> None:
    payload = src.build_payload()

    assert payload["ok"] is True
    assert payload["grade"] == "A+"
    assert payload["passed_drill_count"] == payload["required_drill_count"] == 10
    assert payload["simulation_only"] is True
    assert payload["real_outage_evidence"] is False
    assert payload["live_execution_authority"] is False
    assert len(payload["run_sha256"]) == 64
    assert all(row["containment_verified"] for row in payload["drills"])
    assert all(row["no_duplicate_orders"] for row in payload["drills"])
    assert all(len(row["evidence_sha256"]) == 64 for row in payload["drills"])

    order_lifecycle = next(
        row for row in payload["drills"] if row["drill"] == "order_reject_partial_fill_cancel_replace"
    )
    assert order_lifecycle["evidence"]["event_chain"]["unresolved_count"] == 0
    assert order_lifecycle["evidence"]["replacement_filled"]["state"] == "filled"
