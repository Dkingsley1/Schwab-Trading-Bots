from __future__ import annotations

from scripts.ops.bot_quality_autopilot import _guard_targeted_retrain_queue, _process_exit_code


def _queue(*bot_ids: str) -> dict[str, dict]:
    return {
        bot_id: {
            "bot_id": bot_id,
            "next_step": "targeted_retrain",
            "reasons": ["stale_retrain_label"],
        }
        for bot_id in bot_ids
    }


def test_targeted_retrain_guard_reroutes_sample_starved_candidates() -> None:
    queue = _queue("sample_starved")

    guard = _guard_targeted_retrain_queue(
        queue,
        requalification_rows=[
            {
                "bot_id": "sample_starved",
                "sample_count": 57,
                "eligible_sequences": 1,
            }
        ],
        overfit_rows=[],
        min_sample_count=200,
    )

    assert queue["sample_starved"]["next_step"] == "collect_more_data"
    assert "minimum_sample_floor_not_met" in queue["sample_starved"]["reasons"]
    assert guard["rerouted_count"] == 1


def test_targeted_retrain_guard_reroutes_overfit_candidates() -> None:
    queue = _queue("overfit_guarded")

    _guard_targeted_retrain_queue(
        queue,
        requalification_rows=[
            {
                "bot_id": "overfit_guarded",
                "sample_count": 746,
                "eligible_sequences": 1,
            }
        ],
        overfit_rows=[{"bot_id": "overfit_guarded", "status": "high_accuracy_guarded"}],
        min_sample_count=200,
    )

    assert queue["overfit_guarded"]["next_step"] == "reduce_overfitting"
    assert "overfit_status_high_accuracy_guarded" in queue["overfit_guarded"]["reasons"]


def test_targeted_retrain_guard_preserves_current_evidence_eligible_candidate() -> None:
    queue = _queue("eligible")

    guard = _guard_targeted_retrain_queue(
        queue,
        requalification_rows=[
            {
                "bot_id": "eligible",
                "sample_count": 1122,
                "eligible_sequences": 3,
            }
        ],
        overfit_rows=[{"bot_id": "eligible", "status": "generalization_clean"}],
        min_sample_count=200,
    )

    assert queue["eligible"]["next_step"] == "targeted_retrain"
    assert guard["rerouted_count"] == 0


def test_process_exit_code_treats_evidence_work_as_a_successful_cycle() -> None:
    assert _process_exit_code({"overall_status": "needs_work"}) == 0
    assert _process_exit_code({"overall_status": "ready"}) == 0
    assert _process_exit_code({"overall_status": "blocked"}) == 2
