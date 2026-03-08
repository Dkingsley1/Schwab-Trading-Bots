from core.master_bot import BotStatus, MasterBot


def _status(bot_id: str, *, deleted: bool = False, reason: str = "walk_forward_fail") -> BotStatus:
    return BotStatus(
        bot_id=bot_id,
        bot_role="signal_sub_bot",
        active=False,
        reason=reason,
        weight=0.0,
        preference_score=0.0,
        quality_score=0.6,
        test_accuracy=None,
        candidate_test_accuracy=0.6,
        candidate_quality_score=0.6,
        previous_best_accuracy=None,
        no_improvement_streak=0,
        deleted_from_rotation=deleted,
        delete_reason="deleted" if deleted else "",
        promoted=False,
        promotion_reason="",
        model_path=None,
        log_file=f"/tmp/{bot_id}.json",
    )


def test_floor_override_only_revives_passing_rotation_candidates(tmp_path) -> None:
    master = MasterBot(project_root=str(tmp_path), min_active_bots=2)
    master.walk_forward_map = {
        "brain_refinery_good": {
            "runs": 30,
            "status": "pass",
            "forward_mean": 0.61,
            "delta": 0.0,
            "trading_quality_score": 0.63,
        },
        "brain_refinery_bad": {
            "runs": 30,
            "status": "fail",
            "forward_mean": 0.49,
            "delta": -0.03,
            "trading_quality_score": 0.41,
        },
        "brain_refinery_deleted": {
            "runs": 30,
            "status": "pass",
            "forward_mean": 0.64,
            "delta": 0.0,
            "trading_quality_score": 0.67,
        },
    }

    statuses = [
        _status("brain_refinery_good", reason="correlation_pruned_gt_0.92"),
        _status("brain_refinery_bad", reason="correlation_pruned_gt_0.92"),
        _status("brain_refinery_deleted", deleted=True, reason="rotation_deleted"),
    ]

    result = {status.bot_id: status for status in master._enforce_min_active_bots(statuses)}

    assert result["brain_refinery_good"].active is True
    assert result["brain_refinery_bad"].active is False
    assert result["brain_refinery_deleted"].active is False
