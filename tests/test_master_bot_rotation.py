from core.master_bot import BotStatus, MasterBot
import json


def _status(
    bot_id: str,
    *,
    deleted: bool = False,
    reason: str = "walk_forward_fail",
    role: str = "signal_sub_bot",
    quality: float = 0.6,
) -> BotStatus:
    return BotStatus(
        bot_id=bot_id,
        bot_role=role,
        active=False,
        reason=reason,
        weight=0.0,
        preference_score=0.0,
        quality_score=quality,
        test_accuracy=None,
        candidate_test_accuracy=0.6,
        candidate_quality_score=quality,
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


def test_role_floor_restores_best_infrastructure_canary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MASTER_MIN_ACTIVE_INFRASTRUCTURE_BOTS", "4")
    monkeypatch.setenv("MASTER_INFRA_FLOOR_MIN_QUALITY_SCORE", "0.44")
    master = MasterBot(project_root=str(tmp_path), min_active_bots=0)

    statuses = [
        _status("brain_refinery_v56_meta_ranker", role="infrastructure_sub_bot", reason="within_operating_band", quality=0.87),
        _status("brain_refinery_v68_risk_budget_layer", role="infrastructure_sub_bot", reason="within_operating_band", quality=0.42),
        _status("brain_refinery_v86_risk_budget_allocator_v2", role="infrastructure_sub_bot", reason="within_operating_band", quality=0.50),
        _status("brain_refinery_v67_correlation_penalty_layer", role="infrastructure_sub_bot", reason="graduation_hold:forward_mean<0.520", quality=0.468),
        _status("brain_refinery_v69_cost_aware_execution_filter", role="infrastructure_sub_bot", reason="graduation_hold:forward_mean<0.520", quality=0.465),
    ]
    for st in statuses[:3]:
        st.active = True

    result = {status.bot_id: status for status in master._enforce_role_floors(statuses)}

    assert result["brain_refinery_v67_correlation_penalty_layer"].active is True
    assert result["brain_refinery_v67_correlation_penalty_layer"].reason == "role_floor_infrastructure_sub_bot"
    assert result["brain_refinery_v69_cost_aware_execution_filter"].active is False


def test_train_from_outcomes_preserves_bots_without_fresh_logs(tmp_path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    logs_dir = tmp_path / "logs"
    walk_forward_dir = tmp_path / "governance" / "walk_forward"
    logs_dir.mkdir()
    walk_forward_dir.mkdir(parents=True)
    stale_prev_log_path = logs_dir / "brain_refinery_v10_kept_missing_20260328_000000.json"
    stale_latest_log_path = logs_dir / "brain_refinery_v10_kept_missing_20260329_000000.json"
    fresh_log_path = logs_dir / "brain_refinery_v11_fresh_20260330_010101.json"

    stale_prev_log_path.write_text(
        json.dumps(
            {
                "model_path": "/tmp/brain_refinery_v10_kept_missing_existing.npz",
                "metrics": {
                    "test_accuracy": 0.62,
                    "best_val_f1": 0.61,
                    "test_macro_f1": 0.60,
                    "final_val_loss": 0.44,
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    stale_latest_log_path.write_text(
        json.dumps(
            {
                "model_path": "/tmp/brain_refinery_v10_kept_missing_latest.npz",
                "metrics": {
                    "test_accuracy": 0.63,
                    "best_val_f1": 0.62,
                    "test_macro_f1": 0.61,
                    "final_val_loss": 0.43,
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    registry_path.write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-03-29T00:00:00+00:00",
                "summary": {"total_bots": 3, "active_bots": 2, "inactive_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v10_kept_missing",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "reason": "within_operating_band",
                        "weight": 0.55,
                        "preference_score": 1.2,
                        "quality_score": 0.81,
                        "test_accuracy": 0.63,
                        "candidate_test_accuracy": 0.63,
                        "candidate_quality_score": 0.81,
                        "previous_best_accuracy": 0.63,
                        "no_improvement_streak": 1,
                        "deleted_from_rotation": False,
                        "delete_reason": "",
                        "promoted": False,
                        "promotion_reason": "held_previous_model",
                        "model_path": "/tmp/brain_refinery_v10_kept_missing.npz",
                        "log_file": str(stale_prev_log_path),
                        "candidate_log_file": str(stale_prev_log_path),
                    },
                    {
                        "bot_id": "brain_refinery_v11_fresh",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "reason": "within_operating_band",
                        "weight": 0.45,
                        "preference_score": 1.0,
                        "quality_score": 0.72,
                        "test_accuracy": 0.61,
                        "candidate_test_accuracy": 0.61,
                        "candidate_quality_score": 0.72,
                        "previous_best_accuracy": 0.61,
                        "no_improvement_streak": 0,
                        "deleted_from_rotation": False,
                        "delete_reason": "",
                        "promoted": False,
                        "promotion_reason": "held_previous_model",
                        "model_path": "/tmp/brain_refinery_v11_fresh.npz",
                        "log_file": str(logs_dir / "brain_refinery_v11_fresh_20260329_000000.json"),
                        "candidate_log_file": str(logs_dir / "brain_refinery_v11_fresh_20260329_000000.json"),
                    },
                    {
                        "bot_id": "brain_refinery_v12_deleted",
                        "bot_role": "signal_sub_bot",
                        "active": False,
                        "reason": "deleted_no_improvement_3_retrainings",
                        "weight": 0.0,
                        "preference_score": 0.0,
                        "quality_score": 0.18,
                        "test_accuracy": 0.49,
                        "candidate_test_accuracy": 0.49,
                        "candidate_quality_score": 0.18,
                        "previous_best_accuracy": 0.54,
                        "no_improvement_streak": 3,
                        "deleted_from_rotation": True,
                        "delete_reason": "deleted_no_improvement_3_retrainings",
                        "promoted": False,
                        "promotion_reason": "rotation_deleted",
                        "model_path": "/tmp/brain_refinery_v12_deleted.npz",
                        "log_file": "/tmp/brain_refinery_v12_deleted_20260329_000000.json",
                        "candidate_log_file": "/tmp/brain_refinery_v12_deleted_20260329_000000.json",
                    },
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    (walk_forward_dir / "walk_forward_latest.json").write_text(
        json.dumps(
            {
                "bots": {
                    "brain_refinery_v11_fresh": {
                        "runs": 30,
                        "status": "pass",
                        "forward_mean": 0.63,
                        "delta": 0.01,
                        "trading_quality_score": 0.66,
                    }
                }
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    fresh_log_path.write_text(
        json.dumps(
            {
                "model_path": "/tmp/brain_refinery_v11_fresh_new.npz",
                "metrics": {
                    "test_accuracy": 0.66,
                    "best_val_f1": 0.65,
                    "test_macro_f1": 0.64,
                    "final_val_loss": 0.42,
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    master = MasterBot(project_root=str(tmp_path), min_active_bots=0)
    payload = master.train_from_outcomes()
    rows = {row["bot_id"]: row for row in payload["sub_bots"]}

    assert set(rows) == {
        "brain_refinery_v10_kept_missing",
        "brain_refinery_v11_fresh",
        "brain_refinery_v12_deleted",
    }
    assert payload["summary"]["total_bots"] == 3
    assert payload["summary"]["active_bots"] == 2
    assert rows["brain_refinery_v10_kept_missing"]["active"] is True
    assert rows["brain_refinery_v10_kept_missing"]["reason"] == "within_operating_band"
    assert rows["brain_refinery_v12_deleted"]["deleted_from_rotation"] is True
