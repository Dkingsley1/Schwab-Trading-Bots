import json
from pathlib import Path

import scripts.run_shadow_training_loop as loop
from core.execution_lane_pipeline import evaluate_live_promotion
from scripts.ops.training_probation_isolation import build_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_training_excluded_promoted_bot_is_not_canary_authority() -> None:
    bot = loop.SubBot(
        bot_id="brain_refinery_v80_execution_feasibility_sentinel",
        weight=1.0,
        active=True,
        reason="legacy_promoted",
        test_accuracy=0.50,
        promoted=True,
        training_excluded=True,
    )

    assert loop._is_canary_bot(bot, set()) is False


def test_live_promotion_blocks_quality_probation_promoted_bot(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_gate_latest.json",
        {"promote_ok": True, "coverage_ok": True, "considered_bots": 2},
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "lane_promotion_gate_latest.json",
        {"promote_ok": True, "coverage_ok": True, "lanes": {"default": {"promote_ok": True, "coverage_ok": True}}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "promotion_quality_gate_latest.json",
        {"ok": True, "failed_checks": []},
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v86_risk_budget_allocator_v2",
                    "active": True,
                    "promoted": True,
                    "promotion_status": "probation",
                    "training_excluded": True,
                }
            ]
        },
    )

    out = evaluate_live_promotion(
        project_root=str(tmp_path),
        intent={
            "intent_kind": "paper_mirror",
            "action": "BUY",
            "strategy": "paper_mirror::brain_refinery_v86_risk_budget_allocator_v2",
            "metadata": {"allow_live_promotion": True, "runtime_lane": "default"},
        },
        paper_result={"status": "PAPER_EXECUTED"},
    )

    assert out["promote_ok"] is False
    assert "bot_training_or_quality_excluded" in out["reasons"]
    assert "bot_promotion_status_not_live:probation" in out["reasons"]


def test_probation_isolation_clamps_weak_promoted_authority(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v12_news_shocks",
                    "active": True,
                    "promoted": True,
                    "promotion_status": "promoted",
                    "training_excluded": True,
                    "training_exclusion_reason": "minimum_data_collection_threshold_not_met",
                }
            ]
        },
    )

    payload = build_payload(
        project_root=tmp_path,
        registry_path=registry_path,
        audit_path=tmp_path / "missing_audit.json",
        apply=True,
        include_bot_ids=["brain_refinery_v12_news_shocks"],
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["authority_clamped_count"] == 1
    assert row["promoted"] is False
    assert row["promotion_status"] == "probation"
    assert row["trusted_master_authority"] is False
    assert row["training_exclusion_reason"] == "minimum_data_collection_threshold_not_met"


def test_probation_isolation_uses_full_quality_failed_id_list(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    bot_ids = [f"brain_refinery_v{i}_quality_probe" for i in range(30)]
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": True,
                    "lifecycle_state": "active",
                }
                for bot_id in bot_ids
            ]
        },
    )
    audit_path = tmp_path / "governance" / "health" / "training_registry_audit_latest.json"
    _write_json(
        audit_path,
        {
            "active_quality_failed_bot_ids": bot_ids,
            "active_quality_failed": [{"bot_id": bot_id} for bot_id in bot_ids[:25]],
        },
    )

    payload = build_payload(
        project_root=tmp_path,
        registry_path=registry_path,
        audit_path=audit_path,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["target_count"] == 30
    assert payload["newly_isolated_count"] == 30
    assert all(row["training_excluded"] is True for row in registry["sub_bots"])


def test_probation_isolation_can_isolate_runtime_input_debt_without_deactivating(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    bot_ids = ["brain_refinery_v12_news_shocks", "brain_refinery_v15_liquidity_droughts"]
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": True,
                    "lifecycle_state": "paper_live_data",
                    "data_collection_active": True,
                }
                for bot_id in bot_ids
            ]
        },
    )
    audit_path = tmp_path / "governance" / "health" / "training_registry_audit_latest.json"
    _write_json(
        audit_path,
        {
            "active_quality_failed_bot_ids": [],
            "active_quality_failed": [{"bot_id": "brain_refinery_v999_stale_preview"}],
            "active_sample_starved_bot_ids": bot_ids,
        },
    )

    payload = build_payload(
        project_root=tmp_path,
        registry_path=registry_path,
        audit_path=audit_path,
        apply=True,
        include_runtime_input_debt=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["target_count"] == 2
    assert payload["quality_target_count"] == 0
    assert payload["runtime_input_target_count"] == 2
    assert payload["newly_isolated_count"] == 2
    assert all(row["active"] is True for row in registry["sub_bots"])
    assert all(row["data_collection_active"] is True for row in registry["sub_bots"])
    assert all(row["training_excluded"] is True for row in registry["sub_bots"])
    assert all(row["training_exclusion_reason"] == "runtime_input_debt_isolation" for row in registry["sub_bots"])
