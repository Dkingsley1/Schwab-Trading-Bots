from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "frontier_intelligence_expansion.py"
spec = importlib.util.spec_from_file_location("frontier_intelligence_expansion", MODULE_PATH)
frontier = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(frontier)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1038, "active_bots": 1000},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1037_recursive_awareness_living_framework_map_regression_guard_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Deep Awareness Bot",
                        "slot_kind": "deep_recursive_awareness_living_framework_map_regression_guard",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_frontier_intelligence_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = frontier.build_payload(tmp_path)

    assert payload["system_count"] == 12
    assert payload["bot_count"] == 48
    assert payload["planned_bot_count"] == 48
    assert payload["target_platform_total_bots"] == 1086
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1038_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1085_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.08
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 48000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 180
    assert "counterfactual_causal_lab" in payload["pack"]["intelligence_advancements"]
    assert "operator_copilot_intent_bridge" in payload["pack"]["intelligence_advancements"]


def test_apply_adds_collect_only_frontier_intelligence_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = frontier.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 48
    assert payload["new_total_bots"] == 49
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("frontier_intelligence_version") == frontier.PACK_VERSION
    ]
    assert len(added) == 48
    assert registry["summary"]["frontier_intelligence_bot_count"] == 48
    assert registry["summary"]["data_collection_active_bots"] == 49
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["minimum_training_observations"] == 48000
        assert row["minimum_data_collection_days"] == 180
        assert row["data_collection_capture_mode"] == "sampled"
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1086
        assert row["frontier_intelligence_contract"]["contract_version"] == "frontier_intelligence_layers_v1"
        assert row["training_threshold_policy"]["requires_platform_brain_v6_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "frontier_intelligence_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "frontier_intelligence_latest.json").exists()
    assert (tmp_path / "governance" / "frontier_intelligence" / "counterfactual_causal_lab").is_dir()
    assert "governance/frontier_intelligence/counterfactual_causal_lab" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = frontier.apply_registry(tmp_path)
    second = frontier.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("frontier_intelligence_version") == frontier.PACK_VERSION
    ]

    assert first["added_bot_count"] == 48
    assert second["added_bot_count"] == 0
    assert len(added) == 48
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
