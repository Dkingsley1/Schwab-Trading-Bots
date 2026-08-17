from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "deep_recursive_awareness_expansion.py"
spec = importlib.util.spec_from_file_location("deep_recursive_awareness_expansion", MODULE_PATH)
deep = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(deep)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1000, "active_bots": 945},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1009_seed_apex_frontier_guard",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_deep_recursive_awareness_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = deep.build_payload(tmp_path)

    assert payload["system_count"] == 7
    assert payload["bot_count"] == 28
    assert payload["planned_bot_count"] == 28
    assert payload["target_platform_total_bots"] == 1028
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1010_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1037_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.1
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 36000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 150
    assert "causal_self_diagnosis" in payload["pack"]["awareness_advancements"]
    assert "recursive_platform_map" in payload["pack"]["awareness_advancements"]
    assert {system["slug"] for system in payload["pack"]["systems"]} == {
        "causal_self_diagnosis",
        "predictive_runtime_oracle",
        "experience_memory_core",
        "self_upgrade_critic_board",
        "operator_context_governor",
        "internal_critic_board",
        "recursive_platform_map",
    }


def test_apply_adds_collect_only_recursive_awareness_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = deep.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 28
    assert payload["new_total_bots"] == 29
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("deep_recursive_awareness_version") == deep.PACK_VERSION
    ]
    assert len(added) == 28
    assert registry["summary"]["deep_recursive_awareness_bot_count"] == 28
    assert registry["summary"]["data_collection_active_bots"] == 28
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
        assert row["minimum_training_observations"] == 36000
        assert row["minimum_data_collection_days"] == 150
        assert row["data_collection_capture_mode"] == "sampled"
        assert row["data_collection_compute_guard_mode"] == "sustain"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1028
        assert row["deep_recursive_awareness_contract"]["contract_version"] == "deep_recursive_awareness_layers_v1"
        assert row["training_threshold_policy"]["requires_causal_diagnosis_quality_clearance"] is True
        assert row["training_threshold_policy"]["requires_recursive_map_integrity_clearance"] is True
    assert (tmp_path / "config" / "deep_recursive_awareness_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "deep_recursive_awareness_latest.json").exists()
    assert (tmp_path / "governance" / "deep_recursive_awareness" / "causal_self_diagnosis").is_dir()
    assert "governance/deep_recursive_awareness/causal_self_diagnosis" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = deep.apply_registry(tmp_path)
    second = deep.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("deep_recursive_awareness_version") == deep.PACK_VERSION
    ]

    assert first["added_bot_count"] == 28
    assert second["added_bot_count"] == 0
    assert len(added) == 28
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
