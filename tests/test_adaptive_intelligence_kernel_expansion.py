from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "adaptive_intelligence_kernel_expansion.py"
spec = importlib.util.spec_from_file_location("adaptive_intelligence_kernel_expansion", MODULE_PATH)
kernel = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(kernel)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-03T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v878_seed_bot",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "data_collection_active": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_adaptive_intelligence_kernel(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = kernel.build_payload(tmp_path)

    assert payload["pack_count"] == 1
    assert payload["bot_count"] == 15
    assert payload["planned_bot_count"] == 15
    assert payload["kernel"]["bot_ids"][0].startswith("brain_refinery_v879_")
    assert payload["kernel"]["bot_ids"][-1].startswith("brain_refinery_v893_")
    assert payload["kernel"]["storage_retention_rule"]["dedupe_required"] is True
    assert payload["kernel"]["storage_retention_rule"]["sample_rate"] == 0.20
    assert payload["kernel"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["kernel"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 10000
    assert payload["kernel"]["capacity_check"]["heavy_reasoning_mode"] == "off_hot_path_low_pressure_or_simulation_window_only"
    assert "online_meta_learning" in payload["kernel"]["adaptive_depth"]
    assert "simulation_to_reality_gap_detection" in payload["kernel"]["adaptive_depth"]


def test_apply_adds_collection_only_kernel_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = kernel.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 15
    assert payload["new_total_bots"] == 16
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("adaptive_intelligence_kernel_version") == kernel.KERNEL_VERSION]
    assert len(added) == 15
    assert registry["summary"]["adaptive_intelligence_kernel_bot_count"] == 15
    assert registry["summary"]["structured_capability_pack_bot_count"] == 15
    assert registry["summary"]["data_collection_active_bots"] == 15
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["minimum_training_observations"] == 10000
        assert row["minimum_data_collection_days"] == 35
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "5_to_15_bots_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "advanced_intelligence_layers_v5"
    assert (tmp_path / "config" / "adaptive_intelligence_kernel_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "adaptive_intelligence_kernel_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = kernel.apply_registry(tmp_path)
    second = kernel.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("adaptive_intelligence_kernel_version") == kernel.KERNEL_VERSION]

    assert first["added_bot_count"] == 15
    assert second["added_bot_count"] == 0
    assert len(added) == 15
    assert second["kernel"]["bot_ids"][0] == first["added_bot_ids"][0]
