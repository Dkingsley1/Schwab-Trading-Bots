from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "system_self_awareness_expansion.py"
spec = importlib.util.spec_from_file_location("system_self_awareness_expansion", MODULE_PATH)
self_awareness = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(self_awareness)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-03T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v893_seed_bot",
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


def test_dry_run_plans_system_self_awareness_infrabots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = self_awareness.build_payload(tmp_path)

    assert payload["pack_count"] == 1
    assert payload["bot_count"] == 10
    assert payload["planned_bot_count"] == 10
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v894_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v903_")
    assert payload["pack"]["storage_retention_rule"]["dedupe_required"] is True
    assert payload["pack"]["paper_only_floor"]["execution_enabled"] is False
    assert "resource_awareness" in payload["pack"]["self_awareness_depth"]
    assert "grandmaster_self_model_bridge" in payload["pack"]["self_awareness_depth"]


def test_apply_adds_collection_only_self_awareness_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = self_awareness.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 10
    assert payload["new_total_bots"] == 11
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("system_self_awareness_version") == self_awareness.PACK_VERSION]
    assert len(added) == 10
    assert registry["summary"]["system_self_awareness_bot_count"] == 10
    assert registry["summary"]["data_collection_active_bots"] == 10
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["minimum_training_observations"] == 5000
        assert row["minimum_data_collection_days"] == 21
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "5_to_15_bots_max"
    assert (tmp_path / "config" / "system_self_awareness_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "system_self_awareness_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = self_awareness.apply_registry(tmp_path)
    second = self_awareness.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("system_self_awareness_version") == self_awareness.PACK_VERSION]

    assert first["added_bot_count"] == 10
    assert second["added_bot_count"] == 0
    assert len(added) == 10
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
