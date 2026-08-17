import json
from pathlib import Path

from scripts.ops import sleeve_micro_expansion


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"sub_bots": rows, "summary": {"total_bots": len(rows)}}, ensure_ascii=True), encoding="utf-8")


def test_plan_registry_expansion_is_collect_only_and_bounded() -> None:
    registry = {"sub_bots": [], "summary": {}}

    plan = sleeve_micro_expansion.plan_registry_expansion(registry)

    assert plan["planned_bot_count"] == 10
    assert plan["sleeve_profile_count"] == 10
    assert plan["safety_contract"]["data_collection_only"] is True
    for row in plan["planned_rows"]:
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["data_collection_active"] is True
        assert row["trading_enabled"] is False
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["rotation_blocked"] is True
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["eligible_for_master_vote"] is False
        assert row["weight"] == 0.0
        assert row["micro_expansion_slot"] is True
        assert row["max_daily_mb_per_bot"] == 2


def test_apply_registry_adds_ten_then_becomes_idempotent(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    registry_path = project_root / "master_bot_registry.json"
    config_path = project_root / "config" / "sleeve_micro_expansion_v1.json"
    _write_registry(registry_path, [])

    first = sleeve_micro_expansion.apply_registry(project_root, registry_path=registry_path, config_path=config_path)
    second = sleeve_micro_expansion.apply_registry(project_root, registry_path=registry_path, config_path=config_path)

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload["sub_bots"]
    assert first["applied"] is True
    assert first["added_bot_count"] == 10
    assert second["applied"] is False
    assert second["added_bot_count"] == 0
    assert len(rows) == 10
    assert payload["summary"]["sleeve_micro_expansion_bot_count"] == 10
    assert payload["summary"]["max_bot_version"] == 1647
    assert config_path.exists()
