from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "whole_system_governor.py"
spec = importlib.util.spec_from_file_location("whole_system_governor", MODULE_PATH)
governor = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(governor)


def _write_registry(root: Path) -> None:
    rows = []
    for idx in range(1, 8):
        rows.append(
            {
                "bot_id": f"brain_refinery_v{idx}_test_bot",
                "bot_role": "infrastructure_sub_bot" if idx % 2 else "signal_sub_bot",
                "active": True,
                "data_collection_active": True,
                "training_excluded": True,
                "data_collection_storage_guarded": idx != 7,
                "lifecycle_state": "data_collection_only",
                "capability_pack_slug": "quant_operational_intelligence" if idx <= 4 else "legacy_research",
                "quality_score": 0.2 + idx / 100,
                "live_trading_enabled": False,
                "execution_enabled": False,
            }
        )
    (root / "master_bot_registry.json").write_text(
        json.dumps({"summary": {"total_bots": len(rows), "active_bots": len(rows)}, "sub_bots": rows}) + "\n",
        encoding="utf-8",
    )


def test_build_payload_constructs_seven_layer_governor(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = governor.build_payload(tmp_path)

    assert payload["whole_system_governor_version"] == governor.GOVERNOR_VERSION
    assert payload["layer_count"] == 8
    assert payload["governor"]["authority_boundary"].startswith("advisory_budgeting")
    assert payload["governor"]["policy"]["expansion_requires_clean_scaling_contract"] is True
    assert payload["clean_scaling_control"]["overall_status"] == "missing"
    assert payload["evidence_court"]["required_sections"]
    assert payload["memory_triage_policy"]["hard_limits"]["raw_trace_requires_governor_exception"] is True
    assert payload["operator_decision_packet"]["do_not_do"]
    assert len(payload["sleeve_budgets"]) >= 2


def test_pressure_moves_low_value_collection_to_heartbeat(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps({"overall_status": "degraded", "backpressure": {"total_pending_lines": 300000}}) + "\n",
        encoding="utf-8",
    )

    payload = governor.build_payload(tmp_path)

    assert payload["governor"]["mode"] == "protective"
    assert payload["memory_triage_policy"]["default_capture_tier"] == "thin_digest"
    assert any(budget["capture_tier"] in {"heartbeat", "thin_digest"} for budget in payload["sleeve_budgets"])


def test_apply_writes_artifacts_and_registry_summary(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = governor.apply_governor(tmp_path)

    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "applied"
    assert registry["summary"]["whole_system_governor_version"] == governor.GOVERNOR_VERSION
    assert "whole_system_governor_clean_scaling_status" in registry["summary"]
    assert (tmp_path / "governance" / "health" / "whole_system_governor_latest.json").exists()
    assert (tmp_path / "governance" / "whole_system_governor" / "sleeve_budgets.json").exists()
    assert (tmp_path / "governance" / "whole_system_governor" / "operator_decision_packet.json").exists()
    assert (tmp_path / "governance" / "whole_system_governor" / "clean_scaling_contract.json").exists()
    assert (tmp_path / "exports" / "reports" / "operator" / "whole_system_governor_latest.md").exists()


def test_whole_system_governor_surfaces_clean_scaling_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    (health / "expansion_capacity_planner_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "blocked",
                "clean_scaling_contract": {
                    "overall_status": "blocked",
                    "grade": "C",
                    "mode": "blocked_clean_scaling",
                    "max_clean_wave_size_now": 0,
                    "blocked_dimensions": ["sql_overlay_tail_debt"],
                    "watch_dimensions": ["runtime_headroom"],
                    "dimension_count": 6,
                    "next_action": "clear overlay tails",
                    "clean_scaling_invariants": ["overlay tails must clear"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = governor.build_payload(tmp_path)

    clean = payload["clean_scaling_control"]
    assert clean["overall_status"] == "blocked"
    assert clean["blocked_dimensions"] == ["sql_overlay_tail_debt"]
    assert payload["governor"]["clean_scaling"]["max_clean_wave_size_now"] == 0
    assert any(item["title"] == "Clean scaling gate is not ready" for item in payload["operator_decision_packet"]["attention_queue"])
