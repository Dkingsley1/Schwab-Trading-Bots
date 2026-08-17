from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ops" / "system_expansion_execution_layer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("system_expansion_execution_layer", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load system_expansion_execution_layer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_base_sources(root: Path) -> None:
    health = root / "governance" / "health"
    _write_json(
        health / "system_architecture_contract_graph_latest.json",
        {
            "ok": False,
            "overall_status": "blocked",
            "blocked_nodes": ["system_drift_guard"],
            "degraded_nodes": ["runtime_throttle", "adaptive_regression_guard"],
            "stale_nodes": ["health_fast"],
            "recommended_commands": [
                ["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"],
                ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
            ],
            "nodes": [
                {
                    "node_id": "health_fast",
                    "status": "degraded",
                    "required": True,
                    "artifact_stale": True,
                    "artifact": "governance/health/health_fast_latest.json",
                    "artifact_age_minutes": 42,
                    "artifact_max_age_minutes": 10,
                    "commands": [["./scripts/ops/opsctl.sh", "health-fast", "--json"]],
                },
                {
                    "node_id": "system_drift_guard",
                    "status": "blocked",
                    "required": False,
                    "artifact_stale": False,
                    "commands": [["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]],
                },
            ],
        },
    )
    _write_json(
        health / "schwab_indicator_intelligence_latest.json",
        {
            "ok": True,
            "overall_status": "schwab_indicator_intelligence_ready",
            "coverage": {"catalog_item_count": 2, "study_count": 1, "strategy_count": 1},
            "catalog_items": [
                {
                    "name": "VWAP",
                    "kind": "study",
                    "families": ["volume_flow"],
                    "required_inputs": ["ohlc_price_bars", "volume"],
                },
                {
                    "name": "MACDStrat",
                    "kind": "strategy",
                    "families": ["trend", "strategy_signal"],
                    "required_inputs": ["ohlc_price_bars", "paper_validation_evidence"],
                },
            ],
            "sleeve_applicability_matrix": [
                {
                    "sleeve": "intraday_aggressive",
                    "mapped_item_count": 2,
                    "top_studies": ["VWAP"],
                    "top_strategies": ["MACDStrat"],
                    "mapped_families": ["volume_flow", "trend", "strategy_signal"],
                }
            ],
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "ok": False,
            "overall_status": "degraded",
            "memory_pressure_level": "elevated",
            "compute_pressure_level": "watch",
            "host_saturation_score": 62,
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "ok": False,
            "overall_status": "needs_work",
            "memory_snapshot": {"memory_pressure_state": "green", "swap_used_gb": 0.5, "compressed_store_gb": 5.0},
            "storage_snapshot": {"pressure_index": 0.1},
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"ok": True, "overall_status": "ready", "backpressure": {"total_pending_lines": 1000}})
    _write_json(health / "health_fast_latest.json", {"ok": False, "overall_status": "degraded"})
    _write_json(
        health / "capital_rotation_control_latest.json",
        {
            "ok": True,
            "overall_status": "capital_rotation_ready",
            "sleeve_rotation_plan": [
                {"profile": "quality_growth", "signed_rotation_pressure_norm": 0.42},
                {"profile": "intraday_aggressive", "signed_rotation_pressure_norm": -0.35},
            ],
        },
    )
    _write_json(
        health / "paper_performance_latest.json",
        {"ok": True, "overall_status": "ready", "sleeve_latest": [{"profile": "quality_growth", "net_pnl": 42.0, "win_rate": 0.62}]},
    )
    _write_json(root / "master_bot_registry.json", {"sub_bots": [{"active": True, "sleeve_profile": "intraday_aggressive", "schwab_direct_inputs": ["quotes", "chains"]}]})


def test_build_payload_covers_all_twelve_lanes(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)

    payload = module.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["rollup"]["lane_count"] == 12
    lanes = {row["lane_id"]: row for row in payload["lanes"]}
    assert set(lanes) == {row["lane_id"] for row in module.LANE_DEFINITIONS}
    assert lanes["self_healing_router"]["details"]["route_count"] >= 1
    assert lanes["stale_surface_autofix"]["details"]["stale_count"] == 1
    assert lanes["schwab_indicator_feature_bridge"]["details"]["feature_candidate_count"] == 2
    assert lanes["grandmaster_safe_mode"]["details"]["sleeve_modes"][0]["live_execution_authority"] is False
    assert payload["authority_boundary"] == module.LIVE_LOCK


def test_apply_writes_override_and_operator_memory(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)
    override = tmp_path / "config" / ".env.system_expansion_execution_layer_override"
    memory = tmp_path / "governance" / "system_expansion_execution" / "operator_memory.jsonl"

    payload = module.build_payload(tmp_path, apply=True, override_path=override, memory_path=memory)

    assert payload["write_result"]["applied"] is True
    text = override.read_text(encoding="utf-8")
    assert "SYSTEM_EXPANSION_LANE_COUNT=12" in text
    assert "SYSTEM_EXPANSION_LIVE_EXECUTION_AUTHORITY=0" in text
    assert memory.exists()
    first = json.loads(memory.read_text(encoding="utf-8").splitlines()[0])
    assert first["next_focus"] == "self_healing_router_then_stale_surface_autofix_then_indicator_feature_bridge"
