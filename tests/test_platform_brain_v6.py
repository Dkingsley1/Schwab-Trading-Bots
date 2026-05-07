import json
from pathlib import Path

from scripts.ops import platform_brain_v6 as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _seed_project(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v1", "active": True, "data_collection_active": True, "lifecycle_state": "active"},
                {"bot_id": "frontier_seed", "active": True, "data_collection_active": True, "lifecycle_state": "data_collection_only", "training_excluded": True, "capability_pack_slug": "frontier_seed"},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_throttle_control_latest.json",
        {"overall_status": "blocked", "host_saturation_score": 88.0, "compute_pressure_level": "high", "memory_pressure_level": "elevated"},
    )
    _write_json(project_root / "governance" / "health" / "memory_efficiency_control_latest.json", {"overall_status": "constrained"})
    _write_json(project_root / "governance" / "health" / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}})
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "high", "pressure_index": 1.7, "backpressure": {"total_pending_lines": 26000, "pending_lines_threshold": 15000}},
    )
    _write_json(project_root / "governance" / "health" / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_blocked", "clear_blockers": ["queue_backpressure_active"]})
    _write_json(
        project_root / "governance" / "health" / "platform_stabilization_quality_latest.json",
        {
            "overall_status": "needs_work",
            "sections": {
                "expansion_rehearsal_gate": {
                    "overall_status": "needs_work",
                    "expansion_allowed_now": False,
                    "gate_closed_reasons": ["runtime_not_calm", "storage_or_queue_not_settled"],
                }
            },
        },
    )
    _write_json(project_root / "governance" / "health" / "platform_brain_v5_latest.json", {"overall_status": "needs_work"})
    _write_json(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json", {"overall_status": "degraded"})
    _write_json(project_root / "governance" / "health" / "deep_recursive_awareness_latest.json", {"mode": "applied"})
    _append_jsonl(project_root / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl", {"platform_status": "needs_work"})
    _append_jsonl(project_root / "governance" / "platform_brain_v5" / "reflex_memory" / "reflex_events.jsonl", {"regret_score": 40})


def test_platform_brain_v6_builds_fifteen_foresight_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["section_count"] == 15
    assert set(payload["section_keys"]) == set(src.SECTION_KEYS)
    assert payload["control_count"] == 15
    assert "runtime_not_calm" in payload["gate_blockers"]
    assert "queue_backpressure_active" in payload["gate_blockers"]
    assert payload["sections"]["execution_policy_sandbox"]["live_execution_allowed"] is False
    assert payload["sections"]["formal_safety_guard"]["mlx_default"] is True
    assert payload["recommended_env_overrides"]["PLATFORM_BRAIN_V6_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"
    assert payload["recommended_env_overrides"]["PLATFORM_BRAIN_V6_EXPANSION_ALLOWED_NOW"] == "0"


def test_platform_brain_v6_writes_artifacts_and_memory(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    written = src.write_section_artifacts(tmp_path, payload)
    memory_path = tmp_path / "governance" / "platform_brain_v6" / "foresight_memory" / "events.jsonl"

    assert len(written) == 15
    assert all(Path(value).exists() for value in written.values())
    assert src._append_memory_event(memory_path, payload["latest_foresight_event"]) is True
    assert memory_path.read_text(encoding="utf-8").count("\n") == 1


def test_platform_brain_v6_recommends_safe_intervention_first(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    intervention = payload["sections"]["causal_intervention_planner"]
    narrative = payload["sections"]["operator_narrative_synthesizer"]

    assert intervention["intervention_count"] >= 1
    assert narrative["next_best_command"].startswith("./scripts/ops/opsctl.sh")
    assert payload["sections"]["multi_agent_debate_chamber"]["hold_count"] >= 1
