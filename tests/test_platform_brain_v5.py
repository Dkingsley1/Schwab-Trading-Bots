import json
from pathlib import Path

from scripts.ops import platform_brain_v5 as src


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
                {"bot_id": "alpha_intraday_bot", "active": True, "data_collection_active": True, "lifecycle_state": "active"},
                {"bot_id": "cold_start_macro_bot", "active": True, "data_collection_active": True, "lifecycle_state": "data_collection_only", "training_excluded": True},
                {"bot_id": "inactive_legacy_bot", "active": False, "data_collection_active": False, "lifecycle_state": "inactive"},
            ]
        },
    )
    ranked_priorities = [
        {
            "section": "swap_cpu_capacity_planner",
            "priority_score": 91,
            "recommended_command": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
        },
        {
            "section": "provider_rotation_failover_mesh",
            "priority_score": 82,
            "recommended_command": "./scripts/ops/opsctl.sh platform-intelligence --apply --json",
        },
        {
            "section": "training_readiness_board",
            "priority_score": 73,
            "recommended_command": "./scripts/ops/opsctl.sh train --profile aggressive",
        },
        {
            "section": "operator_auth",
            "priority_score": 64,
            "recommended_command": "open broker auth UI",
        },
    ]
    _write_json(
        project_root / "governance" / "health" / "platform_brain_v4_latest.json",
        {
            "overall_status": "needs_work",
            "pressure_snapshot": {"overall_status": "degraded"},
            "sections": {
                "executive_meta_orchestrator": {
                    "overall_status": "needs_work",
                    "next_best_command": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
                    "ranked_priorities": ranked_priorities,
                },
                "autonomous_priority_ranker": {
                    "overall_status": "needs_work",
                    "priority_count": 4,
                    "ranked_priorities": ranked_priorities,
                },
                "predictive_expansion_simulator": {
                    "overall_status": "ready",
                    "simulations": [
                        {"additional_bots": 25, "recommendation": "defer"},
                        {"additional_bots": 100, "recommendation": "defer"},
                    ],
                },
                "training_scheduler_brain": {
                    "overall_status": "ready",
                    "training_policy": "off_hours_micro_batches",
                },
                "operator_intent_model": {
                    "overall_status": "ready",
                    "inferred_operator_mode": "foreground_app_headroom",
                },
                "critic_council": {
                    "overall_status": "needs_work",
                    "caution_count": 4,
                    "votes": [
                        {"critic": "pressure", "vote": "hold"},
                        {"critic": "provider", "vote": "hold"},
                        {"critic": "data", "vote": "hold"},
                        {"critic": "training", "vote": "hold"},
                    ],
                },
                "bot_portfolio_economist": {
                    "overall_status": "ready",
                    "trainable_bots": 12,
                    "cold_start_bots": 30,
                    "overlap_cluster_count": 5,
                },
                "data_value_engine": {
                    "overall_status": "needs_work",
                    "data_value_score": 35,
                },
                "causal_world_model": {
                    "overall_status": "needs_work",
                    "current_world_state": {"provider_status": "needs_work"},
                },
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {"overall_status": "degraded", "expansion_count": 12, "control_count": 12},
    )
    _append_jsonl(
        project_root / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl",
        {
            "platform_status": "needs_work",
            "top_actions": [
                "pressure-relief",
                "platform-intelligence",
                "pressure-relief",
            ],
        },
    )


def test_platform_brain_v5_builds_all_twelve_reflex_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["section_count"] == 12
    assert set(payload["section_keys"]) == set(src.SECTION_KEYS)
    assert payload["control_count"] == 12
    assert payload["sections"]["reflex_action_router"]["next_best_command"] == "./scripts/ops/opsctl.sh pressure-relief --apply --json"
    assert payload["sections"]["safe_autonomy_boundary"]["live_execution_allowed"] is False
    assert payload["recommended_env_overrides"]["PLATFORM_BRAIN_V5_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"
    assert payload["recommended_env_overrides"]["PAPER_TRADE_LOCK"] == "1"


def test_platform_brain_v5_rolls_thin_inputs_to_watch(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {"overall_status": "watch", "expansion_count": 12, "control_count": 12},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "watch"
    assert payload["ok"] is True
    assert payload["sections"]["temporal_self_model"]["overall_status"] == "thin"
    assert payload["sections"]["data_contract_negotiator"]["overall_status"] == "thin"


def test_platform_brain_v5_low_data_contract_score_is_watch_not_repair_failure() -> None:
    contract = src._data_contract(
        {
            "sections": {
                "data_value_engine": {"data_value_score": 35},
                "causal_world_model": {"current_world_state": {"provider_status": "watch"}},
            }
        },
        {"overall_status": "watch"},
    )

    assert contract["overall_status"] == "watch"
    assert contract["data_value_score"] == 35


def test_platform_brain_v5_writes_artifacts_and_reflex_memory(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    written = src.write_section_artifacts(tmp_path, payload)
    reflex_event = payload["sections"]["regret_and_outcome_ledger"]["latest_reflex_event"]
    memory_path = tmp_path / "governance" / "platform_brain_v5" / "reflex_memory" / "events.jsonl"

    assert len(written) == 12
    assert all(Path(path).exists() for path in written.values())
    assert src._append_reflex_event(memory_path, reflex_event) is True
    assert memory_path.read_text(encoding="utf-8").count("\n") == 1


def test_platform_brain_v5_rehearses_expansion_and_fuses_critics(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    scenarios = payload["sections"]["scenario_rehearsal_lab"]
    roadmap = payload["sections"]["strategic_roadmap_synthesizer"]
    critics = payload["sections"]["critic_ensemble_fusion"]
    reflex = payload["sections"]["reflex_action_router"]

    assert scenarios["scenario_count"] == 5
    assert roadmap["expansion_allowed_now"] is False
    assert critics["fusion_vote"] == "hold_expansion"
    assert critics["caution_count"] >= 3
    assert reflex["safe_reflex_count"] == 2
    assert reflex["operator_review_count"] == 2


def test_platform_brain_v5_treats_caution_only_expansion_hold_as_watch(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    critics = payload["sections"]["critic_ensemble_fusion"]

    assert payload["overall_status"] == "watch"
    assert critics["overall_status"] == "watch"
    assert critics["hard_vote_count"] == 0
    assert critics["severity_policy"] == "caution_votes_hold_expansion_without_degrading_guarded_collection_or_paper"


def test_platform_brain_v5_keeps_blocked_critic_vote_hard() -> None:
    critics = src._critic_fusion(
        {
            "sections": {
                "critic_council": {
                    "overall_status": "blocked",
                    "caution_count": 3,
                    "votes": [
                        {"critic": "provider", "vote": "blocked"},
                        {"critic": "data", "vote": "hold"},
                    ],
                }
            }
        },
        {"safe_reflex_count": 0},
    )

    assert critics["overall_status"] == "needs_work"
    assert critics["hard_vote_count"] == 1
    assert critics["severity_policy"] == "blocked_or_critical_critic_votes_require_operator_repair"


def test_platform_brain_v5_treats_repeated_advisory_regret_as_watch(tmp_path: Path) -> None:
    ledger = src._regret_ledger(
        tmp_path,
        {"v5_reflex_event_count": 2, "repeated_action_themes": [{"action": "refresh", "count": 3}, {"action": "review", "count": 2}]},
        {
            "overall_status": "needs_work",
            "sections": {
                "autonomous_priority_ranker": {
                    "priority_count": 8,
                    "ranked_priorities": [
                        {"section": "quality", "status": "watch"},
                        {"section": "realism", "status": "watch"},
                    ],
                },
                "executive_meta_orchestrator": {"next_best_command": "./scripts/ops/opsctl.sh health-fast --json"},
            },
        },
    )

    assert ledger["overall_status"] == "watch"
    assert ledger["hard_priority_count"] == 0
    assert ledger["severity_policy"] == "high_regret_from_advisory_repetition_is_soak_watch_debt"


def test_platform_brain_v5_keeps_hard_priority_regret_as_needs_work(tmp_path: Path) -> None:
    ledger = src._regret_ledger(
        tmp_path,
        {"v5_reflex_event_count": 0, "repeated_action_themes": [{"action": "repair", "count": 3}]},
        {
            "overall_status": "degraded",
            "sections": {
                "autonomous_priority_ranker": {
                    "priority_count": 8,
                    "ranked_priorities": [{"section": "provider", "status": "critical"}],
                },
                "executive_meta_orchestrator": {"next_best_command": "./scripts/ops/opsctl.sh provider-mesh --json"},
            },
        },
    )

    assert ledger["overall_status"] == "needs_work"
    assert ledger["hard_priority_count"] == 1
    assert ledger["severity_policy"] == "high_regret_with_hard_priority_rows_requires_operator_repair"
