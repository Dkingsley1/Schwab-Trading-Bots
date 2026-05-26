import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import training_data_intake_expansion as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_training_data_intake_builds_focus_from_contract_and_needs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v117_iv_skew_dislocation_overlay"
    registry_path = project_root / "master_bot_registry.json"
    needs_path = project_root / "governance" / "health" / "bot_needs_intelligence_latest.json"
    quality_path = project_root / "governance" / "health" / "training_quality_control_latest.json"
    _write_json(
        registry_path,
        {
            "summary": {},
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "bot_role": "options_sub_bot",
                    "active": True,
                    "data_collection_active": True,
                    "lifecycle_state": "paper_live_data",
                    "minimum_training_observations": 1000,
                    "label_contract": {
                        "label_family": "options_surface",
                        "training_lane": "lane_specific_fast",
                        "required_context": ["options_chain", "iv_surface"],
                    },
                }
            ],
        },
    )
    _write_json(
        needs_path,
        {
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "collect_more_data",
                    "priority": 92,
                    "evidence": {
                        "sample_count": 44,
                        "observation_count": 44,
                        "eligible_sequences": 1,
                        "positive_rate": 0.88,
                        "acted_coverage": 0.6,
                        "quality_score": 0.3,
                    },
                }
            ]
        },
    )
    _write_json(
        quality_path,
        {
            "targeted_actions": {
                "runtime_input_depth_debt_bot_ids": [bot_id],
            }
        },
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        bot_needs_path=needs_path,
        training_quality_path=quality_path,
        apply=False,
    )

    record = payload["focus_records"][0]
    assert payload["collector_count"] == 1
    assert record["bot_id"] == bot_id
    assert "options_chain" in record["focus_context"]
    assert "greeks" in record["expanded_context"]
    assert "sample_starved" in record["weaknesses"]
    assert "label_imbalanced" in record["weaknesses"]
    assert "runtime_depth_debt" in record["weaknesses"]
    assert "label_outcome_join" in record["enrichment_context"]
    assert record["sample_enrichment_plan"]["usable_sample_goal"] == 240
    assert record["sample_enrichment_plan"]["eligible_sequence_goal"] == 8
    assert record["sample_enrichment_plan"]["intensity"] == "high"
    assert record["label_repair_plan"]["required_join_mode"] == "point_in_time_only"
    assert "label_outcome_join" in record["label_repair_plan"]["required_label_outputs"]
    assert record["label_repair_plan"]["balance_targets"]["positive_rate_min"] == 0.35
    assert payload["summaries"]["context_counts"]["options_chain"] == 1


def test_training_data_intake_adds_advanced_quant_section_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v444_quant_pricing_merton_jump_diffusion_bot"
    registry_path = project_root / "master_bot_registry.json"
    needs_path = project_root / "governance" / "health" / "bot_needs_intelligence_latest.json"
    quality_path = project_root / "governance" / "health" / "training_quality_control_latest.json"
    _write_json(
        registry_path,
        {
            "summary": {},
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "data_collection_active": True,
                    "lifecycle_state": "data_collection_only",
                    "minimum_training_observations": 3000,
                    "data_intake_collections": ["quant_model_feature_surface"],
                    "label_contract": {
                        "label_family": "quant_pricing_research",
                        "training_lane": "research_quant_proxy",
                        "required_context": ["quant_model_feature_surface"],
                    },
                }
            ],
        },
    )
    _write_json(
        needs_path,
        {
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "collect_more_data",
                    "priority": 80,
                    "evidence": {
                        "sample_count": 80,
                        "observation_count": 160,
                        "eligible_sequences": 1,
                        "positive_rate": 0.5,
                        "acted_coverage": 0.0,
                        "quality_score": 0.0,
                    },
                }
            ]
        },
    )
    _write_json(quality_path, {"targeted_actions": {}})

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        bot_needs_path=needs_path,
        training_quality_path=quality_path,
        apply=True,
    )

    record = payload["focus_records"][0]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert "advanced_quant_depth_debt" in record["weaknesses"]
    assert "advanced_quant_proxy_gap" in record["weaknesses"]
    assert "advanced_quant_label_gap" in record["weaknesses"]
    assert "model_price_sensitivity_grid" in record["focus_context"]
    assert "point_in_time_label_quality" in record["enrichment_context"]
    assert record["sample_enrichment_plan"]["usable_sample_goal"] == 360
    assert record["sample_enrichment_plan"]["eligible_sequence_goal"] == 12
    assert "pricing_model_dispersion_bucket" in record["label_repair_plan"]["required_label_outputs"]
    assert record["advanced_quant_collection_contract"]["active"] is True
    assert record["advanced_quant_collection_contract"]["section_count"] == 5
    assert row["data_collection_advanced_quant_contract"]["contract_version"] == "advanced_quant_collection_sections_v1"
    assert payload["summaries"]["advanced_quant_contract_count"] == 1


def test_training_data_intake_apply_writes_registry_focus_metadata(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v188_host_resource_pressure_guard"
    registry_path = project_root / "master_bot_registry.json"
    needs_path = project_root / "governance" / "health" / "bot_needs_intelligence_latest.json"
    quality_path = project_root / "governance" / "health" / "training_quality_control_latest.json"
    _write_json(
        registry_path,
        {
            "summary": {},
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "bot_role": "infrastructure_sub_bot",
                    "active": True,
                    "data_collection_active": True,
                    "lifecycle_state": "data_collection_only",
                    "minimum_training_observations": 200,
                    "label_contract": {
                        "label_family": "operational_guard_effect",
                        "required_context": ["runtime_health", "incident_log"],
                    },
                }
            ],
        },
    )
    _write_json(
        needs_path,
        {
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "top_off_walk_forward_runs",
                    "priority": 55,
                    "evidence": {
                        "sample_count": 240,
                        "observation_count": 240,
                        "eligible_sequences": 5,
                        "positive_rate": 0.5,
                        "acted_coverage": 0.1,
                        "quality_score": 0.7,
                    },
                }
            ]
        },
    )
    _write_json(quality_path, {"targeted_actions": {}})

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        bot_needs_path=needs_path,
        training_quality_path=quality_path,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["apply_result"]["registry_updated"] is True
    assert row["data_intake_expansion"]["version"] == "training_data_intake_expansion_v1"
    assert "memory_pressure" in row["data_collection_context_demand"]
    assert "runtime_health" in row["data_collection_focus_context"]
    assert "runtime_health" in row["data_collection_enrichment_context"]
    assert row["data_collection_sample_enrichment_plan"]["usable_sample_goal"] == 200
    assert row["data_collection_label_repair_plan"]["plan_version"] == "label_repair_v1"
    assert "sample_eligibility_reason" in row["data_collection_label_repair_plan"]["required_label_outputs"]


def test_training_data_intake_uses_paper_loss_confirmation_controls(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v48_position_1m_3m"
    registry_path = project_root / "master_bot_registry.json"
    needs_path = project_root / "governance" / "health" / "bot_needs_intelligence_latest.json"
    quality_path = project_root / "governance" / "health" / "training_quality_control_latest.json"
    paper_path = project_root / "governance" / "health" / "paper_profitability_control_latest.json"
    _write_json(
        registry_path,
        {
            "summary": {},
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "data_collection_active": True,
                    "lifecycle_state": "active",
                    "minimum_training_observations": 1000,
                    "label_contract": {
                        "label_family": "multi_day",
                        "required_context": ["daily_bars"],
                    },
                }
            ],
        },
    )
    _write_json(
        needs_path,
        {
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "targeted_quality_retrain",
                    "priority": 50,
                    "evidence": {
                        "sample_count": 260,
                        "observation_count": 1000,
                        "eligible_sequences": 8,
                        "positive_rate": 0.5,
                        "acted_coverage": 0.1,
                        "quality_score": 0.8,
                    },
                }
            ]
        },
    )
    _write_json(quality_path, {"targeted_actions": {}})
    _write_json(
        paper_path,
        {
            "strategy_controls": [
                {
                    "profile": "swing_aggressive",
                    "strategy": "paper_mirror::brain_refinery_v48_position_1m_3m",
                    "bot_id": bot_id,
                    "mode": "paper_quarantine",
                    "ending_net_pnl_total": -750.0,
                    "score_penalty_norm": 1.0,
                    "confirmation_bias_score_norm": 0.72,
                    "loss_causes": ["conflict:low", "event_proximity:low", "fill_quality:unknown"],
                    "confirmation_bias_control": {
                        "active": True,
                        "min_independent_evidence_channels": 4,
                    },
                    "data_intake_enrichment": {
                        "required_context": ["paper_profile_strategy_pair", "cross_asset_confirmation"],
                        "required_label_outputs": ["confirmation_bias_bucket", "independent_evidence_channel_count"],
                    },
                }
            ],
            "scout_collection_contract": {
                "active": True,
                "mode": "collect_first_no_execution",
                "target_bot_ids": [bot_id],
                "required_context": ["paper_position_state", "exit_drag_trace", "no_trade_counterfactual"],
                "required_label_outputs": [
                    "paper_unrealized_drag_bucket",
                    "paper_exit_quality_bucket",
                    "no_trade_counterfactual_outcome",
                ],
                "collection_rules": ["persist exit-drag traces before any scout can leave collection-only mode"],
            },
        },
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        bot_needs_path=needs_path,
        training_quality_path=quality_path,
        paper_profitability_path=paper_path,
        apply=False,
    )

    record = payload["focus_records"][0]
    assert "paper_loss_drag" in record["weaknesses"]
    assert "confirmation_bias" in record["weaknesses"]
    assert "paper_profile_strategy_pair" in record["focus_context"]
    assert "independent_evidence_channel_count" in record["label_repair_plan"]["required_label_outputs"]
    assert "paper_unrealized_drag_bucket" in record["label_repair_plan"]["required_label_outputs"]
    assert "no_trade_counterfactual_outcome" in record["label_repair_plan"]["required_label_outputs"]
    assert "exit_drag_trace" in record["required_context"]
    assert record["profitability_scout_collection"]["active"] is True
    assert record["paper_loss_controls"][0]["profile"] == "swing_aggressive"
