from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "training_labeling_intelligence.py"
spec = importlib.util.spec_from_file_location("training_labeling_intelligence", MODULE_PATH)
tli = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(tli)


def _write_registry(root: Path) -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v1613_autonomic_governance_expansion_stability_stress_oracle_governor_bridge_bot",
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "data_collection_active": True,
            "training_excluded": True,
            "lifecycle_state": "data_collection_only",
            "slot_kind": "autonomic_governance_mesh_expansion_stability_stress_oracle_governor_bridge",
        },
        {
            "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
            "bot_role": "signal_sub_bot",
            "active": True,
            "quality_score": 0.6,
            "label_contract": {
                "label_family": "intraday_fast",
                "primary_horizon": "5m_30m_forward_return",
                "required_context": ["one_minute_bars"],
            },
            "data_label_contract_version": "existing_v1",
        },
    ]
    (root / "master_bot_registry.json").write_text(
        json.dumps({"summary": {"total_bots": len(rows), "active_bots": len(rows), "max_bot_version": 1613}, "sub_bots": rows}) + "\n",
        encoding="utf-8",
    )


def test_dry_run_reports_missing_labels_and_plans_intelligence_layer(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "source_verification_latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "overall_status": "ready",
                "sources": [
                    {
                        "source_id": "market_micro_context",
                        "verification_status": "single_source_verified",
                        "ok": True,
                        "fresh": True,
                    },
                    {
                        "source_id": "free_equity_reference_context",
                        "verification_status": "single_source_verified",
                        "ok": True,
                        "fresh": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = tli.build_payload(tmp_path)

    assert payload["system_count"] == 6
    assert payload["bot_count"] == 24
    assert payload["planned_bot_count"] == 24
    assert payload["target_platform_total_bots"] == 1628
    assert payload["missing_label_contract_count"] == 1
    enrichment = payload["free_label_source_enrichment"]
    assert enrichment["verified_context_count"] >= 1
    assert "one_minute_bars" in enrichment["verified_contexts"]
    assert enrichment["classification_counts"]["free_public_or_verified_proxy"] >= 1
    one_minute = next(row for row in enrichment["context_sources"] if row["context"] == "one_minute_bars")
    assert one_minute["source_confidence_norm"] > 0.0
    assert "sample_eligibility_reason" in one_minute["materialization_contract"]["required_outputs"]
    assert payload["label_materialization_plan"]["contract_count"] == enrichment["required_context_count"]
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1614_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1637_")
    assert payload["universal_label_contract_version"] == tli.UNIVERSAL_LABEL_CONTRACT_VERSION


def test_verified_proxy_routes_clear_live_tape_and_research_context_caveats(tmp_path: Path) -> None:
    contexts = [
        "feed_latency_schema_health",
        "futures_bars",
        "mbo_mbp_depth_snapshot",
        "model_price_sensitivity_grid",
        "opra_nbbo_taq_sip_normalized_events",
        "quant_model_feature_surface",
        "state_filter_diagnostics",
    ]
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "summary": {"total_bots": 1, "active_bots": 1, "max_bot_version": 999},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v999_proxy_caveat_training_bot",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "label_contract": {
                            "label_family": "market_data_tape_normalization_research",
                            "primary_horizon": "proxy_caveat_training",
                            "required_context": contexts,
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    source_ids = [
        "market_micro_context",
        "market_quote_profiles",
        "extended_quant_context",
        "official_macro_context",
        "options_context_mesh",
    ]
    (health / "source_verification_latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "overall_status": "ready",
                "sources": [
                    {
                        "source_id": source_id,
                        "verification_status": "single_source_verified",
                        "ok": True,
                        "fresh": True,
                        "source_confidence_score": 0.9,
                    }
                    for source_id in source_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    (health / "collector_contracts_latest.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    payload = tli.build_payload(tmp_path)

    enrichment = payload["free_label_source_enrichment"]
    materialization = payload["label_materialization_plan"]
    for context in contexts:
        row = next(item for item in enrichment["context_sources"] if item["context"] == context)
        assert row["coverage_status"] == "verified"
        assert row["materialization_contract"]["eligible_for_training"] is True
    assert materialization["blocked_contract_count"] == 0
    assert set(contexts).issubset(set(materialization["ready_contexts"]))


def test_apply_adds_bots_and_normalizes_all_label_contracts(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = tli.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 24
    assert payload["label_contract_summary"]["missing_contracts_before"] == 1
    assert payload["label_contract_summary"]["updated_label_contract_bot_count"] == 1
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    rows = registry["sub_bots"]
    added = [row for row in rows if row.get("capability_pack_slug") == tli.PACK_SLUG]
    assert len(added) == 24
    assert registry["summary"]["training_label_contract_bot_count"] == len(rows)
    assert registry["summary"]["universal_label_contract_bot_count"] == len(rows)
    assert registry["summary"]["training_labeling_intelligence_bot_count"] == 24
    missing_fixed = rows[0]
    preserved = rows[1]
    assert missing_fixed["data_label_contract_version"] == tli.UNIVERSAL_LABEL_CONTRACT_VERSION
    assert missing_fixed["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert preserved["label_contract"]["primary_horizon"] == "5m_30m_forward_return"
    assert preserved["universal_label_contract"]["label_family"] == "intraday_fast"
    assert "one_minute_bars" in preserved["universal_label_contract"]["free_source_context_candidates"]
    for row in added:
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["data_collection_sample_rate"] == 0.01
        assert row["data_collection_max_daily_storage_mb"] == 1
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
        assert row["label_contract"]["free_source_context_policy"] == "point_in_time_verified_free_public_sources_only"
    assert (tmp_path / "config" / "training_labeling_intelligence_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "training_labeling_intelligence_latest.json").exists()
    assert (tmp_path / "governance" / "training_labeling_intelligence" / "label_coverage_latest.json").exists()
    assert (tmp_path / "governance" / "training_labeling_intelligence" / "free_label_source_enrichment_latest.json").exists()
    assert (tmp_path / "governance" / "training_labeling_intelligence" / "label_materialization_plan_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = tli.apply_registry(tmp_path)
    second = tli.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("capability_pack_slug") == tli.PACK_SLUG]

    assert first["added_bot_count"] == 24
    assert second["added_bot_count"] == 0
    assert len(added) == 24
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]


def test_apply_guards_legacy_training_labeling_rows_without_claiming_pack_slots(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "master_bot_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    legacy_bot_id = "brain_refinery_v1661_training_labeling_label_contract_normalizer_telemetry_collector_bot"
    registry["sub_bots"].append(
        {
            "bot_id": legacy_bot_id,
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "lifecycle_state": "active",
            "weight": 0.00001,
            "preference_score": 0.2,
            "paper_trading_enabled": True,
            "execution_enabled": True,
        }
    )
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    dry_run = tli.build_payload(tmp_path)
    payload = tli.apply_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    legacy = next(row for row in registry["sub_bots"] if row["bot_id"] == legacy_bot_id)
    pack_rows = [row for row in registry["sub_bots"] if row.get("capability_pack_slug") == tli.PACK_SLUG]

    assert dry_run["training_labeling_collection_guard"]["noncompliant_before_count"] == 1
    guard = payload["training_labeling_collection_guard"]
    assert guard["legacy_repaired_bot_count"] == 1
    assert legacy_bot_id in guard["legacy_repaired_bot_ids"]
    assert legacy["lifecycle_state"] == "data_collection_only"
    assert legacy["data_collection_active"] is True
    assert legacy["training_excluded"] is True
    assert legacy["exclude_from_training"] is True
    assert legacy["weight"] == 0.0
    assert legacy["paper_trading_enabled"] is False
    assert legacy["execution_enabled"] is False
    assert legacy["rotation_blocked"] is True
    assert legacy["legacy_training_labeling_collection_guard_version"] == tli.PACK_VERSION
    assert legacy.get("slot_kind") in {None, ""}
    assert legacy.get("capability_pack_slug") in {None, ""}
    assert len(pack_rows) == 24
    assert registry["summary"]["training_labeling_intelligence_bot_count"] == 24


def test_targeted_label_repair_overrides_generic_existing_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "master_bot_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sub_bots"].append(
        {
            "bot_id": "brain_refinery_v95_rates_regime_bond_bot",
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "paper_live_data",
            "label_contract": {
                "version": tli.UNIVERSAL_LABEL_CONTRACT_VERSION,
                "label_family": "generic_directional",
                "primary_horizon": "1d_forward_return",
                "required_context": ["price_bars", "volume", "market_context"],
                "source": "inferred_from_registry_identity",
            },
            "data_label_contract_version": tli.UNIVERSAL_LABEL_CONTRACT_VERSION,
        }
    )
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    payload = tli.apply_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    repaired = next(row for row in registry["sub_bots"] if row["bot_id"] == "brain_refinery_v95_rates_regime_bond_bot")

    assert "brain_refinery_v95_rates_regime_bond_bot" in payload["label_contract_summary"]["updated_label_contract_bot_ids"]
    assert repaired["training_label_contract_status"] == "targeted_labeling_repair"
    assert repaired["label_contract"]["label_family"] == "fixed_income_rates"
    assert repaired["label_contract"]["training_lane"] == "slow_lane_balanced"
    assert "rates_curve" in repaired["label_contract"]["required_context"]


def test_missing_advanced_quant_contract_gets_quant_research_family(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "master_bot_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sub_bots"].append(
        {
            "bot_id": "brain_refinery_v444_quant_pricing_merton_jump_diffusion_bot",
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "training_excluded": True,
            "lifecycle_state": "data_collection_only",
        }
    )
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    payload = tli.apply_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    repaired = next(row for row in registry["sub_bots"] if row["bot_id"] == "brain_refinery_v444_quant_pricing_merton_jump_diffusion_bot")

    assert "brain_refinery_v444_quant_pricing_merton_jump_diffusion_bot" in payload["label_contract_summary"]["updated_label_contract_bot_ids"]
    assert repaired["label_contract"]["label_family"] == "quant_pricing_research"
    assert repaired["label_contract"]["training_lane"] == "research_quant_proxy"
    assert "quant_model_feature_surface" in repaired["label_contract"]["required_context"]
    assert "model_price_sensitivity_grid" in repaired["label_contract"]["required_context"]


def test_collect_only_diagnostics_include_training_excluded_paper_live_data(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "master_bot_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sub_bots"].append(
        {
            "bot_id": "brain_refinery_v12_news_shocks",
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "training_excluded": True,
            "lifecycle_state": "paper_live_data",
            "observations": 320,
            "minimum_training_observations": 1000,
        }
    )
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    payload = tli.apply_registry(
        tmp_path,
        materialize_collect_only_diagnostics=True,
        collect_only_diagnostic_min_version=0,
    )

    diagnostics = payload["collect_only_diagnostics"]
    assert "brain_refinery_v12_news_shocks" in diagnostics["written_bot_ids"]
    diag_path = tmp_path / "governance" / "training_diagnostics" / "brain_refinery_v12_news_shocks_latest.json"
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag["training_excluded"] is True
    assert diag["lifecycle_state"] == "paper_live_data"
    assert diag["runtime_meta"]["collection_threshold"]["observations_remaining"] == 680
    depth = diag["runtime_meta"]["label_depth_contract"]
    assert depth["status"] == "collect_and_materialize_label_depth"
    assert depth["observation_gap"] == 680
    assert "abstained_candidate_trace" in depth["required_depth_events"]
    assert diag["runtime_meta"]["usable_sample_bridge"]["policy"] == "do_not_count_estimated_capacity_as_real_training_samples"


def test_collect_only_diagnostics_include_collection_only_even_without_training_excluded(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "master_bot_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sub_bots"].append(
        {
            "bot_id": "brain_refinery_v314_collection_coverage_gap_mapper",
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "data_collection_active": True,
            "training_excluded": False,
            "lifecycle_state": "data_collection_only",
            "data_collection_observations": 1200,
            "minimum_training_observations": 1000,
        }
    )
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    payload = tli.apply_registry(
        tmp_path,
        materialize_collect_only_diagnostics=True,
        collect_only_diagnostic_min_version=300,
    )

    diagnostics = payload["collect_only_diagnostics"]
    assert "brain_refinery_v314_collection_coverage_gap_mapper" in diagnostics["written_bot_ids"]
    diag_path = tmp_path / "governance" / "training_diagnostics" / "brain_refinery_v314_collection_coverage_gap_mapper_latest.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag["training_excluded"] is True
    assert diag["label_depth_status"] == "label_depth_ready_for_real_diagnostic_refresh"


def test_training_process_intelligence_reads_walk_forward_coverage_artifacts(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    health = tmp_path / "governance" / "health"
    walk_forward = tmp_path / "governance" / "walk_forward"
    health.mkdir(parents=True)
    walk_forward.mkdir(parents=True)
    (health / "training_quality_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "needs_attention",
                "targeted_actions": {"targeted_retrain_bot_ids": ["brain_refinery_v17_mixed_regime"]},
            }
        ),
        encoding="utf-8",
    )
    (health / "training_runtime_control_latest.json").write_text(json.dumps({"snapshot_ready": True}), encoding="utf-8")
    (walk_forward / "coverage_gap_closer_latest.json").write_text(
        json.dumps(
            {
                "active_stage_candidates": [
                    {"bot_id": "brain_refinery_v35_dmi_state_machine"},
                    {"bot_id": "brain_refinery_v4_simple"},
                ],
                "autopilot_contract": {"launch_contract": {"coverage_repair_ready": True}},
            }
        ),
        encoding="utf-8",
    )

    intelligence = tli._training_process_intelligence(tmp_path)

    assert intelligence["coverage_repair_bot_ids"] == [
        "brain_refinery_v35_dmi_state_machine",
        "brain_refinery_v4_simple",
    ]
    assert intelligence["selected_target_source"] == "coverage_repair"
    assert intelligence["recommended_retrain_profile"] == "coverage_canary"
