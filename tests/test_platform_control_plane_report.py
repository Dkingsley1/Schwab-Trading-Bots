import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.platform_control_plane_report as report


def test_platform_control_plane_report_aggregates_core_sections(tmp_path) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "walk_forward").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "champion_challenger").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "lifecycle").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "audits").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "feature_versions").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "experiments").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "risk").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "feature_store").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "research").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "migrations").mkdir(parents=True, exist_ok=True)
    (project_root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts").mkdir(parents=True, exist_ok=True)
    (project_root / "exports" / "paper_broker_bridge" / "paper").mkdir(parents=True, exist_ok=True)
    (project_root / "exports" / "state_snapshot_drills").mkdir(parents=True, exist_ok=True)

    (project_root / "governance" / "health" / "training_success_latest.json").write_text(json.dumps({"reason": "trained_ok_but_not_promotable:failed_exit_2"}), encoding="utf-8")
    (project_root / "governance" / "health" / "retrain_scorecard_latest.json").write_text(json.dumps({"lineage": {"lineage_schema_version": 1}}), encoding="utf-8")
    (project_root / "governance" / "health" / "paper_performance_latest.json").write_text(json.dumps({"sleeve_latest": [{"profile": "intraday_aggressive", "tca_summary": {"mean_slippage_gap_bps": 3.0, "poor_or_fair_fill_count": 2}, "top_loss_causes": [{"cause": "spread_regime:wide"}], "ending_net_pnl_total": -4.0}]}), encoding="utf-8")
    (project_root / "governance" / "health" / "point_in_time_event_store_latest.json").write_text(json.dumps({"ok": True, "event_count": 5}), encoding="utf-8")
    (project_root / "governance" / "health" / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True}), encoding="utf-8")
    (project_root / "governance" / "health" / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (project_root / "governance" / "health" / "counterfactual_replay_latest.json").write_text(json.dumps({"profiles_reviewed": ["intraday_aggressive"]}), encoding="utf-8")
    (project_root / "governance" / "health" / "live_readiness_smoke_latest.json").write_text(json.dumps({"preopen_dashboard": {"token_warning_level": "watch"}, "memory_hygiene": {"memory_pressure_kind": "swap_only"}}), encoding="utf-8")
    (project_root / "governance" / "health" / "ingestion_backpressure_latest.json").write_text(json.dumps({"pending_lines_cold": 1200, "cold_lane_recommendation": "offload_shadow_pnl_attribution", "top_cold_pending_files": [{"source_rel": "governance/shadow_intraday_aggressive_equities/shadow_pnl_attribution_20260401.jsonl"}]}), encoding="utf-8")
    (project_root / "governance" / "health" / "resource_guard_latest.json").write_text(json.dumps({"memory_pressure_kind": "swap_only"}), encoding="utf-8")
    (project_root / "governance" / "health" / "training_quality_control_latest.json").write_text(
        json.dumps({"overall_status": "blocked", "top_priorities": ["runtime_input_coverage"]}),
        encoding="utf-8",
    )
    (project_root / "governance" / "feature_store" / "latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "lineage_schema_version": 1,
                "dataset_contract": {"rows_sha256": "rows-hash"},
                "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
                "lane_partitions": [{"lane": "intraday_aggressive", "row_count": 12}],
            }
        ),
        encoding="utf-8",
    )
    (project_root / "governance" / "research" / "multiple_testing_guard_latest.json").write_text(
        json.dumps({"ok": True, "overall_status": "ready", "family_size": 12, "correction_method": "benjamini_hochberg_fdr"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "research" / "decay_monitor_latest.json").write_text(
        json.dumps({"ok": True, "overall_status": "needs_work", "weak_sleeve_count": 1}),
        encoding="utf-8",
    )
    (project_root / "governance" / "migrations" / "latest.json").write_text(
        json.dumps({"ok": True, "summary": {"contract_count": 6}}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "paper_replay_drill_latest.json").write_text(
        json.dumps({"ok": True, "replay_hash": "paper-hash"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "replay_end_to_end_latest.json").write_text(
        json.dumps({"ok": True, "replay_hash": "e2e-hash"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "replay_hash_registry_guard_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "promotion_quality_gate_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "derived_state_latest.json").write_text(
        json.dumps({"risk_level": "medium", "gross_risk_budget": 0.61}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "security_audit_latest.json").write_text(
        json.dumps({"ok": True, "overall_status": "ready"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "secret_scan_latest.json").write_text(
        json.dumps({"findings_count": 0}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "session_ready_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "daily_auto_verify_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "live_reconciliation_slo_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "paper_reconciliation_slo_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "slo_burn_latest.json").write_text(
        json.dumps({"severity": "ok"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "snapshot_coverage_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "runtime_training_snapshot_latest.json").write_text(
        json.dumps({"row_count": 25}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "replay_feature_ablation_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "training_registry_audit_latest.json").write_text(
        json.dumps(
            {
                "active_sample_starved": [{"bot_id": "brain_refinery_v4_simple"}],
                "active_quality_failed": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}],
                "active_stale_diagnostics": [{"bot_id": "brain_refinery_v13_choppy"}],
                "tier_counts": {"active_repair": 1, "active_probation": 1, "active_stale": 1},
                "supportability_counts": {"unsupported_runtime_inputs": 1, "supported_but_quality_failing": 1},
            }
        ),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "training_label_audit_latest.json").write_text(
        json.dumps({"top_actions": ["fix_shared_runtime_input", "tighten_abstention_thresholds"]}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "sql_link_service_progress_latest.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_step": "merge_primary",
                "completed_shard_count": 13,
                "completed_merge_count": 7,
                "merged_rows_this_cycle": 2903552,
            }
        ),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "sql_link_service_latest.json").write_text(
        json.dumps({"sqlite_wal_size_gb": 42.0}),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "storage_maintenance_latest.json").write_text(
        json.dumps({"reason": "resource_guard_blocked"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json").write_text(json.dumps({"promote_ok": False}), encoding="utf-8")
    (project_root / "governance" / "champion_challenger" / "registry.json").write_text(json.dumps({"champion": {"name": "alpha"}, "stages": ["research", "shadow", "paper", "promoted", "live"]}), encoding="utf-8")
    (project_root / "governance" / "lifecycle" / "model_lifecycle_latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "missing_active_artifacts": 1,
                "missing_log_only_artifacts": 3,
                "missing_active_artifacts_total": 4,
                "stale_active_training_diagnostics": 1,
                "repair": {"fixed_count": 2, "registry_updated": True},
            }
        ),
        encoding="utf-8",
    )
    (project_root / "governance" / "feature_versions" / "latest.json").write_text(
        json.dumps({"env_hash": "env-1", "file_hashes": {"master_bot_registry.json": "abc"}}),
        encoding="utf-8",
    )
    (project_root / "governance" / "experiments" / "experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "exp_123",
                "replayability": {
                    "bundle_hash": "bundle-1",
                    "dataset_hash": "dataset-1",
                    "model_hash": "model-1",
                    "replay_hash": "replay-1",
                    "exact_replay_ready": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (project_root / "governance" / "audits" / "registry_mutation_latest.json").write_text(json.dumps({"actor": "test"}), encoding="utf-8")
    (project_root / "governance" / "audits" / "registry_mutation_journal_20260401.jsonl").write_text(
        json.dumps({"event": "mutation"}) + "\n",
        encoding="utf-8",
    )
    (project_root / "governance" / "risk" / "portfolio_risk_latest.json").write_text(
        json.dumps({"risk_level": "medium", "risk_score": 21.0}),
        encoding="utf-8",
    )
    (project_root / "governance" / "risk" / "execution_budget_latest.json").write_text(
        json.dumps({"gross_risk_budget": 0.75, "max_total_actions_per_hour": 18}),
        encoding="utf-8",
    )
    (project_root / "governance" / "watchdog" / "sleeve_slo_latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / ".github" / "workflows" / "ci_guardrails.yml").write_text("name: CI\n", encoding="utf-8")
    (project_root / ".github" / "CODEOWNERS").write_text("* @ops-review\n", encoding="utf-8")
    (project_root / "scripts" / "runbook.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "scripts" / "release_ops.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "exports" / "state_snapshot_drills" / "latest.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    (project_root / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260401.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "metadata": {"source_profile": "intraday_aggressive"},
                "slippage_gap_bps": 2.0,
                "allocation_confidence_scale": 0.7,
                "allocation_conflict_norm": 0.2,
                "tradeability_score": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = report.build_report(project_root, max_rows=100)

    assert payload["point_in_time_data_lineage"]["lineage_schema_version"] == 1
    assert payload["model_registry_and_rollout"]["champion"]["name"] == "alpha"
    assert payload["model_registry_and_rollout"]["promotion_middle_lane_active"] is False
    assert payload["model_registry_and_rollout"]["artifact_hygiene"]["hard_missing_active_artifacts"] == 1
    assert payload["model_registry_and_rollout"]["artifact_hygiene"]["missing_log_only_artifacts"] == 3
    assert payload["model_registry_and_rollout"]["training_registry_audit"]["tier_counts"]["active_repair"] == 1
    assert payload["model_registry_and_rollout"]["training_label_audit"]["top_actions"] == [
        "fix_shared_runtime_input",
        "tighten_abstention_thresholds",
    ]
    assert payload["transaction_cost_analysis"]["top_profiles_by_slippage_gap"][0]["profile"] == "intraday_aggressive"
    assert payload["broker_reconciliation_layer"]["broker_ready"] is True
    assert payload["broker_reconciliation_layer"]["preopen_dashboard"]["token_warning_level"] == "watch"
    assert payload["storage_sql_backlog_shaping"]["cold_lane_recommendation"] == "offload_shadow_pnl_attribution"
    assert payload["storage_sql_backlog_shaping"]["sql_sync"]["status"] == "running"
    assert payload["storage_sql_backlog_shaping"]["sql_sync"]["sqlite_wal_size_gb"] == 42.0
    assert payload["morning_control_plane"]["training_readiness"]["active_sample_starved"] == 1
    assert payload["morning_control_plane"]["training_readiness"]["top_label_actions"] == [
        "fix_shared_runtime_input",
        "tighten_abstention_thresholds",
    ]
    assert payload["morning_control_plane"]["sql_storage"]["sql_sync_step"] == "merge_primary"
    assert payload["institutional_readiness"]["overall_score"] > 0.0
    assert payload["institutional_domains_by_slug"]["point_in_time_data_lineage"]["score"] >= 80.0
    assert payload["institutional_domains_by_slug"]["immutable_experiment_tracking"]["status"] in {"advancing", "strong"}
    assert payload["institutional_domains_by_slug"]["high_fidelity_simulator"]["score"] >= 70.0
    assert payload["institutional_domains_by_slug"]["developer_process"]["status"] in {"advancing", "strong"}
