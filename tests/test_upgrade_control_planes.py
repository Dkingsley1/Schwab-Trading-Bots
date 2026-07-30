import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.storage_tier_policy as storage_tier_src
from scripts.ops import regime_control_plane as regime_src
from scripts.ops import supportability_control as supportability_src
from scripts.ops import training_runtime_control as training_runtime_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_storage_tier_policy_surfaces_hot_path_and_cold_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    decisions = project_root / "decisions" / "live_decisions_20260409.jsonl"
    explanations = project_root / "decision_explanations" / "decision_explanations_20260409.jsonl"
    sql_shard = project_root / "data" / "sql_link_shards" / "jsonl_link_explanations.sqlite3-wal"
    content_blob = project_root / "governance" / "content_store" / "sha256" / "aa" / "blob"
    for path, content in (
        (decisions, "decision\n" * 8),
        (explanations, "explanation\n" * 64),
        (sql_shard, "wal\n" * 64),
        (content_blob, "blob\n" * 128),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    built = storage_tier_src.discover_storage_files(project_root)
    assert sql_shard in built
    assert content_blob in built

    out_path = project_root / "governance" / "health" / "storage_tier_policy_latest.json"
    args = [
        "storage_tier_policy.py",
        "--project-root",
        str(project_root),
        "--top-n",
        "5",
        "--hot-budget-gb",
        "0.0000001",
        "--cold-candidate-min-mb",
        "0.000001",
        "--offload-manifest-min-mb",
        "0.000001",
        "--offload-manifest-file",
        str(project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"),
    ]
    old_argv = sys.argv
    try:
        sys.argv = args
        rc = storage_tier_src.main()
    finally:
        sys.argv = old_argv
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["overall_status"] in {"degraded", "blocked"}
    assert any(row["service_role"] == "artifact_store" for row in payload["cold_path_candidates"])
    assert payload["pressure"]["hot_path_over_budget_bytes"] > 0
    contract = payload["manifest_backed_offload_contract"]
    assert contract["status"] == "planned"
    assert contract["policy_script_is_read_only"] is True
    assert contract["eligible_offload_files"] >= 2
    assert "stateful_sql_compaction_only" in contract["never_delete_classes"]
    manifest = json.loads(Path(contract["manifest_path"]).read_text(encoding="utf-8"))
    entries = {row["relative_path"]: row for row in manifest["entries"]}
    assert entries[str(explanations.relative_to(project_root))]["classification"] == "eligible_manifest_backed_offload"
    assert entries[str(explanations.relative_to(project_root))]["planned_cold_relative_path"].count("decision_explanations/") == 1
    assert entries[str(content_blob.relative_to(project_root))]["classification"] == "eligible_manifest_backed_offload"
    sql_entry = entries[str(sql_shard.relative_to(project_root))]
    assert sql_entry["classification"] == "stateful_sql_compaction_only"
    assert sql_entry["delete_allowed_by_policy"] is False
    assert "sqlite_checkpoint" in sql_entry["allowed_actions"]
    decision_entry = entries[str(decisions.relative_to(project_root))]
    assert decision_entry["classification"] == "keep_hot_critical"
    assert decision_entry["delete_allowed_by_policy"] is False


def test_storage_tier_policy_controls_fixed_hot_budget_with_continuous_run_margin(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    decisions = project_root / "decisions" / "live_decisions_20260627.jsonl"
    explanations = project_root / "decision_explanations" / "decision_explanations_20260627.jsonl"
    for path, content in (
        (decisions, "decision\n" * 128),
        (explanations, "explanation\n" * 256),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "continuous_run_contract": {
                "status": "ready",
                "ready": True,
                "available_margin_gb": 4.0,
            }
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "continuous_run_soak_contract": {
                "status": "ready",
                "ready": True,
                "inputs": {
                    "collector_intake_status": "enforced",
                    "storage_efficiency_status": "ready",
                    "backlog_relief_active": False,
                },
                "forecast": {"continuous_run_margin_gb": 4.0},
            },
            "storage_efficiency_contract": {"overall_status": "ready"},
        },
    )

    args = [
        "storage_tier_policy.py",
        "--project-root",
        str(project_root),
        "--hot-budget-gb",
        "0.0000001",
        "--cold-candidate-min-mb",
        "0.000001",
        "--offload-manifest-min-mb",
        "0.000001",
        "--offload-manifest-file",
        str(project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"),
    ]
    old_argv = sys.argv
    try:
        sys.argv = args
        rc = storage_tier_src.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((health / "storage_tier_policy_latest.json").read_text(encoding="utf-8"))
    contract = payload["hot_path_budget_contract"]

    assert rc == 0
    assert payload["overall_status"] == "ready"
    assert payload["pressure"]["raw_hot_path_over_budget_bytes"] > 0
    assert payload["pressure"]["hot_path_over_budget_bytes"] == 0
    assert payload["pressure"]["hot_budget_bytes"] > payload["pressure"]["configured_hot_budget_bytes"]
    assert contract["status"] == "managed_ready"
    assert contract["active"] is True
    assert contract["blockers"] == []


def test_storage_tier_policy_accepts_clean_optional_collector_intake(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    decisions = project_root / "decisions" / "live_decisions_20260627.jsonl"
    explanations = project_root / "decision_explanations" / "decision_explanations_20260627.jsonl"
    for path, content in (
        (decisions, "decision\n" * 128),
        (explanations, "explanation\n" * 256),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "continuous_run_contract": {
                "status": "ready",
                "ready": True,
                "available_margin_gb": 4.0,
            }
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "collector_intake_enforcement_audit": {
                "status": "not_required",
                "required": False,
                "mismatch_count": 0,
            },
            "continuous_run_soak_contract": {
                "status": "ready",
                "ready": True,
                "inputs": {
                    "collector_intake_status": "not_required",
                    "storage_efficiency_status": "ready",
                    "backlog_relief_active": False,
                },
                "forecast": {"continuous_run_margin_gb": 4.0},
            },
            "storage_efficiency_contract": {"overall_status": "ready"},
        },
    )

    args = [
        "storage_tier_policy.py",
        "--project-root",
        str(project_root),
        "--hot-budget-gb",
        "0.0000001",
        "--cold-candidate-min-mb",
        "0.000001",
        "--offload-manifest-min-mb",
        "0.000001",
        "--offload-manifest-file",
        str(project_root / "governance" / "health" / "storage_tier_offload_manifest_latest.json"),
    ]
    old_argv = sys.argv
    try:
        sys.argv = args
        rc = storage_tier_src.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((health / "storage_tier_policy_latest.json").read_text(encoding="utf-8"))
    contract = payload["hot_path_budget_contract"]

    assert rc == 0
    assert payload["overall_status"] == "ready"
    assert payload["pressure"]["raw_hot_path_over_budget_bytes"] > 0
    assert payload["pressure"]["hot_path_over_budget_bytes"] == 0
    assert contract["active"] is True
    assert contract["blockers"] == []
    assert contract["inputs"]["collector_intake_enforced"] is False
    assert contract["inputs"]["collector_intake_soak_safe"] is True
    assert contract["inputs"]["collector_intake"]["safely_optional"] is True


def test_training_runtime_control_prioritizes_sequence_timeout_retries(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sequence_count": 20,
            "row_count": 500,
            "rows_path": str(project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"),
            "coverage": {"top_modes": [], "top_sequences": []},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 22.8,
            "top_priorities": ["active_supportability"],
            "targeted_actions": {
                "targeted_retrain_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"],
                "quality_probation_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"],
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "retrain_scorecard_latest.json",
        {
            "failure_details": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "reason": "timeout",
                    "stdout_tail": "[RuntimeTraining] loading_sequences run_tag=brain_refinery_v43_intraday_ultrafast_proxy",
                }
            ],
            "retry_pack": {"command": ["./scripts/ops/opsctl.sh", "retrain-force-targeted"]},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "resource_guard_latest.json",
        {"resource_guard_ok": False, "memory_pressure_state": "yellow", "swap_used_gb": 6.2},
    )
    _write_json(
        project_root / "governance" / "health" / "health_gates_latest.json",
        {
            "recommended_operating_mode": "shadow_only",
            "inputs": {
                "backpressure_overload_severe": True,
                "backpressure_pending_lines": 40000,
                "backpressure_oldest_pending_age_seconds": 7200.0,
                "sql_progress_status": "ok",
            },
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 4, "seed_queue": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "priority": 10.0}]},
    )

    payload = training_runtime_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["snapshot_ready"] is True
    assert payload["precompute_targets"][0]["bot_id"] == "brain_refinery_v43_intraday_ultrafast_proxy"
    assert "loading_sequences_timeout" in payload["precompute_targets"][0]["reasons"]
    assert payload["training_launch_contract"]["mode"] == "blocked"
    assert payload["training_launch_contract"]["backpressure_gate"]["severe"] is True
    assert "backpressure_overload_severe" in payload["training_launch_contract"]["launch_blockers"]
    assert "resource_guard_not_green" in payload["training_launch_contract"]["prep_blockers"]


def test_regime_control_plane_combines_sentiment_and_risk(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "sentiment_report_latest.json",
        {
            "ok": True,
            "selected_day_utc": "20260409",
            "event_count": 8,
            "day": {"available": True, "avg_sentiment_hint": -0.6, "mean_shock_hint": 0.8, "day_end_day_utc": "20260409"},
            "week": {"available": True, "avg_sentiment_hint": -0.4, "mean_shock_hint": 0.7},
            "month": {"available": True, "avg_sentiment_hint": -0.2, "mean_shock_hint": 0.6},
            "year": {"available": True, "avg_sentiment_hint": 0.1, "mean_shock_hint": 0.4},
            "daily_sentiment_series": [1, 2, 3],
            "weekly_sentiment_series": [1],
            "monthly_sentiment_series": [1],
            "yearly_sentiment_series": [1],
            "latest_live_macro_snapshot": {"source": "Fed", "speaker": "Powell"},
        },
    )
    _write_json(health_root / "official_macro_context_sync_latest.json", {"sources": {"fed": {"ok": True}, "bls": {"ok": True}}})
    _write_json(health_root / "market_micro_sync_latest.json", {"sources": {"local_micro": {"ok": True}}})
    _write_json(health_root / "fx_market_context_sync_latest.json", {"sources": {"fed_h10": {"ok": True}}})
    _write_json(health_root / "crypto_market_context_sync_latest.json", {"sources": {"coingecko": {"ok": True}}})
    _write_json(health_root / "market_crypto_correlation_sync_latest.json", {"ok": True})
    _write_json(health_root / "derived_state_latest.json", {"risk_score": 72.0, "risk_level": "high", "execution_multiplier": 0.7})
    _write_json(
        health_root / "paper_execution_calibration_latest.json",
        {"metrics": {"mae_bps": 20.0}, "thresholds": {"max_mae_bps": 35.0}},
    )

    payload = regime_src.build_payload(project_root)

    assert payload["stance_label"] in {"bearish", "neutral"}
    assert payload["regime_state"] in {"risk_off_shock", "fragile_transition", "mixed_transition", "risk_off_trend"}
    assert payload["scores"]["risk_norm"] > 0.7


def test_supportability_control_flags_uncovered_students(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {"supportability": {"active_bots": 1, "active_supportable_bots": 0, "active_supportability_score": 0.0, "tier_counts": {"active_probation": 1}}},
    )
    _write_json(
        project_root / "governance" / "lifecycle" / "model_lifecycle_latest.json",
        {"missing_active_artifacts_total": 0, "missing_log_only_artifacts": 0, "stale_active_training_diagnostics": 0, "repair": {"enabled": True, "registry_updated": False}},
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_student_plan_latest.json",
        {
            "summary": {"teacher_count": 0, "student_count": 2, "assignment_count": 2},
            "teachers": [],
            "assignments": [
                {"student_bot_id": "brain_refinery_v10_seasonal", "student_role": "signal_sub_bot", "student_runs": 4, "teachers": []},
                {"student_bot_id": "brain_refinery_v68_risk_budget_layer", "student_role": "infrastructure_sub_bot", "student_runs": 6, "teachers": []},
            ],
        },
    )
    _write_json(
        project_root / "governance" / "health" / "training_requalification_latest.json",
        {"candidate_count": 5, "reactivation_ready_count": 0, "top_candidates": []},
    )

    payload = supportability_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["teacher_student"]["students_without_teachers"] == 2
    assert payload["teacher_student"]["teacher_gap_by_role"][0]["missing_assignments"] >= 1


def test_supportability_control_treats_score_as_percent(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {"supportability": {"active_bots": 10, "active_supportable_bots": 4, "active_supportability_score": 40.0}},
    )
    _write_json(project_root / "governance" / "lifecycle" / "model_lifecycle_latest.json", {"repair": {"enabled": True}})
    _write_json(
        project_root / "governance" / "distillation" / "teacher_student_plan_latest.json",
        {"summary": {"student_count": 0, "assignment_count": 0}, "teachers": [{"teacher_bot_id": "teacher"}], "assignments": []},
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {"summary": {"elite_teacher_count": 1, "qualified_teacher_count": 1}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_requalification_latest.json",
        {"candidate_count": 1, "reactivation_ready_count": 1, "top_candidates": []},
    )

    payload = supportability_src.build_payload(project_root)

    assert payload["overall_status"] == "needs_work"
    assert "expand the supportable active roster" in payload["recommended_actions"][0]
