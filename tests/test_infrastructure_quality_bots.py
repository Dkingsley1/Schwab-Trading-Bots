import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.distill_new_bots as distill_src
from scripts.ops import bot_quality_autopilot as quality_auto_src
from scripts.ops import infrastructure_autofix_bot as infra_src
from scripts.ops import teacher_quality_guard as teacher_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_teacher_quality_guard_prefers_strong_performers_and_excludes_probation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "walk_forward_latest.json",
        {
            "bots": {
                "brain_refinery_v10_seasonal": {
                    "runs": 14,
                    "forward_mean": 0.58,
                    "delta": 0.02,
                    "trading_quality_score": 0.81,
                    "status": "pass",
                },
                "brain_refinery_v43_intraday_ultrafast_proxy": {
                    "runs": 12,
                    "forward_mean": 0.57,
                    "delta": 0.01,
                    "trading_quality_score": 0.79,
                    "status": "pass",
                },
            }
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "test_accuracy": 0.61,
                    "quality_score": 0.82,
                },
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "test_accuracy": 0.60,
                    "quality_score": 0.80,
                },
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {"targeted_actions": {"quality_probation_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"]}},
    )
    _write_json(
        project_root / "governance" / "health" / "paper_performance_latest.json",
        {
            "sleeve_latest": [
                {
                    "top_winning_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v10_seasonal",
                            "ending_net_pnl_total": 250.0,
                        }
                    ]
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_student_plan_latest.json",
        {"assignments": [{"student_bot_id": "brain_refinery_v50_investment_drawdown_risk", "student_role": "signal_sub_bot"}]},
    )

    payload = teacher_src.build_payload(project_root)

    assert payload["overall_status"] in {"ready", "degraded"}
    assert payload["qualified_teachers"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert all(row["bot_id"] != "brain_refinery_v43_intraday_ultrafast_proxy" for row in payload["qualified_teachers"])


def test_distill_new_bots_prefers_curated_teacher_pool(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    walk_forward_path = project_root / "governance" / "walk_forward" / "walk_forward_latest.json"
    registry_path = project_root / "master_bot_registry.json"
    teacher_quality_path = project_root / "governance" / "distillation" / "teacher_quality_latest.json"
    out_path = project_root / "governance" / "distillation" / "teacher_student_plan_latest.json"

    _write_json(
        walk_forward_path,
        {
            "bots": {
                "brain_refinery_v10_seasonal": {"runs": 12, "forward_mean": 0.56, "delta": 0.01, "status": "pass"},
                "brain_refinery_v56_meta_ranker": {"runs": 2, "status": "insufficient_runs"},
            }
        },
    )
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "infrastructure_sub_bot", "active": True, "test_accuracy": 0.60, "quality_score": 0.80},
                {"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot", "active": False},
            ]
        },
    )
    _write_json(
        teacher_quality_path,
        {
            "qualified_teachers": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "infrastructure_sub_bot",
                    "teacher_score": 0.77,
                    "teacher_grade": "elite",
                    "walk_forward_runs": 12,
                    "walk_forward_forward_mean": 0.56,
                    "walk_forward_delta": 0.01,
                }
            ]
        },
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "distill_new_bots.py",
            "--walk-forward",
            str(walk_forward_path),
            "--registry",
            str(registry_path),
            "--teacher-quality",
            str(teacher_quality_path),
            "--out",
            str(out_path),
        ]
        rc = distill_src.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["summary"]["curated_teacher_count"] == 1
    assert payload["assignments"][0]["teachers"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert payload["assignments"][0]["teachers"][0]["teacher_grade"] == "elite"


def test_bot_quality_autopilot_builds_queue_and_teacher_preview(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "targeted_actions": {
                "refresh_diagnostics_bot_ids": ["brain_refinery_v10_seasonal"],
                "repair_runtime_input_bot_ids": ["brain_refinery_v56_meta_ranker"],
                "quality_probation_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"],
                "targeted_retrain_bot_ids": ["brain_refinery_v43_intraday_ultrafast_proxy"],
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "supportability_control_latest.json",
        {
            "teacher_student": {
                "students_without_teachers": 1,
                "uncovered_students": [
                    {"student_bot_id": "brain_refinery_v56_meta_ranker", "student_role": "infrastructure_sub_bot"}
                ],
            }
        },
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "overall_status": "ready",
            "summary": {"qualified_teacher_count": 2, "elite_teacher_count": 1},
            "qualified_teachers": [
                {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v86_risk_budget_allocator_v2", "bot_role": "infrastructure_sub_bot"},
            ],
        },
    )
    _write_json(
        project_root / "governance" / "health" / "training_requalification_latest.json",
        {"top_candidates": [{"bot_id": "brain_refinery_v99_defensive_dividend_concentration", "actions": ["seed_walk_forward_coverage"], "priority": 77.0}]},
    )
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 2})
    _write_json(project_root / "governance" / "health" / "training_runtime_control_latest.json", {"snapshot_ready": False})
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot"},
                {"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v99_defensive_dividend_concentration", "bot_role": "options_sub_bot"},
                {"bot_id": "brain_refinery_v86_risk_budget_allocator_v2", "bot_role": "infrastructure_sub_bot"},
            ]
        },
    )

    payload = quality_auto_src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "blocked"
    assert payload["quality_upgrade_queue"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert payload["assignment_preview"][0]["suggested_teacher_bot_ids"] == ["brain_refinery_v86_risk_budget_allocator_v2"]


def test_bot_quality_autopilot_surfaces_infrastructure_helper_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "health" / "training_quality_control_latest.json", {"overall_status": "degraded", "targeted_actions": {}})
    _write_json(project_root / "governance" / "health" / "supportability_control_latest.json", {"teacher_student": {"students_without_teachers": 0, "uncovered_students": []}})
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "overall_status": "ready",
            "summary": {"qualified_teacher_count": 1, "elite_teacher_count": 1},
            "qualified_teachers": [{"bot_id": "brain_refinery_v86_risk_budget_allocator_v2", "bot_role": "infrastructure_sub_bot"}],
        },
    )
    _write_json(project_root / "governance" / "health" / "training_requalification_latest.json", {"top_candidates": []})
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 1,
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v56_meta_ranker",
                    "bot_role": "infrastructure_sub_bot",
                    "priority": 84.0,
                    "needs_runtime_input_repair": True,
                }
            ],
        },
    )
    _write_json(project_root / "governance" / "health" / "training_runtime_control_latest.json", {"snapshot_ready": True})
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot"},
                {"bot_id": "brain_refinery_v86_risk_budget_allocator_v2", "bot_role": "infrastructure_sub_bot"},
            ]
        },
    )

    payload = quality_auto_src.build_payload(project_root, apply=False)

    assert payload["quality_blockers"]["infrastructure_helper_count"] == 1
    assert payload["infrastructure_helper_queue"][0]["bot_id"] == "brain_refinery_v56_meta_ranker"
    assert payload["infrastructure_helper_queue"][0]["recommended_teacher_bot_ids"] == ["brain_refinery_v86_risk_budget_allocator_v2"]


def test_infrastructure_autofix_bot_builds_safe_apply_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "daily_auto_verify_latest.json", {"failed_checks": ["promotion_quality_gate"]})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "storage": {"retention_debt_gb": 3.2},
            "backpressure": {
                "total_pending_lines": 62000,
                "estimated_total_drain_minutes": 22.5,
            },
        },
    )
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "rows": [
                {
                    "name": "schwab_education_context",
                    "contract_ok": False,
                }
            ]
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "sources": [
                {
                    "source_id": "polygon_unusual_whales_options_context",
                    "verification_status": "single_source_unverified",
                }
            ]
        },
    )
    _write_json(health / "options_flow_context_sync_latest.json", {"ok": False})
    _write_json(health / "schwab_education_context_sync_latest.json", {"ok": False})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "degraded", "lease_budget": {"expires_in_seconds": 120.0}})
    _write_json(health / "blackstart_recovery_latest.json", {"overall_status": "blocked"})
    _write_json(health / "artifact_freshness_slo_latest.json", {"overall_status": "blocked"})
    _write_json(health / "runtime_snapshot_cache_control_latest.json", {"overall_status": "blocked", "cache_health": {"snapshot_ready": False}})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "blocked", "channels": {"any_configured": False}, "critical_backlog": {"unsent_count": 5}})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "supportability_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "blocked"})

    payload = infra_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert payload["overall_status"] == "blocked"
    assert "daily_verify_auto_remediation" in names
    assert "storage_backpressure_autopilot" in names
    assert "schwab_education_refresh" in names
    assert "options_flow_efficiency" in names
    assert "premarket_token_guard" in names
    assert "bot_quality_autopilot" in names
    assert payload["metrics"]["storage_total_pending_lines"] == 62000
    assert payload["metrics"]["storage_total_drain_minutes"] == 22.5


def test_infrastructure_autofix_bot_treats_degraded_child_repairs_as_degraded(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "daily_auto_verify_latest.json", {"failed_checks": ["promotion_quality_gate"]})
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        return {
            "cmd": list(cmd),
            "rc": 2,
            "timed_out": False,
            "payload": {"overall_status": "degraded"},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(infra_src, "_run_json", _fake_run_json)

    payload = infra_src.build_payload(project_root, apply=True, timeout_sec=30)

    assert payload["repair_plan"]
    assert payload["overall_status"] == "degraded"
