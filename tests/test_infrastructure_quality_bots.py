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


def test_teacher_quality_guard_excludes_overfit_risk_teachers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v10_seasonal"
    _write_json(
        project_root / "governance" / "walk_forward" / "walk_forward_latest.json",
        {"bots": {bot_id: {"runs": 14, "forward_mean": 0.59, "delta": 0.02, "trading_quality_score": 0.84, "status": "pass"}}},
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "test_accuracy": 0.62,
                    "quality_score": 0.86,
                }
            ]
        },
    )
    _write_json(project_root / "governance" / "health" / "training_quality_control_latest.json", {"targeted_actions": {}})
    _write_json(project_root / "governance" / "health" / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        project_root / "governance" / "health" / "overfitting_awareness_latest.json",
        {
            "overall_status": "guarded",
            "risk_bot_count": 1,
            "hard_risk_bot_count": 0,
            "bot_risk": [
                {
                    "bot_id": bot_id,
                    "status": "overfit_watch",
                    "risk_score": 0.62,
                    "train_forward_gap": 0.11,
                    "policy": {"may_teach": False, "may_promote": False, "requires_generalization_canary": True},
                }
            ],
        },
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_student_plan_latest.json",
        {"assignments": [{"student_bot_id": "brain_refinery_v50_student", "student_role": "signal_sub_bot"}]},
    )

    payload = teacher_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["summary"]["overfit_blocked_teacher_count"] == 1
    assert all(row["bot_id"] != bot_id for row in payload["qualified_teachers"])
    assert any(row["reason"] == "overfit_risk_blocked" for row in payload["excluded_bots"])


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


def test_bot_quality_autopilot_refreshes_registry_audit_before_quality_control(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "targeted_actions": {}})
    _write_json(health / "supportability_control_latest.json", {"overall_status": "ready", "teacher_student": {"students_without_teachers": 0}})
    _write_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json", {"summary": {"qualified_teacher_count": 1, "elite_teacher_count": 1}})
    _write_json(project_root / "governance" / "health" / "training_runtime_control_latest.json", {"snapshot_ready": True})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})

    calls: list[list[str]] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": list(cmd), "rc": 0, "timed_out": False, "payload": {"overall_status": "ready"}}

    monkeypatch.setattr(quality_auto_src, "_run_json", _fake_run_json)

    quality_auto_src.build_payload(project_root, apply=True, timeout_sec=30)

    joined = [" ".join(cmd) for cmd in calls]
    registry_index = next(idx for idx, text in enumerate(joined) if "training_registry_audit.py" in text)
    quality_index = next(idx for idx, text in enumerate(joined) if "training_quality_control.py" in text)
    assert registry_index < quality_index


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
    assert "storage_pressure_clearance" in names
    assert "schwab_education_refresh" in names
    assert "options_flow_efficiency" in names
    assert "schwab_auth_supervisor" in names
    assert "bot_quality_autopilot" in names
    assert payload["metrics"]["storage_total_pending_lines"] == 62000
    assert payload["metrics"]["storage_total_drain_minutes"] == 22.5


def test_infrastructure_autofix_keeps_quality_debt_repair_advisory_when_supportability_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "recoverable_blocked_keys": [],
            "top_priorities": ["runtime_input_coverage", "active_probation_isolation"],
            "targeted_actions": {
                "repair_runtime_input_bot_ids": ["brain_refinery_v12_news_shocks"],
                "targeted_retrain_bot_ids": ["brain_refinery_v12_news_shocks"],
            },
        },
    )
    _write_json(health / "supportability_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "blocked"})

    payload = infra_src.build_payload(project_root, apply=False)

    repair_names = [row["name"] for row in payload["repair_plan"]]
    advisory_names = [row["name"] for row in payload["advisory_repair_plan"]]
    assert "bot_quality_autopilot" not in repair_names
    assert "bot_quality_autopilot" in advisory_names


def test_infrastructure_autofix_refreshes_required_collector_failures(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failures": ["official_macro_context"],
            "rows": [
                {"name": "market_micro_context", "required": True, "contract_ok": False},
                {"name": "schwab_education_context", "required": False, "contract_ok": False},
            ],
        },
    )
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})

    payload = infra_src.build_payload(project_root, apply=False)
    names = [row["name"] for row in payload["repair_plan"]]

    assert "official_macro_context_refresh" in names
    assert "market_micro_context_refresh" in names
    assert "schwab_education_refresh" in names


def test_infrastructure_autofix_routes_raw_control_plane_and_library_surfaces(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"overall_status": "ready", "failed_guard_count": 0})
    _write_json(health / "host_capability_contract_latest.json", {"overall_status": "blocked"})
    _write_json(health / "library_utilization_router_latest.json", {"overall_status": "blocked"})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready", "library_coverage": {"coverage_ratio": 0.75}})
    _write_json(health / "library_upgrade_route_control_latest.json", {"overall_status": "blocked", "upgrade_plan": {"hard_blocker_count": 1}})
    _write_json(
        health / "coordination_state_latest.json",
        {
            "overall_status": "blocked",
            "artifact_issues": [
                {"name": "required_artifact_stale:halt_trigger_control_plane"},
                {"name": "optional_artifact_stale:shadow_watchdog_tripwire"},
                {"name": "optional_artifact_stale:heavy_livefeed"},
            ],
        },
    )

    payload = infra_src.build_payload(project_root, apply=False)
    names = [row["name"] for row in payload["repair_plan"]]
    advisory_names = [row["name"] for row in payload["advisory_repair_plan"]]

    assert "host_capability_contract" in names
    assert "library_utilization_router" in names
    assert "mlx_intelligence_router" in names
    assert "library_upgrade_route_control" in names
    assert "halt_trigger_control_plane" in names
    assert "shadow_watchdog_tripwire_refresh" in advisory_names
    assert "live_feed_heavy_guard_refresh" in advisory_names
    assert "coordination_state_control" in names


def test_infrastructure_autofix_bounds_system_drift_autopilot_timeout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "system_drift_guard_latest.json",
        {
            "overall_status": "blocked",
            "metrics": {"blocked_surface_count": 1, "degraded_surface_count": 0},
            "surfaces": [{"name": "report_pdf_bundle", "status": "blocked"}],
        },
    )
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})

    payload = infra_src.build_payload(project_root, apply=False)
    drift_step = next(row for row in payload["repair_plan"] if row["name"] == "system_drift_autopilot")
    cmd = drift_step["cmd"]

    timeout_index = cmd.index("--max-step-timeout-seconds")
    assert cmd[timeout_index + 1] == str(infra_src.SYSTEM_DRIFT_AUTOFIX_STEP_TIMEOUT_SECONDS)


def test_infrastructure_autofix_does_not_relaunch_master_when_nested_under_master(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "master_infrastructure_supervisor_latest.json", {"overall_status": "blocked"})
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})
    monkeypatch.setenv(infra_src.REPAIR_CALL_STACK_ENV, "master_infrastructure_supervisor")

    payload = infra_src.build_payload(project_root, apply=False)

    repair_names = [row["name"] for row in payload["repair_plan"]]
    advisory_rows = [row for row in payload["advisory_repair_plan"] if row["name"] == "master_infrastructure_supervisor_refresh"]
    assert "master_infrastructure_supervisor_refresh" not in repair_names
    assert advisory_rows
    assert "nested_under_master_supervisor=1" in advisory_rows[0]["reason"]


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


def test_infrastructure_autofix_apply_respects_run_timeout_budget(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "daily_auto_verify_latest.json", {"failed_checks": ["promotion_quality_gate"]})
    _write_json(health / "remote_alert_control_latest.json", {"channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}})

    monotonic_now = {"value": 1000.0}
    calls: list[tuple[list[str], int]] = []

    monkeypatch.setattr(infra_src.time, "monotonic", lambda: monotonic_now["value"])

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        calls.append((list(cmd), int(timeout_sec)))
        monotonic_now["value"] += float(timeout_sec) + 0.01
        return {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "timeout_sec": int(timeout_sec),
            "payload": {"overall_status": "ready"},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(infra_src, "_run_json", _fake_run_json)

    payload = infra_src.build_payload(project_root, apply=True, timeout_sec=5)

    assert payload["repair_plan"]
    assert calls
    assert calls[0][1] == 5
    assert payload["metrics"]["timeout_budget_exhausted"] is True
    assert any(row["skipped"] and row["reason"] == "run_timeout_budget_exhausted" for row in payload["attempts"])
    assert payload["overall_status"] == "blocked"
