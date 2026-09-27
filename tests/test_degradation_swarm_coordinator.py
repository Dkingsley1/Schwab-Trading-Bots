import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import degradation_swarm_coordinator as swarm_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _contained_fixture(project_root: Path) -> Path:
    health = project_root / "governance" / "health"
    _write_json(
        health / "operator_cockpit_latest.json",
        {
            "overall_status": "ready",
            "degradation_containment": {
                "status": "contained_degradation",
                "contained_degradation_lanes": [
                    "storage_reserve",
                    "source_verification",
                ],
                "uncontained_lanes": [],
                "safe_to_keep_collecting": True,
                "safe_to_submit_paper_orders": False,
                "safe_to_train_or_promote": False,
                "source_context_debt": ["decision_context_mesh"],
                "source_decision_critical_blockers": [],
                "lanes": {
                    "storage_reserve": {
                        "status": "contained",
                        "contained": True,
                        "owner": "storage_tier_policy",
                        "trade_impact": "training delayed by reserve debt",
                    },
                    "source_verification": {
                        "status": "context_debt",
                        "contained": True,
                        "owner": "source_verification_report",
                        "trade_impact": "context claims and promotion blocked",
                    },
                },
            },
        },
    )
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {
                "status": "ok",
                "attention": [
                    "external_backlog_drain_writer_busy",
                    "source_verification_context_debt",
                ],
                "remediation_actions": [
                    {
                        "attention": "external_backlog_drain_writer_busy",
                        "owner": "writer_cycle_coordinator",
                        "command": [
                            "./scripts/ops/opsctl.sh",
                            "writer-cycle-coordinator",
                            "--apply",
                            "--fast-handoff",
                            "--json",
                        ],
                        "success_condition": "writer handoff progresses",
                    },
                    {
                        "attention": "source_verification_context_debt",
                        "owner": "source_verification_report",
                        "command": [
                            "./scripts/ops/opsctl.sh",
                            "source-verification-refresh",
                            "--apply",
                            "--json",
                        ],
                        "success_condition": "context debt clears",
                    },
                ],
                "degradation_containment": {
                    "status": "contained",
                    "hot_path_blocked": False,
                    "containment_rows": [
                        {
                            "attention": "external_backlog_drain_writer_busy",
                            "domain": "storage_backlog",
                            "contained": True,
                            "release_condition": "writer no longer busy",
                        },
                        {
                            "attention": "source_verification_context_debt",
                            "domain": "market_context",
                            "contained": True,
                            "release_condition": "source context debt empty",
                        },
                    ],
                },
            }
        },
    )
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "overall_status": "blocked",
            "degradation_containment": {
                "status": "contained_degradation",
                "hot_path_blocked": False,
                "contained_lanes": ["governance_artifact_freshness"],
                "uncontained_lanes": [],
            },
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall_status": "degraded",
            "containment_contract": {
                "status": "contained_context_debt",
                "decision_context_debt": ["decision_context_mesh"],
                "decision_critical_blockers": [],
            },
            "source_runtime_contract": {
                "decision_critical_sources_ready": True,
                "decision_context_debt": ["decision_context_mesh"],
                "decision_critical_blockers": [],
            },
        },
    )
    return health


def test_degradation_swarm_builds_lane_specific_context(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _contained_fixture(project_root)

    payload = swarm_src.build_payload(project_root)

    assert payload["overall_status"] == "coordinating"
    assert payload["swarm"]["ready_to_swarm"] is True
    assert payload["swarm"]["safe_executable_assignment_count"] > 0
    lanes = {row["lane"] for row in payload["assignments"]}
    assert {"storage_reserve", "source_verification"} <= lanes
    source_assignments = [
        row for row in payload["assignments"] if row["lane"] == "source_verification"
    ]
    assert source_assignments
    assert any(
        row["command"][1] == "source-verification-refresh" for row in source_assignments
    )
    context = source_assignments[0]["degradation_context"]
    assert source_assignments[0]["section"] == "trading_brain"
    assert source_assignments[0]["domain"] == "market_context"
    assert source_assignments[0]["phase_definition"]["authority"]
    assert context["source_context_debt"] == ["decision_context_mesh"]
    assert context["source_decision_critical_blockers"] == []
    assert context["section"] == "trading_brain"
    assert context["lane_definition"]["canonical_owner"] == "source_verification_report"
    assert "do_not_place_live_orders" in context["hard_limits"]


def test_degradation_swarm_operating_model_groups_lanes_by_brain_and_phase(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _contained_fixture(project_root)

    payload = swarm_src.build_payload(project_root)
    operating_model = payload["operating_model"]

    assert set(operating_model["active_sections"]) == {"ops_brain", "trading_brain"}
    assert operating_model["active_lanes"]["storage_reserve"]["section"] == "ops_brain"
    assert (
        operating_model["active_lanes"]["source_verification"]["section"]
        == "trading_brain"
    )
    assert operating_model["phase_definitions"]["repair"]["authority"] == (
        "safe_apply_allowlist_only"
    )
    assert "why" in operating_model["required_owner_output_fields"]
    assert (
        operating_model["swarm_protocol"]["owner_operating_contract_required"] is True
    )
    assert {"observe", "repair", "verify"} <= set(operating_model["active_phase_order"])
    assert operating_model["sections"]["ops_brain"]["assignment_count"] > 0
    assert operating_model["sections"]["trading_brain"]["assignment_count"] > 0
    assert (
        operating_model["active_lanes"]["source_verification"][
            "owner_output_release_signal"
        ]
        == swarm_src.OWNER_OUTPUT_RELEASE_SIGNAL
    )
    assert any(
        row["need"] == "context_sources_need_release_criteria_and_confidence_budget"
        for row in operating_model["refinement_backlog"]
    )
    assert payload["swarm"]["active_sections"] == operating_model["active_sections"]


def test_degradation_swarm_allows_known_owner_verifier_commands() -> None:
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "roster-resilience", "--json"]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "artifact-freshness-slo", "--json"]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "external-backlog-retry-bot", "--apply", "--json"]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "local-storage-reserve-guard", "--apply", "--json"]
    )
    assert swarm_src._command_safe(
        [
            "./scripts/ops/opsctl.sh",
            "raw-training-compaction",
            "--apply",
            "--compaction-workers",
            "4",
            "--json",
        ]
    )
    assert swarm_src._command_safe(
        [
            "./scripts/ops/opsctl.sh",
            "storage-pressure-clearance",
            "--apply",
            "--checkpoint-mode",
            "passive",
            "--json",
        ]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "storage-prune-standby", "--apply", "--json"]
    )
    assert swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "storage-switch-external", "--dry-run"]
    )
    assert not swarm_src._command_safe(
        ["./scripts/ops/opsctl.sh", "storage-switch-external"]
    )


def test_degradation_swarm_storage_backpressure_assignment_is_tightly_bounded() -> None:
    storage_task = next(
        row
        for row in swarm_src._lane_playbooks()["storage_reserve"]
        if row["infrabot_id"] == "storage_backpressure_infrabot"
    )

    assert storage_task["command"] == [
        "./scripts/ops/opsctl.sh",
        "storage-backpressure-autopilot",
        "--apply",
        "--quick-bounded",
        "--wait-timeout-seconds",
        "20",
        "--command-timeout-seconds",
        "60",
        "--backpressure-command-timeout-seconds",
        "30",
        "--json",
    ]
    assert storage_task["resource_lock"] == "single_writer"


def test_degradation_swarm_storage_pipeline_is_stage_ordered() -> None:
    tasks = swarm_src._lane_playbooks()["storage_reserve"]
    by_id = {row["infrabot_id"]: row for row in tasks}

    assert {
        "local_storage_reserve_infrabot",
        "pcore_storage_contract_infrabot",
        "raw_file_compaction_infrabot",
        "sqlite_checkpoint_infrabot",
        "external_route_rehome_planner",
        "verified_standby_prune_infrabot",
        "storage_retention_unison_infrabot",
    } <= set(by_id)
    assert by_id["pcore_storage_contract_infrabot"]["resource_lock"] == (
        "p_core_file_compaction"
    )
    raw_command = by_id["raw_file_compaction_infrabot"]["command"]
    assert raw_command[raw_command.index("--compaction-workers") + 1] == "4"
    assert by_id["sqlite_checkpoint_infrabot"]["resource_lock"] == "single_writer"
    assert by_id["verified_standby_prune_infrabot"]["resource_lock"] == (
        "storage_standby_prune"
    )
    assert by_id["external_route_rehome_planner"]["auto_execute_requested"] is False
    assert by_id["external_route_rehome_planner"]["command"] == [
        "./scripts/ops/opsctl.sh",
        "storage-switch-external",
        "--dry-run",
    ]


def test_degradation_swarm_routes_coordination_state_attention_to_ops_self_audit() -> (
    None
):
    lane = swarm_src._lane_for_runtime_row(
        {"attention": "coordination_state_control_needs_work", "domain": ""}
    )

    assert lane == "ops_self_audit"


def test_degradation_swarm_blocks_execution_when_degradation_is_uncontained(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    health = _contained_fixture(project_root)
    _write_json(
        health / "operator_cockpit_latest.json",
        {
            "overall_status": "degraded",
            "degradation_containment": {
                "status": "uncontained_degradation",
                "contained_degradation_lanes": ["storage_reserve"],
                "uncontained_lanes": ["collection_hot_path"],
                "safe_to_keep_collecting": False,
                "lanes": {
                    "storage_reserve": {"status": "contained", "contained": True},
                    "collection_hot_path": {
                        "status": "degraded",
                        "contained": False,
                    },
                },
            },
        },
    )

    payload = swarm_src.build_payload(project_root)

    assert payload["overall_status"] == "storage_recovery"
    assert payload["swarm"]["ready_to_swarm"] is True
    assert payload["swarm"]["mode"] == (
        "storage_recovery_while_global_degradation_uncontained"
    )
    assert payload["swarm"]["storage_recovery_assignment_count"] > 0
    storage_assignments = [
        row for row in payload["assignments"] if row["lane"] == "storage_reserve"
    ]
    non_storage_assignments = [
        row for row in payload["assignments"] if row["lane"] != "storage_reserve"
    ]
    assert storage_assignments
    assert any(row["safe_execute_allowed"] for row in storage_assignments)
    assert all(row["allowed_when_uncontained"] is True for row in storage_assignments)
    assert all(
        "uncontained_degradation_present" in row["blocked_by"]
        for row in non_storage_assignments
    )


def test_degradation_swarm_executes_bounded_safe_repairs_with_context_env(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    _contained_fixture(project_root)
    calls: list[dict] = []

    def fake_run(command, *, cwd, timeout_seconds, env):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout_seconds": timeout_seconds,
                "env": dict(env),
            }
        )
        return {
            "rc": 0,
            "timed_out": False,
            "stdout": json.dumps({"overall_status": "ready", "ok": True}),
            "stderr": "",
        }

    monkeypatch.setattr(swarm_src, "run_bounded_process_group", fake_run)
    out_path = project_root / "governance" / "health" / "swarm.json"
    context_path = project_root / "governance" / "health" / "swarm_context.json"
    state_path = project_root / "governance" / "health" / "swarm_state.json"
    ledger_path = project_root / "logs" / "swarm_ledger.jsonl"

    payload = swarm_src.build_payload(
        project_root,
        apply=True,
        execute_safe_repairs=True,
        max_execute_actions=2,
        command_timeout_seconds=30,
        out_path=out_path,
        context_path=context_path,
        state_path=state_path,
        lock_path=project_root / "governance" / "locks" / "swarm.lock",
        ledger_path=ledger_path,
    )

    assert payload["overall_status"] == "executed"
    assert payload["execution_summary"]["executed_count"] == 2
    assert context_path.exists()
    assert state_path.exists()
    assert ledger_path.exists()
    assert len(calls) == 2
    assert all(call["env"]["ALLOW_ORDER_EXECUTION"] == "0" for call in calls)
    assert all(
        call["env"]["DEGRADATION_SWARM_CONTEXT_PATH"] == str(context_path)
        for call in calls
    )
    assert all("start-live" not in call["command"] for call in calls)


def test_degradation_swarm_treats_reported_gate_debt_as_followup_not_command_failure() -> (
    None
):
    classification = swarm_src._classify_result(
        {
            "rc": 2,
            "timed_out": False,
            "payload": {"overall_status": "needs_attention", "ok": False},
        }
    )

    assert classification["outcome"] == "completed_with_followups"
    assert classification["success_like"] is True
    assert classification["command_failed"] is False
    assert classification["requires_followup"] is True


def test_degradation_swarm_treats_recent_running_storage_timeout_as_contained_progress(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root
        / "governance"
        / "health"
        / "storage_backpressure_autopilot_latest.json",
        {
            "timestamp_utc": swarm_src.iso_now(),
            "overall_status": "running",
            "ok": True,
            "busy": True,
            "quick_bounded": True,
            "recommended_actions": ["continue bounded storage drain"],
        },
    )

    classification = swarm_src._classify_result(
        {
            "rc": 124,
            "timed_out": True,
            "payload": {},
        },
        assignment={
            "exec_command": [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "storage-backpressure-autopilot",
                "--apply",
                "--quick-bounded",
                "--json",
            ]
        },
        project_root=project_root,
    )

    assert classification["outcome"] == "completed_with_followups"
    assert classification["success_like"] is True
    assert classification["command_failed"] is False
    assert classification["requires_followup"] is True
    assert classification["timeout_progressing"] is True
    assert classification["summary"]["overall_status"] == "running"
    assert (
        classification["summary"]["timeout_progress_classification"]
        == "owner_artifact_recent_ok_and_progressing"
    )


def test_degradation_swarm_treats_external_backlog_timeout_progress_as_contained(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": swarm_src.iso_now(),
            "overall_status": "drain_active",
            "ok": True,
        },
    )

    classification = swarm_src._classify_result(
        {
            "rc": 124,
            "timed_out": True,
            "payload": {},
        },
        assignment={
            "exec_command": [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "external-backlog-drain",
                "--apply",
                "--follow-through",
                "--json",
            ]
        },
        project_root=project_root,
    )

    assert classification["outcome"] == "completed_with_followups"
    assert classification["success_like"] is True
    assert classification["command_failed"] is False
    assert classification["timeout_progressing"] is True
    assert classification["summary"]["overall_status"] == "drain_active"


def test_degradation_swarm_counts_reclassified_timeout_as_progress(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    assignment = {
        "assignment_id": "storage_reserve:external_backlog_drain_infrabot:abc",
        "incident_id": "incident-1",
        "lane": "storage_reserve",
        "infrabot_id": "external_backlog_drain_infrabot",
        "phase": "repair",
        "priority": 10,
        "exec_command": [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "external-backlog-drain",
            "--apply",
            "--follow-through",
            "--json",
        ],
        "max_attempts_per_incident": 2,
        "safe_execute_allowed": True,
    }

    def fake_run(*_args, **_kwargs):
        _write_json(
            project_root
            / "governance"
            / "health"
            / "external_backlog_drain_latest.json",
            {
                "timestamp_utc": swarm_src.iso_now(),
                "overall_status": "drain_active",
                "ok": True,
            },
        )
        return {"rc": 124, "timed_out": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(swarm_src, "run_bounded_process_group", fake_run)
    summary = swarm_src._execute_assignments(
        project_root,
        {"incident_id": "incident-1", "assignments": [assignment]},
        max_execute_actions=1,
        command_timeout_seconds=30,
        context_path=project_root / "governance" / "health" / "context.json",
        state_path=project_root / "governance" / "health" / "swarm_state.json",
        lock_path=project_root / "governance" / "locks" / "swarm.lock",
        ledger_path=project_root / "logs" / "swarm_ledger.jsonl",
    )

    assert summary["executed_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["timed_out_count"] == 0
    assert summary["raw_timed_out_count"] == 1
    assert summary["progress_reclassified_timeout_count"] == 1
    result = summary["results"][0]
    assert result["timed_out"] is True
    assert result["failed"] is False
    assert result["classification"]["timeout_progressing"] is True


def test_degradation_swarm_skips_recent_external_backlog_owner_progress(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": swarm_src.iso_now(),
            "overall_status": "drain_active",
            "ok": True,
        },
    )
    assignment = {
        "assignment_id": "storage_reserve:external_backlog_drain_infrabot:abc",
        "incident_id": "incident-1",
        "lane": "storage_reserve",
        "infrabot_id": "external_backlog_drain_infrabot",
        "phase": "repair",
        "priority": 10,
        "exec_command": [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "external-backlog-drain",
            "--apply",
            "--follow-through",
            "--json",
        ],
        "safe_execute_allowed": True,
    }

    def fake_run(*_args, **_kwargs):
        raise AssertionError("recent owner progress should prevent relaunch")

    monkeypatch.setattr(swarm_src, "run_bounded_process_group", fake_run)
    summary = swarm_src._execute_assignments(
        project_root,
        {"incident_id": "incident-1", "assignments": [assignment]},
        max_execute_actions=1,
        command_timeout_seconds=30,
        context_path=project_root / "governance" / "health" / "context.json",
        state_path=project_root / "governance" / "health" / "swarm_state.json",
        lock_path=project_root / "governance" / "locks" / "swarm.lock",
        ledger_path=project_root / "logs" / "swarm_ledger.jsonl",
    )

    assert summary["executed_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["completed_from_state_count"] == 1
    result = summary["results"][0]
    assert result["reason"] == "recent_owner_progress_detected"
    assert result["classification"]["summary"]["overall_status"] == "drain_active"


def test_degradation_swarm_reclassifies_cooldown_timeout_when_owner_progresses(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    assignment = {
        "assignment_id": "storage_reserve:external_backlog_drain_infrabot:abc",
        "incident_id": "incident-1",
        "lane": "storage_reserve",
        "infrabot_id": "external_backlog_drain_infrabot",
        "exec_command": [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "external-backlog-drain",
            "--apply",
            "--follow-through",
            "--json",
        ],
        "max_attempts_per_incident": 2,
        "safe_execute_allowed": True,
    }
    state_path = project_root / "governance" / "health" / "swarm_state.json"
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "assignments": {
                assignment["assignment_id"]: {
                    "incident_id": "incident-1",
                    "last_outcome": "timeout",
                    "failure_count": 1,
                    "cooldown_until_utc": (
                        swarm_src.datetime.now(swarm_src.timezone.utc)
                        + swarm_src.timedelta(minutes=5)
                    ).isoformat(),
                    "last_summary": {},
                }
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "external_backlog_drain_latest.json",
        {
            "timestamp_utc": swarm_src.iso_now(),
            "overall_status": "drain_active",
            "ok": True,
        },
    )

    def fake_run(*_args, **_kwargs):
        raise AssertionError("cooldown progress should not relaunch the command")

    monkeypatch.setattr(swarm_src, "run_bounded_process_group", fake_run)
    summary = swarm_src._execute_assignments(
        project_root,
        {"incident_id": "incident-1", "assignments": [assignment]},
        max_execute_actions=1,
        command_timeout_seconds=30,
        context_path=project_root / "governance" / "health" / "context.json",
        state_path=state_path,
        lock_path=project_root / "governance" / "locks" / "swarm.lock",
        ledger_path=project_root / "logs" / "swarm_ledger.jsonl",
    )

    assert summary["executed_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["completed_from_state_count"] == 1
    assert summary["followup_count"] == 1
    result = summary["results"][0]
    assert result["reason"] == (
        "cooldown_timeout_reclassified_from_recent_owner_progress"
    )
    assert result["classification"]["summary"]["overall_status"] == "drain_active"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    row = state["assignments"][assignment["assignment_id"]]
    assert row["last_outcome"] == "completed_with_followups"
    assert row["failure_count"] == 0


def test_degradation_swarm_keeps_unknown_timeout_as_retryable_failure() -> None:
    classification = swarm_src._classify_result(
        {"rc": 124, "timed_out": True, "payload": {}}
    )

    assert classification["outcome"] == "timeout"
    assert classification["success_like"] is False
    assert classification["command_failed"] is True
    assert classification["retryable"] is True


def test_degradation_swarm_skips_completed_assignment_for_same_incident() -> None:
    gate = swarm_src._state_gate(
        {
            "assignments": {
                "source_verification:bot:abc": {
                    "incident_id": "incident-1",
                    "last_outcome": "completed_with_followups",
                    "failure_count": 0,
                }
            }
        },
        {
            "assignment_id": "source_verification:bot:abc",
            "incident_id": "incident-1",
        },
        now=swarm_src.datetime.now(swarm_src.timezone.utc),
    )

    assert gate["active"] is True
    assert gate["gate"] == "already_completed"


def test_degradation_swarm_reports_previously_completed_assignments(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    _contained_fixture(project_root)
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, timeout_seconds, env):
        calls.append(command)
        if len(calls) == 1:
            return {
                "rc": 0,
                "timed_out": False,
                "stdout": json.dumps({"overall_status": "ready", "ok": True}),
                "stderr": "",
            }
        return {
            "rc": 2,
            "timed_out": False,
            "stdout": json.dumps({"overall_status": "needs_attention", "ok": False}),
            "stderr": "",
        }

    monkeypatch.setattr(swarm_src, "run_bounded_process_group", fake_run)
    out_path = project_root / "governance" / "health" / "swarm.json"
    context_path = project_root / "governance" / "health" / "swarm_context.json"
    state_path = project_root / "governance" / "health" / "swarm_state.json"
    ledger_path = project_root / "logs" / "swarm_ledger.jsonl"
    lock_path = project_root / "governance" / "locks" / "swarm.lock"

    first = swarm_src.build_payload(
        project_root,
        apply=True,
        execute_safe_repairs=True,
        max_execute_actions=20,
        command_timeout_seconds=30,
        out_path=out_path,
        context_path=context_path,
        state_path=state_path,
        lock_path=lock_path,
        ledger_path=ledger_path,
    )
    executed_count = first["execution_summary"]["executed_count"]
    followup_count = first["execution_summary"]["followup_count"]
    assert executed_count > 1
    assert followup_count == executed_count - 1
    assert first["overall_status"] == "executed_with_followups"

    calls.clear()
    second = swarm_src.build_payload(
        project_root,
        apply=True,
        execute_safe_repairs=True,
        max_execute_actions=20,
        command_timeout_seconds=30,
        out_path=out_path,
        context_path=context_path,
        state_path=state_path,
        lock_path=lock_path,
        ledger_path=ledger_path,
    )

    summary = second["execution_summary"]
    assert calls == []
    assert summary["executed_count"] == 0
    assert summary["skipped_count"] == executed_count
    assert summary["already_completed_skipped_count"] == executed_count
    assert summary["completed_from_state_count"] == executed_count
    assert summary["effective_completed_count"] == executed_count
    assert summary["already_completed_success_count"] == 1
    assert summary["already_completed_followup_count"] == followup_count
    assert summary["followup_count"] == followup_count
    assert second["overall_status"] == "executed_with_followups"


def test_degradation_swarm_incident_id_ignores_volatile_context_details() -> None:
    base = {
        "contained_degradation_lanes": ["source_verification", "storage_reserve"],
        "uncontained_lanes": [],
        "runtime_attention": [
            "source_verification_context_debt",
            "external_backlog_drain_writer_busy",
        ],
        "source_context_debt": ["decision_context_mesh"],
    }
    changed_context = {
        **base,
        "source_context_debt": ["decision_context_mesh", "optional_macro_context"],
        "safe_to_keep_collecting": False,
    }

    assert swarm_src._incident_id(base) == swarm_src._incident_id(changed_context)
