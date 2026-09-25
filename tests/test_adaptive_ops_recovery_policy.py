from __future__ import annotations

import importlib.util
import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ops"
        / "adaptive_ops_recovery_policy.py"
    )
    spec = importlib.util.spec_from_file_location("adaptive_ops_recovery_policy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_critical_recovery_tightens_storage_controls_and_preserves_trade_safety(
    tmp_path: Path,
) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {
                "degradation_containment": {"hot_path_blocked": True},
                "soak_management_context": {"soak_status": "blocked"},
            },
            "storage": {
                "status": "blocked",
                "pressure_index": 12.173,
                "total_pending_lines": 26322,
                "core_pending_lines": 7502,
            },
            "memory": {"status": "blocked"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"status": "blocked", "pressure_index": 12.173},
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {
            "status": "blocked",
            "hard_blockers": ["local_hot_storage_below_hard_reserve"],
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json", {"overall_status": "blocked"}
    )
    _write_json(
        health / "soak_reliability_sentinel_latest.json",
        {
            "overall_status": "blocked",
            "blockers": ["ingestion_storage_control_not_ready"],
        },
    )

    payload = module.build_payload(tmp_path, apply=False, now="2026-09-09T15:40:00Z")

    assert payload["tier"] == "critical_recovery"
    env = payload["env_overrides"]
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE"] == "1"
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW"] == "0"
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE"] == "1"
    assert env["STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS"] == "5"
    assert env["BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB"] == "20"
    assert env["MARKET_DATA_ONLY"] == "1"
    assert env["ALLOW_ORDER_EXECUTION"] == "0"
    assert env["TOP_BOT_ENABLE_LIVE_EXECUTION"] == "0"
    assert env["BOT_PROTECTED_VOLUME_DENYLIST"] == "/Volumes/VIDEO"
    assert payload["platform_ok"] is False


def test_observe_relaxes_to_normal_cadence_when_pressure_is_low(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {"degradation_containment": {"hot_path_blocked": False}},
            "storage": {
                "status": "ready",
                "pressure_index": 0.05,
                "total_pending_lines": 250,
            },
            "memory": {"status": "ready"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"status": "ready", "pressure_index": 0.0},
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {"status": "ready", "hard_blockers": []},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json", {"overall_status": "ready"}
    )
    _write_json(
        health / "soak_reliability_sentinel_latest.json",
        {"overall_status": "ready", "blockers": []},
    )

    payload = module.build_payload(tmp_path, apply=False, now="2026-09-09T15:45:00Z")

    assert payload["tier"] == "observe"
    env = payload["env_overrides"]
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE"] == "0"
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW"] == "1"
    assert (
        env["MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS"]
        == "600"
    )
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.95"
    assert payload["platform_ok"] is False
    assert payload["control_plane_refresh_status"] == "not_run"


def test_control_plane_refresh_plan_repairs_stale_role_before_soak_dashboard(
    tmp_path: Path,
) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    capabilities = tmp_path / "governance" / "collector_capabilities"
    now = module.iso_now()
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "overall": {
                "status": "degraded",
                "degradation_containment": {"hot_path_blocked": False},
                "attention_tiers": {
                    "critical": [],
                    "degraded": ["system_role_contract_stale"],
                },
            },
            "storage": {
                "status": "ready",
                "pressure_index": 0.10,
                "total_pending_lines": 250,
            },
            "memory": {"status": "ready"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"timestamp_utc": now, "status": "ready"},
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {
            "timestamp_utc": now,
            "status": "ready",
            "hard_blockers": [],
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "soak_reliability_sentinel_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "blockers": [],
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "blockers": ["system_role_contract_stale"],
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {"timestamp_utc": now, "hard_gate_triggered": False},
    )
    _write_json(
        health / "halt_trigger_control_plane_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "coordination_state_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        capabilities / "materialized_capabilities_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "collector_capability_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ok": True},
    )

    metrics = module.collect_metrics(tmp_path)
    plan = module.build_control_plane_refresh_plan(metrics)

    assert [step["id"] for step in plan] == [
        "risk_service_boundary",
        "system_role_contract",
        "soak_reliability_sentinel",
        "unattended_soak_readiness",
        "runtime_gate_dashboard",
    ]


def test_control_plane_refresh_plan_orders_health_halt_before_soak(
    tmp_path: Path,
) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    capabilities = tmp_path / "governance" / "collector_capabilities"
    now = module.iso_now()
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "overall": {
                "status": "degraded",
                "degradation_containment": {"hot_path_blocked": False},
                "attention_tiers": {"critical": [], "degraded": []},
            },
            "storage": {"status": "ready", "pressure_index": 0.10},
            "memory": {"status": "ready"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"timestamp_utc": now, "status": "ready"},
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {"timestamp_utc": now, "status": "ready"},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "soak_reliability_sentinel_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "blockers": [
                "health_gates_refresh_due",
                "halt_trigger_control_plane_not_ready",
            ],
        },
    )
    _write_json(
        health / "system_role_contract_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ok": True},
    )
    _write_json(
        capabilities / "materialized_capabilities_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "collector_capability_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ok": True},
    )
    _write_json(
        health / "coordination_state_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "blockers": []},
    )

    metrics = module.collect_metrics(tmp_path)
    plan = module.build_control_plane_refresh_plan(metrics)

    assert [step["id"] for step in plan] == [
        "risk_service_boundary",
        "health_gates",
        "global_halt_refresh",
        "halt_trigger_status",
        "soak_reliability_sentinel",
        "unattended_soak_readiness",
        "runtime_gate_dashboard",
    ]


def test_control_plane_refresh_plan_handles_coordination_refresh_due(
    tmp_path: Path,
) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    capabilities = tmp_path / "governance" / "collector_capabilities"
    now = module.iso_now()
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "overall": {
                "status": "degraded",
                "degradation_containment": {"hot_path_blocked": False},
                "attention_tiers": {"critical": [], "degraded": []},
            },
            "storage": {"status": "ready", "pressure_index": 0.10},
            "memory": {"status": "ready"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"timestamp_utc": now, "status": "ready"},
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {"timestamp_utc": now, "status": "ready"},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "soak_reliability_sentinel_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "blockers": ["coordination_state_refresh_due"],
        },
    )
    _write_json(
        health / "system_role_contract_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ok": True},
    )
    _write_json(
        health / "health_gates_latest.json",
        {"timestamp_utc": now, "hard_gate_triggered": False},
    )
    _write_json(
        health / "halt_trigger_control_plane_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        capabilities / "materialized_capabilities_latest.json",
        {"timestamp_utc": now, "overall_status": "ready"},
    )
    _write_json(
        health / "collector_capability_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "ok": True},
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "blockers": []},
    )

    metrics = module.collect_metrics(tmp_path)
    plan = module.build_control_plane_refresh_plan(metrics)

    assert [step["id"] for step in plan] == [
        "risk_service_boundary",
        "coordination_state",
        "soak_reliability_sentinel",
        "unattended_soak_readiness",
        "runtime_gate_dashboard",
    ]


def test_apply_writes_override_and_health_payload_idempotently(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    override = tmp_path / "config" / ".env.adaptive_ops_recovery_policy_override"
    out_file = health / "adaptive_ops_recovery_policy_latest.json"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "overall": {"degradation_containment": {"hot_path_blocked": True}},
            "storage": {
                "status": "blocked",
                "pressure_index": 6.0,
                "total_pending_lines": 18000,
            },
            "memory": {"status": "ready"},
        },
    )

    payload = module.build_payload(
        tmp_path,
        apply=True,
        out_file=out_file,
        override_file=override,
        now="2026-09-09T15:50:00Z",
        refresh_control_plane=False,
    )
    assert payload["override_changed"] is True
    assert payload["timestamp_utc"] == "2026-09-09T15:50:00Z"
    lifecycle = payload["job_lifecycle"]
    assert lifecycle["job_id"] == "adaptive_ops_recovery_policy"
    assert lifecycle["scheduled"] is False
    assert lifecycle["failed"] is False
    assert lifecycle["terminal_status"] == "completed"
    assert out_file.exists()
    text = override.read_text(encoding="utf-8")
    assert "export ADAPTIVE_OPS_RECOVERY_TIER=critical_recovery" in text
    assert "export MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE=1" in text

    second = module.build_payload(
        tmp_path,
        apply=True,
        out_file=out_file,
        override_file=override,
        now="2026-09-09T15:50:00Z",
        refresh_control_plane=False,
    )
    assert second["override_changed"] is False


@pytest.mark.parametrize(
    "timestamp", [None, "invalid", "2026-09-09T00:00:00", "future"]
)
def test_refresh_requires_valid_producer_time_not_file_mtime(tmp_path, timestamp):
    module = _load_module()
    if timestamp == "future":
        timestamp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    path = tmp_path / "fresh_file.json"
    _write_json(path, {"timestamp_utc": timestamp, "ok": True})
    row = module._health_artifact_metric(
        tmp_path, "test", Path(path.name), max_age_minutes=15
    )
    assert row["stale"] is True
    assert row["refresh_due"] is True


@pytest.mark.parametrize(
    "payload,rc,step,expected",
    [
        ({}, 0, "system_role_contract", False),
        ({"ok": True}, 1, "system_role_contract", False),
        ({"ok": False, "overall_status": "ready"}, 0, "system_role_contract", False),
        ({"ok": True}, 0, "system_role_contract", True),
        ({"overall_status": "blocked"}, 0, "halt_trigger_status", False),
        (
            {
                "overall_status": "blocked",
                "blockers": {"halt_clear": [], "live": ["paper_trade_lock_active"]},
            },
            0,
            "halt_trigger_status",
            True,
        ),
        ({"hard_gate_triggered": False}, 0, "health_gates", True),
        ({"hard_gate_triggered": True}, 0, "health_gates", False),
        (
            {
                "ok": True,
                "continuous_run_soak_contract": {
                    "status": "blocked",
                    "blockers": ["drain_time_above_target"],
                },
            },
            0,
            "ingestion_storage_control",
            False,
        ),
    ],
)
def test_refresh_success_requires_valid_completion(payload, rc, step, expected):
    module = _load_module()
    assert module._refresh_payload_ok(payload, rc, step_id=step) is expected


@pytest.mark.parametrize(
    "max_steps,budget,reason",
    [
        (0, 210, "max_refresh_steps_reached"),
        (10, 0, "refresh_budget_exhausted"),
        (10, 210, "opsctl_missing"),
    ],
)
def test_refresh_reports_every_unfinished_step(tmp_path, max_steps, budget, reason):
    module = _load_module()
    plan = module.build_control_plane_refresh_plan({"soak_blocked": True})
    results = module.run_control_plane_refresh_plan(
        tmp_path, plan, env_overrides={}, max_steps=max_steps, budget_seconds=budget
    )
    assert len(results) == len(plan) > 1
    assert all(row["ok"] is False and row["skip_reason"] == reason for row in results)


def test_refresh_uses_bounded_runner_and_safety_env(tmp_path, monkeypatch):
    module = _load_module()
    opsctl = tmp_path / "scripts/ops/opsctl.sh"
    opsctl.parent.mkdir(parents=True)
    opsctl.touch()
    calls = []

    def bounded(command, **kwargs):
        calls.append(kwargs)
        return {
            "rc": 124,
            "stdout": '{"ok": true}',
            "timed_out": True,
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(module, "run_bounded_process_group", bounded)
    plan = module.build_control_plane_refresh_plan({"dashboard_status": "degraded"})
    results = module.run_control_plane_refresh_plan(
        tmp_path,
        plan,
        env_overrides={"ALLOW_ORDER_EXECUTION": "1"},
        max_steps=10,
        budget_seconds=210,
    )
    assert calls[0]["env"]["ALLOW_ORDER_EXECUTION"] == "0"
    assert results[0]["ok"] is False
    assert results[0]["timeout_cleanup"]["reaped"] is True


def test_metrics_read_canonical_ingestion_status_and_backlog(tmp_path):
    module = _load_module()
    _write_json(
        tmp_path / "governance/health/ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "backpressure": {"total_pending_lines": 27000},
        },
    )
    metrics = module.collect_metrics(tmp_path)
    assert metrics["storage_blocked"] is True
    assert metrics["total_pending_lines"] == 27000
    assert module.choose_tier(metrics) == "critical_recovery"


def test_busy_policy_writer_defers_without_refresh_or_artifact_overwrite(
    tmp_path, monkeypatch, capsys
):
    module = _load_module()
    lock = tmp_path / "governance/locks/adaptive_ops_recovery_policy.lock"
    lock.parent.mkdir(parents=True)

    def unexpected(*args, **kwargs):
        pytest.fail("busy writer must not compute or publish another generation")

    monkeypatch.setattr(module, "build_payload", unexpected)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = module.main(["--project-root", str(tmp_path), "--apply", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "status=deferred" in captured.err
    assert json.loads(captured.out)["reason"] == "policy_writer_busy"
    assert not (tmp_path / module.DEFAULT_OUT_REL).exists()


def test_pressure_override_is_published_before_slow_refresh(tmp_path, monkeypatch):
    module = _load_module()
    _write_json(
        tmp_path / "governance/health/ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "pressure_index": 6.0,
        },
    )

    def refresh(project_root, plan, **kwargs):
        override = (project_root / module.DEFAULT_OVERRIDE_REL).read_text()
        assert "ADAPTIVE_OPS_RECOVERY_TIER=critical_recovery" in override
        return [{"id": step["id"], "ok": False, "skipped": True} for step in plan]

    monkeypatch.setattr(module, "run_control_plane_refresh_plan", refresh)
    payload = module.build_payload(tmp_path, apply=True)
    assert payload["ok"] is False
    assert payload["control_plane_refresh_status"] == "attention"
    assert payload["job_lifecycle"]["terminal_status"] == "completed_with_findings"


def test_rising_backlog_refreshes_storage_before_halt_consumers():
    module = _load_module()
    plan = module.build_control_plane_refresh_plan({"total_pending_lines": 16000})
    assert [step["id"] for step in plan] == [
        "ingestion_storage_control",
        "global_halt_refresh",
        "halt_trigger_status",
        "system_plumbing_control",
        "system_architecture_hardening",
        "soak_reliability_sentinel",
        "unattended_soak_readiness",
        "runtime_gate_dashboard",
    ]
    assert all("--apply" not in step["command"] for step in plan)


def test_adaptive_policy_cannot_relax_owner_intake_caps_or_pauses(tmp_path):
    module = _load_module()
    config = tmp_path / "config"
    config.mkdir()
    owner = config / ".env.storage_pressure_override"
    owner.write_text(
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.16\nTRAINING_RUNTIME_PAUSED_FOR_BACKLOG=1\n"
    )
    (config / ".env.operator_mode_override").write_text(
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.20\nTRAINING_RUNTIME_PAUSED_FOR_BACKLOG=0\n"
    )
    env = module.env_for_tier("observe", project_root=tmp_path)
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.16"
    assert env["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "1"
    owner.write_text(
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.95\nTRAINING_RUNTIME_PAUSED_FOR_BACKLOG=0\n"
    )
    recovered = module.env_for_tier("observe", project_root=tmp_path)
    assert recovered["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.2"
    assert recovered["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "0"


def test_unattended_storage_blocker_selects_soak_recovery(tmp_path):
    module = _load_module()
    _write_json(
        tmp_path / "governance/health/soak_reliability_sentinel_latest.json",
        {"overall_status": "ready"},
    )
    _write_json(
        tmp_path / "governance/health/unattended_soak_readiness_latest.json",
        {
            "overall_status": "blocked",
            "blockers": ["ingestion_soak_contract_not_ready"],
        },
    )
    assert module.choose_tier(module.collect_metrics(tmp_path)) == "soak_recovery"


def test_soak_recovery_uses_two_short_rescored_waves():
    module = _load_module()
    env = module.env_for_tier("soak_recovery")
    assert env["STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES"] == "2"
    assert env["STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS"] == "180"
    assert env["STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS"] == "75"
    assert env["STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES"] == "5000"
    assert env["MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE"] == "1"
    assert env["ALLOW_ORDER_EXECUTION"] == "0"
    for tier in ("observe", "critical_recovery"):
        restored = module.env_for_tier(tier)
        assert restored["STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES"] == "1"
        assert restored["STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES"] == "20000"


def test_paper_guard_refresh_precedes_soak_consumers():
    module = _load_module()
    plan = module.build_control_plane_refresh_plan(
        {"soak_blockers": ["runtime_paper_regression_guard_not_ready"]}
    )
    assert [row["id"] for row in plan] == [
        "runtime_paper_regression_guard",
        "soak_reliability_sentinel",
        "unattended_soak_readiness",
        "runtime_gate_dashboard",
    ]
    assert "--apply" not in plan[0]["command"]


def test_stale_advisory_reports_refresh_in_dependency_order_without_authority():
    module = _load_module()
    names = [
        "bot_profitability_scalability_control",
        "sleeve_scalability_selector",
        "master_grandmaster_evidence_v2",
    ]
    plan = module.build_control_plane_refresh_plan(
        {"dashboard_degraded_attention": [f"{name}_stale" for name in reversed(names)]}
    )
    assert [row["id"] for row in plan] == names + ["runtime_gate_dashboard"]
    assert all("--apply" not in row["command"] for row in plan)
    assert all(row["timeout_seconds"] <= 45 for row in plan)
    assert (
        module.build_control_plane_refresh_plan(
            {
                "dashboard_degraded_attention": [
                    "untrusted_command_stale",
                    "teacher_quality_evidence_pending",
                ]
            }
        )
        == []
    )
