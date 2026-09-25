import json
from pathlib import Path

import pytest

from core.cpu_workload_policy import (
    CPUWorkloadPolicyError,
    cpu_environment_contract,
    load_cpu_workload_policy,
    nice_target_for_class,
    resolve_shadow_workload_class,
    resource_partition,
    runtime_priority_decision,
    taskpolicy_executable,
)
from core import cpu_workload_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_policy_is_locked_and_cannot_claim_runtime_authority() -> None:
    policy = load_cpu_workload_policy(PROJECT_ROOT / "config" / "cpu_workload_policy_v1.json")

    assert policy["policy_locked"] is True
    assert policy["platform_contract"]["hard_affinity_supported_on_macos"] is False
    assert all(value is False for value in policy["authority"].values())


def test_taskpolicy_resolution_accepts_current_macos_sbin_location(monkeypatch) -> None:
    monkeypatch.setattr(cpu_workload_policy.shutil, "which", lambda _name: "/usr/sbin/taskpolicy")
    monkeypatch.setattr(cpu_workload_policy.os, "access", lambda path, _mode: path == "/usr/sbin/taskpolicy")

    assert taskpolicy_executable() == "/usr/sbin/taskpolicy"


def test_taskpolicy_resolution_is_optional_when_binary_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(cpu_workload_policy.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cpu_workload_policy.os, "access", lambda _path, _mode: False)

    assert taskpolicy_executable() == ""


@pytest.mark.parametrize("invalid_boundary", ["fast", 4.5, True, -1, 21])
def test_policy_rejects_malformed_nice_boundaries(
    tmp_path: Path,
    invalid_boundary: object,
) -> None:
    payload = load_cpu_workload_policy()
    payload["workload_classes"]["market_decision"]["nice_ceiling"] = invalid_boundary
    policy_path = tmp_path / "cpu_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        CPUWorkloadPolicyError,
        match="cpu_workload_class_nice_value_invalid:market_decision",
    ):
        load_cpu_workload_policy(policy_path)


def test_m1_max_partition_reserves_one_performance_core() -> None:
    policy = load_cpu_workload_policy()

    partition = resource_partition(policy, performance_core_count=8, efficiency_core_count=2)

    assert partition["critical_shared_performance_workers"] == 7
    assert partition["foreground_performance_core_reserve"] == 1
    assert partition["efficiency_service_workers"] == 2
    assert partition["critical_efficiency_spillover_allowed"] is False


def test_nice_boundaries_separate_critical_and_support_work() -> None:
    policy = load_cpu_workload_policy()

    assert nice_target_for_class(policy, "critical_supervisor", 20) == 0
    assert nice_target_for_class(policy, "paper_execution", 20) == 0
    assert nice_target_for_class(policy, "market_decision", 18) == 4
    assert nice_target_for_class(policy, "data_collection", 0) == 12
    assert nice_target_for_class(policy, "research_training", 4) == 15


def test_priority_decision_never_claims_in_process_elevation() -> None:
    policy = load_cpu_workload_policy()

    decision = runtime_priority_decision(
        policy,
        workload_class="market_decision",
        requested_nice=4,
        current_nice=18,
    )

    assert decision["managed_restart_required"] is True
    assert decision["self_deprioritize_delta"] == 0
    assert decision["hard_affinity_claimed"] is False


def test_priority_decision_treats_ceiling_and_floor_as_boundaries() -> None:
    policy = load_cpu_workload_policy()

    critical = runtime_priority_decision(
        policy,
        workload_class="market_decision",
        requested_nice=4,
        current_nice=0,
    )
    support = runtime_priority_decision(
        policy,
        workload_class="data_collection",
        requested_nice=12,
        current_nice=20,
    )

    assert critical["priority_compliant"] is True
    assert critical["self_deprioritize_delta"] == 0
    assert support["priority_compliant"] is True
    assert support["managed_restart_required"] is False


def test_shadow_resolution_defaults_to_decisions_and_honors_collect_only() -> None:
    policy = load_cpu_workload_policy()

    assert resolve_shadow_workload_class(policy, profile="dividend") == "market_decision"
    assert (
        resolve_shadow_workload_class(
            policy,
            profile="data_plane_backpressure_resilience",
            lifecycle_state="data_collection_only",
        )
        == "data_collection"
    )
    assert resolve_shadow_workload_class(policy, explicit="research_training") == "research_training"
    with pytest.raises(CPUWorkloadPolicyError):
        resolve_shadow_workload_class(policy, explicit="magic_core_pin")


def test_environment_contract_locks_precedence_without_hard_affinity_claim() -> None:
    policy = load_cpu_workload_policy()

    env = cpu_environment_contract(policy, performance_core_count=8, efficiency_core_count=2)

    assert env["BOT_CPU_WORKLOAD_POLICY_LOCKED"] == "1"
    assert env["BOT_CPU_HARD_AFFINITY_SUPPORTED"] == "0"
    assert env["BOT_CPU_CRITICAL_SUPERVISOR_MAX_NICE"] == "0"
    assert env["BOT_CPU_PAPER_EXECUTION_MAX_NICE"] == "0"
    assert env["BOT_CPU_MARKET_DECISION_MAX_NICE"] == "4"
    assert env["BOT_CPU_CRITICAL_SHARED_WORKERS"] == "7"


def test_runtime_env_applies_locked_priority_contract_at_final_precedence() -> None:
    text = (PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh").read_text(encoding="utf-8")

    lock_offset = text.rfind('export BOT_CPU_WORKLOAD_POLICY_LOCKED="1"')
    access_mode_offset = text.index('ACCESS_MODE_RAW="${BOT_RUNTIME_ACCESS_MODE:-native}"')

    assert lock_offset > text.rfind("load_file")
    assert lock_offset < access_mode_offset
    assert 'export PAPER_EXECUTION_RUNTIME_NICE="0"' in text[lock_offset:access_mode_offset]
    assert 'export BOT_CPU_CRITICAL_SUPERVISOR_MAX_NICE="0"' in text[lock_offset:access_mode_offset]
    assert 'export SLEEVE_NICE_SPECIALIZED="12"' in text[lock_offset:access_mode_offset]
    assert 'export RUNTIME_RESEARCH_TRAINING_NICE="15"' in text[lock_offset:access_mode_offset]


def test_shadow_launchers_carry_explicit_workload_classes() -> None:
    market_launchers = (
        "run_parallel_shadows.py",
        "run_dividend_shadow.py",
        "run_bond_shadow.py",
        "run_fx_shadow.py",
        "run_parallel_aggressive_modes.py",
    )
    for filename in market_launchers:
        text = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert '"--runtime-cpu-class"' in text or "'--runtime-cpu-class'" in text
        assert '"market_decision"' in text or "'market_decision'" in text

    specialized = (PROJECT_ROOT / "scripts" / "run_specialized_sleeve_shadow.py").read_text(encoding="utf-8")
    assert '"--runtime-cpu-class"' in specialized
    assert '"data_collection"' in specialized

    opsctl = (PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh").read_text(encoding="utf-8")
    watchdog = (PROJECT_ROOT / "scripts" / "shadow_watchdog.py").read_text(encoding="utf-8")
    assert opsctl.count("--runtime-cpu-class market_decision") >= 3
    assert watchdog.count("--runtime-cpu-class market_decision") >= 3
