import json
import signal
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import runtime_throttle_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_runtime_throttle_classifies_paper_execution_lane_separately() -> None:
    classification = src._classify_process(
        "/repo/.venv/bin/python /repo/scripts/run_execution_lane.py --mode paper"
    )

    assert classification["category"] == "paper_execution"
    assert classification["priority_tier"] == "paper_gate_controlled"
    assert classification["throttle_candidate"] is True


def test_runtime_throttle_classifies_shadow_runners_as_paper_downshift_lanes() -> None:
    classification = src._classify_process(
        "/repo/.venv/bin/python /repo/scripts/run_parallel_shadows.py --broker schwab --interval-seconds 60"
    )

    assert classification["category"] == "paper_execution"
    assert classification["priority_tier"] == "paper_shadow_downshift"
    assert classification["throttle_candidate"] is True


def test_runtime_throttle_classifies_training_requalification_as_research_pressure() -> None:
    classification = src._classify_process(
        "/repo/.venv/bin/python /repo/scripts/ops/training_requalification_lane.py --write-queue --json"
    )

    assert classification["category"] == "research_training"
    assert classification["priority_tier"] == "research_downshift"
    assert classification["throttle_candidate"] is False

    candidates = src._research_training_pressure_candidates(
        [
            {
                "pid": 5152,
                "category": classification["category"],
                "cpu_percent": 79.0,
                "command": "/repo/.venv/bin/python /repo/scripts/ops/training_requalification_lane.py --write-queue --json",
            }
        ],
        profile="sustain",
        compute_pressure_level="normal",
        memory_pressure_level="elevated",
    )

    assert candidates
    assert candidates[0]["pid"] == 5152
    assert candidates[0]["throttle_reason"] == "research_training_loop_under_host_pressure"


def test_runtime_throttle_pauses_blocked_paper_execution_consumer(tmp_path: Path, monkeypatch) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.2,
            "backpressure": {"core_pending_lines": 1200, "total_pending_lines": 1500},
        },
    )
    _write_json(
        health_root / "paper_400_ramp_latest.json",
        {
            "stage": "blocked",
            "ok": False,
            "armed": False,
            "blockers": ["runtime_capacity_not_ready_for_400_paper"],
        },
    )
    runtime_snapshot = {
        "cpu_count": 12,
        "load_averages": {"one_minute": 7.2, "five_minutes": 5.5, "fifteen_minutes": 5.0},
        "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
        "vm_stat": {},
        "top_processes": [
            {
                "pid": 5151,
                "nice": 0,
                "cpu_percent": 88.0,
                "mem_percent": 1.0,
                "elapsed": "00:15:00",
                "command": "python scripts/run_execution_lane.py --mode paper",
                "category": "paper_execution",
                "priority_tier": "paper_gate_controlled",
                "throttle_candidate": True,
            }
        ],
        "category_cpu": {"paper_execution": 88.0},
        "category_counts": {"paper_execution": 1},
    }
    calls: list[list[str]] = []
    kills: list[tuple[int, int]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["paper_execution_policy"]["pause_paper_execution"] is True
    assert payload["paper_execution_pause_candidates"][0]["terminate_when_apply"] is False
    assert payload["runtime_saturation_governor_v2"]["paper_live_data_policy"]["paper_execution_consumer_paused"] is True
    assert "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE=1" in override
    assert "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED=0" in override
    assert "PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED=1" in override
    assert result["paper_execution_pause"]["successful_count"] == 0
    assert (5151, src.signal.SIGTERM) not in kills
    assert any(cmd[:2] == ["renice", "-n"] for cmd in calls)


def test_runtime_throttle_keeps_full_force_paper_open_on_pressure_only_ramp_blocker() -> None:
    policy = src._paper_execution_pressure_pause_policy(
        {
            "artifact_present": True,
            "paper_execution_allowed": False,
            "pause_paper_execution": True,
            "reason": "paper_ramp_blocked",
            "stage": "blocked",
            "armed": False,
            "ok": False,
            "blockers": ["runtime_capacity_not_ready_for_400_paper"],
        },
        {
            "paper_execution_hot": True,
            "paper_hot_low_priority": True,
            "paper_execution_cpu_percent": 90.0,
            "bot_owned_cpu_percent": 210.0,
            "storage_writer_cpu_percent": 80.0,
        },
        throttle_profile="sustain",
        compute_pressure_level="high",
        memory_pressure_level="normal",
        live_read_only=True,
        storage_ready_for_runtime_advisory=True,
        full_force_paper_required=True,
    )

    assert policy["paper_execution_allowed"] is True
    assert policy["pause_paper_execution"] is False
    assert policy["pressure_pause_bypassed"] is True
    assert policy["pressure_pause_bypass_reason"] == "full_force_paper_ramp_pressure_only_blocker"


def test_runtime_throttle_capacity_limits_full_force_paper_instead_of_pausing_cpu_pressure() -> None:
    policy = src._paper_execution_pressure_pause_policy(
        {
            "artifact_present": True,
            "paper_execution_allowed": True,
            "pause_paper_execution": False,
            "reason": "paper_ramp_armed_and_clean",
            "stage": "armed",
            "armed": True,
            "ok": True,
            "blockers": [],
        },
        {
            "paper_execution_hot": True,
            "paper_hot_low_priority": True,
            "paper_execution_pressure_dominant": True,
            "paper_execution_cpu_percent": 180.0,
            "bot_owned_cpu_percent": 190.0,
            "storage_writer_cpu_percent": 0.0,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_cpu_percent": 0.0,
            "protected_work_hot": False,
        },
        throttle_profile="soft_cap",
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
        live_read_only=True,
        storage_ready_for_runtime_advisory=True,
        full_force_paper_required=True,
    )

    assert policy["paper_execution_allowed"] is True
    assert policy["pause_paper_execution"] is False
    assert policy["pressure_pause_bypassed"] is True
    assert policy["pressure_pause_bypass_reason"] == "full_force_paper_ramp_capacity_limited_low_priority_soak"
    assert policy["capacity_limited_paper_execution"] is True

    overrides = src._runtime_env_overrides(
        "soft_cap",
        "normal",
        "elevated",
        paper_capacity_contract={"full_force_stabilization_required": True, "mode": "full_force_buffered"},
        paper_execution_policy=policy,
    )
    assert overrides["PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED"] == "1"
    assert overrides["PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE"] == "0"
    assert overrides["PAPER_EXECUTION_RUNTIME_NICE"] == "20"
    assert overrides["EXECUTION_LANE_BATCH_LIMIT"] == "25"
    assert overrides["EXECUTION_LANE_BATCH_SLEEP_SECONDS"] == "2.0"
    assert overrides["EXECUTION_LANE_BACKLOG_SLEEP_SECONDS"] == "5.0"
    assert overrides["EXECUTION_LANE_HOST_LOAD_SOFT_CAP"] == "6.0"
    assert overrides["EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS"] == "5.0"
    assert overrides["EXECUTION_LANE_MESSAGE_SLEEP_SECONDS"] == "0.04"
    assert overrides["EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS"] == "900"
    assert overrides["EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS"] == "60"


def test_runtime_throttle_downshifts_full_force_paper_without_restart_when_other_lanes_are_hot() -> None:
    policy = src._paper_execution_pressure_pause_policy(
        {
            "artifact_present": True,
            "paper_execution_allowed": True,
            "pause_paper_execution": False,
            "reason": "paper_ramp_armed_and_clean",
            "stage": "armed",
            "armed": True,
            "ok": True,
            "blockers": [],
        },
        {
            "paper_execution_hot": True,
            "paper_hot_low_priority": True,
            "paper_execution_pressure_dominant": True,
            "paper_execution_cpu_percent": 63.6,
            "bot_owned_cpu_percent": 362.1,
            "storage_writer_cpu_percent": 168.4,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_hot": True,
            "research_hot_low_priority": True,
            "research_training_cpu_percent": 130.1,
            "protected_work_hot": False,
        },
        throttle_profile="sustain",
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
        live_read_only=True,
        storage_ready_for_runtime_advisory=True,
        full_force_paper_required=True,
    )

    assert policy["paper_execution_allowed"] is True
    assert policy["pause_paper_execution"] is False
    assert policy["pressure_pause_active"] is False
    assert policy["pressure_pause_bypassed"] is True
    assert policy["pressure_pause_bypass_reason"] == "full_force_paper_ramp_elevated_compute_downshift_without_restart"
    assert policy["capacity_limited_paper_execution"] is True


def test_runtime_throttle_pauses_hot_coinbase_paper_feed_under_cpu_pressure(tmp_path: Path, monkeypatch) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0, "oldest_pending_age_seconds": 0.0},
        },
    )
    _write_json(
        health_root / "paper_400_ramp_latest.json",
        {"stage": "armed", "ok": True, "armed": True, "blockers": []},
    )
    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 8.0, "five_minutes": 6.8, "fifteen_minutes": 18.0},
        "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
        "vm_stat": {},
        "top_processes": [
            {
                "pid": 95776,
                "nice": 19,
                "cpu_percent": 63.0,
                "mem_percent": 0.8,
                "elapsed": "00:09:00",
                "command": "python scripts/run_shadow_training_loop.py --broker coinbase --symbols BTC-USD",
                "category": "paper_execution",
                "priority_tier": "paper_crypto_feed",
                "throttle_candidate": True,
            },
            {
                "pid": 95782,
                "nice": 19,
                "cpu_percent": 47.0,
                "mem_percent": 0.8,
                "elapsed": "00:09:00",
                "command": "python scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures",
                "category": "paper_execution",
                "priority_tier": "paper_crypto_feed",
                "throttle_candidate": True,
            },
        ],
        "category_cpu": {"paper_execution": 110.0},
        "category_counts": {"paper_execution": 2},
    }
    calls: list[list[str]] = []
    kills: list[tuple[int, int]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["paper_execution_policy"]["reason"] == "paper_execution_cpu_pressure"
    assert payload["paper_execution_policy"]["pressure_pause_active"] is True
    assert payload["paper_execution_pause_candidates"]
    assert all(row["terminate_when_apply"] is True for row in payload["paper_execution_pause_candidates"])
    assert "PAPER_CRYPTO_FEED_RUNTIME_PAUSED_FOR_PRESSURE=1" in override
    assert result["paper_execution_pause"]["successful_count"] == 2
    assert (95776, src.signal.SIGTERM) in kills
    assert (95782, src.signal.SIGTERM) in kills


def test_runtime_throttle_pauses_hot_paper_execution_under_elevated_bot_owned_pressure(tmp_path: Path, monkeypatch) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0, "oldest_pending_age_seconds": 0.0},
        },
    )
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "ok": True, "armed": True, "blockers": []})
    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 6.4, "five_minutes": 8.8, "fifteen_minutes": 9.2},
        "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
        "vm_stat": {},
        "top_processes": [
            {
                "pid": 76121,
                "nice": 20,
                "cpu_percent": 96.0,
                "mem_percent": 0.7,
                "elapsed": "00:04:00",
                "command": "python scripts/run_execution_lane.py --mode paper",
                "category": "paper_execution",
                "priority_tier": "paper_gate_controlled",
                "throttle_candidate": True,
            }
        ],
        "category_cpu": {"paper_execution": 96.0},
        "category_counts": {"paper_execution": 1},
    }
    calls: list[list[str]] = []
    kills: list[tuple[int, int]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["compute_pressure_level"] == "elevated"
    assert payload["paper_execution_policy"]["reason"] == "paper_execution_cpu_pressure"
    assert payload["paper_execution_policy"]["pressure_pause_active"] is True
    assert payload["paper_execution_pause_candidates"][0]["terminate_when_apply"] is True
    assert payload["runtime_saturation_governor_v2"]["paper_live_data_policy"]["paper_execution_consumer_paused"] is True
    assert "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED=0" in override
    assert result["paper_execution_pause"]["successful_count"] == 1
    assert (76121, src.signal.SIGTERM) in kills


def test_runtime_guard_sigstops_hot_research_until_training_gate_clears(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = {
        "throttle_profile": "sustain",
        "compute_pressure_level": "high",
        "memory_pressure_level": "normal",
        "runtime_saturation_governor_v2": {
            "training_policy": {
                "training_paused": True,
                "reason": "host_saturation_or_memory_pressure",
            }
        },
    }
    candidates = [
        {
            "pid": 4242,
            "category": "research_training",
            "cpu_percent": 77.0,
            "command": "python scripts/run_shadow_training_loop.py --broker schwab",
        }
    ]
    state_path = tmp_path / "pause_state.json"

    paused = src._apply_research_training_pause(tmp_path, candidates, payload, state_path=state_path)

    assert paused["pause_requested"] is True
    assert paused["successful_count"] == 1
    assert (4242, 0) in calls
    assert (4242, signal.SIGSTOP) in calls
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["paused_processes"][0]["pid"] == 4242

    calls.clear()
    resumed = src._apply_research_training_pause(
        tmp_path,
        [],
        {
            "throttle_profile": "normal",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "runtime_saturation_governor_v2": {"training_policy": {"training_paused": False}},
        },
        state_path=state_path,
    )

    assert resumed["pause_requested"] is False
    assert resumed["resume_successful_count"] == 1
    assert (4242, signal.SIGCONT) in calls


def test_runtime_throttle_includes_medium_hot_research_under_sustain() -> None:
    candidates = src._research_training_pressure_candidates(
        [
            {
                "pid": 5150,
                "category": "research_training",
                "cpu_percent": 13.5,
                "command": "python scripts/run_shadow_training_loop.py --broker schwab",
            }
        ],
        profile="sustain",
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
    )

    assert candidates
    assert candidates[0]["pid"] == 5150


def test_runtime_env_overrides_soft_cap_carries_support_spawn_contract() -> None:
    overrides = src._runtime_env_overrides("soft_cap", "normal", "elevated")

    assert overrides["YTDLP_SUPPORT_NICE"] == "12"
    assert overrides["MACRO_YTDLP_SUPPORT_NICE"] == "12"
    assert overrides["TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM"] == "0"
    assert overrides["SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS"] == "30"


def test_full_force_paper_keeps_cooling_controls_under_soft_cap() -> None:
    overrides = src._runtime_env_overrides(
        "soft_cap",
        "normal",
        "elevated",
        paper_capacity_contract={"full_force_stabilization_required": True, "mode": "full_force_buffered"},
        paper_execution_policy={"artifact_present": True, "paper_execution_allowed": True, "pause_paper_execution": False},
    )

    assert overrides["PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED"] == "1"
    assert overrides["PAPER_EXECUTION_RUNTIME_NICE"] == "20"
    assert overrides["EXECUTION_LANE_BATCH_LIMIT"] == "25"
    assert overrides["EXECUTION_LANE_BATCH_SLEEP_SECONDS"] == "2.0"
    assert overrides["EXECUTION_LANE_BACKLOG_SLEEP_SECONDS"] == "5.0"
    assert overrides["EXECUTION_LANE_HOST_LOAD_SOFT_CAP"] == "6.0"
    assert overrides["EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS"] == "5.0"
    assert overrides["EXECUTION_LANE_MESSAGE_SLEEP_SECONDS"] == "0.04"


def test_drain_friendly_sql_overrides_honor_smooth_load_shape_cap(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_PREPROCESS_WORKERS", "7")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "2")
    monkeypatch.setenv("SQL_LINK_WRITER_NICE", "4")

    overrides = src._drain_friendly_sql_overrides()

    assert overrides["BACKLOG_PCORE_PREPROCESS_WORKERS"] == "2"
    assert overrides["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "2"
    assert overrides["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "2"
    assert overrides["SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"] == "2"
    assert overrides["SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS"] == "20"
    assert overrides["SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM"] == "1"
    assert overrides["SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP"] == "2"
    assert overrides["SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP"] == "2"
    assert overrides["SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP"] == "1"
    assert overrides["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] == "10"
    assert overrides["SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS"] == "240"
    assert overrides["SQL_LINK_WRITER_NICE"] == "4"


def test_runtime_throttle_applies_mac_fluidity_foreground_first_overrides(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.02,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 120,
                "total_pending_lines": 140,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 3.0, "fifteen_minutes": 2.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 54.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"interactive_cotenant": 54.0},
            "category_counts": {"interactive_cotenant": 1},
        },
    )

    assert payload["throttle_profile"] == "soft_cap"
    fluidity = payload["mac_fluidity_contract"]
    assert fluidity["overall_status"] == "ready"
    assert fluidity["fluidity_band"] == "guarded_smooth"
    assert fluidity["env_overrides"]["DATA_COLLECTION_RESOURCE_SAMPLE_RATE"] == "0.30"
    assert fluidity["env_overrides"]["OPS_SUPPORT_JOB_NICE"] == "20"

    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert result["mac_fluidity_contract"]["fluidity_band"] == "guarded_smooth"
    assert "MAC_FLUIDITY_CONTRACT_ENABLED=1" in override
    assert "MAC_FOREGROUND_FIRST=1" in override
    assert "DATA_COLLECTION_RESOURCE_SAMPLE_RATE=0.30" in override
    assert "OPS_SUPPORT_JOB_NICE=20" in override
    assert "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS=60" in override


def test_mac_fluidity_contract_pauses_hot_research_when_foreground_needs_headroom() -> None:
    contract = src._mac_fluidity_contract(
        overall_status="degraded",
        throttle_profile="soft_cap",
        saturation_score=42.0,
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
        storage_pressure_index=0.02,
        storage_total_pending_lines=140,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=0.0,
        storage_oldest_age_threshold_seconds=240.0,
        host_pressure_attribution={
            "foreground_app_cpu_percent": 30.0,
            "macos_system_cpu_percent": 18.0,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_cpu_percent": 72.0,
            "storage_writer_cpu_percent": 20.0,
            "operator_observability_cpu_percent": 0.0,
        },
        cotenant_contract={"active": True, "open_app_count": 1},
        runtime_saturation_governor={"saturation_band": "guarded"},
    )

    assert contract["overall_status"] == "watch"
    assert contract["fluidity_band"] == "guarded_smooth"
    assert contract["research_pause_recommended"] is True
    assert contract["env_overrides"]["MAC_FLUIDITY_RESEARCH_PAUSE"] == "1"
    assert contract["env_overrides"]["TRAINING_RUNTIME_GOVERNOR_MODE"] == "paused_for_mac_fluidity"
    assert contract["env_overrides"]["TRAINING_RUNTIME_MAX_PARALLEL"] == "0"


def test_mac_fluidity_contract_pauses_research_on_writer_contention_even_when_score_ready() -> None:
    contract = src._mac_fluidity_contract(
        overall_status="advisory",
        throttle_profile="soft_cap",
        saturation_score=38.0,
        compute_pressure_level="normal",
        memory_pressure_level="normal",
        storage_pressure_index=0.02,
        storage_total_pending_lines=140,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=0.0,
        storage_oldest_age_threshold_seconds=240.0,
        host_pressure_attribution={
            "foreground_app_cpu_percent": 30.0,
            "macos_system_cpu_percent": 20.0,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_cpu_percent": 62.0,
            "storage_writer_cpu_percent": 80.0,
            "operator_observability_cpu_percent": 0.0,
        },
        cotenant_contract={"active": True, "open_app_count": 1},
        runtime_saturation_governor={"saturation_band": "normal"},
    )

    assert contract["overall_status"] == "ready"
    assert contract["research_pause_recommended"] is True
    assert contract["research_writer_contention"] is True
    assert contract["env_overrides"]["MAC_FLUIDITY_RESEARCH_PAUSE"] == "1"
    assert contract["env_overrides"]["MAC_FLUIDITY_RESEARCH_WRITER_CONTENTION"] == "1"
    assert contract["env_overrides"]["TRAINING_RUNTIME_MAX_PARALLEL"] == "0"
    assert contract["env_overrides"]["RUNTIME_RESEARCH_TRAINING_PAUSE_LIMIT"] == "8"


def test_mac_fluidity_contract_freezes_hot_support_maintenance_when_storage_is_clear() -> None:
    contract = src._mac_fluidity_contract(
        overall_status="advisory",
        throttle_profile="soft_cap",
        saturation_score=42.0,
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
        storage_pressure_index=0.02,
        storage_total_pending_lines=140,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=0.0,
        storage_oldest_age_threshold_seconds=240.0,
        host_pressure_attribution={
            "foreground_app_cpu_percent": 30.0,
            "macos_system_cpu_percent": 18.0,
            "throttle_candidate_support_cpu_percent": 66.0,
            "research_training_cpu_percent": 0.0,
            "storage_writer_cpu_percent": 20.0,
            "operator_observability_cpu_percent": 0.0,
        },
        cotenant_contract={"active": True, "open_app_count": 1},
        runtime_saturation_governor={"saturation_band": "advisory"},
    )

    assert contract["overall_status"] == "watch"
    assert contract["fluidity_band"] == "guarded_smooth"
    assert contract["support_pause_recommended"] is True
    assert contract["env_overrides"]["MAC_FLUIDITY_SUPPORT_PAUSE"] == "1"
    assert contract["env_overrides"]["OPS_SUPPORT_MAINTENANCE_FREEZE"] == "1"
    assert contract["env_overrides"]["SUPPORT_MAINTENANCE_CONCURRENCY"] == "0"


def test_strained_mac_fluidity_preserves_bounded_micro_canary_when_backlog_and_memory_are_clear() -> None:
    governor = src._runtime_saturation_governor_v2(
        saturation_score=46.0,
        throttle_profile="sustain",
        compute_pressure_level="high",
        memory_pressure_level="normal",
        storage_total_pending_lines=0,
        storage_oldest_pending_age_seconds=0.0,
        support_trim_candidates=[],
        research_training_trim_candidates=[],
        paper_execution_policy={"pause_paper_execution": False, "paper_execution_allowed": True},
        paper_execution_pause_candidates=[],
    )

    contract = src._mac_fluidity_contract(
        overall_status="advisory",
        throttle_profile="sustain",
        saturation_score=46.0,
        compute_pressure_level="high",
        memory_pressure_level="normal",
        storage_pressure_index=0.0,
        storage_total_pending_lines=0,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=0.0,
        storage_oldest_age_threshold_seconds=240.0,
        host_pressure_attribution={
            "foreground_app_cpu_percent": 42.0,
            "macos_system_cpu_percent": 36.0,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_cpu_percent": 0.0,
            "paper_execution_cpu_percent": 0.0,
            "protected_live_or_macro_cpu_percent": 0.0,
            "storage_writer_cpu_percent": 0.0,
            "operator_observability_cpu_percent": 12.0,
        },
        cotenant_contract={"active": True, "open_app_count": 1},
        runtime_saturation_governor=governor,
    )

    assert governor["training_policy"]["reason"] == "bounded_compute_pressure_micro_canary"
    assert governor["training_policy"]["max_parallel_trainings"] == 1
    assert contract["fluidity_band"] == "strained"
    assert contract["env_overrides"]["TRAINING_RUNTIME_GOVERNOR_MODE"] == "micro_canary_only"
    assert contract["env_overrides"]["TRAINING_RUNTIME_MAX_PARALLEL"] == "1"
    assert contract["env_overrides"]["MAC_FLUIDITY_BOUNDED_CANARY"] == "1"


def test_mac_fluidity_contract_manages_single_bounded_writer_drain() -> None:
    contract = src._mac_fluidity_contract(
        overall_status="ready",
        throttle_profile="sustain",
        saturation_score=69.85,
        compute_pressure_level="high",
        memory_pressure_level="normal",
        storage_pressure_index=0.137,
        storage_total_pending_lines=2172,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=23.484,
        storage_oldest_age_threshold_seconds=240.0,
        host_pressure_attribution={
            "foreground_app_cpu_percent": 37.5,
            "macos_system_cpu_percent": 18.9,
            "throttle_candidate_support_cpu_percent": 0.0,
            "research_training_cpu_percent": 0.0,
            "paper_execution_cpu_percent": 0.0,
            "protected_live_or_macro_cpu_percent": 0.0,
            "storage_writer_cpu_percent": 97.3,
            "storage_writer_hot": True,
            "operator_observability_cpu_percent": 28.2,
        },
        cotenant_contract={"active": True, "open_app_count": 1},
        runtime_saturation_governor={"saturation_band": "guarded"},
    )

    assert contract["overall_status"] == "watch"
    assert contract["fluidity_band"] == "guarded_smooth"
    assert contract["fluidity_score"] >= 86.0
    assert contract["measurements"]["bounded_writer_fluidity_managed"] is True
    assert contract["support_pause_recommended"] is False
    assert contract["research_pause_recommended"] is False


def test_sql_writer_fluidity_contract_bounds_governance_tail_shard() -> None:
    contract = src._sql_writer_fluidity_contract(
        throttle_profile="sustain",
        compute_pressure_level="elevated",
        memory_pressure_level="normal",
        saturation_score=61.0,
        storage_drain_active=True,
        storage_pressure_index=0.0,
        storage_total_pending_lines=62,
        storage_pending_threshold=15000,
        storage_oldest_pending_age_seconds=0.0,
        host_pressure_attribution={
            "storage_writer_cpu_percent": 93.0,
            "storage_writer_hot": True,
            "throttle_candidate_support_cpu_percent": 60.0,
            "research_training_cpu_percent": 0.0,
            "operator_observability_cpu_percent": 0.0,
        },
        mac_fluidity_contract={
            "overall_status": "watch",
            "fluidity_band": "guarded_smooth",
            "fluidity_score": 80.0,
        },
        current_sql_overrides={"SQL_LINK_SERVICE_PREPROCESS_WORKERS": "1"},
    )

    assert contract["active"] is True
    assert contract["tier"] == "guarded_relief"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS"] == "20"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM"] == "1"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP"] == "1"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP"] == "1"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP"] == "1"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] == "8"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS"] == "180"
    assert contract["env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_SQLITE_BATCH_MAX_BYTES"] == str(12 * 1024 * 1024)
    assert contract["env_overrides"]["INGEST_HOST_LOAD_SOFT_CAP"] == "6.0"
    assert contract["env_overrides"]["INGEST_HOST_LOAD_SLEEP_SECONDS"] == "0.50"
    assert contract["env_overrides"]["INGEST_FLUSH_SLEEP_SECONDS"] == "0.05"
    assert contract["env_overrides"]["INGEST_FILE_SLEEP_SECONDS"] == "0.25"


def test_runtime_throttle_control_protects_core_lanes_and_flags_support_jobs(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "yellow",
            "memory_pressure_kind": "none",
            "swap_used_gb": 9.5,
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_profile": "pro_balanced",
        },
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            }
        },
    )
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "host_contract": {
                "chip": "Apple M5 Max",
                "memory_architecture": "unified",
                "shared_cpu_gpu_memory_pool": True,
            }
        },
    )

    runtime_snapshot = {
        "cpu_count": 12,
        "load_averages": {"one_minute": 13.2, "five_minutes": 11.4, "fifteen_minutes": 9.8},
        "thermal": {
            "thermal_warning_active": False,
            "performance_warning_active": False,
            "cpu_power_warning_active": False,
        },
        "vm_stat": {"pages_throttled": 0},
        "top_processes": [
            {
                "pid": 101,
                "cpu_percent": 86.5,
                "mem_percent": 2.0,
                "elapsed": "00:42:11",
                "command": "python scripts/run_execution_lane.py --mode paper",
                "category": "live_execution",
                "priority_tier": "protected",
                "throttle_candidate": False,
            },
            {
                "pid": 202,
                "cpu_percent": 70.2,
                "mem_percent": 1.4,
                "elapsed": "01:15:03",
                "command": "python scripts/ops/sql_queue_retention.py --vacuum",
                "category": "support_maintenance",
                "priority_tier": "throttle_first",
                "throttle_candidate": True,
            },
            {
                "pid": 303,
                "cpu_percent": 26.1,
                "mem_percent": 3.8,
                "elapsed": "02:01:33",
                "command": "python scripts/run_shadow_training_loop.py --broker schwab --profile dividend",
                "category": "research_training",
                "priority_tier": "protected",
                "throttle_candidate": False,
            },
            {
                "pid": 404,
                "cpu_percent": 39.2,
                "mem_percent": 0.6,
                "elapsed": "10:10:10",
                "command": "WindowServer",
                "category": "interactive_cotenant",
                "priority_tier": "external_cotenant",
                "throttle_candidate": False,
            },
        ],
        "category_cpu": {
            "live_execution": 86.5,
            "support_maintenance": 70.2,
            "research_training": 26.1,
            "interactive_cotenant": 39.2,
        },
        "category_counts": {
            "live_execution": 1,
            "support_maintenance": 1,
            "research_training": 1,
            "interactive_cotenant": 1,
        },
    }

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)

    assert payload["overall_status"] == "degraded"
    assert payload["throttle_profile"] == "sustain"
    assert payload["host_saturation_score"] >= 56.0
    assert payload["protected_workloads"]["categories"] == ["live_execution"]
    assert payload["support_trim_candidates"][0]["category"] == "support_maintenance"
    governor = payload["runtime_saturation_governor_v2"]
    assert governor["mode"] == "runtime_saturation_governor_v2"
    assert governor["training_policy"]["training_paused"] is True
    assert governor["training_policy"]["max_parallel_trainings"] == 0
    assert governor["paper_live_data_policy"]["protect_paper_execution_queue"] is True
    assert payload["upgrade_track"]["upgradeable"] is True
    assert any("off-hours throttle windows" in action for action in payload["recommended_actions"])


def test_runtime_throttle_control_escalates_to_protect_live_when_thermal_pressure_hits(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "swap_used_gb": 24.0,
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_profile": "constrained",
        },
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            }
        },
    )

    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 15.0, "five_minutes": 13.5, "fifteen_minutes": 11.8},
        "thermal": {
            "thermal_warning_active": True,
            "performance_warning_active": True,
            "cpu_power_warning_active": False,
        },
        "vm_stat": {"pages_throttled": 0},
        "top_processes": [],
        "category_cpu": {},
        "category_counts": {},
    }

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)

    assert payload["overall_status"] == "blocked"
    assert payload["throttle_profile"] == "protect_live"
    assert payload["memory_pressure_level"] == "high"
    assert payload["release_contract"]["live_lane_should_be_read_only"] is True


def test_runtime_throttle_does_not_call_memory_high_when_memory_efficiency_is_storage_blocked(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "swap_used_gb": 1.5,
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_profile": "constrained",
            "reasons": ["storage_pressure_critical", "co_running_heavy_competition"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.5,
            },
            "cotenant_awareness": {
                "memory_pressure_clear": True,
                "storage_pressure_clear": False,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.0, "five_minutes": 4.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    assert payload["memory_pressure_level"] == "normal"


def test_runtime_throttle_counts_only_explicit_paper_live_data_capacity(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "collector_stability_only",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "paper_runtime_stability_mode": "full_force_guarded",
                },
                {
                    "bot_id": "legacy_paper_live_data",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_live_data_enabled": True,
                },
                {
                    "bot_id": "legacy_paper_execution",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_execution_allowed": True,
                },
            ]
        },
    )

    counts = src._registry_capacity_counts(tmp_path, registry_path=tmp_path / "master_bot_registry.json")

    assert counts["active_bot_count"] == 3
    assert counts["paper_tagged_count"] == 2


def test_runtime_throttle_apply_cools_sql_writer_when_backlog_is_green(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "swap_used_gb": 24.0,
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "blocked"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"release_contract": {"live_lane_should_be_read_only": True}},
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_operating_mode": "maintenance_drain_window",
            "pressure_index": 0.081,
            "severity": "stable",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {"core_pending_lines": 586, "total_pending_lines": 586},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 3.0, "five_minutes": 3.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )

    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["throttle_profile"] == "protect_live"
    assert payload["storage_stabilization"]["drain_friendly_sql_required"] is True
    assert result["storage_drain_active"] is True
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_HOST_COOLING_ACTIVE"] == "1"
    assert "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE=1" in override
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=180" in override
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=12" not in override
    assert "OPS_SUPPORT_JOB_NICE=20" in override
    assert "YTDLP_SUPPORT_NICE=20" in override
    assert "MACRO_YTDLP_SUPPORT_NICE=20" in override
    assert "SUPPORT_MAINTENANCE_CONCURRENCY=1" in override
    assert "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM=1" in override
    assert "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS=75" in override


def test_runtime_throttle_writes_idle_sql_cooling_when_backlog_is_clean() -> None:
    overrides = src._sql_overrides_for_runtime_pressure(
        "sustain",
        storage_drain_active=False,
        storage_pressure={
            "pressure_index": 0.0,
            "total_pending_lines": 0,
            "core_pending_lines": 0,
            "oldest_pending_age_seconds": 0.0,
        },
        sql_writer_coordination={},
    )

    assert overrides["SQL_LINK_SERVICE_IDLE_BACKLOG_COOLDOWN"] == "1"
    assert overrides["BACKLOG_PCORE_PREPROCESS_WORKERS"] == "1"
    assert overrides["SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"] == "1"
    assert overrides["SQL_LINK_WRITER_BACKGROUND_POLICY"] == "1"
    assert overrides["SQL_LINK_WRITER_NICE"] == "18"
    assert overrides["SQL_LINK_CHILD_WRITER_CPU_POLICY"] == "foreground_safe_idle_backlog"


def test_runtime_throttle_apply_retires_hot_clean_backlog_sql_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    kills: list[tuple[int, int]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = {
        "throttle_profile": "sustain",
        "memory_pressure_level": "elevated",
        "compute_pressure_level": "elevated",
        "host_saturation_score": 69.02,
        "runtime_snapshot": {
            "storage_pressure": {
                "pressure_index": 0.0,
                "severity": "stable",
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "oldest_pending_age_seconds": 0.0,
            }
        },
        "storage_stabilization": {
            "drain_friendly_sql_required": False,
            "total_pending_lines": 0,
            "core_pending_lines": 0,
            "sql_writer_coordination": {
                "concentrated_core_drain": False,
                "total_pending_lines": 7,
            },
        },
        "host_pressure_attribution": {
            "storage_writer_cpu_percent": 303.3,
            "storage_writer_hot": True,
        },
        "p_core_runtime_feedback": {},
        "paper_capacity_contract": {},
        "paper_execution_policy": {},
        "support_trim_candidates": [],
        "research_training_trim_candidates": [],
        "paper_execution_pause_candidates": [],
        "top_processes": [
            {
                "pid": 34211,
                "nice": 0,
                "cpu_percent": 94.7,
                "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
            {
                "pid": 34217,
                "nice": 0,
                "cpu_percent": 92.2,
                "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
            {
                "pid": 34299,
                "nice": 0,
                "cpu_percent": 31.0,
                "command": "python scripts/ops/sql_link_shard_manager.py --json",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
        ],
    }

    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_IDLE_BACKLOG_COOLDOWN"] == "1"
    assert result["sql_writer_fluidity_contract"]["reason"] == "storage_writer_heat_after_clean_backlog_is_being_retired"
    assert result["storage_writer_cooling"]["successful_count"] == 2
    assert any(cmd[:3] == ["renice", "-n", "18"] and cmd[-1] == "34211" for cmd in calls)
    assert (34211, src.signal.SIGTERM) in kills
    assert (34217, src.signal.SIGTERM) in kills
    assert (34299, src.signal.SIGTERM) not in kills
    assert "SQL_LINK_SERVICE_IDLE_BACKLOG_COOLDOWN=1" in override
    assert "SQL_LINK_WRITER_BACKGROUND_POLICY=1" in override
    assert "SQL_LINK_WRITER_NICE=18" in override


def test_runtime_throttle_cools_excess_sql_children_when_fluidity_lane_cap_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    kills: list[tuple[int, int]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", fake_kill)

    payload = {
        "throttle_profile": "sustain",
        "memory_pressure_level": "normal",
        "compute_pressure_level": "high",
        "host_saturation_score": 70.0,
        "runtime_snapshot": {
            "storage_pressure": {
                "pressure_index": 0.0,
                "severity": "stable",
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "oldest_pending_age_seconds": 0.0,
            }
        },
        "storage_stabilization": {
            "drain_friendly_sql_required": True,
            "total_pending_lines": 0,
            "core_pending_lines": 0,
            "sql_writer_coordination": {
                "concentrated_core_drain": False,
                "total_pending_lines": 1151,
            },
        },
        "host_pressure_attribution": {
            "storage_writer_cpu_percent": 140.0,
            "storage_writer_hot": True,
        },
        "sql_writer_fluidity_contract": {
            "active": True,
            "overall_status": "guarded",
            "tier": "guarded_relief",
            "reason": "storage_writer_heat_is_reducing_runtime_fluidity",
            "measurements": {
                "storage_writer_cpu_percent": 140.0,
                "memory_pressure_level": "normal",
                "current_sql_lane_cap": 7,
                "recommended_sql_lane_cap": 2,
            },
            "env_overrides": {
                "SQL_LINK_SERVICE_FLUIDITY_GOVERNOR_ACTIVE": "1",
                "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "2",
                "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "2",
            },
        },
        "p_core_runtime_feedback": {},
        "paper_capacity_contract": {},
        "paper_execution_policy": {},
        "support_trim_candidates": [],
        "research_training_trim_candidates": [],
        "paper_execution_pause_candidates": [],
        "top_processes": [
            {
                "pid": 45201,
                "nice": 4,
                "cpu_percent": 64.0,
                "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
            {
                "pid": 45202,
                "nice": 4,
                "cpu_percent": 52.0,
                "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
            {
                "pid": 45203,
                "nice": 4,
                "cpu_percent": 18.0,
                "command": "python scripts/ops/sql_link_shard_manager.py --json",
                "category": "storage_writer",
                "priority_tier": "backlog_writer",
                "throttle_candidate": False,
            },
        ],
    }

    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )

    assert result["sql_writer_fluidity_contract"]["tier"] == "guarded_relief"
    assert result["storage_writer_cooling"]["reason"] == "fluidity_lane_cap_writer_cooling"
    assert result["storage_writer_cooling"]["successful_count"] == 2
    assert (45201, src.signal.SIGTERM) in kills
    assert (45202, src.signal.SIGTERM) in kills
    assert (45203, src.signal.SIGTERM) not in kills


def test_runtime_throttle_coordinates_concentrated_core_sql_drain(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "swap_used_gb": 24.0,
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "blocked"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"release_contract": {"live_lane_should_be_read_only": True}},
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_operating_mode": "maintenance_drain_window",
            "pressure_index": 0.081,
            "severity": "stable",
            "storage": {"backlog_drain_status": "handoff_requested"},
            "backpressure": {"core_pending_lines": 33631, "total_pending_lines": 33651},
        },
    )
    _write_json(
        health_root / "backpressure_drainer_fleet_latest.json",
        {
            "active_drainer": {
                "name": "core_decision_drainer",
                "concentration": {
                    "total_pending_lines": 33623,
                    "top1_share": 0.544806,
                    "top3_share": 0.92871,
                    "concentrated": True,
                },
            },
            "service_request": {
                "env_overrides": {"SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1"}
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 3.0, "five_minutes": 3.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )

    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")
    coordination = payload["storage_stabilization"]["sql_writer_coordination"]

    assert coordination["concentrated_core_drain"] is True
    assert coordination["recommended_merge_max_seconds_per_cycle"] == 90
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "1"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
    assert "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN=1" in override
    assert "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS=420" in override
    assert "SQL_LINK_SERVICE_PREPROCESS_WORKERS=1" in override
    assert "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE=90" in override
    assert "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE=12000" in override


def test_runtime_throttle_apply_preserves_selected_writer_lane_cap_under_storage_pressure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_PREPROCESS_WORKERS", "7")
    payload = {
        "throttle_profile": "protect_live",
        "memory_pressure_level": "normal",
        "compute_pressure_level": "high",
        "runtime_snapshot": {
            "storage_pressure": {
                "pressure_index": 22.0,
                "severity": "critical",
                "total_pending_lines": 509,
                "core_pending_lines": 509,
            }
        },
        "storage_stabilization": {
            "drain_friendly_sql_required": True,
            "sql_writer_coordination": {"concentrated_core_drain": False},
        },
        "p_core_runtime_feedback": {
            "preprocess_worker_budget": 3,
            "shard_link_writer_lanes": 3,
        },
        "paper_capacity_contract": {},
        "paper_execution_policy": {},
        "support_trim_candidates": [],
        "research_training_trim_candidates": [],
        "paper_execution_pause_candidates": [],
    }

    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )

    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "3"
    assert "BACKLOG_PCORE_PREPROCESS_WORKERS=3" in override
    assert "SQL_LINK_SERVICE_PREPROCESS_WORKERS=3" in override
    assert "SQL_LINK_SERVICE_SHARD_WRITER_LANES=3" in override
    assert "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES=3" in override
    assert "BACKLOG_PCORE_PREPROCESS_WORKERS=7" not in override


def test_runtime_throttle_apply_preserves_selected_writer_lanes_when_storage_overlay_is_clear(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_PREPROCESS_WORKERS", "7")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "1")
    payload = {
        "throttle_profile": "soft_cap",
        "memory_pressure_level": "normal",
        "compute_pressure_level": "elevated",
        "runtime_snapshot": {
            "storage_pressure": {
                "pressure_index": 0.025,
                "severity": "stable",
                "total_pending_lines": 374,
                "core_pending_lines": 374,
                "oldest_pending_age_seconds": 0.0,
            }
        },
        "storage_stabilization": {
            "drain_friendly_sql_required": True,
            "sql_writer_coordination": {"concentrated_core_drain": False, "total_pending_lines": 374},
        },
        "p_core_runtime_feedback": {
            "preprocess_worker_budget": 7,
            "shard_link_writer_lanes": 7,
        },
        "paper_capacity_contract": {},
        "paper_execution_policy": {},
        "support_trim_candidates": [],
        "research_training_trim_candidates": [],
        "paper_execution_pause_candidates": [],
    }

    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )

    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE" not in result["drain_friendly_sql_overrides"]
    assert result["drain_friendly_sql_overrides"]["BACKLOG_PCORE_PREPROCESS_WORKERS"] == "7"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "7"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "7"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"] == "7"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS"] == "20"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM"] == "1"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP"] == "1"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"] == "10"
    assert result["drain_friendly_sql_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS"] == "240"
    assert "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE=1" not in override
    assert "BACKLOG_PCORE_PREPROCESS_WORKERS=7" in override
    assert "SQL_LINK_SERVICE_SHARD_WRITER_LANES=7" in override
    assert "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES=7" in override
    assert "SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS=20" in override
    assert "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM=1" in override
    assert "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP=1" in override
    assert "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES=10" in override
    assert "BACKLOG_PCORE_PREPROCESS_WORKERS=1" not in override


def test_runtime_payload_ignores_stale_lane_cap_when_pcore_operator_override_is_active(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "1")
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"release_contract": {"live_lane_should_be_read_only": True}},
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.0,
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "backlog_relief_contract": {
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "policy": "p_core_preprocess_single_sql_writer",
                    "preprocess_worker_budget": 7,
                    "shard_link_writer_lanes": 7,
                    "primary_merge_writer_count": 1,
                    "writer_lane_policy": "parallel_child_shard_writers_on_p_core_budget_single_serial_primary_merge",
                    "p_core_burst_intelligence": {"mode": "operator_override"},
                }
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 3.0, "five_minutes": 3.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    feedback = payload["p_core_runtime_feedback"]
    assert feedback["preprocess_worker_budget"] == 7
    assert feedback["shard_link_writer_lanes"] == 7
    assert feedback["configured_max_shard_writer_lanes"] == 1
    assert feedback["configured_smooth_cap_applied"] is False
    assert feedback["configured_smooth_cap_ignored_for_operator_override"] is True


def test_runtime_throttle_marks_bounded_sql_overlay_support_pressure_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 2.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"release_contract": {"live_lane_should_be_read_only": True}},
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "severity": "critical",
            "pressure_index": 24.535,
            "recommended_operating_mode": "maintenance_only",
            "storage": {"backlog_drain_status": "blocked"},
            "backpressure": {
                "overlay_adjusted": True,
                "overlay_pressure_clear": False,
                "core_pending_lines": 2591,
                "total_pending_lines": 2731,
                "oldest_pending_age_seconds": 5888.295,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.484,
                },
            },
        },
    )
    _write_json(
        health_root / "paper_400_ramp_latest.json",
        {"stage": "armed", "ok": True, "armed": True, "blockers": []},
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.39, "five_minutes": 5.86, "fifteen_minutes": 5.74},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 2201,
                    "nice": 14,
                    "cpu_percent": 70.4,
                    "mem_percent": 0.2,
                    "elapsed": "00:30",
                    "command": "yt-dlp --flat-playlist --dump-single-json https://www.youtube.com/@federalreserve/streams",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 2202,
                    "nice": 0,
                    "cpu_percent": 111.6,
                    "mem_percent": 1.8,
                    "elapsed": "08:00",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"support_maintenance": 70.4, "interactive_cotenant": 111.6},
            "category_counts": {"support_maintenance": 1, "interactive_cotenant": 1},
        },
    )

    advisory = payload["soft_cap_advisory_reclassification"]
    assert payload["overall_status"] == "ready"
    assert advisory["active"] is True
    assert advisory["reason"] == "niced_support_pressure_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["overlay_runtime_relief_active"] is True
    assert advisory["measurements"]["support_low_priority_guarded_ready"] is True
    assert payload["runtime_snapshot"]["storage_pressure"]["overlay_relief_contract"]["active"] is True
    assert payload["mac_fluidity_contract"]["storage_clear_for_fluidity"] is True
    assert payload["mac_fluidity_contract"]["measurements"]["overlay_fluidity_managed"] is True


def test_heavy_livefeed_is_operator_observability_not_support_trim() -> None:
    row = src._classify_process(
        "/bin/zsh /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_feed_tail.sh --source all --heavy"
    )

    assert row["category"] == "operator_observability"
    assert row["priority_tier"] == "operator_visible"
    assert row["throttle_candidate"] is False

    tail_row = src._classify_process("tail -n 120 -F logs/a.log logs/b.log")

    assert tail_row["category"] == "operator_observability"
    assert tail_row["priority_tier"] == "operator_visible"
    assert tail_row["throttle_candidate"] is False

    tail_c_row = src._classify_process("tail -c 262144 -F logs/a.log logs/b.log")

    assert tail_c_row["category"] == "operator_observability"
    assert tail_c_row["priority_tier"] == "operator_visible"
    assert tail_c_row["throttle_candidate"] is False

    awk_row = src._classify_process("awk -v max 1100 -v limit 0 -v color 1")

    assert awk_row["category"] == "operator_observability"
    assert awk_row["priority_tier"] == "operator_visible"
    assert awk_row["throttle_candidate"] is False


def test_runtime_throttle_classifies_swap_governor_as_support() -> None:
    row = src._classify_process(
        "/opt/homebrew/bin/python scripts/ops/swap_pressure_governor.py --apply --json"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True


def test_runtime_throttle_classifies_compactors_and_retention_as_support() -> None:
    for command in (
        "/opt/homebrew/bin/python scripts/ops/governance_telemetry_compactor.py --apply --json",
        "/opt/homebrew/bin/python scripts/ops/governance_lifecycle_compactor.py --apply --json",
        "/opt/homebrew/bin/python scripts/ops/decision_log_compactor.py --apply --json",
        "/opt/homebrew/bin/python scripts/ops/raw_training_compaction_intelligence.py --apply --json",
        "/opt/homebrew/bin/python scripts/ops/retention_intelligence_v2.py --json",
        "/opt/homebrew/bin/python scripts/collect_dividend_drip_state.py --json",
    ):
        row = src._classify_process(command)
        assert row["category"] == "support_maintenance"
        assert row["priority_tier"] == "throttle_first"
        assert row["throttle_candidate"] is True


def test_runtime_throttle_classifies_sql_linkers_as_storage_writer() -> None:
    row = src._classify_process(
        "/opt/homebrew/bin/python scripts/link_jsonl_to_sql.py --mode sqlite"
    )
    manager = src._classify_process(
        "/opt/homebrew/bin/python scripts/ops/sql_link_shard_manager.py --once --json"
    )

    assert row["category"] == "storage_writer"
    assert row["priority_tier"] == "backlog_writer"
    assert row["throttle_candidate"] is False
    assert manager["category"] == "storage_writer"
    assert manager["priority_tier"] == "backlog_writer"
    assert manager["throttle_candidate"] is False


def test_runtime_throttle_surfaces_p_core_backlog_feedback(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {})
    _write_json(health_root / "memory_efficiency_control_latest.json", {})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "pressure_index": 1.2,
            "severity": "critical",
            "backpressure": {"core_pending_lines": 42000, "total_pending_lines": 53000},
            "backlog_relief_contract": {
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "policy": "p_core_preprocess_single_sql_writer",
                    "preprocess_worker_budget": 4,
                    "training_pcore_gate": {"allowed_when_backlog_green": True, "max_workers": 2},
                }
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 2.0, "five_minutes": 2.0, "fifteen_minutes": 2.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {"pid": 1, "nice": 8, "cpu_percent": 5.0, "category": "support_maintenance", "priority_tier": "throttle_first"},
                {"pid": 2, "nice": 0, "cpu_percent": 4.0, "category": "operator_observability", "priority_tier": "operator_visible"},
            ],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    feedback = payload["p_core_runtime_feedback"]
    assert feedback["active"] is True
    assert feedback["preprocess_worker_budget"] == 4
    assert feedback["single_writer_only"] is True
    assert feedback["top_process_nice_distribution"] == {"8": 1, "0": 1}


def test_runtime_throttle_p_core_feedback_honors_smooth_lane_cap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "2")
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {})
    _write_json(health_root / "memory_efficiency_control_latest.json", {})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "pressure_index": 1.2,
            "severity": "critical",
            "backpressure": {"core_pending_lines": 42000, "total_pending_lines": 53000},
            "backlog_relief_contract": {
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "policy": "p_core_preprocess_single_sql_writer",
                    "preprocess_worker_budget": 4,
                    "shard_link_writer_lanes": 4,
                }
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 2.0, "five_minutes": 2.0, "fifteen_minutes": 2.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    feedback = payload["p_core_runtime_feedback"]
    assert feedback["preprocess_worker_budget"] == 2
    assert feedback["shard_link_writer_lanes"] == 2
    assert feedback["configured_max_shard_writer_lanes"] == 2
    assert feedback["configured_smooth_cap_applied"] is True


def test_project_path_does_not_misclassify_maintenance_as_pycharm() -> None:
    row = src._classify_process(
        "/opt/homebrew/Cellar/python@3.12/3.12.12_2/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python "
        "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/sql_hot_retention.py --vacuum"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True


def test_pycharm_app_is_still_interactive_cotenant() -> None:
    row = src._classify_process("/Applications/PyCharm.app/Contents/MacOS/pycharm")

    assert row["category"] == "interactive_cotenant"
    assert row["priority_tier"] == "external_cotenant"
    assert row["throttle_candidate"] is False


def test_operator_observers_are_not_unknown_pressure() -> None:
    for command in (
        "/opt/homebrew/bin/btop",
        "/Library/Frameworks/Python.framework/Versions/3.14/bin/asitop --interval 3",
        "/System/Applications/Utilities/Activity Monitor.app/Contents/MacOS/Activity Monitor",
        "/opt/homebrew/bin/python -m pytest tests/test_runtime_throttle_control.py -q",
    ):
        row = src._classify_process(command)
        assert row["category"] == "operator_observability"
        assert row["priority_tier"] == "operator_visible"
        assert row["throttle_candidate"] is False


def test_support_throttle_uses_support_nice_not_research_nice(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_THROTTLE_RESEARCH_NICE", "6")

    target = src._target_nice_for_candidate(
        {"category": "support_maintenance", "throttle_candidate": True},
        {"OPS_SUPPORT_JOB_NICE": "16"},
    )

    assert target == 16


def test_paper_shadow_downshift_uses_paper_nice_not_research_nice(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_THROTTLE_RESEARCH_NICE", "2")

    target = src._target_nice_for_candidate(
        {"category": "paper_execution", "priority_tier": "paper_shadow_downshift", "throttle_candidate": True},
        {},
    )

    assert target == 12


def test_runtime_throttle_attributes_macos_system_pressure(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 1.2})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})

    classified = src._classify_process("/usr/libexec/spotlightknowledged.updater -u")
    assert classified["category"] == "system_cotenant"
    assert classified["priority_tier"] == "external_system"
    assert classified["throttle_candidate"] is False

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 9.0, "five_minutes": 6.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 707,
                    "nice": 0,
                    "cpu_percent": 91.0,
                    "mem_percent": 0.4,
                    "elapsed": "02:03",
                    "command": "/usr/libexec/spotlightknowledged.updater -u",
                    **classified,
                },
                {
                    "pid": 808,
                    "nice": 6,
                    "cpu_percent": 18.0,
                    "mem_percent": 0.8,
                    "elapsed": "00:30",
                    "command": "python scripts/link_jsonl_to_sql.py --mode sqlite",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {"system_cotenant": 91.0, "support_maintenance": 18.0},
            "category_counts": {"system_cotenant": 1, "support_maintenance": 1},
        },
    )

    attribution = payload["host_pressure_attribution"]
    assert payload["throttle_domains"]["system_cotenant"]["cpu_percent"] == 91.0
    assert attribution["external_pressure_dominant"] is True
    assert attribution["system_cotenant_hot"] is True
    assert attribution["dominant_bucket"] == "macos_system"
    assert attribution["hot_external_processes"][0]["pid"] == 707
    assert any("Spotlight" in action for action in payload["recommended_actions"])


def test_runtime_throttle_attributes_pmset_log_as_macos_system_pressure(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 1.2})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})

    classified = src._classify_process("/usr/bin/pmset -g log")
    assert classified["category"] == "system_cotenant"
    assert classified["priority_tier"] == "external_system"
    assert classified["throttle_candidate"] is False

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 5.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {"pid": 909, "nice": 0, "cpu_percent": 55.0, "mem_percent": 0.1, "elapsed": "00:04", "command": "/usr/bin/pmset -g log", **classified}
            ],
            "category_cpu": {"system_cotenant": 55.0},
            "category_counts": {"system_cotenant": 1},
        },
    )

    attribution = payload["host_pressure_attribution"]
    assert attribution["system_cotenant_hot"] is True
    assert attribution["dominant_bucket"] == "macos_system"
    assert attribution["unknown_cpu_percent"] == 0.0


def test_runtime_throttle_marks_system_secondary_when_bot_owned_support_dominates(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 1.2})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 5.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 1001,
                    "nice": 20,
                    "cpu_percent": 84.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python scripts/link_jsonl_to_sql.py --mode sqlite",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 1002,
                    "nice": 20,
                    "cpu_percent": 82.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python scripts/link_jsonl_to_sql.py --mode sqlite",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 1003,
                    "nice": 5,
                    "cpu_percent": 80.0,
                    "mem_percent": 0.5,
                    "elapsed": "00:20",
                    "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
                    "category": "research_training",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
                {
                    "pid": 1004,
                    "nice": 0,
                    "cpu_percent": 100.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/usr/bin/pmset -g log",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "support_maintenance": 166.0,
                "research_training": 80.0,
                "system_cotenant": 90.0,
            },
            "category_counts": {"support_maintenance": 2, "research_training": 1, "system_cotenant": 1},
        },
    )

    attribution = payload["host_pressure_attribution"]
    assert attribution["dominant_bucket"] == "bot_owned"
    assert attribution["bot_owned_pressure_dominant"] is True
    assert attribution["system_secondary_to_bot_owned"] is True
    assert attribution["support_trim_required"] is True
    assert "trim support maintenance" in " ".join(payload["recommended_actions"])


def test_runtime_throttle_reclassifies_bounded_foreground_system_mix_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "pressure_index": 0.01, "backpressure": {"core_pending_lines": 20, "total_pending_lines": 25}},
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 5.5, "five_minutes": 4.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 1101,
                    "nice": 0,
                    "cpu_percent": 90.0,
                    "mem_percent": 0.5,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_user",
                    "throttle_candidate": False,
                },
                {
                    "pid": 1102,
                    "nice": 0,
                    "cpu_percent": 44.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/System/Library/PrivateFrameworks/CoreSuggestions.framework/Versions/A/Support/suggestd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"interactive_cotenant": 90.0, "system_cotenant": 44.0},
            "category_counts": {"interactive_cotenant": 1, "system_cotenant": 1},
        },
    )

    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "foreground_and_macos_system_mix_is_guarded_advisory"
    assert advisory["measurements"]["foreground_system_guarded"] is True


def test_runtime_throttle_reclassifies_guarded_external_cotenant_mix_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.429,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 6442,
                "total_pending_lines": 6442,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 24.578,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 10.6, "five_minutes": 8.9, "fifteen_minutes": 8.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 2101,
                    "nice": 0,
                    "cpu_percent": 123.5,
                    "mem_percent": 1.0,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2102,
                    "nice": 0,
                    "cpu_percent": 94.3,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/System/Library/PrivateFrameworks/CoreSuggestions.framework/Versions/A/Support/suggestd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2103,
                    "nice": 0,
                    "cpu_percent": 46.3,
                    "mem_percent": 0.2,
                    "elapsed": "00:20",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2104,
                    "nice": 0,
                    "cpu_percent": 13.7,
                    "mem_percent": 0.2,
                    "elapsed": "00:20",
                    "command": "python scripts/run_all_sleeves.py",
                    "category": "live_execution",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "interactive_cotenant": 123.5,
                "system_cotenant": 94.3,
                "operator_observability": 46.3,
                "live_execution": 13.7,
            },
            "category_counts": {
                "interactive_cotenant": 1,
                "system_cotenant": 1,
                "operator_observability": 1,
                "live_execution": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["runtime_saturation_governor_v2"]["saturation_band"] == "guarded"
    assert payload["runtime_saturation_governor_v2"]["training_policy"]["training_paused"] is True
    assert payload["host_pressure_attribution"]["external_pressure_dominant"] is True
    assert payload["host_pressure_attribution"]["bot_owned_pressure_dominant"] is False
    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "external_cotenant_pressure_is_guarded_advisory_not_bot_runtime_degradation"
    assert advisory["measurements"]["external_cotenant_guarded"] is True


def test_runtime_throttle_reclassifies_external_high_compute_as_capacity_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.4,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 5000,
                "total_pending_lines": 5000,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 120.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.8, "five_minutes": 9.5, "fifteen_minutes": 8.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 2201,
                    "nice": 0,
                    "cpu_percent": 90.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2202,
                    "nice": 0,
                    "cpu_percent": 95.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/System/Library/PrivateFrameworks/CoreSuggestions.framework/Versions/A/Support/suggestd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2203,
                    "nice": 0,
                    "cpu_percent": 70.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python other_tool.py",
                    "category": "unclassified",
                    "priority_tier": "observe",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2204,
                    "nice": 20,
                    "cpu_percent": 10.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python scripts/ops/creative_cotenant_guard.py apply --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {
                "interactive_cotenant": 100.0,
                "system_cotenant": 95.0,
                "unclassified": 70.0,
                "support_maintenance": 10.0,
            },
            "category_counts": {
                "interactive_cotenant": 1,
                "system_cotenant": 1,
                "unclassified": 1,
                "support_maintenance": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "high"
    assert payload["runtime_saturation_governor_v2"]["saturation_band"] == "guarded"
    assert payload["runtime_saturation_governor_v2"]["training_policy"]["training_paused"] is True
    assert payload["host_pressure_attribution"]["external_pressure_dominant"] is True
    assert payload["host_pressure_attribution"]["bot_owned_pressure_dominant"] is False
    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation"
    assert advisory["measurements"]["external_high_compute_guarded"] is True


def test_runtime_throttle_marks_single_green_storage_writer_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.8, "five_minutes": 7.0, "fifteen_minutes": 2.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 95.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"storage_writer": 95.0},
            "category_counts": {"storage_writer": 1},
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "high"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is True
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert payload["paper_capacity_contract"]["compute_pressure_limited"] is False


def test_runtime_throttle_marks_soft_cap_storage_writer_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.8, "five_minutes": 8.1, "fifteen_minutes": 9.2},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 95.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 302,
                    "nice": 0,
                    "cpu_percent": 25.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"storage_writer": 95.0, "operator_observability": 25.0},
            "category_counts": {"storage_writer": 1, "operator_observability": 1},
        },
    )

    assert payload["throttle_profile"] == "soft_cap"
    assert payload["compute_pressure_level"] == "elevated"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is True


def test_runtime_throttle_keeps_single_bounded_writer_ready_above_legacy_ceiling(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.0, "five_minutes": 8.0, "fifteen_minutes": 8.8},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 95.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"storage_writer": 95.0},
            "category_counts": {"storage_writer": 1},
        },
    )

    assert payload["host_saturation_score"] > 62.0
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is True


def test_runtime_throttle_marks_normal_compute_storage_writer_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 3.0, "fifteen_minutes": 2.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 92.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 302,
                    "nice": 0,
                    "cpu_percent": 28.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 45.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "/usr/libexec/syspolicyd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"storage_writer": 92.0, "operator_observability": 28.0, "system_cotenant": 45.0},
            "category_counts": {"storage_writer": 1, "operator_observability": 1, "system_cotenant": 1},
        },
    )

    assert payload["compute_pressure_level"] == "normal"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is True


def test_runtime_throttle_marks_clear_storage_writer_burst_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.009,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.05, "five_minutes": 4.2, "fifteen_minutes": 4.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 117.5,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 302,
                    "nice": 0,
                    "cpu_percent": 24.4,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 147.7,
                    "mem_percent": 1.0,
                    "elapsed": "00:10",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"storage_writer": 117.5, "operator_observability": 24.4, "interactive_cotenant": 147.7},
            "category_counts": {"storage_writer": 1, "operator_observability": 1, "interactive_cotenant": 1},
        },
    )

    assert payload["compute_pressure_level"] == "normal"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "bounded_storage_writer_burst_after_clear_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["storage_writer_burst_complete_guarded_ready"] is True
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is False


def test_runtime_throttle_marks_pending_support_pressure_live_read_only_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 9.8, "five_minutes": 6.2, "fifteen_minutes": 6.7},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 501,
                    "nice": 0,
                    "cpu_percent": 49.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/ops/storage_failback_sync.py --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                }
            ],
            "category_cpu": {"support_maintenance": 49.0},
            "category_counts": {"support_maintenance": 1},
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "support_throttle_pending_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["support_throttle_pending_guarded"] is True
    assert advisory["measurements"]["support_throttle_pending_guarded_ready"] is True


def test_runtime_throttle_marks_niced_support_pressure_live_read_only_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 9.8, "five_minutes": 6.2, "fifteen_minutes": 6.7},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 501,
                    "nice": 20,
                    "cpu_percent": 145.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/ops/creative_cotenant_guard.py apply --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                }
            ],
            "category_cpu": {"support_maintenance": 145.0},
            "category_counts": {"support_maintenance": 1},
        },
    )

    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "niced_support_pressure_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["support_low_priority_guarded_ready"] is True


def test_runtime_throttle_marks_bounded_writer_plus_support_pressure_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.8, "five_minutes": 6.2, "fifteen_minutes": 8.1},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 78.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 501,
                    "nice": 0,
                    "cpu_percent": 50.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/ops/storage_failback_sync.py --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {"storage_writer": 78.0, "support_maintenance": 50.0},
            "category_counts": {"storage_writer": 1, "support_maintenance": 1},
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "bounded_writer_and_support_throttle_pending_is_guarded_runtime_ready"
    assert advisory["measurements"]["bounded_writer_with_support_guarded_ready"] is True
    assert advisory["measurements"]["support_throttle_pending_guarded_ready"] is True


def test_runtime_throttle_marks_bounded_writer_plus_low_priority_paper_shadow_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.8, "five_minutes": 6.2, "fifteen_minutes": 8.1},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 0,
                    "cpu_percent": 92.0,
                    "mem_percent": 0.3,
                    "elapsed": "01:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 601,
                    "nice": 12,
                    "cpu_percent": 78.0,
                    "mem_percent": 0.4,
                    "elapsed": "00:04",
                    "command": "python scripts/run_parallel_shadows.py --broker schwab",
                    "category": "paper_execution",
                    "priority_tier": "paper_shadow_downshift",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {"storage_writer": 92.0, "paper_execution": 78.0},
            "category_counts": {"storage_writer": 1, "paper_execution": 1},
        },
    )

    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "bounded_writer_and_low_priority_paper_shadow_is_guarded_runtime_ready"
    assert advisory["measurements"]["bounded_writer_with_paper_shadow_guarded_ready"] is True
    assert advisory["measurements"]["paper_hot_low_priority"] is True


def test_runtime_throttle_treats_armed_paper_ramp_writer_heat_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.385,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 1422,
                "total_pending_lines": 2568,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 92.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.1, "five_minutes": 5.2, "fifteen_minutes": 5.4},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 4,
                    "cpu_percent": 170.0,
                    "mem_percent": 0.3,
                    "elapsed": "02:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 601,
                    "nice": 12,
                    "cpu_percent": 96.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 701,
                    "nice": 0,
                    "cpu_percent": 27.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "storage_writer": 170.0,
                "paper_execution": 96.0,
                "operator_observability": 27.0,
            },
            "category_counts": {
                "storage_writer": 1,
                "paper_execution": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["release_contract"]["effective_live_read_only_reason"] == "paper_trade_lock"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "full_force_paper_ramp_writer_pressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["full_force_paper_ramp_guarded_ready"] is True
    assert advisory["measurements"]["paper_execution_allowed"] is True
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True


def test_runtime_throttle_keeps_high_compute_full_paper_ramp_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.02,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 46,
                "total_pending_lines": 169,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.8, "five_minutes": 9.5, "fifteen_minutes": 8.2},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 20,
                    "cpu_percent": 88.0,
                    "mem_percent": 0.3,
                    "elapsed": "02:00",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": True,
                },
                {
                    "pid": 601,
                    "nice": 20,
                    "cpu_percent": 74.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 701,
                    "nice": 0,
                    "cpu_percent": 42.0,
                    "mem_percent": 0.3,
                    "elapsed": "00:02",
                    "command": "python scripts/resource_guard.py --profile collect",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 801,
                    "nice": 19,
                    "cpu_percent": 46.0,
                    "mem_percent": 0.3,
                    "elapsed": "00:03",
                    "command": "python scripts/run_shadow_training_loop.py --broker schwab",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "storage_writer": 88.0,
                "paper_execution": 74.0,
                "support_maintenance": 42.0,
                "research_training": 46.0,
            },
            "category_counts": {
                "storage_writer": 1,
                "paper_execution": 1,
                "support_maintenance": 1,
                "research_training": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "high"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "full_force_paper_ramp_writer_pressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["full_force_paper_ramp_guarded_ready"] is True
    assert advisory["measurements"]["paper_hot_low_priority"] is True
    assert advisory["measurements"]["support_jobs_hot"] is True
    assert advisory["measurements"]["research_training_hot"] is True
    assert advisory["measurements"]["research_hot_low_priority"] is True
    assert payload["paper_execution_policy"]["paper_execution_allowed"] is True
    assert payload["paper_execution_policy"]["pause_paper_execution"] is False
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert payload["paper_capacity_contract"]["compute_pressure_limited"] is False


def test_runtime_throttle_keeps_high_compute_paper_only_full_ramp_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.0,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.8, "five_minutes": 9.5, "fifteen_minutes": 8.2},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 601,
                    "nice": 20,
                    "cpu_percent": 105.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {"paper_execution": 105.0},
            "category_counts": {"paper_execution": 1},
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "high"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "full_force_paper_ramp_pressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["full_force_paper_ramp_guarded_ready"] is True
    assert advisory["measurements"]["storage_writer_hot"] is False
    assert payload["paper_execution_policy"]["paper_execution_allowed"] is True
    assert payload["paper_execution_policy"]["pause_paper_execution"] is False
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert payload["paper_capacity_contract"]["compute_pressure_limited"] is False


def test_runtime_throttle_marks_low_priority_paper_heat_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.081,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 1221,
                "total_pending_lines": 1647,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.6,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.0, "five_minutes": 4.2, "fifteen_minutes": 4.1},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 601,
                    "nice": 12,
                    "cpu_percent": 74.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 701,
                    "nice": 0,
                    "cpu_percent": 48.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:10",
                    "command": "/usr/libexec/syspolicyd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 801,
                    "nice": 0,
                    "cpu_percent": 22.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/live_feed_tail.sh --source all",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "paper_execution": 74.0,
                "system_cotenant": 48.0,
                "operator_observability": 22.0,
            },
            "category_counts": {
                "paper_execution": 1,
                "system_cotenant": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "low_priority_paper_execution_pressure_is_guarded_advisory"
    assert advisory["measurements"]["paper_lane_low_priority_guarded"] is True
    assert advisory["measurements"]["paper_execution_allowed"] is True


def test_runtime_throttle_marks_paper_heat_with_green_os_memory_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "needs_work",
            "reasons": ["compressed_memory_high"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "normal",
                "memory_free_pct": 91.0,
                "swap_used_gb": 7.4,
                "compressor_gb": 0.5,
            },
        },
    )
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.0,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.0, "five_minutes": 4.2, "fifteen_minutes": 4.1},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 601,
                    "nice": 12,
                    "cpu_percent": 78.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 801,
                    "nice": 0,
                    "cpu_percent": 17.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/live_feed_tail.sh --source all",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "paper_execution": 78.0,
                "operator_observability": 17.0,
            },
            "category_counts": {
                "paper_execution": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "low_priority_paper_execution_pressure_is_guarded_advisory"
    assert advisory["measurements"]["paper_ramp_memory_guarded"] is True
    assert advisory["measurements"]["paper_lane_low_priority_guarded"] is True


def test_runtime_throttle_marks_clean_backlog_writer_cooldown_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "needs_work",
            "reasons": ["compressed_memory_high"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "normal",
                "memory_free_pct": 91.0,
                "swap_used_gb": 7.4,
                "compressor_gb": 0.5,
            },
        },
    )
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.0,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 5.0, "five_minutes": 5.0, "fifteen_minutes": 4.5},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 501,
                    "nice": 0,
                    "cpu_percent": 89.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:30",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 701,
                    "nice": 0,
                    "cpu_percent": 31.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/live_feed_tail.sh --source all",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "storage_writer": 89.0,
                "operator_observability": 31.0,
            },
            "category_counts": {
                "storage_writer": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "bounded_storage_writer_after_green_backpressure_is_guarded_advisory"
    assert advisory["measurements"]["storage_writer_cooling_guarded_advisory"] is True
    assert advisory["measurements"]["plain_storage_clear_guarded_ready"] is True


def test_runtime_throttle_marks_bounded_writer_paper_research_mix_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.358,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 2298,
                "total_pending_lines": 2336,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 86.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.8, "five_minutes": 6.2, "fifteen_minutes": 5.9},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 301,
                    "nice": 4,
                    "cpu_percent": 99.7,
                    "mem_percent": 0.3,
                    "elapsed": "03:20",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 401,
                    "nice": 12,
                    "cpu_percent": 33.6,
                    "mem_percent": 1.1,
                    "elapsed": "05:55",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 402,
                    "nice": 19,
                    "cpu_percent": 28.1,
                    "mem_percent": 1.4,
                    "elapsed": "09:02",
                    "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
                {
                    "pid": 403,
                    "nice": 20,
                    "cpu_percent": 16.1,
                    "mem_percent": 0.1,
                    "elapsed": "00:01",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "storage_writer": 99.7,
                "paper_execution": 33.6,
                "research_training": 28.1,
                "operator_observability": 16.1,
            },
            "category_counts": {
                "storage_writer": 1,
                "paper_execution": 1,
                "research_training": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "soft_cap"
    assert payload["compute_pressure_level"] == "elevated"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "bounded_bot_owned_writer_paper_research_is_guarded_runtime_ready"
    assert advisory["measurements"]["bounded_bot_owned_runtime_guarded_ready"] is True
    assert advisory["measurements"]["storage_writer_cooling_guarded_ready"] is False
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True


def test_runtime_throttle_marks_bounded_read_only_protected_lane_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "stable",
            "pressure_index": 0.025,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.2, "five_minutes": 6.0, "fifteen_minutes": 7.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 401,
                    "nice": 5,
                    "cpu_percent": 62.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:02",
                    "command": "python scripts/run_all_sleeves.py --with-aggressive-modes",
                    "category": "live_execution",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
                {
                    "pid": 402,
                    "nice": 0,
                    "cpu_percent": 82.0,
                    "mem_percent": 2.0,
                    "elapsed": "12:00",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 403,
                    "nice": 0,
                    "cpu_percent": 9.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:02",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "live_execution": 62.0,
                "interactive_cotenant": 82.0,
                "operator_observability": 9.0,
            },
            "category_counts": {
                "live_execution": 1,
                "interactive_cotenant": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "soft_cap"
    assert payload["compute_pressure_level"] == "elevated"
    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "bounded_read_only_protected_lane_after_green_backpressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["bounded_protected_lane_guarded_ready"] is True
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True


def test_runtime_throttle_reclassifies_external_high_compute_with_bounded_storage_overlay(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "elevated",
            "pressure_index": 0.774,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 6090,
                "total_pending_lines": 6090,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 185.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{i}", "active": True, "lifecycle_state": "active"}
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.5, "five_minutes": 10.0, "fifteen_minutes": 8.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 2201,
                    "nice": 0,
                    "cpu_percent": 48.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2202,
                    "nice": 0,
                    "cpu_percent": 25.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/usr/libexec/syspolicyd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 2203,
                    "nice": 20,
                    "cpu_percent": 3.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "interactive_cotenant": 48.0,
                "system_cotenant": 25.0,
                "research_training": 3.0,
            },
            "category_counts": {
                "interactive_cotenant": 1,
                "system_cotenant": 1,
                "research_training": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "high"
    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory"
    assert advisory["measurements"]["bounded_storage_overlay_guarded"] is True
    assert advisory["measurements"]["storage_ready_for_runtime_advisory"] is True
    assert payload["paper_capacity_contract"]["attribution_capacity_advisory"] is True
    assert payload["paper_capacity_contract"]["compute_pressure_limited"] is False
    assert payload["paper_capacity_contract"]["pressure_limited"] is False


def test_runtime_throttle_reclassifies_external_cotenant_with_bounded_storage_overlay(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "elevated",
            "pressure_index": 0.774,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 6090,
                "total_pending_lines": 6090,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 185.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 9.0, "five_minutes": 8.0, "fifteen_minutes": 6.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 3201,
                    "nice": 0,
                    "cpu_percent": 55.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 3202,
                    "nice": 0,
                    "cpu_percent": 25.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "python scripts/run_all_sleeves.py --broker schwab",
                    "category": "live_execution",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "interactive_cotenant": 55.0,
                "live_execution": 25.0,
            },
            "category_counts": {
                "interactive_cotenant": 1,
                "live_execution": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["compute_pressure_level"] == "elevated"
    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "external_cotenant_with_bounded_storage_overlay_is_advisory_not_bot_runtime_degradation"
    assert advisory["measurements"]["external_cotenant_guarded"] is True
    assert advisory["measurements"]["bounded_storage_overlay_guarded"] is True


def test_runtime_throttle_promotes_low_bot_owned_bounded_overlay_to_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "severity": "elevated",
            "pressure_index": 0.774,
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 6090,
                "total_pending_lines": 6090,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 185.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 9.0, "five_minutes": 8.0, "fifteen_minutes": 6.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 4301,
                    "nice": 0,
                    "cpu_percent": 88.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:20",
                    "command": "/System/Library/PrivateFrameworks/CoreSuggestions.framework/Versions/A/Support/suggestd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 4302,
                    "nice": 0,
                    "cpu_percent": 43.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:20",
                    "command": "/Applications/Codex.app/Contents/MacOS/Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 4303,
                    "nice": 0,
                    "cpu_percent": 5.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:20",
                    "command": "python scripts/run_all_sleeves.py --broker schwab",
                    "category": "live_execution",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "system_cotenant": 88.0,
                "interactive_cotenant": 43.0,
                "live_execution": 5.0,
                "operator_observability": 4.0,
            },
            "category_counts": {
                "system_cotenant": 1,
                "interactive_cotenant": 1,
                "live_execution": 1,
                "operator_observability": 1,
            },
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["to_status"] == "ready"
    assert advisory["reason"] == "external_cotenant_with_bounded_storage_overlay_is_guarded_runtime_ready"
    assert advisory["measurements"]["runtime_ready_guarded"] is True


def test_runtime_throttle_reclassifies_niced_support_secondary_system_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "pressure_index": 0.01, "backpressure": {"core_pending_lines": 20, "total_pending_lines": 25}},
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 7.5, "five_minutes": 5.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 1201,
                    "nice": 20,
                    "cpu_percent": 96.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:30",
                    "command": "python scripts/build_runtime_training_snapshot.py --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 1202,
                    "nice": 0,
                    "cpu_percent": 42.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:30",
                    "command": "/System/Library/PrivateFrameworks/CoreSuggestions.framework/Versions/A/Support/suggestd",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"support_maintenance": 96.0, "system_cotenant": 42.0},
            "category_counts": {"support_maintenance": 1, "system_cotenant": 1},
        },
    )

    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "niced_support_maintenance_with_secondary_system_pressure_is_guarded_advisory"
    assert advisory["measurements"]["support_low_priority_guarded"] is True


def test_storage_control_plane_helpers_are_support_throttle_candidates() -> None:
    snapshot_row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/build_runtime_training_snapshot.py --json"
    )
    assert snapshot_row["category"] == "support_maintenance"
    assert snapshot_row["priority_tier"] == "throttle_first"
    assert snapshot_row["throttle_candidate"] is True

    row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/storage_resilience_control.py --fast --json"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True

    coverage_row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/snapshot_coverage_sentinel.py --json"
    )

    assert coverage_row["category"] == "support_maintenance"
    assert coverage_row["priority_tier"] == "throttle_first"
    assert coverage_row["throttle_candidate"] is True


def test_guard_ops_helpers_are_support_throttle_candidates() -> None:
    for command in (
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/premarket_token_guard.py --json",
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/guard_intelligence_layer.py --apply --json",
    ):
        row = src._classify_process(command)
        assert row["category"] == "support_maintenance"
        assert row["priority_tier"] == "throttle_first"
        assert row["throttle_candidate"] is True


def test_schwab_auth_supervisor_is_protected_not_support_throttle() -> None:
    row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/schwab_auth_supervisor.py --apply --json"
    )

    assert row["category"] == "live_execution"
    assert row["priority_tier"] == "protected_if_live"
    assert row["throttle_candidate"] is False


def test_heavy_diagnostics_are_support_throttle_candidates() -> None:
    divergence = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/data_source_divergence_bot.py --json"
    )
    assert divergence["category"] == "support_maintenance"
    assert divergence["priority_tier"] == "throttle_first"
    assert divergence["throttle_candidate"] is True

    mlx_audit = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/mlx_runtime_audit.py --json"
    )
    assert mlx_audit["category"] == "support_maintenance"
    assert mlx_audit["priority_tier"] == "throttle_first"
    assert mlx_audit["throttle_candidate"] is True

    mlx_router = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/mlx_intelligence_router.py --json"
    )
    assert mlx_router["category"] == "support_maintenance"
    assert mlx_router["priority_tier"] == "throttle_first"
    assert mlx_router["throttle_candidate"] is True

    library_router = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/library_utilization_router.py --json"
    )
    assert library_router["category"] == "support_maintenance"
    assert library_router["priority_tier"] == "throttle_first"
    assert library_router["throttle_candidate"] is True


def test_report_and_cleanup_helpers_are_support_throttle_candidates() -> None:
    report = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/paper_performance_report.py --day 20260503 --json-only"
    )
    assert report["category"] == "support_maintenance"
    assert report["priority_tier"] == "throttle_first"
    assert report["throttle_candidate"] is True

    sqlite_maintenance = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/sqlite_performance_maintenance.py --checkpoint-only --json"
    )
    assert sqlite_maintenance["category"] == "support_maintenance"
    assert sqlite_maintenance["priority_tier"] == "throttle_first"
    assert sqlite_maintenance["throttle_candidate"] is True

    collector_contracts = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/collector_contracts.py --json"
    )
    assert collector_contracts["category"] == "support_maintenance"
    assert collector_contracts["priority_tier"] == "throttle_first"
    assert collector_contracts["throttle_candidate"] is True

    infra_autofix = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/infrastructure_autofix_bot.py --apply --json"
    )
    assert infra_autofix["category"] == "support_maintenance"
    assert infra_autofix["priority_tier"] == "throttle_first"
    assert infra_autofix["throttle_candidate"] is True

    coverage_gap = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/coverage_gap_closer.py --apply-stage --json"
    )
    assert coverage_gap["category"] == "support_maintenance"
    assert coverage_gap["priority_tier"] == "throttle_first"
    assert coverage_gap["throttle_candidate"] is True


def test_full_force_paper_capacity_adds_buffered_runtime_overrides(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "shadow_only",
            "pressure_index": 0.08,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 1200, "total_pending_lines": 1200},
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_capacity_test_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only" if i % 2 else "shadow_candidate",
                    "sleeve_profile": "options" if i % 7 == 0 else "intraday_aggressive",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.4, "five_minutes": 2.0, "fifteen_minutes": 1.8},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))

    assert payload["paper_capacity_contract"]["full_force_stabilization_required"] is True
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS=240" in override
    assert "JSONL_BUFFER_MAX_ITEMS=240" in override
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=12" in override
    assert result["collector_guard"]["full_force_paper_stabilization"] is True
    assert registry["sub_bots"][0]["paper_execution_queue_policy"] == "buffered_jsonl_batching"


def test_runtime_throttle_collector_guard_uses_candidate_for_canonical_source_by_default(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    candidate_path = tmp_path / "governance" / "health" / "runtime_throttle_registry_candidate_latest.json"
    guard_path = tmp_path / "governance" / "health" / "runtime_throttle_source_write_guard_latest.json"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v1_runtime_candidate_guard",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "paper_runtime_stability_mode": "full_force_guarded",
                }
            ]
        },
    )
    monkeypatch.setattr(src, "SOURCE_REGISTRY_PATH", registry_path)
    source_before = registry_path.read_text(encoding="utf-8")
    payload = {
        "throttle_profile": "sustain",
        "memory_pressure_level": "normal",
        "compute_pressure_level": "high",
        "paper_capacity_contract": {
            "full_force_stabilization_required": True,
            "mode": "full_force_buffered",
            "runtime_policy": {"control_refresh_seconds": 240},
        },
    }

    result = src._apply_registry_collector_guard(
        tmp_path,
        payload,
        registry_path=registry_path,
        candidate_registry_path=candidate_path,
        source_write_guard_path=guard_path,
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert result["changed_count"] == 1
    assert result["registry_source_write_blocked"] is True
    assert result["registry_source_written"] is False
    assert registry_path.read_text(encoding="utf-8") == source_before
    assert candidate["sub_bots"][0]["paper_runtime_stability_mode"] == "full_force_buffered"
    assert guard_path.exists()

    allowed = src._apply_registry_collector_guard(
        tmp_path,
        payload,
        registry_path=registry_path,
        candidate_registry_path=candidate_path,
        source_write_guard_path=guard_path,
        allow_source_registry_write=True,
    )
    source = json.loads(registry_path.read_text(encoding="utf-8"))

    assert allowed["registry_source_write_blocked"] is False
    assert allowed["registry_source_written"] is True
    assert source["sub_bots"][0]["paper_runtime_stability_mode"] == "full_force_buffered"


def test_runtime_throttle_does_not_protect_live_on_cool_raw_live_sql_overlay_pressure(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_operating_mode": "maintenance_only",
            "pressure_index": 4.1,
            "severity": "critical",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 61463,
                "total_pending_lines": 62187,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 3335,
                    "total_pending_lines": 4039,
                    "oldest_pending_age_seconds": 553.671,
                },
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_overlay_capacity_test_bot",
                    "active": True,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.0, "five_minutes": 1.7, "fifteen_minutes": 1.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    assert payload["throttle_profile"] != "protect_live"
    assert payload["paper_capacity_contract"]["storage_pressure_limited"] is False
    assert payload["paper_capacity_contract"]["storage_overlay_capacity_relief"]["active"] is True
    assert payload["runtime_snapshot"]["storage_pressure"]["overlay_capacity_relief"] is True


def test_runtime_throttle_uses_overlay_raw_live_estimate_for_capacity_relief(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_operating_mode": "maintenance_drain_window",
            "pressure_index": 32.377,
            "severity": "critical",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 672,
                "total_pending_lines": 1581,
                "overlay_adjusted": True,
                "effective_raw_live": {
                    "core_pending_lines": 672,
                    "total_pending_lines": 1581,
                    "oldest_pending_age_seconds": 7770.598,
                    "source": "sql_ingestion_overlay_pressure",
                    "raw_live_estimate": {
                        "core_pending_lines": 22,
                        "total_pending_lines": 929,
                        "oldest_pending_age_seconds": 0.0,
                    },
                },
                "effective_raw_live_source": "sql_ingestion_overlay_pressure",
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_overlay_estimate_capacity_test_bot",
                    "active": True,
                    "paper_trade_enabled": i < 400,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.0, "five_minutes": 1.7, "fifteen_minutes": 1.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    overlay = payload["paper_capacity_contract"]["storage_overlay_capacity_relief"]
    assert payload["throttle_profile"] != "protect_live"
    assert payload["paper_capacity_contract"]["storage_pressure_limited"] is False
    assert overlay["active"] is True
    assert overlay["raw_live"]["source"] == "effective_raw_live.raw_live_estimate"
    assert payload["runtime_snapshot"]["storage_pressure"]["overlay_capacity_relief"] is True


def test_runtime_throttle_prefers_bounded_effective_raw_live_for_capacity_relief(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "recommended_operating_mode": "maintenance_drain_window",
            "pressure_index": 1.894,
            "severity": "high",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 5980,
                "total_pending_lines": 5980,
                "overlay_adjusted": True,
                "effective_raw_live": {
                    "core_pending_lines": 5980,
                    "total_pending_lines": 5980,
                    "oldest_pending_age_seconds": 454.443,
                    "source": "sql_ingestion_overlay_pressure",
                    "reconciled_from_raw_live": True,
                    "raw_live_estimate": {
                        "core_pending_lines": 18881,
                        "total_pending_lines": 1235466,
                        "oldest_pending_age_seconds": 106.561,
                    },
                },
                "effective_raw_live_source": "sql_ingestion_overlay_pressure",
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_bounded_overlay_capacity_test_bot",
                    "active": True,
                    "paper_trade_enabled": i < 400,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.0, "five_minutes": 1.7, "fifteen_minutes": 1.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    overlay = payload["paper_capacity_contract"]["storage_overlay_capacity_relief"]
    assert payload["throttle_profile"] != "protect_live"
    assert payload["paper_capacity_contract"]["storage_pressure_limited"] is False
    assert overlay["active"] is True
    assert overlay["raw_live"]["source"] == "sql_ingestion_overlay_pressure"
    assert overlay["raw_live"]["total_pending_lines"] == 5980
    assert payload["runtime_snapshot"]["storage_pressure"]["overlay_capacity_relief"] is True


def test_runtime_throttle_uses_explicit_empty_sql_overlay_for_storage_relief(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_operating_mode": "maintenance_only",
            "pressure_index": 259.871,
            "severity": "critical",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 7686,
                "total_pending_lines": 12585,
                "oldest_pending_age_seconds": 62368.944,
                "effective_raw_live": {
                    "core_pending_lines": 7686,
                    "total_pending_lines": 12585,
                    "oldest_pending_age_seconds": 62368.944,
                },
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "fresh_source_count": 18,
                "explicit_empty_source_count": 18,
                "stale_pending_lines": 0,
                "total_pending_lines": 0,
                "files_with_pending": 0,
                "oldest_pending_age_seconds": 0.0,
                "top_pending_files": [],
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.0, "five_minutes": 1.7, "fifteen_minutes": 1.6},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )

    overlay = payload["runtime_snapshot"]["storage_pressure"]["overlay_relief_contract"]
    assert payload["throttle_profile"] != "protect_live"
    assert payload["runtime_snapshot"]["storage_pressure"]["pressure_index"] == 0.0
    assert overlay["active"] is True
    assert overlay["direct_sql_overlay_clear"] is True
    assert overlay["raw_storage_pressure_index"] > 1.0
    assert payload["mac_fluidity_contract"]["measurements"]["storage_total_pending_lines"] == 0


def test_runtime_throttle_preserves_lower_capability_pack_sampling(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "yellow", "swap_used_gb": 9.0})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.08,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 1200, "total_pending_lines": 1200},
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v849_coordination_lineage_genome_mapper_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "data_collection_sample_rate": 0.9,
                    "data_collection_max_daily_mb": 150,
                    "freshness_slo_seconds": 60,
                    "capability_pack_contract": {
                        "storage_retention_rule": {
                            "sample_rate": 0.18,
                            "max_daily_mb_per_bot": 28,
                        }
                    },
                }
            ]
        },
    )
    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.0, "five_minutes": 7.0, "fifteen_minutes": 6.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {"interactive_cotenant": 40.0},
            "category_counts": {"interactive_cotenant": 1},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert result["collector_guard"]["policy"]["sample_rate"] == 0.3
    assert row["data_collection_sample_rate"] == 0.18
    assert row["data_collection_max_daily_mb"] == 28


def test_runtime_throttle_consumes_memory_cotenant_awareness(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "cotenant_awareness": {
                "active": True,
                "mode": "managed_cotenant",
                "co_running_level": "interactive",
                "creative_level": "none",
                "open_app_count": 2,
                "open_apps": ["PyCharm", "Final Cut Pro"],
                "co_running_classes": ["developer", "creative"],
                "memory_pressure_clear": True,
                "storage_pressure_clear": True,
            },
        },
    )
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["overall_status"] == "advisory"
    assert payload["throttle_profile"] == "soft_cap"
    assert payload["cotenant_awareness_contract"]["profile_adjusted"] is True
    assert "RUNTIME_COTENANT_AWARE=1" in override
    assert result["env_override_count"] >= 5


def test_runtime_throttle_marks_low_pressure_external_soft_cap_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.316,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 709, "total_pending_lines": 944},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 4.0, "fifteen_minutes": 4.8},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 36.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"interactive_cotenant": 36.0},
            "category_counts": {"interactive_cotenant": 1},
        },
    )

    assert payload["throttle_profile"] == "soft_cap"
    assert payload["overall_status"] == "advisory"
    assert payload["host_pressure_attribution"]["external_pressure_dominant"] is True
    assert payload["soft_cap_advisory_reclassification"]["active"] is True


def test_runtime_throttle_marks_guarded_foreground_sustain_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.66,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 9949,
                "total_pending_lines": 16493,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 1.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.0, "five_minutes": 5.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 110.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"interactive_cotenant": 110.0},
            "category_counts": {"interactive_cotenant": 1},
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "advisory"
    assert payload["ok"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "foreground_cotenant_pressure_is_guarded_advisory_not_bot_runtime_degradation"
    assert advisory["measurements"]["foreground_guarded"] is True


def test_runtime_throttle_marks_foreground_pressure_with_clean_storage_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.025,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 11.25, "five_minutes": 8.7, "fifteen_minutes": 7.8},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 71.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
                {
                    "pid": 304,
                    "nice": 0,
                    "cpu_percent": 29.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:02",
                    "command": "/System/Library/PrivateFrameworks/PhotoLibraryServices.framework/photolibraryd",
                    "category": "unclassified",
                    "priority_tier": "observe",
                    "throttle_candidate": False,
                },
                {
                    "pid": 305,
                    "nice": 0,
                    "cpu_percent": 21.3,
                    "mem_percent": 0.2,
                    "elapsed": "00:02",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"interactive_cotenant": 71.0, "unclassified": 29.0, "operator_observability": 21.3},
            "category_counts": {"interactive_cotenant": 1, "unclassified": 1, "operator_observability": 1},
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["to_status"] == "ready"
    assert advisory["reason"] == "foreground_cotenant_pressure_is_guarded_runtime_ready"
    assert advisory["measurements"]["plain_storage_clear_guarded_ready"] is True
    assert advisory["measurements"]["runtime_ready_guarded"] is True
    assert advisory["measurements"]["bot_owned_non_operator_cpu_percent"] == 0.0


def test_runtime_throttle_marks_system_cotenant_pressure_with_clean_storage_as_guarded_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.025,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 4.3, "five_minutes": 6.8, "fifteen_minutes": 7.2},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 60.7,
                    "mem_percent": 0.2,
                    "elapsed": "00:02",
                    "command": "/System/Library/Frameworks/CoreServices.framework/mds_stores",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 304,
                    "nice": 0,
                    "cpu_percent": 26.3,
                    "mem_percent": 0.2,
                    "elapsed": "00:02",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"system_cotenant": 60.7, "operator_observability": 26.3},
            "category_counts": {"system_cotenant": 1, "operator_observability": 1},
        },
    )

    assert payload["overall_status"] == "ready"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["to_status"] == "ready"
    assert advisory["reason"] == "external_cotenant_pressure_with_clean_storage_is_guarded_runtime_ready"
    assert advisory["measurements"]["plain_external_live_read_only_guarded_ready"] is True
    assert advisory["measurements"]["runtime_ready_guarded"] is True


def test_runtime_throttle_marks_niced_support_pressure_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 68, "total_pending_lines": 77},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.0, "five_minutes": 5.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 16,
                    "cpu_percent": 85.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "python scripts/link_jsonl_to_sql.py",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                }
            ],
            "category_cpu": {"support_maintenance": 85.0},
            "category_counts": {"support_maintenance": 1},
        },
    )

    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "advisory"
    assert payload["host_pressure_attribution"]["support_hot_low_priority"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "support_pressure_is_already_niced_and_guarded_advisory"
    assert advisory["measurements"]["support_low_priority_guarded"] is True
    assert advisory["measurements"]["storage_fresh_overflow"] is True


def test_runtime_throttle_marks_bounded_writer_support_and_read_only_protected_lane_ready(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.137,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 2059,
                "total_pending_lines": 2172,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 23.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 6.0, "five_minutes": 4.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 801,
                    "nice": 0,
                    "cpu_percent": 98.0,
                    "mem_percent": 0.4,
                    "elapsed": "00:03",
                    "command": "python scripts/link_jsonl_to_sql.py",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
                {
                    "pid": 802,
                    "nice": 16,
                    "cpu_percent": 42.0,
                    "mem_percent": 0.3,
                    "elapsed": "00:03",
                    "command": "python scripts/ops/creative_cotenant_guard.py",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 803,
                    "nice": 0,
                    "cpu_percent": 64.0,
                    "mem_percent": 0.4,
                    "elapsed": "00:03",
                    "command": "python scripts/ops/live_macro_auto_watch.py --json",
                    "category": "macro_capture",
                    "priority_tier": "protected_if_live",
                    "throttle_candidate": False,
                },
                {
                    "pid": 804,
                    "nice": 0,
                    "cpu_percent": 27.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:03",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
                {
                    "pid": 805,
                    "nice": 0,
                    "cpu_percent": 40.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:03",
                    "command": "/usr/libexec/sysmond",
                    "category": "system_cotenant",
                    "priority_tier": "external_system",
                    "throttle_candidate": False,
                },
                {
                    "pid": 806,
                    "nice": 0,
                    "cpu_percent": 27.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:03",
                    "command": "Codex",
                    "category": "interactive_cotenant",
                    "priority_tier": "external_cotenant",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "storage_writer": 98.0,
                "support_maintenance": 42.0,
                "macro_capture": 64.0,
                "operator_observability": 27.0,
                "system_cotenant": 40.0,
                "interactive_cotenant": 27.0,
            },
            "category_counts": {
                "storage_writer": 1,
                "support_maintenance": 1,
                "macro_capture": 1,
                "operator_observability": 1,
                "system_cotenant": 1,
                "interactive_cotenant": 1,
            },
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "bounded_writer_support_and_read_only_protected_lane_is_guarded_runtime_ready"
    assert advisory["measurements"]["bounded_writer_support_protected_guarded_ready"] is True
    assert payload["mac_fluidity_contract"]["overall_status"] == "ready"


def test_runtime_throttle_marks_operator_observability_pressure_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 68, "total_pending_lines": 77},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 12.0, "five_minutes": 5.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 0,
                    "cpu_percent": 92.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "python scripts/ops/system_intelligence_coordinator.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"operator_observability": 92.0},
            "category_counts": {"operator_observability": 1},
        },
    )

    assert payload["overall_status"] == "advisory"
    assert payload["host_pressure_attribution"]["protected_work_hot"] is False
    assert payload["host_pressure_attribution"]["operator_observability_hot"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "operator_observability_pressure_is_guarded_advisory"
    assert advisory["measurements"]["operator_observability_guarded"] is True


def test_runtime_throttle_keeps_paper_capacity_open_under_operator_observability_high_compute(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.04,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 603,
                "total_pending_lines": 792,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 13.0, "five_minutes": 8.0, "fifteen_minutes": 4.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 304,
                    "nice": 0,
                    "cpu_percent": 61.4,
                    "mem_percent": 0.4,
                    "elapsed": "00:03",
                    "command": "python scripts/ops/runtime_throttle_control.py --json",
                    "category": "operator_observability",
                    "priority_tier": "operator_visible",
                    "throttle_candidate": False,
                },
                {
                    "pid": 305,
                    "nice": 0,
                    "cpu_percent": 7.6,
                    "mem_percent": 0.4,
                    "elapsed": "00:03",
                    "command": "python scripts/run_all_sleeves.py",
                    "category": "live_execution",
                    "priority_tier": "protected",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {"operator_observability": 61.4, "live_execution": 7.6},
            "category_counts": {"operator_observability": 1, "live_execution": 1},
        },
    )

    assert payload["compute_pressure_level"] == "high"
    assert payload["overall_status"] == "advisory"
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "operator_observability_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation"
    assert advisory["measurements"]["operator_observability_high_compute_guarded"] is True
    assert payload["paper_execution_policy"]["paper_execution_allowed"] is True
    assert payload["paper_execution_policy"]["pause_paper_execution"] is False
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert payload["paper_capacity_contract"]["compute_pressure_limited"] is False


def test_runtime_throttle_marks_bounded_protected_work_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 68, "total_pending_lines": 77},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 7.0, "five_minutes": 4.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 303,
                    "nice": 5,
                    "cpu_percent": 52.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:02",
                    "command": "python scripts/ops/live_macro_auto_watch.py --json",
                    "category": "macro_capture",
                    "priority_tier": "protected_if_live",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"macro_capture": 52.0},
            "category_counts": {"macro_capture": 1},
        },
    )

    assert payload["overall_status"] == "advisory"
    assert payload["host_pressure_attribution"]["protected_work_hot"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "protected_live_or_macro_work_is_guarded_advisory"
    assert advisory["measurements"]["protected_work_guarded"] is True


def test_runtime_throttle_marks_niced_research_pressure_as_advisory(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.1,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 120, "total_pending_lines": 180},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 7.0, "five_minutes": 5.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 505,
                    "nice": 18,
                    "cpu_percent": 58.0,
                    "mem_percent": 1.0,
                    "elapsed": "00:04",
                    "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                }
            ],
            "category_cpu": {"research_training": 58.0},
            "category_counts": {"research_training": 1},
        },
    )

    assert payload["overall_status"] == "advisory"
    assert payload["host_pressure_attribution"]["protected_work_hot"] is False
    assert payload["host_pressure_attribution"]["research_training_hot"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["active"] is True
    assert advisory["reason"] == "research_training_pressure_is_already_niced_and_guarded_advisory"
    assert advisory["measurements"]["research_low_priority_guarded"] is True


def test_runtime_throttle_downgrades_protect_live_for_stoppable_background_research(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.025,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {
                "core_pending_lines": 374,
                "total_pending_lines": 374,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 20.0, "five_minutes": 13.0, "fifteen_minutes": 12.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 701,
                    "nice": 18,
                    "cpu_percent": 46.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/run_shadow_training_loop.py --broker schwab",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
                {
                    "pid": 702,
                    "nice": 18,
                    "cpu_percent": 45.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
                {
                    "pid": 703,
                    "nice": 0,
                    "cpu_percent": 100.0,
                    "mem_percent": 0.1,
                    "elapsed": "00:04",
                    "command": "python scripts/ops/creative_cotenant_guard.py apply --json",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
            ],
            "category_cpu": {"research_training": 91.0, "support_maintenance": 100.0},
            "category_counts": {"research_training": 2, "support_maintenance": 1},
        },
    )

    assert payload["protect_live_autonomic_reclassification"]["active"] is True
    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "advisory"
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "research_training_pressure_is_already_niced_and_guarded_advisory"
    assert advisory["measurements"]["research_low_priority_guarded"] is True


def test_runtime_throttle_downgrades_protect_live_when_paper_lane_is_guarded(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(health_root / "PAPER_TRADE_LOCK.flag", {"policy": "live_data_paper_trade_only"})
    _write_json(health_root / "paper_400_ramp_latest.json", {"stage": "armed", "armed": True, "ok": True, "blockers": []})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.136,
            "severity": "stable",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 276,
                "total_pending_lines": 387,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 32.0,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": f"paper_capacity_bot_{idx}", "active": True, "lifecycle_state": "active"}
                for idx in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 20.0, "five_minutes": 13.0, "fifteen_minutes": 12.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [
                {
                    "pid": 601,
                    "nice": 19,
                    "cpu_percent": 41.0,
                    "mem_percent": 0.3,
                    "elapsed": "00:04",
                    "command": "python scripts/run_execution_lane.py --mode paper",
                    "category": "paper_execution",
                    "priority_tier": "paper_gate_controlled",
                    "throttle_candidate": True,
                },
                {
                    "pid": 701,
                    "nice": 20,
                    "cpu_percent": 96.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:04",
                    "command": "python scripts/resource_guard.py --profile options",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 702,
                    "nice": 20,
                    "cpu_percent": 52.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:04",
                    "command": "python scripts/resource_guard.py --profile options",
                    "category": "support_maintenance",
                    "priority_tier": "throttle_first",
                    "throttle_candidate": True,
                },
                {
                    "pid": 801,
                    "nice": 19,
                    "cpu_percent": 36.0,
                    "mem_percent": 0.2,
                    "elapsed": "00:04",
                    "command": "python scripts/run_shadow_training_loop.py --broker schwab",
                    "category": "research_training",
                    "priority_tier": "research_downshift",
                    "throttle_candidate": False,
                },
                {
                    "pid": 901,
                    "nice": 4,
                    "cpu_percent": 79.0,
                    "mem_percent": 0.3,
                    "elapsed": "00:04",
                    "command": "python scripts/link_jsonl_to_sql.py --project-root /repo --mode sqlite",
                    "category": "storage_writer",
                    "priority_tier": "backlog_writer",
                    "throttle_candidate": False,
                },
            ],
            "category_cpu": {
                "paper_execution": 41.0,
                "support_maintenance": 148.0,
                "research_training": 36.0,
                "storage_writer": 79.0,
            },
            "category_counts": {
                "paper_execution": 1,
                "support_maintenance": 2,
                "research_training": 1,
                "storage_writer": 1,
            },
        },
    )

    assert payload["protect_live_autonomic_reclassification"]["active"] is True
    assert payload["protect_live_autonomic_reclassification"]["paper_lane_guarded_for_autonomic_relief"] is True
    assert payload["throttle_profile"] == "sustain"
    assert payload["overall_status"] == "advisory"
    assert payload["paper_execution_policy"]["paper_execution_allowed"] is True
    assert payload["paper_execution_policy"]["pause_paper_execution"] is False
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    advisory = payload["soft_cap_advisory_reclassification"]
    assert advisory["reason"] == "support_pressure_is_already_niced_and_guarded_advisory"
    assert advisory["measurements"]["support_low_priority_guarded"] is True


def test_runtime_throttle_consumes_mlx_intelligence_router_caps(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health_root / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "advisory",
            "library_coverage": {"coverage_ratio": 1.0},
            "route_coverage": {"route_coverage_ratio": 1.0},
            "runtime_caps": {
                "profile": "foreground_safe",
                "max_concurrent_mlx_jobs": 2,
                "tensor_batch_cap": 48,
                "embedding_batch_cap": 96,
                "graph_node_cap": 9000,
                "audio_minutes_per_job_cap": 30,
                "heavy_vlm_enabled": True,
                "compile_mode": "direct_stable",
            },
            "recommended_runtime_env": {
                "MLX_INTELLIGENCE_ROUTER_ENABLED": "1",
                "MLX_INTELLIGENCE_PROFILE": "foreground_safe",
                "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": "2",
                "MLX_INTELLIGENCE_TENSOR_BATCH_CAP": "48",
                "MLX_INTELLIGENCE_SCHEDULER_MODE": "bounded_direct_stable",
                "MLX_INTELLIGENCE_ALLOWED_LANES": "tensor_quant_core,embedding_memory",
                "MLX_INTELLIGENCE_TOTAL_MEMORY_BUDGET_MB": "1400",
                "MLX_INTELLIGENCE_REOPEN_STAGE": "bounded_direct_stable",
                "MLX_INTELLIGENCE_TOKEN_BUDGET": "10",
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["mlx_intelligence_contract"]["active"] is True
    assert payload["mlx_intelligence_contract"]["library_coverage_ratio"] == 1.0
    assert "MLX_INTELLIGENCE_ROUTER_ENABLED=1" in override
    assert "MLX_INTELLIGENCE_TENSOR_BATCH_CAP=48" in override
    assert "MLX_INTELLIGENCE_SCHEDULER_MODE=bounded_direct_stable" in override
    assert "MLX_INTELLIGENCE_ALLOWED_LANES=tensor_quant_core,embedding_memory" in override
    assert "MLX_INTELLIGENCE_REOPEN_STAGE=bounded_direct_stable" in override
    assert "MLX_INTELLIGENCE_TOKEN_BUDGET=10" in override
    assert result["env_override_count"] >= 4


def test_runtime_throttle_keeps_blocked_mlx_router_safety_caps_active() -> None:
    contract = src._mlx_intelligence_contract(
        {
            "overall_status": "blocked",
            "library_coverage": {"coverage_ratio": 0.75},
            "route_coverage": {"route_coverage_ratio": 0.8},
            "runtime_caps": {
                "profile": "protect_live",
                "max_concurrent_mlx_jobs": 1,
                "tensor_batch_cap": 16,
                "embedding_batch_cap": 32,
                "graph_node_cap": 3000,
                "audio_minutes_per_job_cap": 12,
                "heavy_vlm_enabled": False,
                "compile_mode": "off",
                "p_core_allocation_aware": True,
                "p_core_allocation_mode": "foreground_protect",
                "p_core_preprocess_workers": 4,
                "p_core_coordination_policy": "backlog_burst_owns_p_cores_mlx_runs_light",
            },
        }
    )

    assert contract["active"] is True
    assert contract["status"] == "blocked"
    assert contract["p_core_allocation_aware"] is True
    assert contract["p_core_allocation_mode"] == "foreground_protect"
    assert contract["p_core_preprocess_workers"] == 4


def test_runtime_throttle_consumes_library_utilization_router_caps_and_keeps_mlx_default(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health_root / "library_utilization_router_latest.json",
        {
            "overall_status": "advisory",
            "coverage": {
                "coverage_ratio": 1.0,
                "locked_runtime_ok_ratio": 1.0,
                "managed_non_mlx_package_count": 80,
            },
            "runtime_caps": {
                "profile": "foreground_safe",
                "max_async_request_concurrency": 8,
                "max_sql_writer_workers": 2,
                "max_dataframe_workers": 2,
                "max_portable_model_replay_jobs": 0,
                "max_report_render_jobs": 1,
            },
            "recommended_runtime_env": {
                "LIBRARY_UTILIZATION_ROUTER_ENABLED": "1",
                "LIBRARY_UTILIZATION_PROFILE": "foreground_safe",
                "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
                "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
                "PORTABLE_MODEL_REPLAY_POLICY": "canary_or_off_hours_only",
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["library_utilization_contract"]["active"] is True
    assert payload["library_utilization_contract"]["default_ml_backend"] == "mlx"
    assert "LIBRARY_UTILIZATION_ROUTER_ENABLED=1" in override
    assert "PRIMARY_ML_RUNTIME_BACKEND=mlx" in override
    assert "PORTABLE_MODEL_REPLAY_POLICY=canary_or_off_hours_only" in override
    assert result["env_override_count"] >= 5



def test_protect_live_downshifts_simulated_shadow_training_loops(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_CPU_EFFICIENCY_SATURATION_GUARD", "0")
    monkeypatch.delenv("RUNTIME_THROTTLE_RESEARCH_NICE", raising=False)
    monkeypatch.delenv("RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND", raising=False)
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "yellow", "swap_used_gb": 9.0})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "recommended_operating_mode": "market_hours_backlog_protection",
            "pressure_index": 2.0,
            "severity": "high",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {"core_pending_lines": 30000, "total_pending_lines": 30000},
        },
    )
    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 13.5, "five_minutes": 12.0, "fifteen_minutes": 11.0},
        "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
        "vm_stat": {},
        "top_processes": [
            {
                "pid": 4242,
                "cpu_percent": 13.0,
                "mem_percent": 6.7,
                "elapsed": "39:34",
                "command": "python scripts/run_shadow_training_loop.py --broker schwab --interval-seconds 15 --max-iterations 0 --simulate",
                "category": "research_training",
                "priority_tier": "protected",
                "throttle_candidate": False,
            }
        ],
        "category_cpu": {"research_training": 13.0},
        "category_counts": {"research_training": 1},
    }
    calls: list[list[str]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", lambda pid, sig: None)

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["throttle_profile"] == "protect_live"
    assert payload["research_training_trim_candidates"][0]["throttle_reason"] == "simulated_training_loop_under_host_pressure"
    assert result["process_throttle"]["attempted_count"] == 1
    assert any(cmd[:3] == ["renice", "-n", "20"] for cmd in calls)
    assert "SHADOW_LOOP_PRESSURE_INTERVAL_FLOOR_ENABLED=1" in override
    assert "SHADOW_LOOP_PROTECT_LIVE_EXTRA_INTERVAL_SECONDS=30" in override


def test_runtime_throttle_pauses_all_hot_research_candidates_up_to_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_RESEARCH_TRAINING_PAUSE_LIMIT", "4")
    signals: list[tuple[int, signal.Signals | int]] = []

    def fake_kill(pid: int, sig: signal.Signals | int) -> None:
        signals.append((pid, sig))

    payload = {
        "runtime_saturation_governor_v2": {
            "training_policy": {"training_paused": True, "reason": "host_saturation_or_memory_pressure"}
        },
        "throttle_profile": "sustain",
        "compute_pressure_level": "elevated",
        "memory_pressure_level": "normal",
    }
    candidates = [
        {
            "pid": pid,
            "category": "research_training",
            "cpu_percent": cpu,
            "command": f"python scripts/run_shadow_training_loop.py --broker schwab --profile p{pid}",
        }
        for pid, cpu in ((101, 93.0), (102, 88.0), (103, 79.0))
    ]

    monkeypatch.setattr(src.os, "kill", fake_kill)

    result = src._apply_research_training_pause(
        tmp_path,
        candidates,
        payload,
        state_path=tmp_path / "runtime_research_pause_state.json",
    )

    assert result["attempted_count"] == 3
    assert result["successful_count"] == 3
    assert [pid for pid, sig in signals if sig == signal.SIGSTOP] == [101, 102, 103]


def test_runtime_throttle_pauses_research_for_mac_fluidity_watch(tmp_path: Path, monkeypatch) -> None:
    signals: list[tuple[int, signal.Signals | int]] = []

    def fake_kill(pid: int, sig: signal.Signals | int) -> None:
        signals.append((pid, sig))

    payload = {
        "runtime_saturation_governor_v2": {
            "training_policy": {"training_paused": False, "reason": "runtime_training_ready"}
        },
        "throttle_profile": "soft_cap",
        "compute_pressure_level": "elevated",
        "memory_pressure_level": "normal",
        "mac_fluidity_contract": {
            "overall_status": "watch",
            "fluidity_band": "guarded_smooth",
            "fluidity_score": 88.5,
            "research_pause_recommended": True,
        },
    }
    candidates = [
        {
            "pid": 701,
            "category": "research_training",
            "cpu_percent": 72.0,
            "command": "python scripts/run_shadow_training_loop.py --broker schwab",
        },
        {
            "pid": 702,
            "category": "research_training",
            "cpu_percent": 43.5,
            "command": "python scripts/run_shadow_training_loop.py --broker coinbase",
        },
    ]

    monkeypatch.setattr(src.os, "kill", fake_kill)

    result = src._apply_research_training_pause(
        tmp_path,
        candidates,
        payload,
        state_path=tmp_path / "runtime_research_pause_state.json",
    )

    assert result["pause_requested"] is True
    assert result["reason"] == "mac_fluidity_research_pause"
    assert result["successful_count"] == 2
    assert [pid for pid, sig in signals if sig == signal.SIGSTOP] == [701, 702]


def test_runtime_throttle_pauses_support_maintenance_for_mac_fluidity_watch(tmp_path: Path, monkeypatch) -> None:
    signals: list[tuple[int, signal.Signals | int]] = []

    def fake_kill(pid: int, sig: signal.Signals | int) -> None:
        signals.append((pid, sig))

    payload = {
        "throttle_profile": "soft_cap",
        "compute_pressure_level": "elevated",
        "memory_pressure_level": "normal",
        "mac_fluidity_contract": {
            "overall_status": "watch",
            "fluidity_band": "guarded_smooth",
            "fluidity_score": 83.0,
            "support_pause_recommended": True,
        },
    }
    candidates = [
        {
            "pid": 801,
            "category": "support_maintenance",
            "cpu_percent": 66.0,
            "command": "python scripts/ops/swap_pressure_governor.py --json",
        },
        {
            "pid": 802,
            "category": "storage_writer",
            "cpu_percent": 88.0,
            "command": "python scripts/ops/sql_link_writer_service.py --json",
        },
    ]

    monkeypatch.setattr(src.os, "kill", fake_kill)

    result = src._apply_support_maintenance_pause(
        tmp_path,
        candidates,
        payload,
        state_path=tmp_path / "runtime_support_pause_state.json",
    )

    assert result["pause_requested"] is True
    assert result["reason"] == "mac_fluidity_support_pause"
    assert result["successful_count"] == 1
    assert [pid for pid, sig in signals if sig == signal.SIGSTOP] == [801]


def test_efficiency_guard_keeps_research_throttle_off_background_taskpolicy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_CPU_EFFICIENCY_SATURATION_GUARD", "1")
    monkeypatch.setenv("SLEEVE_NICE_SPECIALIZED", "8")
    monkeypatch.delenv("RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND", raising=False)
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "yellow", "swap_used_gb": 9.0})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 13.5, "five_minutes": 12.0, "fifteen_minutes": 11.0},
        "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
        "vm_stat": {},
        "top_processes": [
            {
                "pid": 4242,
                "nice": 8,
                "cpu_percent": 33.0,
                "mem_percent": 6.7,
                "elapsed": "39:34",
                "command": "python scripts/run_shadow_training_loop.py --broker schwab --profile gpu_quant_acceleration",
                "category": "research_training",
                "priority_tier": "protected",
                "throttle_candidate": False,
            }
        ],
        "category_cpu": {"research_training": 33.0},
        "category_counts": {"research_training": 1},
    }
    calls: list[list[str]] = []

    def fake_run_apply(command: list[str]) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(src, "_run_apply_command", fake_run_apply)
    monkeypatch.setattr(src.os, "kill", lambda pid, sig: None)

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=4,
    )

    process = result["process_throttle"]["processes"][0]
    assert process["target_nice"] == 20
    assert process["renice"].get("skipped") is not True
    assert process["taskpolicy"]["skipped"] is True
    assert any(cmd[:3] == ["renice", "-n", "12"] for cmd in calls)
