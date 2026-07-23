import json
from pathlib import Path

import scripts.run_shadow_training_loop as loop


def _reset_dynamic_override_cache() -> None:
    loop._DYNAMIC_STORAGE_OVERRIDE_CACHE.clear()
    loop._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {
            "checked_at_monotonic": 0.0,
            "fingerprint": (),
            "values": {},
        }
    )


def test_dynamic_storage_flag_reads_override_files_in_order(tmp_path, monkeypatch) -> None:
    pressure = tmp_path / ".env.storage_pressure_override"
    storage = tmp_path / ".env.storage_override"
    monkeypatch.setattr(loop, "STORAGE_PRESSURE_OVERRIDE_PATH", pressure)
    monkeypatch.setattr(loop, "STORAGE_OVERRIDE_PATH", storage)
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (pressure, storage))

    pressure.write_text("LOG_API_CALLS=0\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    assert loop._dynamic_storage_flag("LOG_API_CALLS", True) is False

    storage.write_text("LOG_API_CALLS=1\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    assert loop._dynamic_storage_flag("LOG_API_CALLS", True) is True


def test_runtime_research_self_nice_reads_runtime_override(tmp_path, monkeypatch) -> None:
    runtime_override = tmp_path / ".env.runtime_resource_guard_override"
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (runtime_override,))
    runtime_override.write_text("RUNTIME_RESEARCH_TRAINING_NICE=20\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    state = {"nice": 0}

    def fake_nice(delta: int) -> int:
        if int(delta) != 0:
            state["nice"] += int(delta)
        return state["nice"]

    monkeypatch.setattr(loop.os, "nice", fake_nice)

    result = loop._apply_runtime_research_self_nice()

    assert result["applied"] is True
    assert result["previous_nice"] == 0
    assert result["target_nice"] == 20
    assert result["current_nice"] == 20


def test_runtime_research_self_nice_skips_when_already_low_priority(tmp_path, monkeypatch) -> None:
    runtime_override = tmp_path / ".env.runtime_resource_guard_override"
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (runtime_override,))
    runtime_override.write_text("RUNTIME_THROTTLE_RESEARCH_NICE=12\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    monkeypatch.setattr(loop.os, "nice", lambda delta: 20)

    result = loop._apply_runtime_research_self_nice()

    assert result["applied"] is False
    assert result["reason"] == "current_nice_at_or_above_target"
    assert result["current_nice"] == 20
    assert result["target_nice"] == 12


def test_log_api_call_and_loop_state_respect_dynamic_storage_flags(tmp_path, monkeypatch) -> None:
    pressure = tmp_path / ".env.storage_pressure_override"
    storage = tmp_path / ".env.storage_override"
    monkeypatch.setattr(loop, "STORAGE_PRESSURE_OVERRIDE_PATH", pressure)
    monkeypatch.setattr(loop, "STORAGE_OVERRIDE_PATH", storage)
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (pressure, storage))
    monkeypatch.setattr(loop, "_shadow_profile_name", lambda: "default")
    monkeypatch.setattr(loop, "_shadow_domain_name", lambda broker=None: "equities")
    monkeypatch.setattr(loop, "_event_bus_path", lambda project_root: str(tmp_path / "events.jsonl"))

    rows: list[tuple[str, dict]] = []

    def _capture_append(path: str, row: dict) -> None:
        rows.append((path, row))

    monkeypatch.setattr(loop, "_append_jsonl", _capture_append)

    pressure.write_text("LOG_API_CALLS=0\nLOG_LOOP_STATE=0\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    loop._log_api_call(
        project_root=str(tmp_path),
        broker="schwab",
        symbol="SPY",
        endpoint="/quotes",
        status="ok",
        latency_ms=12.0,
    )
    loop._emit_loop_state(
        project_root=str(tmp_path),
        broker="schwab",
        prev_state="idle",
        new_state="running",
        iter_count=1,
    )
    assert rows == []

    pressure.write_text("LOG_API_CALLS=1\nLOG_LOOP_STATE=1\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    loop._log_api_call(
        project_root=str(tmp_path),
        broker="schwab",
        symbol="SPY",
        endpoint="/quotes",
        status="ok",
        latency_ms=12.0,
    )
    loop._emit_loop_state(
        project_root=str(tmp_path),
        broker="schwab",
        prev_state="idle",
        new_state="running",
        iter_count=1,
    )
    assert len(rows) == 3


def test_resolve_hot_channel_write_targets_prefers_channel_primary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        loop,
        "_dynamic_storage_overrides",
        lambda: {
            "CHANNEL_LOG_PRIMARY_MODE": "channel",
            "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "0",
        },
    )

    path = str(tmp_path / "governance" / "events" / "runtime_events_20260417.jsonl")
    primary, mirrors = loop._resolve_hot_channel_write_targets(path, channel="runtime")

    assert primary.endswith("governance/channels/runtime/default_equities_schwab/runtime_20260417.jsonl")
    assert mirrors == []


def test_resolve_hot_channel_write_targets_can_keep_legacy_mirror(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        loop,
        "_dynamic_storage_overrides",
        lambda: {
            "CHANNEL_LOG_PRIMARY_MODE": "channel",
            "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "1",
        },
    )

    path = str(tmp_path / "governance" / "events" / "api_calls_default_equities_schwab_20260417.jsonl")
    primary, mirrors = loop._resolve_hot_channel_write_targets(path, channel="api")

    assert primary.endswith("governance/channels/api/default_equities_schwab/api_20260417.jsonl")
    assert mirrors == [path]


def test_paper_options_profile_allowlist_defaults_exclude_slow_sleeves(monkeypatch) -> None:
    monkeypatch.delenv("TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES", raising=False)

    assert loop._paper_options_profile_allowed("default") is True
    assert loop._paper_options_profile_allowed("aggressive") is True
    assert loop._paper_options_profile_allowed("options_on_futures") is True
    assert loop._paper_options_profile_allowed("options_on_futures_aggressive") is True
    assert loop._paper_options_profile_allowed("conservative") is False
    assert loop._paper_options_profile_allowed("dividend") is False
    assert loop._paper_options_profile_allowed("schwab_futures") is False


def test_paper_options_profile_allowlist_honors_override(monkeypatch) -> None:
    monkeypatch.setenv("TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES", "conservative,dividend")

    assert loop._paper_options_profile_allowed("conservative") is True
    assert loop._paper_options_profile_allowed("dividend") is True
    assert loop._paper_options_profile_allowed("aggressive") is False



def test_external_interval_floor_uses_runtime_and_settlement_pressure(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "runtime_throttle_control_latest.json").write_text(
        json.dumps({"throttle_profile": "protect_live", "compute_pressure_level": "high"}),
        encoding="utf-8",
    )
    (health / "platform_settlement_stabilization_latest.json").write_text(
        json.dumps({"sections": {"queue_decay_meter": {"queue_backpressure_active": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHADOW_LOOP_PROTECT_LIVE_EXTRA_INTERVAL_SECONDS", "32")
    monkeypatch.setenv("SHADOW_LOOP_QUEUE_BACKPRESSURE_EXTRA_INTERVAL_SECONDS", "11")
    monkeypatch.setenv("SHADOW_LOOP_MAX_DYNAMIC_EXTRA_INTERVAL_SECONDS", "40")

    assert loop._external_ingestion_extra_interval_seconds(str(tmp_path)) == 32


def test_external_interval_floor_uses_sustain_training_pause(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "runtime_throttle_control_latest.json").write_text(
        json.dumps(
            {
                "throttle_profile": "sustain",
                "compute_pressure_level": "elevated",
                "runtime_saturation_governor_v2": {"training_policy": {"training_paused": True}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHADOW_LOOP_SUSTAIN_EXTRA_INTERVAL_SECONDS", "21")
    monkeypatch.setenv("SHADOW_LOOP_HIGH_COMPUTE_EXTRA_INTERVAL_SECONDS", "17")
    monkeypatch.setenv("SHADOW_LOOP_MAX_DYNAMIC_EXTRA_INTERVAL_SECONDS", "40")

    assert loop._external_ingestion_extra_interval_seconds(str(tmp_path)) == 21


def test_runtime_training_pause_contract_uses_governor_and_override(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "runtime_throttle_control_latest.json").write_text(
        json.dumps(
            {
                "runtime_saturation_governor_v2": {
                    "training_policy": {
                        "training_paused": True,
                        "reason": "host_saturation_or_memory_pressure",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime_override = tmp_path / ".env.runtime_resource_guard_override"
    runtime_override.write_text("SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS=77\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (runtime_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["sleep_seconds"] == 77
    assert contract["reason"] == "host_saturation_or_memory_pressure"
    assert contract["runtime_training_paused"] is True


def test_runtime_training_pause_contract_hard_pauses_for_backlog_override(tmp_path, monkeypatch) -> None:
    storage_override = tmp_path / ".env.storage_pressure_override"
    storage_override.write_text(
        "\n".join(
            [
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=1",
                "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG=1",
                "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS=45",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (storage_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["sleep_seconds"] == 45
    assert contract["reason"] == "storage_backpressure_backlog"
    assert contract["runtime_training_paused"] is False
    assert contract["backlog_paused"] is True
    assert contract["training_paused_for_backlog"] is True
    assert contract["heavy_collectors_paused_for_backlog"] is True


def test_runtime_training_pause_contract_ignores_stale_env_when_storage_override_is_authoritative(
    tmp_path, monkeypatch
) -> None:
    storage_override = tmp_path / ".env.storage_pressure_override"
    storage_override.write_text("TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=0\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (storage_override,))
    monkeypatch.setenv("HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG", "1")
    monkeypatch.setenv("SHADOW_LOOP_PAUSED_FOR_BACKLOG", "1")
    monkeypatch.setenv("TRAINING_RUNTIME_GOVERNOR_MODE", "paused_for_backlog")
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is False
    assert contract["backlog_paused"] is False
    assert contract["training_paused_for_backlog"] is False
    assert contract["heavy_collectors_paused_for_backlog"] is False
    assert contract["governor_mode"] == ""


def test_paper_live_data_runtime_pause_bypass_allows_host_headroom_only(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "paper_400_ramp_latest.json").write_text(
        json.dumps({"ok": True, "armed": True}),
        encoding="utf-8",
    )
    (health / "runtime_throttle_control_latest.json").write_text(
        json.dumps(
            {
                "runtime_saturation_governor_v2": {
                    "paper_live_data_policy": {
                        "paper_execution_allowed": True,
                        "paper_execution_consumer_paused": False,
                        "protect_paper_execution_queue": True,
                        "protect_live_execution_read_only": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOP_BOT_PAPER_TRADING_ENABLED", "1")
    monkeypatch.delenv("PAPER_BROKER_BRIDGE_ENABLED", raising=False)
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")

    host_pause = {
        "paused": True,
        "reason": "runtime_host_headroom",
        "host_headroom_paused": True,
        "backlog_paused": False,
        "training_paused_for_backlog": False,
        "heavy_collectors_paused_for_backlog": False,
        "governor_mode": "micro_canary_only",
    }
    host_contract = loop._paper_live_data_runtime_pause_bypass_contract(
        str(tmp_path),
        broker="coinbase",
        profile="default",
        runtime_pause=host_pause,
    )
    assert host_contract["allowed"] is True
    assert host_contract["blockers"] == []
    assert (
        loop._paper_live_data_runtime_pause_bypass(
            str(tmp_path),
            broker="coinbase",
            profile="default",
            runtime_pause=host_pause,
        )
        is True
    )

    backlog_pause = {**host_pause, "backlog_paused": True, "reason": "storage_backpressure_backlog"}
    backlog_contract = loop._paper_live_data_runtime_pause_bypass_contract(
        str(tmp_path),
        broker="coinbase",
        profile="default",
        runtime_pause=backlog_pause,
    )
    assert backlog_contract["allowed"] is False
    assert "backlog_paused" in backlog_contract["blockers"]
    assert (
        loop._paper_live_data_runtime_pause_bypass(
            str(tmp_path),
            broker="coinbase",
            profile="default",
            runtime_pause=backlog_pause,
        )
        is False
    )


def test_external_interval_floor_can_be_disabled(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "runtime_throttle_control_latest.json").write_text(
        json.dumps({"throttle_profile": "protect_live", "compute_pressure_level": "high"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHADOW_LOOP_PRESSURE_INTERVAL_FLOOR_ENABLED", "0")

    assert loop._external_ingestion_extra_interval_seconds(str(tmp_path)) == 0
