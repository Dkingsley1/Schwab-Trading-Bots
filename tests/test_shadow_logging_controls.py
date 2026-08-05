import inspect
import json
from datetime import datetime, timedelta, timezone
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


def test_collection_duty_cycle_extends_busy_loop_without_restart(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_dynamic_storage_overrides",
        lambda: {
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.20",
            "SHADOW_LOOP_DUTY_CYCLE_MAX_INTERVAL_SECONDS": "900",
        },
    )

    contract = loop._collection_duty_cycle_contract(loop_seconds=10.0, interval_seconds=30.0)

    assert contract["active"] is True
    assert contract["applied"] is True
    assert contract["effective_ratio"] == 0.20
    assert contract["target_cycle_seconds"] == 50.0
    assert contract["sleep_seconds"] == 40.0


def test_collection_duty_cycle_disabled_preserves_normal_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_dynamic_storage_overrides",
        lambda: {
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "0",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.20",
        },
    )

    contract = loop._collection_duty_cycle_contract(loop_seconds=10.0, interval_seconds=30.0)

    assert contract["active"] is False
    assert contract["applied"] is False
    assert contract["sleep_seconds"] == 20.0


def test_collection_duty_cycle_bounds_bad_ratio_and_max_cycle(monkeypatch) -> None:
    overrides = {
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "nan",
        "SHADOW_LOOP_DUTY_CYCLE_MAX_INTERVAL_SECONDS": "60",
    }
    monkeypatch.setattr(loop, "_dynamic_storage_overrides", lambda: overrides)

    malformed = loop._collection_duty_cycle_contract(loop_seconds=20.0, interval_seconds=30.0)
    assert malformed["parse_error"] is True
    assert malformed["effective_ratio"] == 0.16
    assert malformed["target_cycle_seconds"] == 60.0
    assert malformed["sleep_seconds"] == 40.0

    overrides["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] = "0"
    bounded = loop._collection_duty_cycle_contract(loop_seconds=1.0, interval_seconds=5.0)
    assert bounded["effective_ratio"] == 0.05
    assert bounded["target_cycle_seconds"] == 20.0


def test_shadow_loop_connects_duty_cycle_to_sleep_and_telemetry() -> None:
    source = inspect.getsource(loop.run_loop)

    assert "_collection_duty_cycle_contract(" in source
    assert '"collector_duty_cycle": duty_cycle' in source
    assert 'sleep_s = float(duty_cycle["sleep_seconds"])' in source


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


def test_bulk_risk_attribution_keeps_one_canonical_file_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        loop,
        "_dynamic_storage_overrides",
        lambda: {
            "CHANNEL_LOG_PRIMARY_MODE": "channel",
            "RISK_CHANNEL_MIRROR_ENABLED": "0",
        },
    )

    path = str(tmp_path / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260802.jsonl")
    primary, mirrors = loop._resolve_hot_channel_write_targets(path, channel="risk")

    assert primary == path
    assert mirrors == []


def test_shadow_broker_context_precedes_generic_data_broker(monkeypatch) -> None:
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("SHADOW_BROKER", "coinbase")
    monkeypatch.setenv("SHADOW_DOMAIN", "crypto")

    ctx = loop.build_shadow_context()

    assert ctx.broker == "coinbase"
    assert ctx.domain == "crypto"


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


def test_runtime_pause_catches_new_raw_backlog_before_storage_refresh(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": (now - timedelta(minutes=5)).isoformat(),
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0},
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "pending_lines": 50000,
                "pending_lines_total": 625000,
            }
        ),
        encoding="utf-8",
    )
    empty_override = tmp_path / "empty.env"
    empty_override.write_text("SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES=15000\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (empty_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["backlog_paused"] is True
    assert contract["reason"] == "fresh_raw_backpressure_newer_than_storage_control"
    assert contract["fresh_backlog_pause"]["raw_total_pending_lines"] == 625000


def test_fresh_managed_storage_clear_prevents_raw_false_positive(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {
                    "overlay_adjusted": True,
                    "total_pending_lines": 0,
                    "core_pending_lines": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": (now - timedelta(seconds=5)).isoformat(),
                "pending_lines": 50000,
                "pending_lines_total": 625000,
            }
        ),
        encoding="utf-8",
    )
    empty_override = tmp_path / "empty.env"
    empty_override.write_text("SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES=15000\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (empty_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is False
    assert contract["backlog_paused"] is False
    assert contract["fresh_backlog_pause"]["source"] == "none"


def test_stale_storage_pressure_stays_latched_until_fresh_clear(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": (now - timedelta(minutes=10)).isoformat(),
                "overall_status": "degraded",
                "severity": "critical",
                "backpressure": {"total_pending_lines": 50000, "core_pending_lines": 50000},
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "pending_lines": 0,
                "pending_lines_total": 0,
            }
        ),
        encoding="utf-8",
    )
    empty_override = tmp_path / "empty.env"
    empty_override.write_text("SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES=15000\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (empty_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["backlog_paused"] is True
    assert contract["reason"] == "stale_storage_pressure_requires_fresh_clear"
    assert contract["fresh_backlog_pause"]["stale_pressure_latched"] is True
    assert contract["fresh_backlog_pause"]["clear_confirmed"] is False


def test_stale_backlog_controls_fail_closed(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0},
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "pending_lines": 0,
                "pending_lines_total": 0,
            }
        ),
        encoding="utf-8",
    )
    empty_override = tmp_path / "empty.env"
    empty_override.write_text("SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES=15000\n", encoding="utf-8")
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (empty_override,))
    _reset_dynamic_override_cache()

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["reason"] == "backlog_control_evidence_stale"
    assert contract["fresh_backlog_pause"]["control_evidence_stale"] is True
    assert contract["fresh_backlog_pause"]["clear_confirmed"] is False


def test_collector_refreshes_due_backlog_truth_before_admission(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    scripts = tmp_path / "scripts"
    health.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "ingestion_backpressure_guard.py").write_text("# test guard\n", encoding="utf-8")
    stale = datetime.now(timezone.utc) - timedelta(minutes=2)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": stale.isoformat(),
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0},
            }
        ),
        encoding="utf-8",
    )
    raw_path = health / "ingestion_backpressure_latest.json"
    raw_path.write_text(
        json.dumps(
            {
                "timestamp_utc": stale.isoformat(),
                "pending_lines": 0,
                "pending_lines_total": 0,
            }
        ),
        encoding="utf-8",
    )
    empty_override = tmp_path / "empty.env"
    empty_override.write_text(
        "SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES=15000\n"
        "SHADOW_LOOP_BACKLOG_REFRESH_MAX_AGE_SECONDS=5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "DYNAMIC_STORAGE_OVERRIDE_PATHS", (empty_override,))
    _reset_dynamic_override_cache()

    def _refresh(*_args, **_kwargs):
        raw_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "pending_lines": 22000,
                    "pending_lines_total": 22000,
                }
            ),
            encoding="utf-8",
        )
        return loop.subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(loop.subprocess, "run", _refresh)

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["reason"] == "fresh_raw_backpressure_newer_than_storage_control"
    assert contract["fresh_backlog_pause"]["refresh_contract"]["attempted"] is True
    assert contract["fresh_backlog_pause"]["refresh_contract"]["evidence_fresh"] is True
    assert contract["fresh_backlog_pause"]["raw_total_pending_lines"] == 22000


def test_collector_fails_closed_while_backlog_refresh_is_in_progress(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0},
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_backpressure_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "pending_lines": 0,
                "pending_lines_total": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loop,
        "_refresh_backlog_evidence_if_due",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "available": True,
            "refresh_due": True,
            "evidence_fresh": False,
            "in_progress": True,
        },
    )

    contract = loop._runtime_training_pause_contract(str(tmp_path))

    assert contract["paused"] is True
    assert contract["reason"] == "backlog_evidence_refresh_pending"
    assert contract["fresh_backlog_pause"]["refresh_pending"] is True


def test_collector_resume_stagger_is_bounded_and_stable() -> None:
    first = loop._collector_resume_stagger_seconds(
        broker="schwab",
        profile="dividend",
        instance="dividend_equities_schwab",
        max_seconds=180,
    )
    second = loop._collector_resume_stagger_seconds(
        broker="schwab",
        profile="dividend",
        instance="dividend_equities_schwab",
        max_seconds=180,
    )
    other = loop._collector_resume_stagger_seconds(
        broker="schwab",
        profile="bond",
        instance="bond_equities_schwab",
        max_seconds=180,
    )

    assert 0 <= first <= 180
    assert first == second
    assert 0 <= other <= 180


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
