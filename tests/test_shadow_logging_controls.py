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

    pressure.write_text("LOG_API_CALLS=0\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    assert loop._dynamic_storage_flag("LOG_API_CALLS", True) is False

    storage.write_text("LOG_API_CALLS=1\n", encoding="utf-8")
    _reset_dynamic_override_cache()
    assert loop._dynamic_storage_flag("LOG_API_CALLS", True) is True


def test_log_api_call_and_loop_state_respect_dynamic_storage_flags(tmp_path, monkeypatch) -> None:
    pressure = tmp_path / ".env.storage_pressure_override"
    storage = tmp_path / ".env.storage_override"
    monkeypatch.setattr(loop, "STORAGE_PRESSURE_OVERRIDE_PATH", pressure)
    monkeypatch.setattr(loop, "STORAGE_OVERRIDE_PATH", storage)
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
