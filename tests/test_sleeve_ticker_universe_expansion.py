from pathlib import Path

from scripts.ops import sleeve_ticker_universe_expansion as src


def test_sleeve_ticker_universe_expands_applicable_sleeves() -> None:
    payload = src.build_payload(Path("/tmp/project"))
    env = payload["env_overrides"]

    assert payload["overall_status"] == "ready"
    assert payload["universe_version"] == "sleeve_ticker_universe_v2"
    assert payload["symbol_counts"]["SHADOW_SYMBOLS_CORE"] >= 170
    assert payload["symbol_counts"]["SHADOW_SYMBOLS_VOLATILE"] >= 60
    assert payload["symbol_counts"]["SHADOW_SYMBOLS_DEFENSIVE"] >= 100
    assert payload["symbol_counts"]["COINBASE_WATCH_SYMBOLS"] >= 30
    assert payload["unique_symbol_count"] == 1000
    assert payload["group_slot_count"] == sum(payload["symbol_counts"].values())
    assert "NVDA" in env["SHADOW_SYMBOLS_CORE"]
    assert "SPYM" in env["SHADOW_SYMBOLS_CORE"]
    assert "SPLG" not in env["SHADOW_SYMBOLS_CORE"]
    assert "PANW" in env["SHADOW_SYMBOLS_CORE"]
    assert "DELL" in env["SHADOW_SYMBOLS_CORE"]
    assert "SNDK" in env["SHADOW_SYMBOLS_CORE"]
    assert "VRT" in env["SHADOW_SYMBOLS_CORE"]
    assert "CRWV" in env["SHADOW_SYMBOLS_CORE"]
    assert "MTUM" in env["SHADOW_SYMBOLS_CORE"]
    assert "BRK.B" in env["SHADOW_SYMBOLS_CORE"]
    assert "NDAQ" in env["SHADOW_SYMBOLS_CORE"]
    assert "PCG" in env["SHADOW_SYMBOLS_DEFENSIVE"]
    assert "EWI" in env["SHADOW_SYMBOLS_COMMOD_FX_INTL"]
    assert "QUAL" in env["LONG_TERM_SECTOR_SYMBOLS"]
    assert "KBE" in env["LONG_TERM_SECTOR_SYMBOLS"]
    assert "VGT" in env["LONG_TERM_SECTOR_SYMBOLS"]
    assert "BITX" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "HUT" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "HIVE" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "IREN" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "APLD" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "VXX" in env["SHADOW_SYMBOLS_VOLATILE"]
    assert "TLT" in env["BOND_SYMBOLS"]
    assert "SHV" in env["BOND_SYMBOLS"]
    assert "BKLN" in env["BOND_SYMBOLS"]
    assert "JAAA" in env["BOND_CONTEXT_SYMBOLS"]
    assert "BTC-USD" in env["COINBASE_WATCH_SYMBOLS"]
    assert "SUI-USD" in env["COINBASE_WATCH_SYMBOLS"]
    assert len(env["TICKER_UNIVERSE_ALL_SYMBOLS"].split(",")) == 1000
    assert len(env["TICKER_UNIVERSE_HOT_SYMBOLS"].split(",")) == 150
    assert len(env["TICKER_UNIVERSE_STANDARD_SYMBOLS"].split(",")) == 500
    assert len(env["TICKER_UNIVERSE_SLOW_SYMBOLS"].split(",")) == 500
    assert "ACWI" in env["TICKER_UNIVERSE_ALL_SYMBOLS"]
    assert "SNDK" in env["TICKER_UNIVERSE_ALL_SYMBOLS"]
    assert env["TICKER_NEWS_MAX_SYMBOLS"] == "1000"
    assert env["TICKER_NEWS_LIMIT_PER_SYMBOL"] == "4"
    assert env["FREE_EQUITY_REFERENCE_MAX_SYMBOLS"] == "240"
    assert env["TICKER_UNIVERSE_STORAGE_PROFILE"] == "tiered_1000_guarded"
    assert "SLEEVE_TICKER_UNIVERSE_ENABLED" in env
    assert payload["data_intake_routes"]["market_micro_context"]["mode"] == "runtime_env_MARKET_MICRO_SYMBOLS_bounded_500"
    assert payload["data_intake_routes"]["ticker_news_context"]["mode"] == "runtime_env_TICKER_NEWS_MAX_SYMBOLS_1000"
    assert payload["expansion_sections"]["ai_power_data_center_infrastructure"] == [
        "VRT",
        "GEV",
        "CEG",
        "NRG",
        "VST",
        "PWR",
        "ALAB",
        "CRWV",
        "CLS",
        "JBL",
        "SANM",
        "COHR",
    ]
    assert len(payload["expansion_sections"]["liquid_us_equities"]) == 251
    assert payload["tier_contract"]["target_symbol_count"] == 1000
    assert payload["tier_contract"]["slow_symbol_count"] == 500
    assert payload["storage_optimization_env"]["RETENTION_STALE_PURGE_MAX_GB"] == "8"
    assert payload["safety_contract"]["adds_live_execution"] is False
    assert payload["safety_contract"]["unique_symbol_target"] == 1000
    assert payload["safety_contract"]["slow_tier_deferred_on_storage_pressure"] is True


def test_sleeve_ticker_universe_apply_writes_override_and_health(tmp_path: Path) -> None:
    out = tmp_path / "governance" / "health" / "sleeve_ticker_universe_latest.json"
    override = tmp_path / "config" / ".env.sleeve_ticker_universe_override"
    payload = src.apply_payload(tmp_path, src.build_payload(tmp_path), out_path=out, override_path=override)
    text = override.read_text(encoding="utf-8")

    assert payload["apply_result"]["applied"] is True
    assert out.exists()
    assert "SHADOW_SYMBOLS_CORE=" in text
    assert "COINBASE_WEBSOCKET_SYMBOLS=" in text
    assert "SLEEVE_TICKER_UNIVERSE_ENABLED=1" in text
