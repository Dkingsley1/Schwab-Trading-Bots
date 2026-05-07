from pathlib import Path

from scripts.ops import sleeve_ticker_universe_expansion as src


def test_sleeve_ticker_universe_expands_applicable_sleeves() -> None:
    payload = src.build_payload(Path("/tmp/project"))
    env = payload["env_overrides"]

    assert payload["overall_status"] == "ready"
    assert payload["symbol_counts"]["SHADOW_SYMBOLS_CORE"] >= 90
    assert payload["symbol_counts"]["SHADOW_SYMBOLS_DEFENSIVE"] >= 60
    assert payload["symbol_counts"]["COINBASE_WATCH_SYMBOLS"] >= 15
    assert "NVDA" in env["SHADOW_SYMBOLS_CORE"]
    assert "TLT" in env["BOND_SYMBOLS"]
    assert "BTC-USD" in env["COINBASE_WATCH_SYMBOLS"]
    assert "SLEEVE_TICKER_UNIVERSE_ENABLED" in env
    assert payload["safety_contract"]["adds_live_execution"] is False


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
