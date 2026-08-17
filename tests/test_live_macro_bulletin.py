import importlib.util
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_macro_bulletin.py")
spec = importlib.util.spec_from_file_location("live_macro_bulletin", MODULE_PATH)
live_macro_bulletin = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live_macro_bulletin)


def test_build_live_macro_payload_company_template_is_symbol_scoped():
    payload = live_macro_bulletin.build_live_macro_payload(
        template="earnings_call",
        headline="AAPL earnings call",
        summary="Raised guidance and discussed margin expansion.",
        speaker="Tim Cook",
        source="Apple",
        symbols="AAPL",
        impact="high",
    )

    assert payload["template"] == "earnings_call"
    assert payload["broad_market"] is False
    assert payload["symbols"] == ["AAPL"]
    assert payload["shock_hint"] >= 0.76


def test_build_live_macro_payload_policy_testimony_defaults_to_macro_scope():
    payload = live_macro_bulletin.build_live_macro_payload(
        template="policy_testimony",
        headline="Treasury testimony",
        summary="Officials discussed issuance and market functioning.",
        speaker="Treasury Secretary",
        source="U.S. Treasury",
        impact="medium",
    )

    assert payload["template"] == "policy_testimony"
    assert payload["broad_market"] is True
    assert "TLT" in payload["symbols"]
    assert "XLF" in payload["symbols"]
