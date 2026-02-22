from scripts.shadow_preflight import _parse_symbols


def test_preflight_parse_symbols() -> None:
    assert _parse_symbols("SPY, QQQ, ,IWM") == ["SPY", "QQQ", "IWM"]
