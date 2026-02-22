from scripts.run_shadow_training_loop import _merge_symbol_groups, _parse_symbols


def test_parse_symbols_dedup_and_empty_filter() -> None:
    symbols = _parse_symbols(" SPY , QQQ,SPY , , IWM ")
    assert symbols == ["SPY", "QQQ", "IWM"]


def test_merge_symbol_groups_dedup() -> None:
    symbols = _merge_symbol_groups("SPY,QQQ", "QQQ,TLT", "TLT,GLD")
    assert symbols == ["SPY", "QQQ", "TLT", "GLD"]
