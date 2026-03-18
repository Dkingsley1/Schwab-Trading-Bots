import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "strategy_attribution_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("strategy_attribution_report", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load strategy_attribution_report module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_strategy_attribution_report_rolls_up_lane_layer_and_symbols(tmp_path):
    module = _load_module()
    lane_dir = tmp_path / "governance" / "shadow_intraday_aggressive_equities"
    lane_dir.mkdir(parents=True)
    path = lane_dir / "shadow_pnl_attribution_20260318.jsonl"
    rows = [
        {
            "timestamp_utc": "2026-03-18T14:30:00+00:00",
            "symbol": "NVDA",
            "bot_id": "brain_refinery_v56_meta_ranker",
            "layer": "grand_master",
            "action": "BUY",
            "return_1m": 0.012,
            "pnl_proxy": 0.011,
        },
        {
            "timestamp_utc": "2026-03-18T14:31:00+00:00",
            "symbol": "AAPL",
            "bot_id": "brain_refinery_v64_regime_router_layer",
            "layer": "sub_bot",
            "action": "SELL",
            "return_1m": -0.006,
            "pnl_proxy": 0.004,
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

    payload = module.build_strategy_attribution_report(tmp_path, day="20260318")

    assert payload["ok"] is True
    assert payload["row_count"] == 2
    assert payload["file_count"] == 1
    assert payload["top_lane"] == "shadow_intraday_aggressive_equities"
    assert payload["top_layer"] == "grand_master"
    assert payload["total_pnl_proxy"] == 0.015
    assert payload["by_lane"][0]["samples"] == 2
    assert payload["top_positive_symbols"][0]["symbol"] == "NVDA"

