import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "post_trade_analysis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("post_trade_analysis", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load post_trade_analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_post_trade_analysis_combines_attribution_runtime_calibration_and_softguard(tmp_path):
    module = _load_module()

    lane_dir = tmp_path / "governance" / "shadow_intraday_aggressive_equities"
    lane_dir.mkdir(parents=True)
    attribution_path = lane_dir / "shadow_pnl_attribution_20260318.jsonl"
    attribution_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T14:30:00+00:00",
                "symbol": "NVDA",
                "bot_id": "brain_refinery_v56_meta_ranker",
                "layer": "grand_master",
                "action": "BUY",
                "return_1m": 0.01,
                "pnl_proxy": 0.008,
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    softguard_dir = tmp_path / "governance" / "events"
    softguard_dir.mkdir(parents=True)
    softguard_path = softguard_dir / "live_softguard_20260318.jsonl"
    softguard_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T12:32:02+00:00",
                "event": "global_halt_set",
                "reason": "softguard_api_circuit_opened",
                "mode_label": "shadow_intraday_aggressive_equities",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_runner(cmd, cwd):
        script_name = Path(cmd[1]).name
        if script_name == "paper_execution_calibration_report.py":
            return (
                0,
                {
                    "ok": True,
                    "samples": 4,
                    "metrics": {"mae_bps": 12.5, "p95_bps": 20.0},
                    "thresholds": {"max_mae_bps": 35.0},
                },
                "",
            )
        if script_name == "daily_runtime_summary.py":
            return (
                0,
                {
                    "decision": {"rows": 18, "stale_windows": 1},
                    "watchdog": {"restarts": 2},
                },
                "",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    payload = module.build_post_trade_analysis(
        tmp_path,
        day="20260318",
        hours=24,
        runner=fake_runner,
    )

    assert payload["ok"] is True
    assert payload["summary"]["top_lane"] == "shadow_intraday_aggressive_equities"
    assert payload["summary"]["paper_mae_bps"] == 12.5
    assert payload["summary"]["decision_rows"] == 18
    assert payload["summary"]["global_halt_events"] == 1
    assert "Softguard logged 1 halt events" in payload["assessment"][3]


def test_build_post_trade_analysis_reuses_fresh_strategy_cache(tmp_path):
    module = _load_module()

    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True)
    strategy_cache = {
        "timestamp_utc": module._utc_now(),
        "schema_version": 2,
        "ok": True,
        "day": "20260318",
        "row_count": 7,
        "file_count": 1,
        "latest_event_timestamp_utc": "2026-03-18T14:30:00+00:00",
        "total_pnl_proxy": 1.25,
        "top_lane": "cached_lane",
        "top_layer": "cached_layer",
        "action_counts": {"BUY": 7},
        "by_lane": [{"lane": "cached_lane", "samples": 7, "pnl_proxy_sum": 1.25}],
        "by_layer": [],
        "top_positive_symbols": [],
        "top_negative_symbols": [],
        "top_positive_bots": [],
        "top_negative_bots": [],
        "processing": {"mode": "incremental"},
    }
    (health_dir / "strategy_attribution_latest.json").write_text(json.dumps(strategy_cache), encoding="utf-8")

    def fake_runner(cmd, cwd):
        script_name = Path(cmd[1]).name
        if script_name == "paper_execution_calibration_report.py":
            return 0, {"ok": True, "samples": 1, "metrics": {"mae_bps": 1.0, "p95_bps": 1.0}}, ""
        if script_name == "daily_runtime_summary.py":
            return 0, {"decision": {"rows": 1, "stale_windows": 0}, "watchdog": {"restarts": 0}}, ""
        raise AssertionError(f"unexpected command: {cmd}")

    payload = module.build_post_trade_analysis(tmp_path, day="20260318", hours=24, runner=fake_runner)

    assert payload["summary"]["top_lane"] == "cached_lane"
    assert payload["strategy_attribution"]["row_count"] == 7
    assert payload["sources"]["strategy_attribution_cache_used"] is True
