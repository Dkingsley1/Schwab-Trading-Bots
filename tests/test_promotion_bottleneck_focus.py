import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import promotion_bottleneck_focus as focus


def test_promotion_bottleneck_focus_includes_training_failures_and_weak_sleeves(tmp_path, monkeypatch) -> None:
    readiness = tmp_path / "promotion_readiness_latest.json"
    history = tmp_path / "promotion_readiness_history.jsonl"
    diagnostics = tmp_path / "canary_diagnostics_latest.json"
    regime = tmp_path / "regime_segmented_latest.json"
    training = tmp_path / "training_success_latest.json"
    paper = tmp_path / "paper_performance_latest.json"
    out = tmp_path / "promotion_bottleneck_latest.json"

    readiness.write_text(
        json.dumps(
            {
                "promote_ok": False,
                "fail_share": 0.6,
                "max_fail_share": 0.25,
                "readiness_margin": -0.35,
                "failed_by_segment": {"shock": 4, "mean_revert": 2},
            }
        ),
        encoding="utf-8",
    )
    history.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_utc": "2026-03-30T20:00:00+00:00", "fail_share": 0.8}),
                json.dumps({"timestamp_utc": "2026-03-31T20:00:00+00:00", "fail_share": 0.6}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics.write_text(
        json.dumps({"top_failing_bots": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "fail_days": 8}]}),
        encoding="utf-8",
    )
    regime.write_text(
        json.dumps({"segments": {"shock": {"pass_rate": 0.2}, "mean_revert": {"pass_rate": 0.4}}}),
        encoding="utf-8",
    )
    training.write_text(
        json.dumps(
            {
                "failure_details": [
                    {
                        "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                        "reason": "RuntimeError: runtime_training_quality_guard_failed run_tag=brain_refinery_v43_intraday_ultrafast_proxy long_precision=0.0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    paper.write_text(
        json.dumps(
            {
                "sleeve_latest": [
                    {
                        "profile": "intraday_aggressive",
                        "ending_net_pnl_total": -14.348,
                        "win_rate": 0.25,
                        "winning_strategy_count": 2,
                        "losing_strategy_count": 6,
                        "flat_strategy_count": 3,
                        "top_losing_strategies": [
                            {
                                "strategy": "paper_mirror::brain_refinery_v43_intraday_ultrafast_proxy",
                                "ending_net_pnl_total": -2.93,
                            }
                        ],
                        "top_winning_strategies": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion_bottleneck_focus.py",
            "--readiness-file",
            str(readiness),
            "--history-jsonl",
            str(history),
            "--diagnostics-file",
            str(diagnostics),
            "--regime-file",
            str(regime),
            "--training-success-file",
            str(training),
            "--paper-performance-file",
            str(paper),
            "--out-file",
            str(out),
        ],
    )

    rc = focus.main()
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["latest_training_failures"][0]["bot_id"] == "brain_refinery_v43_intraday_ultrafast_proxy"
    assert payload["latest_training_failures"][0]["observed_live_sleeves"] == ["intraday_aggressive"]
    assert payload["weak_sleeves"][0]["profile"] == "intraday_aggressive"
    assert payload["recommended_retrain_profile"]["RETRAIN_INCLUDE_BOT_IDS"] == "brain_refinery_v43_intraday_ultrafast_proxy"
    assert payload["recommended_retrain_profile"]["RETRAIN_SKIP_MASTER_UPDATE"] is True
