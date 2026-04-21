import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.decay_monitor as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_decay_monitor_flags_weak_sleeves_and_negative_slope(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "ok": True,
            "history_daily_series": [
                {"day_utc": "20260405", "ending_net_pnl_total": 12.0, "change_vs_previous_day": 12.0},
                {"day_utc": "20260406", "ending_net_pnl_total": 4.0, "change_vs_previous_day": -8.0},
            ],
            "period_change_series": [{"label": "7D", "window_days": 7, "change": -3.0, "available_days": 2}],
            "sleeve_daily_series": {
                "default": [{"day_utc": "20260406", "ending_net_pnl_total": -2.0, "change_vs_previous_day": -2.0}]
            },
            "sleeve_latest": [
                {"profile": "default", "data_status": "current", "ending_net_pnl_total": -2.0, "win_rate": 0.33, "top_loss_causes": [{"cause": "spread"}]},
                {"profile": "dividend", "data_status": "current", "ending_net_pnl_total": 5.0, "win_rate": 0.67},
            ],
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False},
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "needs_work"
    assert payload["weak_sleeve_count"] == 1
    assert payload["pnl_slope"] == -8.0

