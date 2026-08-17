import json
from pathlib import Path

from scripts.ops import profitability_benchmark_hurdle as hurdle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_benchmark_hurdle_requires_candidate_bound_cash_and_passive_outperformance(tmp_path: Path) -> None:
    config = {
        "benchmark_hurdle": {
            "artifact": "governance/research/profitability_benchmark_hurdle_latest.json",
            "capture_artifact": "governance/research/profitability_benchmark_capture_latest.json",
            "series": "governance/research/profitability_benchmark_returns.jsonl",
            "minimum_common_days": 3,
            "minimum_excess_return_bps": 0.0,
            "maximum_drawdown_ratio_to_passive": 1.0,
            "cash_annual_rate": 0.04,
            "capture": {"symbol": "SPY"},
        }
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "health" / "profitability_independent_validator_latest.json",
        {
            "evidence_ready": True,
            "candidate_binding": {"candidate_id": "candidate-1", "bound": True},
            "recomputed": {
                "daily": [
                    {"day_utc": f"2026-08-0{day}", "active_return_bps": 10.0}
                    for day in range(1, 4)
                ]
            },
        },
    )
    series = tmp_path / "governance" / "research" / "profitability_benchmark_returns.jsonl"
    series.parent.mkdir(parents=True, exist_ok=True)
    series.write_text(
        "\n".join(
            json.dumps(
                {
                    "day_utc": f"2026-08-0{day}",
                    "candidate_id": "candidate-1",
                    "candidate_full_session": True,
                    "passive_return_bps": 1.0,
                    "cash_return_bps": 0.5,
                }
            )
            for day in range(1, 4)
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hurdle.build_payload(tmp_path, config_path=config_path)

    assert payload["evidence_ready"] is True
    assert payload["checks"] == {
        "minimum_common_days": True,
        "return_hurdle": True,
        "drawdown_hurdle": True,
    }

    series.write_text(
        json.dumps(
            {
                "day_utc": "2026-08-01",
                "candidate_id": "different-candidate",
                "candidate_full_session": True,
                "passive_return_bps": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rejected = hurdle.build_payload(tmp_path, config_path=config_path)

    assert rejected["evidence_ready"] is False
    assert rejected["rejected_candidate_row_count"] == 1
