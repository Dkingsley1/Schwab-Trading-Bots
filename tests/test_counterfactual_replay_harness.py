import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.counterfactual_replay_harness as harness


def test_counterfactual_replay_harness_builds_candidates(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": "2026-04-01T14:00:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "model_score": 0.71,
            "threshold": 0.60,
            "tradeability_score": 0.50,
            "allocation_conflict_norm": 0.10,
            "realized_pnl_total": 1.0,
            "unrealized_pnl_total": 0.0,
        },
        {
            "timestamp_utc": "2026-04-01T14:05:00+00:00",
            "symbol": "MSFT",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "model_score": 0.58,
            "threshold": 0.60,
            "tradeability_score": 0.20,
            "allocation_conflict_norm": 0.80,
            "realized_pnl_total": -1.0,
            "unrealized_pnl_total": 0.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["ok"] is True
    assert "intraday_aggressive" in payload["profiles_reviewed"]
    assert payload["candidate_count"] > 0
    assert payload["top_candidates"]
    assert payload["processing"]["mode"] == "rebuild"


def test_counterfactual_replay_harness_reuses_state_incrementally(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file = project_root / "governance" / "health" / "counterfactual_replay_state.json"
    log_path = log_dir / "paper_bridge_orders_20260401.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "metadata": {"source_profile": "intraday_aggressive"},
                "model_score": 0.72,
                "threshold": 0.60,
                "tradeability_score": 0.60,
                "allocation_conflict_norm": 0.10,
                "realized_pnl_total": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)
    assert first["processing"]["mode"] == "rebuild"
    assert first["processing"]["full_files"] == 1

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": "2026-04-01T14:05:00+00:00",
                    "symbol": "MSFT",
                    "metadata": {"source_profile": "intraday_aggressive"},
                    "model_score": 0.74,
                    "threshold": 0.60,
                    "tradeability_score": 0.65,
                    "allocation_conflict_norm": 0.05,
                    "realized_pnl_total": 2.0,
                }
            )
            + "\n"
        )

    second = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)
    assert second["processing"]["mode"] == "incremental"
    assert second["processing"]["incremental_files"] == 1
    assert second["processing"]["row_buffer_size"] == 2
