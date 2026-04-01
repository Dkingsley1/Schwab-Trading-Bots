import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.promotion_readiness_summary as readiness


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_promotion_readiness_summary_emits_targeted_retrain_hint(tmp_path: Path, monkeypatch, capsys) -> None:
    gate_path = tmp_path / "promotion_gate_latest.json"
    walk_forward_path = tmp_path / "walk_forward_latest.json"
    history_path = tmp_path / "promotion_readiness_history.jsonl"
    latest_path = tmp_path / "promotion_readiness_latest.json"
    fail_list_path = tmp_path / "promotion_fail_bots_latest.json"

    _write_json(
        gate_path,
        {
            "promote_ok": False,
            "coverage_ok": True,
            "considered_bots": 5,
            "failed_bots": 3,
            "fail_share": 0.6,
            "thresholds": {
                "max_fail_share": 0.25,
                "min_forward_mean": 0.53,
                "min_delta": -0.01,
                "min_runs_per_bot": 24,
                "min_considered_bots": 4,
            },
            "fail_examples": [
                {
                    "bot_id": "brain_refinery_v12_news_shocks",
                    "runs": 26,
                    "forward_mean": 0.60,
                    "delta": -0.04,
                    "failed_gates": {"delta": True},
                },
                {
                    "bot_id": "brain_refinery_v27_term_structure_vol",
                    "runs": 28,
                    "forward_mean": 0.50,
                    "delta": 0.0,
                    "failed_gates": {"forward_mean": True},
                },
                {
                    "bot_id": "brain_refinery_v48_position_1m_3m",
                    "runs": 27,
                    "forward_mean": 0.51,
                    "delta": 0.0,
                    "failed_gates": {"forward_mean": True},
                },
            ],
            "near_pass_examples": [
                {
                    "bot_id": "brain_refinery_v48_position_1m_3m",
                    "runs": 27,
                    "forward_mean": 0.51,
                    "delta": 0.0,
                    "failed_gates": {"forward_mean": True},
                    "near_pass_reason": "forward_mean_within_slack:0.01",
                }
            ],
        },
    )
    _write_json(
        walk_forward_path,
        {
            "bots": {
                "brain_refinery_v12_news_shocks": {"segment": "shock"},
                "brain_refinery_v27_term_structure_vol": {"segment": "shock"},
                "brain_refinery_v48_position_1m_3m": {"segment": "liquidity"},
            }
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion_readiness_summary.py",
            "--gate-file",
            str(gate_path),
            "--walk-forward-file",
            str(walk_forward_path),
            "--history-jsonl",
            str(history_path),
            "--latest-out",
            str(latest_path),
            "--fail-list-out",
            str(fail_list_path),
            "--json",
        ],
    )

    rc = readiness.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["recommended_retrain"]["include_bot_ids"][0] == "brain_refinery_v12_news_shocks"
    assert payload["recommended_retrain"]["regime_focus"].startswith("shock")
    assert payload["recommended_retrain"]["watchlist_bot_ids"] == ["brain_refinery_v48_position_1m_3m"]
    assert payload["failed_by_segment"]["shock"] == 2


def test_promotion_readiness_summary_preserves_zero_fail_share(tmp_path: Path, monkeypatch, capsys) -> None:
    gate_path = tmp_path / "promotion_gate_latest.json"
    walk_forward_path = tmp_path / "walk_forward_latest.json"
    history_path = tmp_path / "promotion_readiness_history.jsonl"
    latest_path = tmp_path / "promotion_readiness_latest.json"
    fail_list_path = tmp_path / "promotion_fail_bots_latest.json"

    _write_json(
        gate_path,
        {
            "promote_ok": True,
            "coverage_ok": True,
            "considered_bots": 4,
            "failed_bots": 0,
            "fail_share": 0.0,
            "thresholds": {
                "max_fail_share": 0.25,
                "min_forward_mean": 0.53,
                "min_delta": -0.01,
                "min_runs_per_bot": 12,
                "min_considered_bots": 4,
            },
            "fail_examples": [],
            "near_pass_examples": [],
        },
    )
    _write_json(walk_forward_path, {"bots": {}})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion_readiness_summary.py",
            "--gate-file",
            str(gate_path),
            "--walk-forward-file",
            str(walk_forward_path),
            "--history-jsonl",
            str(history_path),
            "--latest-out",
            str(latest_path),
            "--fail-list-out",
            str(fail_list_path),
            "--json",
        ],
    )

    rc = readiness.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["promote_ok"] is True
    assert payload["fail_share"] == 0.0
