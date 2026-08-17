import json
from pathlib import Path

from scripts.ops import independent_fill_evidence_acquisition as acquisition
from scripts.ops import market_replay_fill_capture as replay


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _seed_candidate(project_root: Path) -> None:
    _write_json(
        project_root / "governance/runtime/production_candidate_state.json",
        {
            "candidate_id": "pc-test-g1",
            "generation": 1,
            "scope_windows_started_utc": {
                "execution": "2026-08-11T11:59:00+00:00",
                "data": "2026-08-11T11:59:00+00:00",
                "dependencies": "2026-08-11T11:59:00+00:00",
            },
        },
    )


def test_capture_uses_later_broker_quote_and_materializes_independent_evidence(tmp_path: Path) -> None:
    _seed_candidate(tmp_path)
    _write_jsonl(
        tmp_path / "exports/paper_broker_bridge/paper/paper_bridge_orders_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:00:00+00:00",
                "status": "PAPER_EXECUTED",
                "symbol": "SPY",
                "action": "BUY",
                "quantity": 2,
                "reference_price": 100.0,
                "expected_fill_price": 100.05,
                "source_broker": "schwab",
                "metadata": {"source_profile": "baseline", "snapshot_id": "snap-order"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "decision_explanations/baseline/decision_explanations_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:00:05+00:00",
                "symbol": "SPY",
                "schema_valid": True,
                "source_quality_label": "broker_native",
                "source_quality_score": 1.0,
                "source_broker": "schwab",
                "features": {"last_price": 100.5, "spread_bps": 10.0},
                "metadata": {"snapshot_id": "snap-order"},
            },
            {
                "timestamp_utc": "2026-08-11T12:01:00+00:00",
                "symbol": "SPY",
                "schema_valid": True,
                "source_quality_label": "broker_native",
                "source_quality_score": 0.99,
                "source_broker": "schwab",
                "source_provider": "schwab_market_data",
                "features": {"last_price": 101.0, "spread_bps": 20.0},
                "metadata": {"snapshot_id": "snap-later"},
            },
        ],
    )

    payload = replay.build_payload(tmp_path, apply=True)

    assert payload["overall_status"] == "ready"
    assert payload["capture_count"] == 1
    inbox = Path(payload["inbox_file"])
    row = json.loads(inbox.read_text(encoding="utf-8"))
    assert row["paper_fill_source"] == "market_replay_fill"
    assert row["observed_at_utc"] == "2026-08-11T12:01:00+00:00"
    assert row["fill_price"] == 101.101
    assert row["provenance"]["replay_dataset_id"]

    acquired = acquisition.build_payload(tmp_path, apply=True)
    assert acquired["candidate_eligible_ledger_records"] == 1
    assert acquired["conflict_count"] == 0


def test_replay_identity_is_stable_when_source_files_move(tmp_path: Path) -> None:
    row = {"timestamp_utc": "2026-08-11T12:00:00+00:00", "symbol": "SPY", "value": 1}
    first = replay._row_id(row, tmp_path / "hot.jsonl", 1, namespace="market_observation")
    second = replay._row_id(row, tmp_path / "cold.jsonl.gz", 900, namespace="market_observation")

    assert first == second


def test_capture_short_circuits_quote_scan_without_candidate_orders(tmp_path: Path, monkeypatch) -> None:
    _seed_candidate(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("quote scan should not run without candidate paper orders")

    monkeypatch.setattr(replay, "_observation_rows", fail_if_called)
    payload = replay.build_payload(tmp_path)

    assert payload["overall_status"] == "waiting_for_paper_orders"
    assert payload["market_observation_count"] == 0
    assert payload["capture_count"] == 0


def test_capture_rejects_model_or_low_quality_market_observations(tmp_path: Path) -> None:
    _seed_candidate(tmp_path)
    _write_jsonl(
        tmp_path / "exports/paper_broker_bridge/paper/paper_bridge_orders_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:00:00+00:00",
                "status": "PAPER_EXECUTED",
                "symbol": "QQQ",
                "action": "SELL",
                "quantity": 1,
                "reference_price": 500.0,
                "expected_fill_price": 499.9,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "decision_explanations/baseline/decision_explanations_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:01:00+00:00",
                "symbol": "QQQ",
                "source_quality_label": "model_derived",
                "source_quality_score": 1.0,
                "features": {"last_price": 499.0, "spread_bps": 10.0},
            },
            {
                "timestamp_utc": "2026-08-11T12:02:00+00:00",
                "symbol": "QQQ",
                "source_quality_label": "broker_native",
                "source_quality_score": 0.5,
                "features": {"last_price": 498.0, "spread_bps": 10.0},
            },
        ],
    )

    payload = replay.build_payload(tmp_path)

    assert payload["overall_status"] == "waiting_for_observations"
    assert payload["market_observation_count"] == 0
    assert payload["capture_count"] == 0


def test_capture_retains_matched_order_without_rescanning_observations(tmp_path: Path, monkeypatch) -> None:
    _seed_candidate(tmp_path)
    _write_jsonl(
        tmp_path / "exports/paper_broker_bridge/paper/paper_bridge_orders_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:00:00+00:00",
                "status": "PAPER_EXECUTED",
                "symbol": "SPY",
                "action": "BUY",
                "quantity": 1,
                "reference_price": 100.0,
                "expected_fill_price": 100.1,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "decision_explanations/baseline/decision_explanations_20260811.jsonl",
        [
            {
                "timestamp_utc": "2026-08-11T12:01:00+00:00",
                "symbol": "SPY",
                "schema_valid": True,
                "source_quality_label": "broker_native",
                "source_quality_score": 1.0,
                "features": {"last_price": 100.2, "spread_bps": 5.0},
            }
        ],
    )
    first = replay.build_payload(tmp_path, apply=True)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("matched orders must not trigger another observation scan")

    monkeypatch.setattr(replay, "_observation_rows", fail_if_called)
    second = replay.build_payload(tmp_path, apply=True)

    assert first["capture_count"] == 1
    assert second["capture_count"] == 1
    assert second["retained_capture_count"] == 1
    assert second["new_capture_count"] == 0
    assert second["pending_order_count"] == 0
    assert second["observation_source_files"] == []
