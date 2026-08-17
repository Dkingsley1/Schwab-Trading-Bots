import gzip
import json
from pathlib import Path

from scripts.ops import profitability_independent_validator as validator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_independent_validator_reconciles_canonical_and_bridge_mirror_once(tmp_path: Path) -> None:
    config = json.loads(validator.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / validator.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "scope_windows_started_utc": {"execution": "2026-08-06T20:00:00+00:00"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_cutoff_utc": "2026-08-06T20:00:00+00:00",
                "evidence_through_utc": "2026-08-06T22:00:00+00:00",
            },
            "post_cost_expectancy": {
                "sample_count": 1,
                "total_post_cost_pnl_delta": 2.0,
                "execution_notional_total": 100.0,
                "max_cumulative_drawdown_post_cost_pnl": 0.0,
            },
        },
    )
    common = {
        "symbol": "SPY",
        "action": "BUY",
        "decision_id": "decision-1",
        "paper_book_id": "book-1",
        "paper_pnl_schema_version": 2,
        "post_cost_pnl_delta": 2.0,
        "post_cost_return_bps": 200.0,
        "execution_notional": 100.0,
        "expected_execution_cost_amount": 0.1,
    }
    trade_path = tmp_path / "exports" / "trade_logs" / "paper" / "paper_trades_paper.jsonl"
    bridge_path = tmp_path / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260806.jsonl"
    trade_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    trade_path.write_text(
        json.dumps({**common, "timestamp_utc": "2026-08-06T21:00:00+00:00", "message_id": "trade"}) + "\n",
        encoding="utf-8",
    )
    bridge_path.write_text(
        json.dumps({**common, "timestamp_utc": "2026-08-06T21:00:00.001000+00:00", "message_id": "bridge"}) + "\n",
        encoding="utf-8",
    )

    payload = validator.build_payload(tmp_path, config_path=config_path)

    assert payload["evidence_ready"] is True
    assert payload["recomputed"]["sample_count"] == 1
    assert payload["scan"]["duplicate_rows"] == 1
    assert all(payload["comparisons"].values())
    assert payload["risk_of_ruin"]["available"] is False
    assert payload["control_contract"]["paper_report_snapshot_watermark_enforced"] is True


def test_independent_validator_fails_closed_on_report_mismatch(tmp_path: Path) -> None:
    config = json.loads(validator.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["independent_validator"]["source_paths"] = ["paper.jsonl"]
    config["independent_validator"]["source_globs"] = []
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "scope_windows_started_utc": {"execution": "2026-08-06T20:00:00+00:00"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_cutoff_utc": "2026-08-06T20:00:00+00:00",
                "evidence_through_utc": "2026-08-06T22:00:00+00:00",
            },
            "post_cost_expectancy": {
                "sample_count": 2,
                "total_post_cost_pnl_delta": 4.0,
                "execution_notional_total": 200.0,
                "max_cumulative_drawdown_post_cost_pnl": 0.0,
            }
        },
    )
    (tmp_path / "paper.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-06T21:00:00+00:00",
                "decision_id": "decision-1",
                "paper_book_id": "book-1",
                "post_cost_pnl_delta": 2.0,
                "post_cost_return_bps": 200.0,
                "execution_notional": 100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = validator.build_payload(tmp_path, config_path=config_path)

    assert payload["evidence_ready"] is False
    assert "sample_count" in payload["blockers"]


def test_independent_validator_normalizes_empty_candidate_window_without_granting_evidence(
    tmp_path: Path,
) -> None:
    config = json.loads(validator.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["independent_validator"]["source_paths"] = ["paper.jsonl"]
    config["independent_validator"]["source_globs"] = []
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "scope_windows_started_utc": {"execution": "2026-08-06T20:00:00+00:00"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_cutoff_utc": "2026-08-06T20:00:00+00:00",
                "evidence_through_utc": "2026-08-06T22:00:00+00:00",
            },
            "post_cost_expectancy": {
                "status": "no_schema_v2_trade_deltas",
                "sample_count": 0,
            },
        },
    )
    (tmp_path / "paper.jsonl").write_text("", encoding="utf-8")

    payload = validator.build_payload(tmp_path, config_path=config_path)

    assert payload["overall_status"] == "evidence_pending"
    assert payload["evidence_ready"] is False
    assert payload["reported_empty_window_normalized"] is True
    assert all(payload["comparisons"].values())
    assert payload["blockers"] == ["candidate_bound_post_cost_rows_pending"]


def test_independent_validator_defers_rows_after_primary_report_watermark(tmp_path: Path) -> None:
    config = json.loads(validator.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["independent_validator"]["source_paths"] = ["paper.jsonl"]
    config["independent_validator"]["source_globs"] = []
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "scope_windows_started_utc": {"execution": "2026-08-06T20:00:00+00:00"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_cutoff_utc": "2026-08-06T20:00:00+00:00",
                "evidence_through_utc": "2026-08-06T21:30:00+00:00",
            },
            "post_cost_expectancy": {
                "sample_count": 1,
                "total_post_cost_pnl_delta": 2.0,
                "execution_notional_total": 100.0,
                "max_cumulative_drawdown_post_cost_pnl": 0.0,
            },
        },
    )
    rows = [
        {
            "timestamp_utc": "2026-08-06T21:00:00+00:00",
            "decision_id": "included",
            "paper_pnl_schema_version": 2,
            "post_cost_pnl_delta": 2.0,
            "post_cost_return_bps": 200.0,
            "execution_notional": 100.0,
        },
        {
            "timestamp_utc": "2026-08-06T22:00:00+00:00",
            "decision_id": "next-refresh",
            "paper_pnl_schema_version": 2,
            "post_cost_pnl_delta": 3.0,
            "post_cost_return_bps": 300.0,
            "execution_notional": 100.0,
        },
    ]
    (tmp_path / "paper.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = validator.build_payload(tmp_path, config_path=config_path)

    assert payload["evidence_ready"] is True
    assert payload["recomputed"]["sample_count"] == 1
    assert payload["scan"]["post_snapshot_rows"] == 1
    assert all(payload["comparisons"].values())
    assert payload["candidate_binding"]["evidence_through_utc"] == "2026-08-06T21:30:00+00:00"


def test_independent_validator_reconciles_external_conflict_snapshot_and_excludes_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_root = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    config = json.loads(validator.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["independent_validator"]["source_paths"] = []
    config["independent_validator"]["source_globs"] = []
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "scope_windows_started_utc": {"execution": "2026-08-06T20:00:00+00:00"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_cutoff_utc": "2026-08-06T20:00:00+00:00",
                "evidence_through_utc": "2026-08-06T22:00:00+00:00",
            },
            "post_cost_expectancy": {
                "sample_count": 1,
                "total_post_cost_pnl_delta": 2.0,
                "execution_notional_total": 100.0,
                "max_cumulative_drawdown_post_cost_pnl": 0.0,
            },
        },
    )
    path = (
        external_root
        / "exports"
        / "trade_logs"
        / "paper"
        / "paper_trades_paper.jsonl.gz.local_fallback.1"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    real_row = {
        "timestamp_utc": "2026-08-06T21:00:00+00:00",
        "decision_id": "real",
        "paper_pnl_schema_version": 2,
        "post_cost_pnl_delta": 2.0,
        "post_cost_return_bps": 200.0,
        "execution_notional": 100.0,
    }
    calibration_row = {
        **real_row,
        "decision_id": "calibration",
        "post_cost_pnl_delta": 500.0,
        "independent_fill_evidence": True,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(real_row) + "\n")
        handle.write(json.dumps(calibration_row) + "\n")

    payload = validator.build_payload(tmp_path, config_path=config_path)

    assert payload["evidence_ready"] is True
    assert payload["source_paths"] == [str(path)]
    assert payload["recomputed"]["sample_count"] == 1
    assert payload["scan"]["calibration_rows_excluded"] == 1
    assert payload["control_contract"]["active_local_fallback_and_external_trade_roots_reconciled"] is True
