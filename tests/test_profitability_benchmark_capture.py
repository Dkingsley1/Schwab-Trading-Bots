import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import profitability_benchmark_capture as capture


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_benchmark_capture_appends_one_immutable_candidate_day(tmp_path: Path) -> None:
    config = json.loads(capture.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / capture.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-1", "generation": 1},
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {"profitability_evidence_window": {"candidate_cutoff_utc": "2026-08-06T12:00:00+00:00"}},
    )
    source = tmp_path / "governance" / "shadow_default_equities" / "master_control_20260806.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    first = {
        "timestamp_utc": "2026-08-06T20:10:00+00:00",
        "symbol": "SPY",
        "broker": "schwab",
        "market": {"last_price": 510.0, "prev_close": 500.0},
        "source_broker": "schwab",
        "source_provider": "schwab",
        "source_quality_label": "broker_native",
        "source_quality_score": 0.95,
    }
    source.write_text(json.dumps(first) + "\n", encoding="utf-8")
    now = datetime(2026, 8, 6, 21, 0, tzinfo=timezone.utc)

    payload = capture.build_payload(tmp_path, config_path=config_path, apply=True, now=now)
    series_path = Path(payload["series_path"])
    first_series = series_path.read_text(encoding="utf-8")

    assert payload["appended_days"] == ["2026-08-06"]
    assert payload["candidate_day_count"] == 1
    row = json.loads(first_series)
    assert row["passive_return_bps"] == 200.0
    assert row["point_in_time_immutable"] is True
    assert row["candidate_full_session"] is True

    later = {
        **first,
        "timestamp_utc": "2026-08-06T20:30:00+00:00",
        "market": {"last_price": 515.0, "prev_close": 500.0},
    }
    source.write_text(json.dumps(first) + "\n" + json.dumps(later) + "\n", encoding="utf-8")
    rerun = capture.build_payload(tmp_path, config_path=config_path, apply=True, now=now)

    assert rerun["appended_days"] == []
    assert series_path.read_text(encoding="utf-8") == first_series


def test_benchmark_capture_does_not_write_before_close_boundary(tmp_path: Path) -> None:
    config = json.loads(capture.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / capture.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-1", "generation": 1},
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {"profitability_evidence_window": {"candidate_cutoff_utc": "2026-08-06T15:00:00+00:00"}},
    )
    source = tmp_path / "governance" / "shadow_default_equities" / "master_control_20260806.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-06T18:00:00+00:00",
                "symbol": "SPY",
                "market": {"last_price": 505.0, "prev_close": 500.0},
                "source_quality_label": "broker_native",
                "source_quality_score": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = capture.build_payload(
        tmp_path,
        config_path=config_path,
        apply=True,
        now=datetime(2026, 8, 6, 18, 30, tzinfo=timezone.utc),
    )

    assert payload["appended_days"] == []
    assert payload["candidate_day_count"] == 0
    assert not Path(payload["series_path"]).exists()


def test_benchmark_capture_rejects_mid_session_candidate_freeze_day(tmp_path: Path) -> None:
    config = json.loads(capture.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / capture.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-1", "generation": 1},
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {"profitability_evidence_window": {"candidate_cutoff_utc": "2026-08-06T16:00:00+00:00"}},
    )
    source = tmp_path / "governance" / "shadow_default_equities" / "master_control_20260806.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-06T20:10:00+00:00",
                "symbol": "SPY",
                "market": {"last_price": 510.0, "prev_close": 500.0},
                "source_quality_label": "broker_native",
                "source_quality_score": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = capture.build_payload(
        tmp_path,
        config_path=config_path,
        apply=True,
        now=datetime(2026, 8, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["candidate_day_count"] == 0
    assert payload["scan"]["partial_candidate_days_rejected"] == 1
