import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "evidence_packet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evidence_packet", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load evidence_packet")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_evidence_packet_builds_truthful_repeatable_readiness_packet(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    challenger = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health / "paper_performance_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "available_days": [f"202605{day:02d}" for day in range(3, 32)] + ["20260601"],
            "day": {
                "day_utc": "20260601",
                "executions": 120,
                "buy_count": 80,
                "sell_count": 40,
                "unique_symbols": 25,
                "ending_net_pnl_total": 12.5,
                "ending_realized_pnl_total": 4.5,
                "ending_unrealized_pnl_total": 8.0,
                "change_vs_previous_day": 1.25,
                "realized_change_vs_previous_day": 0.75,
            },
            "history_daily_series": [
                {
                    "day_utc": f"202605{day:02d}",
                    "executions": 10,
                    "change_vs_previous_day": 1.0,
                    "ending_realized_pnl_total": 0.2,
                }
                for day in range(3, 32)
            ]
            + [
                {
                    "day_utc": "20260601",
                    "executions": 120,
                    "change_vs_previous_day": 1.25,
                    "ending_realized_pnl_total": 4.5,
                }
            ],
            "source_kind": "paper_broker_bridge",
            "active_paper_profile_count_today": 8,
        },
    )
    _write_json(
        health / "sleeve_profitability_dashboard_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall_status": "ready",
            "profitability_grade": "A",
            "totals": {
                "sleeve_count": 4,
                "execution_count": 120,
                "realized_pnl_total": 4.5,
                "unrealized_pnl_total": 8.0,
                "net_pnl_total": 12.5,
            },
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "profitability_grade": "A",
            "raw_profitability_grade": "B",
            "financial_profitability_grade": "B",
            "paper_summary": {
                "day_utc": "20260601",
                "all_sleeve_realized_pnl_total": 4.5,
                "all_sleeve_unrealized_pnl_total": 8.0,
            },
            "profit_harvest_report_card": {
                "grade": "B",
                "raw_outcome_grade": "B",
                "control_grade": "A",
                "current_realized_profit_share_norm": 0.36,
                "target_realized_profit_share_norm": 0.35,
            },
        },
    )
    _write_json(
        health / "income_operating_platform_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall_status": "ready",
            "income_operating_grade": "A",
            "income_operating_score": 90.0,
            "paper_only": True,
            "live_execution_allowed": False,
            "hard_blockers": [],
            "blockers": [],
        },
    )
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall": {"status": "ready", "attention": []},
            "memory": {"status": "ready", "memory_pressure_state": "green", "swap_used_gb": 0.0},
            "storage": {"status": "ready", "pressure_profile": "normal"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall_status": "ready",
            "backpressure": {
                "total_pending_lines": 20,
                "raw_live": {"total_pending_lines": 10},
                "oldest_pending_age_seconds": 3,
                "estimated_total_drain_minutes": 0.1,
            },
        },
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall_status": "ready",
            "training_quality_score": 95,
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "overall_status": "ready",
            "launch_allowed": True,
        },
    )
    _write_json(
        health / "promotion_quality_gate_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "ok": True,
            "failed_checks": [],
            "details": {"promotion_candidate_ids": ["brain_refinery_v10_seasonal"]},
        },
    )
    _write_json(
        challenger / "promotion_packet_latest.json",
        {
            "timestamp_utc": "2026-06-01T12:00:00+00:00",
            "packet_complete": True,
            "ready_for_committee": True,
            "trained_models_complete": True,
            "replayability_contract": {"exact_replay_ready": True},
            "signature": {"status": "verified"},
        },
    )

    payload = module.build_payload(tmp_path, now_utc=datetime(2026, 6, 1, tzinfo=timezone.utc))
    markdown = module.render_markdown(payload)

    assert payload["overall_status"] in {"ready", "watch"}
    assert payload["readiness_score"] >= 85.0
    assert payload["contract"]["read_only"] is True
    assert payload["contract"]["protected_volumes"]["VIDEO"] == "never_touched"
    assert "paper-mode proof" in payload["truth_statement"]
    assert payload["track_record_windows"][0]["status"] == "credible_window"
    assert payload["sleeve_attribution"]["sleeve_count"] == 4
    assert payload["harvest_and_realization"]["computed_all_sleeve_realized_share"] >= 0.35
    assert payload["training_and_promotion"]["packet_signature_verified"] is True
    assert "# Trading System Evidence Packet" in markdown


def test_evidence_packet_surfaces_missing_long_soak_without_failing(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(health / "paper_performance_latest.json", {"available_days": ["20260601"], "day": {"day_utc": "20260601"}})

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "needs_work"
    assert "thirty_day_paper_track_record" in payload["blockers"]
    assert payload["recommended_commands"]


def test_evidence_packet_uses_effective_raw_live_backpressure(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(health / "paper_performance_latest.json", {"available_days": ["20260601"], "day": {"day_utc": "20260601"}})
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "memory": {"status": "ready", "memory_pressure_state": "green"},
            "storage": {"status": "ready", "pressure_profile": "steady_state"},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "total_pending_lines": 0,
                "raw_live": {"total_pending_lines": 25000},
                "effective_raw_live": {"total_pending_lines": 0, "source": "fresh_empty_sql_ingestion_overlay"},
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
            },
        },
    )

    payload = module.build_payload(tmp_path)

    risk_ops = payload["risk_and_operations"]
    assert risk_ops["raw_live_pending_lines"] == 0
    assert risk_ops["raw_live_pending_lines_raw"] == 25000
    assert risk_ops["raw_live_pending_lines_source"] == "fresh_empty_sql_ingestion_overlay"
