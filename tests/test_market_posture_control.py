from __future__ import annotations

import json
from pathlib import Path

import scripts.run_shadow_training_loop as loop
from scripts.ops import market_posture_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _decision(symbol: str, *, action: str = "HOLD", risk_off: float = 0.24, risk_on: float = 0.16, defensive: float = 0.22) -> dict:
    return {
        "timestamp_utc": "2026-05-29T20:00:00+00:00",
        "symbol": symbol,
        "master_action": action,
        "portfolio": {"dispatch_qty": 0.0},
        "flow_awareness_features": {
            "flow_risk_off_norm": risk_off,
            "flow_risk_on_norm": risk_on,
            "flow_defensive_rotation_norm": defensive,
        },
        "allocation_confidence": {
            "allocation_trade_edge_norm": 0.0,
            "allocation_confidence_norm": 0.0,
        },
        "lead_lag_features": {"lead_lag_confirmation_norm": 0.20},
        "circuit_breakers": {"lane_kill_switch_active": False},
    }


def test_market_posture_control_detects_defensive_hold_and_writes_override(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "sleeve_profitability_dashboard_latest.json", {"paper_day": "20260529", "totals": {"net_pnl_total": -12.0}})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "protective_tightening", "profitability_grade": "A+"})
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "bond_equities_schwab" / "decision_20260529.jsonl",
        [_decision("TLT"), _decision("IEF"), _decision("SGOV")],
    )
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "aggressive_equities_schwab" / "decision_20260529.jsonl",
        [_decision("SPY", risk_off=0.20, risk_on=0.18, defensive=0.18)],
    )

    payload = src.build_payload(tmp_path)
    applied = src.apply_payload(tmp_path, payload)
    override_text = (tmp_path / "config" / ".env.market_posture_control_override").read_text(encoding="utf-8")

    assert applied["posture_state"] == "defensive_hold_momentum_faded"
    assert applied["overall_status"] == "defensive_posture_active"
    assert applied["env_overrides"]["MARKET_POSTURE_CONTROL_ENABLED"] == "1"
    assert "MARKET_POSTURE_STATE=defensive_hold_momentum_faded" in override_text
    assert applied["runtime_contract"]["block_aggressive_new_buys_until_re_risk_confirmation"] is True


def test_market_posture_control_prefers_live_decision_day_over_stale_paper_day(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "sleeve_profitability_dashboard_latest.json", {"paper_day": "20260529", "totals": {}})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "ready", "profitability_grade": "A"})
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "schwab_futures_equities_schwab" / "decision_20260530.jsonl",
        [_decision("/ES", action="HOLD", risk_off=0.10, risk_on=0.20, defensive=0.05)],
    )

    payload = src.build_payload(tmp_path)

    assert payload["paper_day"] == "20260530"
    assert payload["profile_summaries"]["schwab_futures"]["exists"] is True
    assert payload["profile_summaries"]["schwab_futures"]["rows_sampled"] == 1


def test_runtime_market_posture_blocks_aggressive_buy_without_rerisk(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_POSTURE_CONTROL_ENABLED", "1")
    monkeypatch.setenv("MARKET_POSTURE_STATE", "defensive_hold_momentum_faded")
    monkeypatch.setenv("MARKET_POSTURE_BUY_CONFIRMATION_FLOOR", "0.58")
    monkeypatch.setenv("MARKET_POSTURE_BUY_RISK_ON_EDGE_DELTA", "0.08")

    action, score, reasons, overlay = loop._market_posture_runtime_control(
        profile="intraday_aggressive",
        action="BUY",
        score=0.72,
        threshold=0.55,
        reasons=[],
        features={
            "flow_risk_on_norm": 0.18,
            "flow_risk_off_norm": 0.24,
            "flow_defensive_rotation_norm": 0.21,
            "core_cross_asset_confirmation_norm": 0.32,
        },
    )

    assert action == "HOLD"
    assert score < 0.72
    assert overlay["market_posture_defensive_state_norm"] == 1.0
    assert overlay["market_posture_rerisk_ok_norm"] == 0.0
    assert any("market_posture_block_aggressive_buy" in reason for reason in reasons)


def test_runtime_market_posture_allows_reconfirmed_aggressive_buy(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_POSTURE_CONTROL_ENABLED", "1")
    monkeypatch.setenv("MARKET_POSTURE_STATE", "defensive_hold_momentum_faded")
    monkeypatch.setenv("MARKET_POSTURE_BUY_CONFIRMATION_FLOOR", "0.58")
    monkeypatch.setenv("MARKET_POSTURE_BUY_RISK_ON_EDGE_DELTA", "0.08")

    action, score, reasons, overlay = loop._market_posture_runtime_control(
        profile="intraday_aggressive",
        action="BUY",
        score=0.72,
        threshold=0.55,
        reasons=[],
        features={
            "flow_risk_on_norm": 0.42,
            "flow_risk_off_norm": 0.20,
            "flow_defensive_rotation_norm": 0.12,
            "core_cross_asset_confirmation_norm": 0.64,
        },
    )

    assert action == "BUY"
    assert score == 0.72
    assert overlay["market_posture_rerisk_ok_norm"] == 1.0
    assert reasons == []
