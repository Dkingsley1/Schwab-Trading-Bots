from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import market_cycle_extraction_engine as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _decision(
    symbol: str,
    *,
    action: str,
    risk_on: float,
    risk_off: float,
    defensive: float,
    trend: float,
    stress: float,
    edge: float = 0.35,
    confidence: float = 0.65,
) -> dict:
    return {
        "timestamp_utc": "2026-07-01T14:30:00+00:00",
        "symbol": symbol,
        "master_action": action,
        "flow_awareness_features": {
            "flow_risk_on_norm": risk_on,
            "flow_risk_off_norm": risk_off,
            "flow_defensive_rotation_norm": defensive,
            "flow_stress_norm": stress,
            "flow_conviction_norm": trend,
        },
        "allocation_confidence": {
            "allocation_trade_edge_norm": edge,
            "allocation_confidence_norm": confidence,
        },
        "lead_lag_features": {"lead_lag_confirmation_norm": trend},
        "core_cross_asset_confirmation_norm": trend,
    }


def _base_context(root: Path, *, regime_state: str, stance: float, shock: float, risk_norm: float = 0.2) -> None:
    health = root / "governance" / "health"
    _write_json(
        health / "regime_control_plane_latest.json",
        {
            "overall_status": "ready",
            "regime_state": regime_state,
            "stance_label": "bullish" if stance > 0 else "bearish",
            "stance_score": stance,
            "scores": {
                "shock_score": shock,
                "risk_norm": risk_norm,
                "cross_asset_confidence": 0.86,
            },
        },
    )
    _write_json(health / "market_posture_control_latest.json", {"overall_status": "ready", "posture_state": "balanced_observe"})
    _write_json(health / "derived_state_latest.json", {"risk_score": risk_norm * 100.0})


def test_market_cycle_engine_classifies_expansion_shadow_only(tmp_path: Path) -> None:
    _base_context(tmp_path, regime_state="risk_on_trend", stance=0.55, shock=0.18)
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "aggressive_equities_schwab" / "decision_20260701.jsonl",
        [
            _decision("SPY", action="BUY", risk_on=0.72, risk_off=0.16, defensive=0.08, trend=0.70, stress=0.18),
            _decision("QQQ", action="BUY", risk_on=0.68, risk_off=0.18, defensive=0.10, trend=0.66, stress=0.20),
        ],
    )
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "schwab_futures_equities_schwab" / "decision_20260701.jsonl",
        [_decision("/ES", action="BUY", risk_on=0.74, risk_off=0.14, defensive=0.07, trend=0.72, stress=0.16)],
    )

    payload = src.build_payload(tmp_path, sample_limit=20, min_rows=3)

    assert payload["overall_status"] == "ready"
    assert payload["cycle_phase"] == "expansion"
    assert payload["market_regime"] == "risk_on_trend"
    assert payload["shadow_contract"]["shadow_only"] is True
    assert payload["shadow_contract"]["live_execution_allowed"] is False
    assert payload["shadow_contract"]["order_routing_allowed"] is False
    assert payload["sleeve_bias"]["paper_or_live_budget_changes_allowed"] is False


def test_market_cycle_engine_classifies_contraction_from_risk_off(tmp_path: Path) -> None:
    _base_context(tmp_path, regime_state="risk_off_shock", stance=-0.55, shock=0.76, risk_norm=0.72)
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "bond_equities_schwab" / "decision_20260701.jsonl",
        [
            _decision("TLT", action="HOLD", risk_on=0.18, risk_off=0.74, defensive=0.68, trend=0.34, stress=0.78, edge=0.12, confidence=0.48),
            _decision("IEF", action="HOLD", risk_on=0.16, risk_off=0.78, defensive=0.70, trend=0.30, stress=0.80, edge=0.10, confidence=0.45),
        ],
    )
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "aggressive_equities_schwab" / "decision_20260701.jsonl",
        [_decision("SPY", action="SELL", risk_on=0.14, risk_off=0.82, defensive=0.62, trend=0.28, stress=0.84, edge=0.05, confidence=0.42)],
    )

    payload = src.build_payload(tmp_path, sample_limit=20, min_rows=3)

    assert payload["overall_status"] == "ready"
    assert payload["cycle_phase"] == "contraction"
    assert payload["market_regime"] == "risk_off_shock"
    assert payload["sleeve_bias"]["bias_delta"]["aggressive"] < 0
    assert payload["sleeve_bias"]["bias_delta"]["bond"] > 0


def test_market_cycle_engine_writes_artifact_and_history(tmp_path: Path) -> None:
    _base_context(tmp_path, regime_state="risk_on_trend", stance=0.40, shock=0.22)
    _append_jsonl(
        tmp_path / "governance" / "channels" / "decision" / "aggressive_equities_schwab" / "decision_20260701.jsonl",
        [_decision("SPY", action="BUY", risk_on=0.70, risk_off=0.20, defensive=0.10, trend=0.62, stress=0.22)],
    )
    out = tmp_path / "governance" / "health" / "market_cycle_state_latest.json"
    history = tmp_path / "governance" / "market_cycle" / "market_cycle_state_history.jsonl"

    rc = src.main([
        "--project-root",
        str(tmp_path),
        "--sample-limit",
        "20",
        "--min-rows",
        "1",
        "--out-file",
        str(out),
        "--history-file",
        str(history),
        "--json",
    ])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    history_rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload["mode"] == "shadow_observation_only"
    assert payload["shadow_contract"]["writes_runtime_env_overrides"] is False
    assert history_rows[-1]["cycle_phase"] == payload["cycle_phase"]
