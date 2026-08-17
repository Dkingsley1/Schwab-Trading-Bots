#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "regime_control_plane_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "regime" / "regime_control_plane_history.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(value), float(hi)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp11(value: float) -> float:
    return _clamp(value, -1.0, 1.0)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _context_health_ratio(payload: dict[str, Any], *, source_key: str = "sources") -> float:
    sources = payload.get(source_key)
    if not isinstance(sources, dict) or not sources:
        return 0.0
    total = 0
    ok = 0
    for row in sources.values():
        if not isinstance(row, dict):
            continue
        total += 1
        if bool(row.get("ok", False)):
            ok += 1
    return (float(ok) / float(total)) if total > 0 else 0.0


def _weighted_sentiment(sentiment: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    weights = (
        ("day", 0.4),
        ("week", 0.25),
        ("month", 0.2),
        ("year", 0.15),
    )
    sentiment_score = 0.0
    shock_score = 0.0
    weight_used = 0.0
    components: dict[str, float] = {}
    for key, weight in weights:
        row = sentiment.get(key)
        if not isinstance(row, dict) or not bool(row.get("available", False)):
            continue
        avg_sentiment = _clamp11(_safe_float(row.get("avg_sentiment_hint"), 0.0))
        mean_shock = _clamp01(_safe_float(row.get("mean_shock_hint"), 0.0))
        sentiment_score += avg_sentiment * weight
        shock_score += mean_shock * weight
        weight_used += weight
        components[f"{key}_sentiment"] = round(avg_sentiment, 6)
        components[f"{key}_shock"] = round(mean_shock, 6)
    if weight_used <= 0.0:
        return 0.0, 0.0, components
    sentiment_score /= weight_used
    shock_score /= weight_used
    return _clamp11(sentiment_score), _clamp01(shock_score), components


def _regime_state(*, stance_score: float, shock_score: float, risk_norm: float, execution_stress: float) -> str:
    if shock_score >= 0.75 and stance_score <= -0.2:
        return "risk_off_shock"
    if risk_norm >= 0.65 and execution_stress >= 0.5:
        return "fragile_transition"
    if stance_score >= 0.35 and shock_score <= 0.45:
        return "risk_on_trend"
    if stance_score <= -0.35:
        return "risk_off_trend"
    if abs(stance_score) < 0.2 and shock_score <= 0.45:
        return "rangebound_transition"
    return "mixed_transition"


def _stance_label(score: float) -> str:
    if score >= 0.3:
        return "bullish"
    if score <= -0.3:
        return "bearish"
    return "neutral"


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_row: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            if lines:
                parsed = json.loads(lines[-1])
                if isinstance(parsed, dict):
                    last_row = parsed
        except Exception:
            last_row = {}
    if (
        last_row.get("history_key") == row.get("history_key")
        and str(last_row.get("regime_state") or "") == str(row.get("regime_state") or "")
        and abs(_safe_float(last_row.get("stance_score"), 0.0) - _safe_float(row.get("stance_score"), 0.0)) < 1e-9
    ):
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    sentiment = _load_json(health_root / "sentiment_report_latest.json")
    official_macro = _load_json(health_root / "official_macro_context_sync_latest.json")
    market_micro = _load_json(health_root / "market_micro_sync_latest.json")
    fx_context = _load_json(health_root / "fx_market_context_sync_latest.json")
    crypto_context = _load_json(health_root / "crypto_market_context_sync_latest.json")
    market_crypto_corr = _load_json(health_root / "market_crypto_correlation_sync_latest.json")
    derived_state = _load_json(health_root / "derived_state_latest.json")
    calibration = _load_json(health_root / "paper_execution_calibration_latest.json")

    sentiment_score, shock_score, sentiment_components = _weighted_sentiment(sentiment)
    risk_norm = _clamp01(_safe_float(derived_state.get("risk_score"), 0.0) / 100.0)
    execution_stress = _clamp01(
        _safe_float((calibration.get("metrics") or {}).get("mae_bps"), 0.0)
        / max(_safe_float((calibration.get("thresholds") or {}).get("max_mae_bps"), 35.0), 1.0)
    )
    cross_asset_confidence = (
        _context_health_ratio(official_macro)
        + _context_health_ratio(market_micro)
        + _context_health_ratio(fx_context)
        + _context_health_ratio(crypto_context)
        + (1.0 if bool(market_crypto_corr.get("ok", False)) else 0.0)
    ) / 5.0
    stance_score = _clamp11(
        (0.6 * sentiment_score)
        - (0.25 * shock_score)
        - (0.2 * risk_norm)
        - (0.15 * execution_stress)
        + (0.1 * ((cross_asset_confidence - 0.5) * 2.0))
    )
    stance_label = _stance_label(stance_score)
    regime_state = _regime_state(
        stance_score=stance_score,
        shock_score=shock_score,
        risk_norm=risk_norm,
        execution_stress=execution_stress,
    )
    event_count = _safe_int(sentiment.get("event_count"), 0)
    overall_status = "ready"
    if event_count < 5 or cross_asset_confidence < 0.6:
        overall_status = "thin"
    if not sentiment or not bool(sentiment.get("ok", False)):
        overall_status = "degraded"

    selected_day_utc = str(sentiment.get("selected_day_utc") or datetime.now(timezone.utc).strftime("%Y%m%d"))
    effective_day_utc = str(
        ((sentiment.get("day") or {}).get("day_end_day_utc"))
        or ((sentiment.get("latest_event") or {}).get("day_utc"))
        or selected_day_utc
    )

    recommended_actions: list[str] = []
    if event_count < 5:
        recommended_actions.append("backfill more historical regime memory so the regime layer is not dominated by a handful of macro events")
    if shock_score >= 0.7:
        recommended_actions.append("raise abstention thresholds and emphasize defensive sleeves while the regime shock score remains elevated")
    if risk_norm >= 0.6:
        recommended_actions.append("treat allocator and execution budgets as regime-aware risk brakes until the derived risk score normalizes")
    if cross_asset_confidence < 0.75:
        recommended_actions.append("repair or refresh context feeds so regime stance is confirmed by macro, FX, crypto, and market-micro sources together")
    if execution_stress >= 0.5:
        recommended_actions.append("feed execution slippage stress back into regime posture so aggressive trading only resumes after realized fills stabilize")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "overall_status": overall_status,
        "selected_day_utc": selected_day_utc,
        "effective_day_utc": effective_day_utc,
        "regime_state": regime_state,
        "stance_label": stance_label,
        "stance_score": round(float(stance_score), 6),
        "scores": {
            "sentiment_score": round(float(sentiment_score), 6),
            "shock_score": round(float(shock_score), 6),
            "risk_norm": round(float(risk_norm), 6),
            "execution_stress_norm": round(float(execution_stress), 6),
            "cross_asset_confidence": round(float(cross_asset_confidence), 6),
        },
        "sentiment_components": sentiment_components,
        "data_depth": {
            "event_count": event_count,
            "daily_points": len(sentiment.get("daily_sentiment_series") or []),
            "weekly_points": len(sentiment.get("weekly_sentiment_series") or []),
            "monthly_points": len(sentiment.get("monthly_sentiment_series") or []),
            "yearly_points": len(sentiment.get("yearly_sentiment_series") or []),
        },
        "sources": {
            "official_macro_ok_ratio": round(_context_health_ratio(official_macro), 6),
            "market_micro_ok_ratio": round(_context_health_ratio(market_micro), 6),
            "fx_context_ok_ratio": round(_context_health_ratio(fx_context), 6),
            "crypto_context_ok_ratio": round(_context_health_ratio(crypto_context), 6),
            "market_crypto_correlation_ok": bool(market_crypto_corr.get("ok", False)),
        },
        "latest_snapshot": {
            "headline": str(((sentiment.get("latest_live_macro_snapshot") or {}).get("headline")) or ""),
            "source": str(((sentiment.get("latest_live_macro_snapshot") or {}).get("source")) or ""),
            "speaker": str(((sentiment.get("latest_live_macro_snapshot") or {}).get("speaker")) or ""),
            "risk_level": str(derived_state.get("risk_level") or ""),
            "execution_multiplier": round(_safe_float(derived_state.get("execution_multiplier"), 0.0), 6),
        },
        "recommended_actions": _ordered_unique(recommended_actions),
        "source_files": {
            "sentiment_report": str(health_root / "sentiment_report_latest.json"),
            "official_macro_context_sync": str(health_root / "official_macro_context_sync_latest.json"),
            "market_micro_sync": str(health_root / "market_micro_sync_latest.json"),
            "fx_market_context_sync": str(health_root / "fx_market_context_sync_latest.json"),
            "crypto_market_context_sync": str(health_root / "crypto_market_context_sync_latest.json"),
            "market_crypto_correlation_sync": str(health_root / "market_crypto_correlation_sync_latest.json"),
            "derived_state": str(health_root / "derived_state_latest.json"),
            "paper_execution_calibration": str(health_root / "paper_execution_calibration_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a regime control plane from sentiment, macro, cross-asset, and execution stress signals.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    history_row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "history_key": payload.get("effective_day_utc"),
        "selected_day_utc": payload.get("selected_day_utc"),
        "effective_day_utc": payload.get("effective_day_utc"),
        "regime_state": payload.get("regime_state"),
        "stance_label": payload.get("stance_label"),
        "stance_score": payload.get("stance_score"),
        "scores": payload.get("scores"),
    }
    _append_history(Path(args.history_file).expanduser(), history_row)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "regime_control_plane "
            f"regime_state={payload.get('regime_state', '')} "
            f"stance_label={payload.get('stance_label', '')} "
            f"overall_status={payload.get('overall_status', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
