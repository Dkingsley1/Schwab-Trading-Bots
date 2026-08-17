#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_STUDY_PATH = PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "position_opportunity_watch_latest.json"
ACTIVE_ROLL_STATES = {"roll_window_active", "operator_price_review", "urgent_roll_review"}


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _decision_age_seconds(context: dict[str, Any], now: datetime) -> float | None:
    timestamp = _parse_timestamp(context.get("timestamp_utc"))
    if timestamp is None:
        return None
    return max((now - timestamp).total_seconds(), 0.0)


def _observation(row: dict[str, Any], *, now: datetime, max_decision_age_seconds: float) -> dict[str, Any]:
    underlying = str(row.get("underlying") or "").strip().upper()
    context = _dict(row.get("chart_context"))
    market = _dict(context.get("market"))
    stance = _dict(context.get("stance"))
    source_profile = str(context.get("profile") or "").strip()
    decision_age = _decision_age_seconds(context, now)
    master_action = str(stance.get("master_action") or "HOLD").strip().upper()
    master_score = _safe_float(stance.get("master_score"), 0.5)
    master_vote = _safe_float(stance.get("master_vote"), 0.0)
    directional_trigger = _safe_float(stance.get("directional_trigger"), 1.0)
    deployability = _safe_float(stance.get("deployability"), 0.0)
    equity_quantity = _safe_float(row.get("equity_quantity"), 0.0)
    roll_status = str(stance.get("roll_watch_status") or "").strip().lower()
    roll_severity = str(stance.get("roll_watch_severity") or "").strip().lower()

    state = "abstain"
    position_action = "HOLD"
    reason = "model_abstained"
    candidate_kind = "none"

    if roll_status in ACTIVE_ROLL_STATES or roll_severity == "critical":
        state = "review_candidate"
        position_action = "ROLL_REVIEW"
        candidate_kind = "covered_call_risk_management"
        reason = f"covered_call_{roll_status or roll_severity}"
    elif source_profile in {"", "broker_position_mark"} or decision_age is None:
        reason = "no_fresh_model_decision_for_position"
    elif decision_age > max(float(max_decision_age_seconds), 0.0):
        reason = "position_model_decision_stale"
    elif master_action not in {"BUY", "SELL"}:
        reason = "position_model_action_hold"
    elif deployability and deployability < 0.52:
        reason = "position_model_deployability_below_floor"
    elif master_action == "BUY":
        state = "paper_candidate"
        position_action = "ADD" if equity_quantity >= 0.0 else "REDUCE_SHORT"
        candidate_kind = "position_aware_directional"
        reason = "fresh_existing_model_buy_decision"
    elif equity_quantity > 0.0:
        state = "paper_candidate"
        position_action = "REDUCE"
        candidate_kind = "position_aware_directional"
        reason = "fresh_existing_model_sell_decision"
    else:
        reason = "sell_signal_without_long_equity_position"

    return {
        "underlying": underlying,
        "state": state,
        "candidate_kind": candidate_kind,
        "position_action": position_action,
        "reason": reason,
        "accounts": sorted(str(item) for item in (row.get("accounts") or []) if str(item)),
        "equity_quantity": round(equity_quantity, 6),
        "option_contract_net": round(_safe_float(row.get("option_contract_net"), 0.0), 6),
        "market_value": round(_safe_float(row.get("market_value"), 0.0), 4),
        "market": {
            "last_price": market.get("last_price"),
            "spread_bps": market.get("spread_bps"),
            "range_pos": market.get("range_pos"),
        },
        "model_context": {
            "source_profile": source_profile,
            "timestamp_utc": context.get("timestamp_utc"),
            "age_seconds": round(decision_age, 3) if decision_age is not None else None,
            "master_action": master_action,
            "master_score": round(master_score, 6),
            "master_vote": round(master_vote, 6),
            "directional_trigger": round(directional_trigger, 6),
            "deployability": round(deployability, 6),
        },
        "execution_contract": {
            "direct_intent_publish_allowed": False,
            "paper_candidate_only": True,
            "live_execution_allowed": False,
            "quantity_recommendation": None,
            "required_route": "existing_sleeve_decision_to_standard_paper_execution_gateway",
        },
    }


def evaluate(
    study: dict[str, Any],
    *,
    max_decision_age_seconds: float = 1800.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows = [row for row in (study.get("underlyings") or []) if isinstance(row, dict)]
    study_ok = bool(study.get("ok", False))
    observations = [
        _observation(row, now=current, max_decision_age_seconds=max_decision_age_seconds)
        for row in rows
        if str(row.get("underlying") or "").strip()
    ]
    candidates = [row for row in observations if str(row.get("state")) in {"paper_candidate", "review_candidate"}]
    abstentions = [row for row in observations if str(row.get("state")) == "abstain"]
    status = "ready" if study_ok else "blocked"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": study_ok,
        "overall_status": status,
        "mode": "position_aware_paper_observation",
        "source_timestamp_utc": study.get("timestamp_utc"),
        "observed_underlying_count": len(observations),
        "candidate_count": len(candidates),
        "abstention_count": len(abstentions),
        "watch_symbols": sorted({str(row.get("underlying")) for row in observations if row.get("underlying")}),
        "candidates": candidates,
        "observations": observations,
        "safety_contract": {
            "uses_existing_model_decisions_only": True,
            "abstention_is_valid": True,
            "does_not_create_order_quantities": True,
            "does_not_publish_execution_intents": True,
            "paper_execution_requires_standard_gateway": True,
            "live_execution_allowed": False,
            "covered_call_rolls_are_review_only": True,
        },
        "regression_contract": {
            "every_visible_position_underlying_is_observed": len(observations) == len(rows),
            "max_decision_age_seconds": float(max_decision_age_seconds),
            "stale_or_missing_decisions_force_abstention": True,
            "sell_signals_cannot_open_uncovered_short_positions": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observe held Schwab positions for gated paper-trade opportunities.")
    parser.add_argument("--study-file", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max-decision-age-seconds", type=float, default=1800.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = evaluate(
        load_json(Path(args.study_file)),
        max_decision_age_seconds=max(float(args.max_decision_age_seconds), 0.0),
    )
    payload["source"] = str(Path(args.study_file))
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "position_opportunity_watch "
            f"status={payload.get('overall_status')} "
            f"observed={payload.get('observed_underlying_count')} "
            f"candidates={payload.get('candidate_count')} "
            f"abstentions={payload.get('abstention_count')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
