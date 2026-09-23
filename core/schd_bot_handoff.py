"""Validate a native SCHD decision for a separately operator-confirmed market test."""

from datetime import timedelta

from core.decision_price_evidence import (
    build_evidence,
    digest,
    session_bounds,
    timestamp,
)
from core.order_intent import canonical_payload_sha256
from core.supervised_broker_test import build_market_request, number


def decision_binding(packet):
    return digest(
        {
            key: packet.get(key)
            for key in ("candidate_id", "decision", "quote", "candles", "price_basis")
        }
    )


def handoff_blockers(packet, quote, *, request, candidate_id, now):
    blockers = []
    try:
        report = build_evidence(packet, now=now)
        blockers.extend(report["blockers"])
        decision = packet["decision"]
        action = request["orderLegCollection"][0]["instruction"]
        native = packet["native_validation"]
        if (
            packet.get("evidence_kind") != "recorded"
            or packet.get("candidate_id") != candidate_id
            or decision.get("action") != action
            or decision.get("strategy") != "grand_master_bot"
            or not native.get("receipt", {}).get("raw_row_sha256")
            or native.get("action_changed") is not False
            or native.get("gates_changed") is not False
        ):
            blockers.append("exact_native_bot_decision_required")
        bounds = session_bounds(now)
        if not bounds or not bounds[0] + timedelta(seconds=75) <= now <= bounds[
            1
        ] - timedelta(seconds=75):
            blockers.append("market_test_regular_session_buffer_required")
        if (
            quote.get("symbol") != "SCHD"
            or quote.get("source_provider") != "schwab_api"
            or quote.get("realtime") is not True
            or quote.get("transport", {}).get("ok") is not True
            or not 0
            <= (now - timestamp(quote["provider_timestamp_utc"])).total_seconds()
            <= 15
        ):
            blockers.append("fresh_realtime_market_quote_required")
        bid, ask = number(quote["bid_price"]), number(quote["ask_price"])
        for side in ("bid", "ask"):
            if (
                number(quote[f"{side}_size"]) < 1
                or not 0
                <= (now - timestamp(quote[f"{side}_timestamp_utc"])).total_seconds()
                <= 15
            ):
                blockers.append(f"fresh_market_{side}_and_size_required")
        if bid <= 0 or ask <= bid or (ask - bid) / ((ask + bid) / 2) * 10000 > 25:
            blockers.append("market_quote_spread_invalid")
        side = "ask" if action == "BUY" else "bid"
        touch = ask if action == "BUY" else bid
        original = number(packet["quote"][side])
        if original <= 0 or abs(touch - original) / original * 10000 > 35:
            blockers.append("market_quote_moved_beyond_reviewed_bound")
        # Admission estimate only. MARKET orders have no broker-enforced price cap.
        if action == "BUY" and ask * number("1.0035") + 1 > 100:
            blockers.append("estimated_market_cost_above_test_budget")
    except (KeyError, ValueError, TypeError, ArithmeticError, IndexError):
        blockers.append("native_market_handoff_evidence_incomplete")
    return list(dict.fromkeys(blockers))


def prepare_handoff(plan, packet, quote, *, action, candidate_id, now):
    request = build_market_request(plan, action=action)
    receipt = {
        "packet": packet,
        "packet_sha256": digest(packet),
        "decision_binding_sha256": decision_binding(packet),
        "current_quote": quote,
        "request_sha256": canonical_payload_sha256(request),
        "candidate_id": candidate_id,
        "market_price_not_guaranteed": True,
        "automatic_wait_or_retry": False,
    }
    blockers = validate_handoff(
        receipt, request=request, candidate_id=candidate_id, now=now
    )
    return {
        "request": request,
        "receipt": receipt,
        "blockers": blockers,
        "ready": not blockers,
    }


def validate_handoff(receipt, *, request, candidate_id, now):
    try:
        if (
            request.get("orderType") != "MARKET"
            or request.get("session") != "NORMAL"
            or receipt.get("request_sha256") != canonical_payload_sha256(request)
            or receipt.get("packet_sha256") != digest(receipt["packet"])
            or receipt.get("decision_binding_sha256")
            != decision_binding(receipt["packet"])
            or receipt.get("candidate_id") != candidate_id
            or receipt.get("market_price_not_guaranteed") is not True
            or receipt.get("automatic_wait_or_retry") is not False
        ):
            return ["native_market_handoff_binding_invalid"]
        return handoff_blockers(
            receipt["packet"],
            receipt["current_quote"],
            request=request,
            candidate_id=candidate_id,
            now=now,
        )
    except (KeyError, ValueError, TypeError):
        return ["native_market_handoff_missing_or_invalid"]
