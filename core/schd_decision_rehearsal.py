"""Isolated one-share conditional simulation. No broker client or order gateway."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
import json

from core.decision_price_evidence import (
    build_evidence,
    digest,
    number,
    quote_evidence,
    session_bounds,
    timestamp,
)
from core.execution_simulator import simulate_execution

AUTHORITY = {
    "live_execution_authority": False,
    "broker_mutation_attempted": False,
    "autonomous_execution": False,
    "production_promotion_credit": False,
    "strategy_profitability_proven": False,
    "account_reconciliation_certified": False,
}
TERMINAL = {"complete", "cancelled", "reconciliation_required"}


def money(value):
    return str(Decimal(str(value)).quantize(Decimal("0.000001")))


def initial_state(*, candidate_id, evidence_kind, implementation_sha256):
    if not candidate_id or evidence_kind not in {"recorded", "synthetic"}:
        raise ValueError("candidate_and_explicit_evidence_kind_required")
    return {
        "schema_version": 1,
        "purpose": "schd_conditional_simulation_only",
        "symbol": "SCHD",
        "account": "isolated_virtual_cash_not_a_broker_account",
        "candidate_id": candidate_id,
        "evidence_kind": evidence_kind,
        "implementation_sha256": implementation_sha256,
        "phase": "waiting_entry",
        "initial_virtual_cash_usd": "100.000000",
        "virtual_cash_usd": "100.000000",
        "virtual_shares": 0,
        "orders": [],
        "pending": None,
        "last_report": None,
        "reconciliation": None,
        "fill_assumption": "Conditional scenario assumes one whole share at the simulator's estimated price, not an observed fill. Model ratios/probabilities are diagnostics, not share quantities or realized outcomes.",
        **AUTHORITY,
    }


def validate_state(state):
    if (
        state.get("schema_version") != 1
        or state.get("purpose") != "schd_conditional_simulation_only"
        or state.get("symbol") != "SCHD"
        or state.get("evidence_kind") not in {"recorded", "synthetic"}
        or state.get("phase")
        not in TERMINAL | {"waiting_entry", "pending_buy", "holding", "pending_sell"}
        or any(state.get(k) is not False for k in AUTHORITY)
    ):
        raise ValueError("invalid_rehearsal_state")
    orders = state.get("orders", [])
    if not isinstance(orders, list) or len(orders) > 2:
        raise ValueError("invalid_rehearsal_orders")
    cash, shares = Decimal("100.000000"), 0
    for expected, order in zip(("BUY", "SELL"), orders):
        if order.get("side") != expected or order.get("quantity") != 1:
            raise ValueError("rehearsal_order_scope_mismatch")
        price = Decimal(order["modeled_fill_price_usd"])
        if not price.is_finite() or price <= 0:
            raise ValueError("invalid_modeled_price")
        cash += -price if expected == "BUY" else price
        shares += 1 if expected == "BUY" else -1
    if (
        cash < 0
        or Decimal(state["virtual_cash_usd"]) != cash
        or state["virtual_shares"] != shares
    ):
        raise ValueError("rehearsal_ledger_mismatch")
    required_count = {
        "waiting_entry": 0,
        "pending_buy": 0,
        "holding": 1,
        "pending_sell": 1,
        "complete": 2,
    }
    if (
        state["phase"] in required_count
        and len(orders) != required_count[state["phase"]]
    ):
        raise ValueError("rehearsal_phase_mismatch")
    if state["phase"].startswith("pending_") != bool(state.get("pending")):
        raise ValueError("rehearsal_pending_mismatch")


def advance(state, packet, *, now, implementation_sha256):
    validate_state(state)
    if (
        state["implementation_sha256"] != implementation_sha256
        or packet.get("candidate_id") != state["candidate_id"]
        or packet.get("evidence_kind") != state["evidence_kind"]
    ):
        raise ValueError("candidate_source_or_evidence_kind_changed")
    result = deepcopy(state)
    if result["phase"] in TERMINAL:
        return result
    if result.get("pending"):
        pending = result["pending"]
        if now >= timestamp(pending["expires_at_utc"]):
            result.update(
                phase="cancelled",
                pending=None,
                wait_reason="simulated_order_expired_no_retry",
            )
            return result
        raw_quote = packet.get("fill_quote")
        if not raw_quote:
            result["wait_reason"] = "later_independent_quote_required"
            return result
        try:
            quote = quote_evidence(raw_quote, asof=now)
            when = timestamp(quote["timestamp_utc"])
            bounds = session_bounds(when)
            if (
                when
                < timestamp(pending["submitted_at_utc"]) + timedelta(milliseconds=120)
                or quote["snapshot_id"] == pending["snapshot_id"]
            ):
                raise ValueError("later_independent_quote_required")
            if (
                bounds is None
                or not bounds[0] <= when < bounds[1]
                or quote["spread_bps"] > 25
            ):
                raise ValueError("fill_quote_session_or_spread_invalid")
        except (KeyError, TypeError, ValueError) as exc:
            result["wait_reason"] = str(exc)
            return result
        outcome = packet.get("scenario_outcome", "assumed_full_fill")
        if outcome != "assumed_full_fill":
            if state["evidence_kind"] != "synthetic" or outcome not in {
                "no_fill",
                "rejected",
                "partial",
            }:
                raise ValueError("unsupported_scenario_outcome")
            result.update(
                phase=(
                    "reconciliation_required" if outcome == "partial" else "cancelled"
                ),
                pending=None,
                wait_reason=f"synthetic_{outcome}_no_retry",
            )
            return result
        inputs = dict(
            action=pending["side"],
            last_price=quote["last"],
            return_1m=0,
            volatility_1m=0,
            spread_bps=quote["spread_bps"],
            bid_price=quote["bid"],
            ask_price=quote["ask"],
            bid_size=quote["bid_size"],
            ask_size=quote["ask_size"],
            order_size=1,
            broker="schwab",
            market_kind="equities",
            symbol="SCHD",
            session="regular",
            order_type="market",
            asset_class="equity",
            quote_age_ms=quote["age_seconds"] * 1000,
        )
        # Expected fill already embeds fees/friction. Do not subtract fees again.
        model = asdict(simulate_execution(**inputs))
        price = Decimal(money(number(model["expected_fill_price"], positive=True)))
        cash = Decimal(result["virtual_cash_usd"])
        if (
            model["paper_execution_status"] != "simulated_fill"
            or model["quote_crossed_or_locked"]
            or (pending["side"] == "BUY" and price + 1 > cash)
        ):
            result.update(
                phase="cancelled",
                pending=None,
                wait_reason="model_reject_or_virtual_budget_exceeded",
            )
            return result
        order = dict(
            side=pending["side"],
            quantity=1,
            order_type="MARKET",
            decision_id=pending["decision_id"],
            modeled_fill_price_usd=str(price),
            decision_evidence=pending["decision_evidence"],
            submitted_at_utc=pending["submitted_at_utc"],
            modeled_at_utc=now.isoformat(),
            fill_quote=quote,
            model_inputs=inputs,
            execution_model=model,
            fill_status="conditional_assumed_full_share_not_observed",
            model_input_limits=[
                "No measured 1m volatility; zero baseline is not a worst-case cost bound",
                "No independently calibrated fill probability or fee schedule",
            ],
        )
        result["orders"].append(order)
        result["pending"] = None
        result.pop("wait_reason", None)
        if pending["side"] == "BUY":
            result.update(
                phase="holding", virtual_shares=1, virtual_cash_usd=str(cash - price)
            )
        else:
            cash += price
            result.update(
                phase="complete", virtual_shares=0, virtual_cash_usd=str(cash)
            )
            buy_price = Decimal(result["orders"][0]["modeled_fill_price_usd"])
            result["reconciliation"] = {
                "state": "conditional_virtual_ledger_balanced",
                "bought_shares": 1,
                "sold_shares": 1,
                "remaining_shares": 0,
                "modeled_net_pnl_usd": str(cash - Decimal("100")),
                "cash_delta_equals_modeled_pnl": cash - Decimal("100")
                == price - buy_price,
                "sold_above_modeled_buy": price > buy_price,
                "dividend_income_usd": "0",
                "fees": "Embedded once in modeled fill prices",
                "broker_fill_and_account_evidence": "not_observed",
            }
        validate_state(result)
        return result
    report = build_evidence(packet, now=now)
    result["last_report"] = report
    if report["blockers"]:
        result["wait_reason"] = "evidence_incomplete_or_bot_wait"
        return result
    decision = packet["decision"]
    expected = "BUY" if result["phase"] == "waiting_entry" else "SELL"
    if decision["action"] != expected:
        result["wait_reason"] = f"awaiting_bot_{expected.lower()}_decision"
        return result
    if result["orders"]:
        entry = result["orders"][0]
        if decision["decision_id"] == entry["decision_id"] or timestamp(
            decision["timestamp_utc"]
        ) <= timestamp(entry["modeled_at_utc"]):
            result["wait_reason"] = "new_post_entry_exit_decision_required"
            return result
    result.update(
        phase="pending_buy" if expected == "BUY" else "pending_sell",
        pending={
            "side": expected,
            "decision_id": decision["decision_id"],
            "snapshot_id": report["quote"]["snapshot_id"],
            "submitted_at_utc": now.isoformat(),
            "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
            "decision_evidence": report,
        },
    )
    result.pop("wait_reason", None)
    validate_state(result)
    return result


def render_markdown(state, chart_paths=None):
    lines = [
        "# SCHD Decision Rehearsal",
        "",
        "SIMULATION ONLY. No broker orders or account balance verification.",
        f"Evidence: {state['evidence_kind']}. Phase: {state['phase']}.",
        f"Candidate: {state['candidate_id']}. Implementation: {state['implementation_sha256']}.",
        f"Virtual shares: {state['virtual_shares']}; virtual cash: ${state['virtual_cash_usd']}.",
        state["fill_assumption"],
        "",
        f"Wait reason: {state.get('wait_reason', 'none')}.",
    ]
    reports = [order["decision_evidence"] for order in state["orders"]]
    chart = state.get("decision_chart_report") or {}
    if chart.get("report_path"):
        lines += [
            "", "## Original Decision Chart", "",
            f"[Open decision chart report]({chart['report_path']})",
            "This sidecar preserves original decision-time evidence. Proposed actions are not broker fills.",
        ]
    if state.get("last_report") and not any(
        r["input_sha256"] == state["last_report"]["input_sha256"] for r in reports
    ):
        reports.append(state["last_report"])
    if not reports:
        lines += ["", "No recorded bot decision or market evidence has been evaluated."]
    for report in reports:
        bot = report["bot_record"]
        sample = report.get("recorded_decision_sample")
        if sample:
            record = sample["record"]
            age = sample["decision_age_seconds"]
            lines += [
                "", "## Recorded Decision Explanation Sample", "",
                f"Recorded action: **{record['action']}** at {record['timestamp_utc']}.",
                f"Decision age when this report was generated: {age:.1f} seconds; "
                + ("within" if 0 <= age <= 120 else "outside")
                + " the 120-second decision-age limit (not overall clearance).",
                f"Model score: {record.get('model_score')}; recorded threshold: {record.get('threshold')}. "
                "The score is not a calibrated probability of profit.",
                "These are the bot's recorded reasons, not a reconstructed explanation:",
                *[f"- {reason}" for reason in record.get("reasons", [])],
                "",
                "The Schwab charts below are separately retrieved context. They are not proof "
                "the bot used these candles. HOLD remains no trade, even when the log's "
                "gate-disposition field says EXECUTE. No entry price is recommended by this sample.",
                "", "### Decision Provenance", "```json",
                json.dumps(sample, indent=2), "```",
            ]
        lines += [
            "",
            f"## {bot['action']} / {report['decision_status']}",
            f"As of {report['as_of_utc']}; decision {bot['decision_id']}; strategy {bot['strategy']}.",
            f"Input SHA-256: {report['input_sha256']}",
            report["attribution"],
            "",
            "### Recorded Bot Rationale",
            "```json",
            json.dumps(bot, indent=2),
            "```",
            "",
            "### Quote And Freshness",
            "```json",
            json.dumps(report["quote"], indent=2),
            "```",
        ]
        if report.get("chart_source"):
            lines += [
                "",
                "### Schwab Chart Provenance",
                "```json",
                json.dumps(report["chart_source"], indent=2),
                "```",
            ]
        if report.get("native_validation"):
            lines += [
                "",
                "### Native Decision Connection",
                "The action is preserved exactly. EXECUTE in a decision log describes gate checks; HOLD still means no trade.",
                "```json",
                json.dumps(report["native_validation"], indent=2),
                "```",
            ]
        for name in ("1m", "5m", "15m", "1h", "1d", "1M", "180d", "1Y"):
            frame = report["timeframes"].get(name, {"status": "missing"})
            lines += [
                "",
                f"### {name} Candles: {frame['status']}",
                *(
                    [
                        f"![SCHD {name} candles]({chart_paths[report['input_sha256']][name]})",
                        "",
                    ]
                    if chart_paths
                    and name in chart_paths.get(report["input_sha256"], {})
                    else []
                ),
                "| Close Time UTC | Open | High | Low | Close | Volume | Body | Upper Wick | Lower Wick | Close In Range |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in frame.get("recent_candles", []):
                keys = (
                    "end_utc",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "body_usd",
                    "upper_wick_usd",
                    "lower_wick_usd",
                    "close_position_in_candle",
                )
                lines.append(
                    "| "
                    + " | ".join(
                        f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k))
                        for k in keys
                    )
                    + " |"
                )
            details = {
                k: v
                for k, v in frame.items()
                if k not in {"recent_candles", "period_candles", "chart_candles"}
            }
            lines += ["```json", json.dumps(details, indent=2), "```"]
        for title, key in (
            ("Supporting Directional Context", "context_supporting_direction"),
            ("Opposing Or Neutral Context", "context_opposing_or_neutral"),
            ("Blockers", "blockers"),
            ("Invalidation", "invalidation"),
            ("Unverified Context", "unverified_context"),
            ("Corporate Actions", "corporate_action_context"),
            ("Definitions", "definitions"),
        ):
            lines += [
                "",
                f"### {title}",
                "```json",
                json.dumps(report[key], indent=2),
                "```",
            ]
    lines += [
        "",
        "## Simulated Orders And Reconciliation",
        "```json",
        json.dumps(
            {
                "orders": [
                    {k: v for k, v in o.items() if k != "decision_evidence"}
                    for o in state["orders"]
                ],
                "reconciliation": state["reconciliation"],
                **AUTHORITY,
            },
            indent=2,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)
