from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.runtime_layers import CircuitBreaker
from core.execution_simulator import simulate_execution


TRADE_ACTIONS = {
    "BUY",
    "SELL",
    "SELL_SHORT",
    "BUY_TO_COVER",
    "BUY_TO_OPEN",
    "BUY_TO_CLOSE",
    "SELL_TO_OPEN",
    "SELL_TO_CLOSE",
}


@dataclass
class GuardDecision:
    ok: bool
    gate: str
    reason: str
    details: Dict[str, Any]


@dataclass
class LiveRiskConfig:
    max_position_qty_per_symbol: float
    max_order_notional: float
    max_open_orders_total: int
    max_open_orders_per_symbol: int
    daily_loss_cap: float
    api_fail_limit: int
    api_cooldown_seconds: int
    trade_min_interval_seconds: float
    trade_min_interval_global_seconds: float
    max_slippage_bps: float
    max_fill_deviation_bps: float

    @classmethod
    def from_env(cls) -> "LiveRiskConfig":
        return cls(
            max_position_qty_per_symbol=max(float(os.getenv("LIVE_MAX_POSITION_QTY_PER_SYMBOL", "250")), 0.0),
            max_order_notional=max(float(os.getenv("LIVE_MAX_ORDER_NOTIONAL", "25000")), 0.0),
            max_open_orders_total=max(int(os.getenv("LIVE_MAX_OPEN_ORDERS_TOTAL", "30")), 1),
            max_open_orders_per_symbol=max(int(os.getenv("LIVE_MAX_OPEN_ORDERS_PER_SYMBOL", "3")), 1),
            daily_loss_cap=max(float(os.getenv("LIVE_MAX_DAILY_LOSS", "1000")), 0.0),
            api_fail_limit=max(int(os.getenv("LIVE_API_FAIL_LIMIT", "3")), 1),
            api_cooldown_seconds=max(int(os.getenv("LIVE_API_COOLDOWN_SECONDS", "120")), 1),
            trade_min_interval_seconds=max(float(os.getenv("LIVE_TRADE_MIN_INTERVAL_SECONDS", "8")), 0.0),
            trade_min_interval_global_seconds=max(float(os.getenv("LIVE_TRADE_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")), 0.0),
            max_slippage_bps=max(float(os.getenv("LIVE_MAX_SLIPPAGE_BPS", "35")), 0.0),
            max_fill_deviation_bps=max(float(os.getenv("LIVE_MAX_FILL_DEVIATION_BPS", "45")), 0.0),
        )


class LiveExecutionGuard:
    def __init__(self, config: LiveRiskConfig) -> None:
        self.config = config
        self._api_breaker = CircuitBreaker(
            fail_limit=config.api_fail_limit,
            cooldown_seconds=config.api_cooldown_seconds,
        )

        self._positions: Dict[str, Dict[str, float]] = {}
        self._broker_positions: Dict[str, float] = {}
        self._open_orders: Dict[str, Dict[str, Any]] = {}
        self._open_orders_by_symbol: Dict[str, int] = {}

        self._last_trade_symbol_ts: Dict[str, float] = {}
        self._last_trade_global_ts = 0.0

        self._daily_key = self._utc_day_key(time.time())
        self._realized_pnl_today = 0.0

        self._fill_count = 0
        self._fill_slippage_bps_sum = 0.0
        self._fill_deviation_bps_sum = 0.0
        self._fill_deviation_violations = 0

    @staticmethod
    def _utc_day_key(now_ts: float) -> str:
        return datetime.fromtimestamp(float(now_ts), tz=timezone.utc).strftime("%Y%m%d")

    @staticmethod
    def _signed_quantity(action: str, quantity: float) -> float:
        side = str(action or "").strip().upper()
        qty = max(float(quantity or 0.0), 0.0)
        if side in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
            return qty
        if side in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
            return -qty
        return 0.0

    def _roll_day(self, now_ts: float) -> None:
        day_key = self._utc_day_key(now_ts)
        if day_key == self._daily_key:
            return
        self._daily_key = day_key
        self._realized_pnl_today = 0.0

        self._fill_count = 0
        self._fill_slippage_bps_sum = 0.0
        self._fill_deviation_bps_sum = 0.0
        self._fill_deviation_violations = 0

    def allow_api_call(self, key: str = "broker_api") -> bool:
        return self._api_breaker.allow(key)

    def record_api_success(self, key: str = "broker_api") -> None:
        self._api_breaker.record_success(key)

    def record_api_failure(self, key: str = "broker_api") -> bool:
        return self._api_breaker.record_failure(key)

    def pre_trade_check(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        reference_price: float,
        intended_price: float = 0.0,
        now_ts: Optional[float] = None,
    ) -> GuardDecision:
        side = str(action or "").strip().upper()
        if side not in TRADE_ACTIONS:
            return GuardDecision(ok=True, gate="non_trade_action", reason="non_trade_action", details={})

        now_value = time.time() if now_ts is None else float(now_ts)
        self._roll_day(now_value)

        symbol_key = str(symbol or "").strip().upper()
        qty = max(float(quantity or 0.0), 0.0)
        ref = max(float(reference_price or 0.0), 0.0)
        intended = max(float(intended_price or 0.0), 0.0)

        if not self.allow_api_call("broker_api"):
            return GuardDecision(
                ok=False,
                gate="api_circuit_breaker",
                reason="api_circuit_open",
                details={
                    "cooldown_seconds": self.config.api_cooldown_seconds,
                    "api_fail_limit": self.config.api_fail_limit,
                },
            )

        if self.config.daily_loss_cap > 0.0 and self._realized_pnl_today <= -abs(self.config.daily_loss_cap):
            return GuardDecision(
                ok=False,
                gate="daily_loss_cap",
                reason="daily_loss_cap_reached",
                details={
                    "realized_pnl_today": float(self._realized_pnl_today),
                    "max_daily_loss": float(self.config.daily_loss_cap),
                },
            )

        if self.config.trade_min_interval_global_seconds > 0.0:
            since_global = now_value - self._last_trade_global_ts
            if self._last_trade_global_ts > 0.0 and since_global < self.config.trade_min_interval_global_seconds:
                return GuardDecision(
                    ok=False,
                    gate="trade_throttle_global",
                    reason="trade_global_min_interval",
                    details={
                        "since_last_global_trade_seconds": round(max(since_global, 0.0), 6),
                        "required_seconds": float(self.config.trade_min_interval_global_seconds),
                    },
                )

        if self.config.trade_min_interval_seconds > 0.0:
            last_symbol_ts = float(self._last_trade_symbol_ts.get(symbol_key, 0.0) or 0.0)
            since_symbol = now_value - last_symbol_ts
            if last_symbol_ts > 0.0 and since_symbol < self.config.trade_min_interval_seconds:
                return GuardDecision(
                    ok=False,
                    gate="trade_throttle_symbol",
                    reason="trade_symbol_min_interval",
                    details={
                        "symbol": symbol_key,
                        "since_last_symbol_trade_seconds": round(max(since_symbol, 0.0), 6),
                        "required_seconds": float(self.config.trade_min_interval_seconds),
                    },
                )

        open_orders_total = len(self._open_orders)
        if open_orders_total >= self.config.max_open_orders_total:
            return GuardDecision(
                ok=False,
                gate="open_order_limit_total",
                reason="open_order_limit_total",
                details={
                    "open_orders_total": int(open_orders_total),
                    "max_open_orders_total": int(self.config.max_open_orders_total),
                },
            )

        symbol_open_orders = int(self._open_orders_by_symbol.get(symbol_key, 0) or 0)
        if symbol_open_orders >= self.config.max_open_orders_per_symbol:
            return GuardDecision(
                ok=False,
                gate="open_order_limit_symbol",
                reason="open_order_limit_symbol",
                details={
                    "symbol": symbol_key,
                    "open_orders_symbol": int(symbol_open_orders),
                    "max_open_orders_per_symbol": int(self.config.max_open_orders_per_symbol),
                },
            )

        signed_qty = self._signed_quantity(side, qty)
        position = self._positions.get(symbol_key, {"qty": 0.0, "avg_price": 0.0})
        current_qty = float(position.get("qty", 0.0) or 0.0)
        projected_qty = current_qty + signed_qty

        if abs(projected_qty) > self.config.max_position_qty_per_symbol:
            return GuardDecision(
                ok=False,
                gate="position_limit",
                reason="projected_position_limit",
                details={
                    "symbol": symbol_key,
                    "current_qty": float(current_qty),
                    "signed_qty": float(signed_qty),
                    "projected_qty": float(projected_qty),
                    "max_position_qty_per_symbol": float(self.config.max_position_qty_per_symbol),
                },
            )

        if ref > 0.0 and self.config.max_order_notional > 0.0:
            order_notional = abs(ref * qty)
            if order_notional > self.config.max_order_notional:
                return GuardDecision(
                    ok=False,
                    gate="order_notional_limit",
                    reason="order_notional_limit",
                    details={
                        "symbol": symbol_key,
                        "reference_price": float(ref),
                        "quantity": float(qty),
                        "order_notional": float(order_notional),
                        "max_order_notional": float(self.config.max_order_notional),
                    },
                )

        if ref > 0.0 and intended > 0.0 and self.config.max_slippage_bps > 0.0:
            if side in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
                adverse_slippage_bps = max(((intended - ref) / ref) * 10000.0, 0.0)
            elif side in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
                adverse_slippage_bps = max(((ref - intended) / ref) * 10000.0, 0.0)
            else:
                adverse_slippage_bps = 0.0

            if adverse_slippage_bps > self.config.max_slippage_bps:
                return GuardDecision(
                    ok=False,
                    gate="slippage_limit",
                    reason="adverse_slippage_limit",
                    details={
                        "symbol": symbol_key,
                        "reference_price": float(ref),
                        "intended_price": float(intended),
                        "adverse_slippage_bps": round(float(adverse_slippage_bps), 6),
                        "max_slippage_bps": float(self.config.max_slippage_bps),
                    },
                )

        return GuardDecision(ok=True, gate="ok", reason="ok", details={"symbol": symbol_key})

    def mark_trade_submitted(self, *, symbol: str, now_ts: Optional[float] = None) -> None:
        now_value = time.time() if now_ts is None else float(now_ts)
        symbol_key = str(symbol or "").strip().upper()
        self._last_trade_symbol_ts[symbol_key] = now_value
        self._last_trade_global_ts = now_value

    def model_expected_fill(
        self,
        *,
        action: str,
        reference_price: float,
        quantity: float,
        spread_bps: float = 8.0,
        volatility_1m: float = 0.0,
        latency_ms: float = 120.0,
        bid_size: float = 1000.0,
        ask_size: float = 1000.0,
    ) -> Dict[str, float]:
        sim = simulate_execution(
            action=str(action or "HOLD").strip().upper(),
            last_price=max(float(reference_price or 0.0), 0.0),
            return_1m=0.0,
            spread_bps=max(float(spread_bps or 0.0), 0.0),
            volatility_1m=max(float(volatility_1m or 0.0), 0.0),
            latency_ms=max(float(latency_ms or 0.0), 0.0),
            bid_size=max(float(bid_size or 0.0), 0.0),
            ask_size=max(float(ask_size or 0.0), 0.0),
            order_size=max(float(quantity or 0.0), 0.0),
        )
        return {
            "expected_fill_price": float(sim.expected_fill_price),
            "expected_slippage_bps": float(sim.slippage_bps),
            "impact_bps": float(sim.impact_bps),
            "latency_ms": float(sim.latency_ms),
        }

    def evaluate_fill_quality(
        self,
        *,
        action: str,
        actual_fill_price: float,
        expected_fill_price: float,
    ) -> Dict[str, Any]:
        side = str(action or "").strip().upper()
        actual = max(float(actual_fill_price or 0.0), 0.0)
        expected = max(float(expected_fill_price or 0.0), 0.0)
        if expected <= 0.0 or actual <= 0.0:
            return {
                "ok": True,
                "fill_deviation_bps": 0.0,
                "max_fill_deviation_bps": float(self.config.max_fill_deviation_bps),
                "reason": "insufficient_prices",
            }

        if side in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
            adverse_bps = max(((actual - expected) / expected) * 10000.0, 0.0)
        elif side in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
            adverse_bps = max(((expected - actual) / expected) * 10000.0, 0.0)
        else:
            adverse_bps = 0.0

        ok = (self.config.max_fill_deviation_bps <= 0.0) or (adverse_bps <= self.config.max_fill_deviation_bps)
        return {
            "ok": bool(ok),
            "fill_deviation_bps": round(float(adverse_bps), 6),
            "max_fill_deviation_bps": float(self.config.max_fill_deviation_bps),
            "reason": "ok" if ok else "fill_deviation_limit",
        }

    def register_open_order(self, *, order_id: str, symbol: str, action: str, quantity: float) -> None:
        oid = str(order_id or "").strip()
        if not oid:
            return
        symbol_key = str(symbol or "").strip().upper()
        if oid in self._open_orders:
            return
        self._open_orders[oid] = {
            "symbol": symbol_key,
            "action": str(action or "").strip().upper(),
            "quantity": float(max(float(quantity or 0.0), 0.0)),
        }
        self._open_orders_by_symbol[symbol_key] = int(self._open_orders_by_symbol.get(symbol_key, 0) or 0) + 1

    def close_open_order(self, order_id: str) -> None:
        oid = str(order_id or "").strip()
        if not oid:
            return
        row = self._open_orders.pop(oid, None)
        if not row:
            return
        symbol_key = str(row.get("symbol", "")).upper()
        if symbol_key in self._open_orders_by_symbol:
            self._open_orders_by_symbol[symbol_key] = max(int(self._open_orders_by_symbol[symbol_key]) - 1, 0)

    def record_realized_pnl(self, pnl_delta: float, now_ts: Optional[float] = None) -> None:
        now_value = time.time() if now_ts is None else float(now_ts)
        self._roll_day(now_value)
        self._realized_pnl_today += float(pnl_delta or 0.0)

    def record_fill(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        fill_price: float,
        expected_fill_price: float = 0.0,
        reference_price: float = 0.0,
        now_ts: Optional[float] = None,
    ) -> Dict[str, float]:
        now_value = time.time() if now_ts is None else float(now_ts)
        self._roll_day(now_value)

        symbol_key = str(symbol or "").strip().upper()
        position = self._positions.get(symbol_key, {"qty": 0.0, "avg_price": 0.0})
        prev_qty = float(position.get("qty", 0.0) or 0.0)
        prev_avg = float(position.get("avg_price", 0.0) or 0.0)

        signed_qty = self._signed_quantity(action, quantity)
        price = max(float(fill_price or 0.0), 0.0)

        realized_delta = 0.0
        new_qty = prev_qty
        new_avg = prev_avg

        if signed_qty != 0.0 and price > 0.0:
            if prev_qty == 0.0 or (prev_qty > 0.0 and signed_qty > 0.0) or (prev_qty < 0.0 and signed_qty < 0.0):
                total_abs = abs(prev_qty) + abs(signed_qty)
                new_qty = prev_qty + signed_qty
                if total_abs > 0.0 and new_qty != 0.0:
                    new_avg = ((abs(prev_qty) * prev_avg) + (abs(signed_qty) * price)) / total_abs
                else:
                    new_avg = 0.0
            else:
                closing_qty = min(abs(prev_qty), abs(signed_qty))
                if prev_qty > 0.0:
                    realized_delta = (price - prev_avg) * closing_qty
                else:
                    realized_delta = (prev_avg - price) * closing_qty

                residual_abs = abs(signed_qty) - closing_qty
                if residual_abs > 0.0:
                    new_qty = (1.0 if signed_qty > 0.0 else -1.0) * residual_abs
                    new_avg = price
                else:
                    new_qty = prev_qty + signed_qty
                    if new_qty == 0.0:
                        new_avg = 0.0
                    else:
                        new_avg = prev_avg

        self._positions[symbol_key] = {"qty": float(new_qty), "avg_price": float(new_avg)}
        self._realized_pnl_today += float(realized_delta)

        ref_price = max(float(reference_price or 0.0), 0.0)
        exp_price = max(float(expected_fill_price or 0.0), 0.0)
        if exp_price <= 0.0 and ref_price > 0.0:
            exp = self.model_expected_fill(
                action=action,
                reference_price=ref_price,
                quantity=quantity,
            )
            exp_price = max(float(exp.get("expected_fill_price", 0.0) or 0.0), 0.0)

        fill_quality = self.evaluate_fill_quality(
            action=action,
            actual_fill_price=price,
            expected_fill_price=exp_price if exp_price > 0.0 else price,
        )

        realized_slippage_bps = 0.0
        if ref_price > 0.0 and price > 0.0:
            side = str(action or "").strip().upper()
            if side in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
                realized_slippage_bps = max(((price - ref_price) / ref_price) * 10000.0, 0.0)
            elif side in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
                realized_slippage_bps = max(((ref_price - price) / ref_price) * 10000.0, 0.0)

        self._fill_count += 1
        self._fill_slippage_bps_sum += float(realized_slippage_bps)
        self._fill_deviation_bps_sum += float(fill_quality.get("fill_deviation_bps", 0.0) or 0.0)
        if not bool(fill_quality.get("ok", True)):
            self._fill_deviation_violations += 1

        return {
            "symbol": symbol_key,
            "position_qty": float(new_qty),
            "position_avg_price": float(new_avg),
            "realized_pnl_delta": float(realized_delta),
            "realized_pnl_today": float(self._realized_pnl_today),
            "fill_price": float(price),
            "expected_fill_price": float(exp_price),
            "realized_slippage_bps": round(float(realized_slippage_bps), 6),
            "fill_quality": fill_quality,
        }

    def local_position_qty(self, symbol: str) -> float:
        symbol_key = str(symbol or "").strip().upper()
        row = self._positions.get(symbol_key, {"qty": 0.0})
        return float(row.get("qty", 0.0) or 0.0)

    def set_local_position(self, *, symbol: str, quantity: float, avg_price: Optional[float] = None) -> Dict[str, float]:
        symbol_key = str(symbol or "").strip().upper()
        prior = self._positions.get(symbol_key, {"qty": 0.0, "avg_price": 0.0})
        prev_avg = float(prior.get("avg_price", 0.0) or 0.0)
        next_avg = prev_avg if avg_price is None else max(float(avg_price or 0.0), 0.0)
        self._positions[symbol_key] = {
            "qty": float(quantity or 0.0),
            "avg_price": float(next_avg),
        }
        return {
            "symbol": symbol_key,
            "position_qty": float(self._positions[symbol_key]["qty"]),
            "position_avg_price": float(self._positions[symbol_key]["avg_price"]),
        }

    def reconcile_broker_position(
        self,
        *,
        symbol: str,
        broker_qty: float,
        tolerance: float = 0.0001,
        manual_adjustment_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol_key = str(symbol or "").strip().upper()
        broker_q = float(broker_qty or 0.0)
        local_q = self.local_position_qty(symbol_key)
        diff = broker_q - local_q
        tol = max(float(tolerance), 0.0)
        manual_tol = tol if manual_adjustment_tolerance is None else max(float(manual_adjustment_tolerance), 0.0)
        mismatch = abs(diff) > tol
        manual_adjustment = mismatch and (abs(diff) <= manual_tol)
        ok = not mismatch
        status = "match" if ok else ("manual_adjustment_detected" if manual_adjustment else "mismatch")
        self._broker_positions[symbol_key] = broker_q
        return {
            "symbol": symbol_key,
            "ok": bool(ok),
            "local_qty": float(local_q),
            "broker_qty": float(broker_q),
            "difference": float(diff),
            "tolerance": float(tol),
            "manual_adjustment_tolerance": float(manual_tol),
            "manual_adjustment_detected": bool(manual_adjustment),
            "status": status,
        }

    def reconcile_order_lifecycle(
        self,
        *,
        broker_open_orders: Optional[list[Dict[str, Any]]] = None,
        position_tolerance: float = 0.0001,
        position_manual_adjustment_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        broker_rows = broker_open_orders if isinstance(broker_open_orders, list) else []
        broker_ids = {str(r.get("order_id", "")).strip() for r in broker_rows if str(r.get("order_id", "")).strip()}
        local_ids = set(self._open_orders.keys())

        missing_on_broker = sorted(local_ids - broker_ids)
        missing_local = sorted(broker_ids - local_ids)

        position_checks: list[Dict[str, Any]] = []
        tol = max(float(position_tolerance), 0.0)
        for symbol, broker_qty in self._broker_positions.items():
            rec = self.reconcile_broker_position(
                symbol=symbol,
                broker_qty=float(broker_qty),
                tolerance=tol,
                manual_adjustment_tolerance=position_manual_adjustment_tolerance,
            )
            position_checks.append(rec)

        pos_ok = all(bool(r.get("ok", False)) for r in position_checks) if position_checks else True
        ok = (len(missing_on_broker) == 0) and (len(missing_local) == 0) and pos_ok

        return {
            "ok": bool(ok),
            "missing_on_broker": missing_on_broker,
            "missing_local": missing_local,
            "position_checks": position_checks,
            "open_orders_local_total": int(len(local_ids)),
            "open_orders_broker_total": int(len(broker_ids)),
        }

    def open_order_ids(self) -> list[str]:
        return sorted(self._open_orders.keys())

    def snapshot(self) -> Dict[str, Any]:
        return {
            "daily_key": self._daily_key,
            "realized_pnl_today": float(self._realized_pnl_today),
            "open_orders_total": int(len(self._open_orders)),
            "open_orders_by_symbol": dict(self._open_orders_by_symbol),
            "positions": {k: dict(v) for k, v in self._positions.items()},
            "broker_positions": dict(self._broker_positions),
            "config": {
                "max_position_qty_per_symbol": float(self.config.max_position_qty_per_symbol),
                "max_order_notional": float(self.config.max_order_notional),
                "max_open_orders_total": int(self.config.max_open_orders_total),
                "max_open_orders_per_symbol": int(self.config.max_open_orders_per_symbol),
                "daily_loss_cap": float(self.config.daily_loss_cap),
                "api_fail_limit": int(self.config.api_fail_limit),
                "api_cooldown_seconds": int(self.config.api_cooldown_seconds),
                "trade_min_interval_seconds": float(self.config.trade_min_interval_seconds),
                "trade_min_interval_global_seconds": float(self.config.trade_min_interval_global_seconds),
                "max_slippage_bps": float(self.config.max_slippage_bps),
                "max_fill_deviation_bps": float(self.config.max_fill_deviation_bps),
            },
            "fill_modeling": {
                "fill_count": int(self._fill_count),
                "avg_realized_slippage_bps": round(float(self._fill_slippage_bps_sum / self._fill_count), 6) if self._fill_count > 0 else 0.0,
                "avg_fill_deviation_bps": round(float(self._fill_deviation_bps_sum / self._fill_count), 6) if self._fill_count > 0 else 0.0,
                "fill_deviation_violations": int(self._fill_deviation_violations),
            },
        }
