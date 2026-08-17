from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    "CLOSE",
    "ROLL",
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
    min_execution_realism_score: float = 25.0
    min_effective_fill_ratio: float = 0.50
    max_reject_probability: float = 0.80
    max_cancel_probability: float = 0.85
    max_stale_quote_probability: float = 0.80
    allow_new_short_positions: bool = False

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
            min_execution_realism_score=max(float(os.getenv("LIVE_MIN_EXECUTION_REALISM_SCORE", "25")), 0.0),
            min_effective_fill_ratio=max(float(os.getenv("LIVE_MIN_EFFECTIVE_FILL_RATIO", "0.50")), 0.0),
            max_reject_probability=max(float(os.getenv("LIVE_MAX_REJECT_PROBABILITY", "0.80")), 0.0),
            max_cancel_probability=max(float(os.getenv("LIVE_MAX_CANCEL_PROBABILITY", "0.85")), 0.0),
            max_stale_quote_probability=max(float(os.getenv("LIVE_MAX_STALE_QUOTE_PROBABILITY", "0.80")), 0.0),
            allow_new_short_positions=_truthy(os.getenv("LIVE_ALLOW_NEW_SHORT_POSITIONS", "0"), False),
        )


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


def _project_path(project_root: str | Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else Path(project_root) / path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_production_firewall_policy(project_root: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(project_root) / "config" / "production_readiness_control_v1.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    policy = payload.get("live_execution_risk_firewall")
    return (policy if isinstance(policy, dict) else {}), config_path


def production_order_firewall_check(
    *,
    project_root: str | Path,
    symbol: str,
    action: str,
    quantity: float,
    order_spec: Dict[str, Any],
    reference_price: float = 0.0,
    risk_reducing_exit: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> GuardDecision:
    env_map = env if isinstance(env, dict) else dict(os.environ)
    policy, config_path = _load_production_firewall_policy(project_root)
    allow_env = str(policy.get("allow_order_execution_env") or "ALLOW_ORDER_EXECUTION")
    market_data_env = str(policy.get("market_data_only_env") or "MARKET_DATA_ONLY")
    market_data_default = bool(policy.get("market_data_only_default", True))

    execution_armed = _truthy(env_map.get(allow_env), False)
    market_data_only = _truthy(env_map.get(market_data_env), market_data_default)
    blockers: list[str] = []
    if not execution_armed:
        blockers.append("live_execution_not_armed")
    if market_data_only:
        blockers.append("market_data_only_active")

    production_excellence: dict[str, Any] = {}
    excellence_path = _project_path(
        project_root,
        policy.get("production_excellence_artifact")
        or "governance/health/production_excellence_control_latest.json",
    )
    if bool(policy.get("require_production_excellence_for_live_submit", True)) and not risk_reducing_exit:
        try:
            loaded = json.loads(excellence_path.read_text(encoding="utf-8"))
            production_excellence = loaded if isinstance(loaded, dict) else {}
        except Exception:
            production_excellence = {}
        if not bool(
            production_excellence.get("ten_out_of_ten_ready", False)
            and production_excellence.get("live_money_consideration_ready", False)
        ):
            blockers.append("production_excellence_not_ready")

    transition_integrity: dict[str, Any] = {}
    transition_path = _project_path(
        project_root,
        policy.get("live_transition_integrity_artifact")
        or "governance/health/live_transition_integrity_control_latest.json",
    )
    if bool(policy.get("require_live_transition_integrity_for_live_submit", False)) and not risk_reducing_exit:
        try:
            loaded = json.loads(transition_path.read_text(encoding="utf-8"))
            transition_integrity = loaded if isinstance(loaded, dict) else {}
        except Exception:
            transition_integrity = {}
        if not bool(
            str(transition_integrity.get("control_grade") or "").strip().upper() in {"A+", "A++"}
            and transition_integrity.get("ready_for_live_transition", False)
        ):
            blockers.append("live_transition_integrity_not_ready")

    active_halt_flags: list[str] = []
    if _truthy(env_map.get("OPERATOR_STOP"), False):
        active_halt_flags.append("env:OPERATOR_STOP")
    if _truthy(env_map.get("GLOBAL_TRADING_HALT"), False):
        active_halt_flags.append("env:GLOBAL_TRADING_HALT")
    for raw_path in _string_list(policy.get("halt_flags")):
        path = _project_path(project_root, raw_path)
        if path.exists():
            active_halt_flags.append(str(path))
    if active_halt_flags and not risk_reducing_exit:
        blockers.append("halt_flags_active")

    missing_safety_flags: list[str] = []
    for raw_path in _string_list(policy.get("required_safety_flags")):
        path = _project_path(project_root, raw_path)
        if not path.exists():
            missing_safety_flags.append(str(path))
    if missing_safety_flags and not risk_reducing_exit:
        blockers.append("required_safety_flag_missing")

    qty = max(float(quantity or 0.0), 0.0)
    max_qty = float(policy.get("max_order_quantity") or 0.0)
    if not risk_reducing_exit and max_qty > 0.0 and qty > max_qty:
        blockers.append("quantity_exceeds_cap")

    order_price = 0.0
    try:
        order_price = float((order_spec or {}).get("price") or 0.0)
    except Exception:
        order_price = 0.0
    if order_price <= 0.0:
        try:
            order_price = max(float(reference_price or 0.0), 0.0)
        except Exception:
            order_price = 0.0
    max_notional = float(policy.get("max_single_order_notional") or 0.0)
    legs = (order_spec or {}).get("orderLegCollection")
    asset_types = [
            str(((leg or {}).get("instrument") or {}).get("assetType") or "").upper()
            for leg in legs
            if isinstance(leg, dict)
        ] if isinstance(legs, list) else []
    instructions = [
        str((leg or {}).get("instruction") or "").upper()
        for leg in legs
        if isinstance(leg, dict)
    ] if isinstance(legs, list) else []
    leg_symbols = [
        str(((leg or {}).get("instrument") or {}).get("symbol") or "").strip().upper()
        for leg in legs
        if isinstance(leg, dict)
    ] if isinstance(legs, list) else []
    allowed_asset_types = {str(item).upper() for item in _string_list(policy.get("allowed_asset_types"))}
    allowed_instructions = {str(item).upper() for item in _string_list(policy.get("allowed_instructions"))}
    if allowed_asset_types and (not asset_types or any(item not in allowed_asset_types for item in asset_types)):
        blockers.append("asset_type_not_allowed")
    effective_instructions = instructions or [str(action or "").strip().upper()]
    if not risk_reducing_exit and allowed_instructions and any(item not in allowed_instructions for item in effective_instructions):
        blockers.append("instruction_not_allowed")

    symbol_key = str(symbol or "").strip().upper()
    if not leg_symbols or any(item != symbol_key for item in leg_symbols):
        blockers.append("order_symbol_mismatch")
    allowlist_path = _project_path(project_root, policy.get("canary_allowlist_path") or "")
    canary_allowlist: list[str] = []
    if allowlist_path.exists():
        try:
            loaded_allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
            raw_symbols = loaded_allowlist.get("symbols", []) if isinstance(loaded_allowlist, dict) else []
            canary_allowlist = [str(item).strip().upper() for item in raw_symbols if str(item).strip()]
        except Exception:
            canary_allowlist = []
    is_new_entry = str(action or "").strip().upper() in {"BUY", "BUY_TO_OPEN", "SELL_SHORT", "SELL_TO_OPEN"}
    if is_new_entry and not risk_reducing_exit and (not canary_allowlist or symbol_key not in set(canary_allowlist)):
        blockers.append("symbol_not_in_live_canary_allowlist")

    if not risk_reducing_exit and max_notional > 0.0 and order_price > 0.0:
        multiplier = 100.0 if "OPTION" in asset_types else 1.0
        notional = abs(order_price * qty * multiplier)
        if notional > max_notional:
            blockers.append("notional_exceeds_cap")
    elif not risk_reducing_exit and max_notional > 0.0:
        notional = 0.0
        blockers.append("reference_price_required_for_notional_cap")
    else:
        notional = 0.0

    details = {
        "symbol": str(symbol or "").strip().upper(),
        "action": str(action or "").strip().upper(),
        "quantity": float(qty),
        "execution_armed": bool(execution_armed),
        "market_data_only": bool(market_data_only),
        "allow_order_execution_env": allow_env,
        "market_data_only_env": market_data_env,
        "active_halt_flags": active_halt_flags,
        "missing_safety_flags": missing_safety_flags,
        "order_price": float(order_price),
        "estimated_notional": float(notional),
        "asset_types": asset_types,
        "instructions": effective_instructions,
        "leg_symbols": leg_symbols,
        "risk_reducing_exit": bool(risk_reducing_exit),
        "allowed_asset_types": sorted(allowed_asset_types),
        "allowed_instructions": sorted(allowed_instructions),
        "canary_allowlist_path": str(allowlist_path),
        "canary_allowlist": canary_allowlist,
        "production_excellence_path": str(excellence_path),
        "production_excellence_ready": bool(production_excellence.get("ten_out_of_ten_ready", False)),
        "live_transition_integrity_path": str(transition_path),
        "live_transition_control_ready": bool(
            str(transition_integrity.get("control_grade") or "").strip().upper() in {"A+", "A++"}
        ),
        "live_transition_runtime_ready": bool(transition_integrity.get("ready_for_live_transition", False)),
        "config_path": str(config_path),
        "policy": "reject_by_default_until_production_firewall_is_armed_and_clear; verified emergency exits remain risk reducing",
    }
    if blockers:
        return GuardDecision(
            ok=False,
            gate="production_order_firewall",
            reason=blockers[0],
            details={**details, "blockers": blockers},
        )
    return GuardDecision(ok=True, gate="production_order_firewall", reason="ok", details=details)


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
        notional_multiplier: float = 1.0,
        now_ts: Optional[float] = None,
        enforce_execution_realism: bool = False,
        spread_bps: float = 8.0,
        volatility_1m: float = 0.0,
        latency_ms: float = 120.0,
        bid_size: float = 1000.0,
        ask_size: float = 1000.0,
        broker: str = "",
        market_kind: str = "",
        session: str = "regular",
        order_type: str = "market",
        asset_class: str = "",
        sleeve: str = "",
        quote_age_ms: float = 0.0,
        market_volume: float = 0.0,
        avg_daily_volume: float = 0.0,
        open_interest: float = 0.0,
        live_fill_slippage_bps: float = 0.0,
        enforce_long_only: bool = True,
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

        if enforce_long_only and projected_qty < 0.0 and not self.config.allow_new_short_positions:
            return GuardDecision(
                ok=False,
                gate="short_position_limit",
                reason="new_short_positions_disabled",
                details={
                    "symbol": symbol_key,
                    "current_qty": float(current_qty),
                    "signed_qty": float(signed_qty),
                    "projected_qty": float(projected_qty),
                    "allow_new_short_positions": False,
                },
            )

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
            notional_scale = max(float(notional_multiplier or 1.0), 1.0)
            order_notional = abs(ref * qty * notional_scale)
            if order_notional > self.config.max_order_notional:
                return GuardDecision(
                    ok=False,
                    gate="order_notional_limit",
                    reason="order_notional_limit",
                    details={
                        "symbol": symbol_key,
                        "reference_price": float(ref),
                        "quantity": float(qty),
                        "notional_multiplier": float(notional_scale),
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

        if enforce_execution_realism:
            sim = simulate_execution(
                action=side,
                last_price=ref,
                return_1m=0.0,
                spread_bps=max(float(spread_bps or 0.0), 0.0),
                volatility_1m=max(float(volatility_1m or 0.0), 0.0),
                latency_ms=max(float(latency_ms or 0.0), 0.0),
                bid_size=max(float(bid_size or 0.0), 0.0),
                ask_size=max(float(ask_size or 0.0), 0.0),
                order_size=qty,
                broker=broker,
                market_kind=market_kind,
                symbol=symbol_key,
                session=session,
                order_type=order_type,
                live_fill_slippage_bps=max(float(live_fill_slippage_bps or 0.0), 0.0),
                asset_class=asset_class,
                sleeve=sleeve,
                quote_age_ms=max(float(quote_age_ms or 0.0), 0.0),
                market_volume=max(float(market_volume or 0.0), 0.0),
                avg_daily_volume=max(float(avg_daily_volume or 0.0), 0.0),
                open_interest=max(float(open_interest or 0.0), 0.0),
            )
            reasons: list[str] = []
            if str(sim.paper_execution_status) == "stale_quote_rejected":
                reasons.append("simulated_stale_quote_rejected")
            elif str(sim.paper_execution_status) == "rejected":
                reasons.append("simulated_order_rejected")
            if float(sim.paper_execution_score) < float(self.config.min_execution_realism_score):
                reasons.append("execution_realism_score_below_floor")
            if float(sim.effective_fill_ratio) < float(self.config.min_effective_fill_ratio):
                reasons.append("effective_fill_ratio_below_floor")
            if float(sim.reject_probability) > float(self.config.max_reject_probability):
                reasons.append("reject_probability_above_cap")
            if float(sim.cancel_probability) > float(self.config.max_cancel_probability):
                reasons.append("cancel_probability_above_cap")
            if float(sim.stale_quote_probability) > float(self.config.max_stale_quote_probability):
                reasons.append("stale_quote_probability_above_cap")
            if reasons:
                return GuardDecision(
                    ok=False,
                    gate="execution_realism_guard",
                    reason=reasons[0],
                    details={
                        "symbol": symbol_key,
                        "reasons": reasons,
                        "paper_execution_status": str(sim.paper_execution_status),
                        "paper_execution_score": round(float(sim.paper_execution_score), 6),
                        "effective_fill_ratio": round(float(sim.effective_fill_ratio), 6),
                        "reject_probability": round(float(sim.reject_probability), 6),
                        "cancel_probability": round(float(sim.cancel_probability), 6),
                        "stale_quote_probability": round(float(sim.stale_quote_probability), 6),
                        "expected_fill_price": float(sim.expected_fill_price),
                        "slippage_bps": round(float(sim.slippage_bps), 6),
                        "thresholds": {
                            "min_execution_realism_score": float(self.config.min_execution_realism_score),
                            "min_effective_fill_ratio": float(self.config.min_effective_fill_ratio),
                            "max_reject_probability": float(self.config.max_reject_probability),
                            "max_cancel_probability": float(self.config.max_cancel_probability),
                            "max_stale_quote_probability": float(self.config.max_stale_quote_probability),
                        },
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
            "partial_fill_ratio": float(sim.partial_fill_ratio),
            "spread_jump_penalty_bps": float(sim.spread_jump_penalty_bps),
            "symbol_curve_multiplier": float(sim.symbol_curve_multiplier),
            "fill_quality_bucket": str(sim.fill_quality_bucket),
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
                "min_execution_realism_score": float(self.config.min_execution_realism_score),
                "min_effective_fill_ratio": float(self.config.min_effective_fill_ratio),
                "max_reject_probability": float(self.config.max_reject_probability),
                "max_cancel_probability": float(self.config.max_cancel_probability),
                "max_stale_quote_probability": float(self.config.max_stale_quote_probability),
            },
            "fill_modeling": {
                "fill_count": int(self._fill_count),
                "avg_realized_slippage_bps": round(float(self._fill_slippage_bps_sum / self._fill_count), 6) if self._fill_count > 0 else 0.0,
                "avg_fill_deviation_bps": round(float(self._fill_deviation_bps_sum / self._fill_count), 6) if self._fill_count > 0 else 0.0,
                "fill_deviation_violations": int(self._fill_deviation_violations),
            },
        }
