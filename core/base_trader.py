# --- THE GENETIC CODE ---
# This file handles plumbing so strategy modules can focus on signal generation.

import json
import os
import random
import re
import shutil
import urllib.request
import uuid
import time
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from typing import Any, Dict, List, Optional, Tuple

from schwab.auth import easy_client

from core.decision_logger import DecisionLogger
from core.live_execution_controls import LiveExecutionGuard, LiveRiskConfig
from core.path_registry import auth_events_path, decision_explanations_paths, execution_guard_path, live_softguard_path

from core.accountability import current_correlation, now_utc_iso, safe_append_jsonl, safe_append_channel_event, safe_write_json_atomic
from core.halt_flags import write_halt_flag_atomic


class BaseTrader:
    def __init__(self, api_key: str, app_secret: str, callback_url: str, mode: str = "shadow"):
        self.api_key = api_key
        self.app_secret = app_secret
        self.callback_url = callback_url
        self.token_path = "token.json"
        self.client = None

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.mode = "shadow"
        self.profile = os.getenv("SHADOW_PROFILE", "").strip().lower()
        self.shadow_domain = ""
        self.mode_label = "shadow"
        self.paper_log_path = ""
        self.live_log_path = ""
        self.paper_bridge_enabled = os.getenv("PAPER_BROKER_BRIDGE_ENABLED", "0").strip() == "1"
        self.paper_bridge_mode = os.getenv("PAPER_BROKER_BRIDGE_MODE", "jsonl").strip().lower()
        if self.paper_bridge_mode not in {"jsonl", "webhook", "both"}:
            self.paper_bridge_mode = "jsonl"
        self.paper_bridge_url = os.getenv("PAPER_BROKER_BRIDGE_URL", "").strip()
        self.paper_bridge_timeout_seconds = max(float(os.getenv("PAPER_BROKER_BRIDGE_TIMEOUT_SECONDS", "4.0")), 0.5)
        self.paper_bridge_source = os.getenv("PAPER_BROKER_BRIDGE_SOURCE", "local_paper_mirror").strip() or "local_paper_mirror"
        self.paper_bridge_log_dir = ""
        self._paper_bridge_warned_missing_url = False
        self._paper_positions: Dict[str, Dict[str, float]] = {}
        self._paper_realized_total = 0.0

        self.live_risk_config = LiveRiskConfig.from_env()
        self.live_guard = LiveExecutionGuard(self.live_risk_config)
        self.live_account_hash = os.getenv("SCHWAB_ACCOUNT_HASH", "").strip()
        self.live_position_reconcile_tolerance = max(
            float(os.getenv("LIVE_POSITION_RECONCILE_TOLERANCE", "0.0001")),
            0.0,
        )
        self.live_pretrade_reconcile_required = os.getenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "1").strip() == "1"
        self.live_pretrade_reconcile_block_on_error = os.getenv("LIVE_PRETRADE_RECONCILE_BLOCK_ON_ERROR", "1").strip() == "1"
        self.live_pretrade_reconcile_block_on_mismatch = os.getenv("LIVE_PRETRADE_RECONCILE_BLOCK_ON_MISMATCH", "1").strip() == "1"
        self.live_pretrade_reconcile_sync_local = os.getenv("LIVE_PRETRADE_RECONCILE_SYNC_LOCAL", "1").strip() == "1"
        self.live_manual_trade_awareness_enabled = os.getenv("LIVE_MANUAL_TRADE_AWARE_ENABLED", "1").strip() == "1"
        self.live_manual_trade_qty_tolerance = max(
            float(os.getenv("LIVE_MANUAL_TRADE_QTY_TOLERANCE", os.getenv("LIVE_POSITION_RECONCILE_TOLERANCE", "0.0001"))),
            0.0,
        )
        self.live_manual_trade_auto_sync_local = os.getenv("LIVE_MANUAL_TRADE_AUTO_SYNC_LOCAL", "1").strip() == "1"

        self.global_halt_flag_path = os.path.join(self.project_root, "governance", "health", "GLOBAL_TRADING_HALT.flag")
        self.operator_stop_flag_path = os.getenv(
            "OPERATOR_STOP_FLAG_PATH",
            os.path.join(self.project_root, "governance", "health", "OPERATOR_STOP.flag"),
        ).strip()
        self.live_softguard_auto_halt_on_api_circuit = os.getenv("LIVE_SOFTGUARD_AUTO_HALT_ON_API_CIRCUIT", "1").strip() == "1"
        self.live_softguard_auto_halt_on_position_mismatch = os.getenv("LIVE_SOFTGUARD_AUTO_HALT_ON_POSITION_MISMATCH", "1").strip() == "1"
        self.live_softguard_auto_cancel_on_operator_stop = os.getenv("LIVE_SOFTGUARD_AUTO_CANCEL_ON_OPERATOR_STOP", "1").strip() == "1"
        self.live_softguard_auto_cancel_on_global_halt = os.getenv("LIVE_SOFTGUARD_AUTO_CANCEL_ON_GLOBAL_HALT", "1").strip() == "1"
        self.live_softguard_auto_cancel_cooldown_seconds = max(
            float(os.getenv("LIVE_SOFTGUARD_AUTO_CANCEL_COOLDOWN_SECONDS", "30")),
            0.0,
        )
        self.live_softguard_emergency_liquidation_enabled = os.getenv("LIVE_SOFTGUARD_EMERGENCY_LIQUIDATION_ENABLED", "0").strip() == "1"
        self.live_softguard_emergency_liquidation_cooldown_seconds = max(
            float(os.getenv("LIVE_SOFTGUARD_EMERGENCY_LIQUIDATION_COOLDOWN_SECONDS", "300")),
            0.0,
        )
        self._softguard_last_auto_cancel_ts = 0.0
        self._softguard_last_emergency_liq_ts = 0.0

        self.live_api_retry_attempts = max(int(os.getenv("LIVE_API_RETRY_ATTEMPTS", "4")), 1)
        self.live_api_retry_backoff_seconds = max(float(os.getenv("LIVE_API_RETRY_BACKOFF_SECONDS", "0.35")), 0.0)
        self.live_api_retry_backoff_multiplier = max(float(os.getenv("LIVE_API_RETRY_BACKOFF_MULTIPLIER", "2.0")), 1.0)
        self.live_api_retry_max_backoff_seconds = max(float(os.getenv("LIVE_API_RETRY_MAX_BACKOFF_SECONDS", "3.0")), 0.0)
        self.live_api_retry_jitter_seconds = max(float(os.getenv("LIVE_API_RETRY_JITTER_SECONDS", "0.1")), 0.0)
        retryable_codes = os.getenv("LIVE_API_RETRYABLE_STATUS_CODES", "408,425,429,500,502,503,504").strip()
        parsed_codes: set[int] = set()
        for token in retryable_codes.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed_codes.add(int(token))
            except Exception:
                continue
        self.live_api_retryable_status_codes = parsed_codes or {408, 425, 429, 500, 502, 503, 504}

        # Safety defaults: no order execution unless explicitly enabled.
        self.execution_enabled = os.getenv("ALLOW_ORDER_EXECUTION", "0").strip() == "1"
        self.market_data_only = os.getenv("MARKET_DATA_ONLY", "1").strip() == "1"

        self.decision_logger = DecisionLogger(self.project_root)
        self.set_mode(mode)

    def _resolve_shadow_domain(self) -> str:
        raw = os.getenv("SHADOW_DOMAIN", "").strip().lower()
        if raw in {"equities", "crypto"}:
            return raw

        broker = os.getenv("DATA_BROKER", "schwab").strip().lower()
        return "crypto" if broker == "coinbase" else "equities"

    def set_mode(self, mode: str) -> None:
        mode = mode.lower().strip()
        if mode not in {"shadow", "paper", "live"}:
            raise ValueError("Mode must be one of: shadow, paper, live")
        self.mode = mode
        self.mode_label = self.mode
        if self.mode == "shadow" and self.profile:
            self.mode_label = f"shadow_{self.profile}"

        if self.mode == "shadow":
            self.shadow_domain = self._resolve_shadow_domain()
            if self.shadow_domain:
                self.mode_label = f"{self.mode_label}_{self.shadow_domain}"

        # Keep decisions and execution artifacts isolated by mode/profile.
        paper_file = f"paper_trades_{self.mode_label}.jsonl"
        live_file = f"live_orders_{self.mode_label}.jsonl"
        legacy_paper = os.path.join(self.project_root, paper_file)
        legacy_live = os.path.join(self.project_root, live_file)

        self.paper_log_path = self._resolve_trade_log_path(paper_file)
        self.live_log_path = self._resolve_trade_log_path(live_file)

        # Keep legacy root-level filenames available for compatibility while storing
        # data under routed storage (`exports/...`) so failover/failback captures them.
        self._ensure_legacy_trade_log_link(legacy_paper, self.paper_log_path)
        self._ensure_legacy_trade_log_link(legacy_live, self.live_log_path)

        self.paper_bridge_log_dir = os.path.join(self.project_root, "exports", "paper_broker_bridge", self.mode_label)
        self.decision_logger = DecisionLogger(self.project_root, subdir=os.path.join("decisions", self.mode_label))

    def _resolve_trade_log_path(self, file_name: str) -> str:
        legacy_path = os.path.join(self.project_root, file_name)
        routed_dir = os.path.join(self.project_root, "exports", "trade_logs", self.mode_label)
        try:
            os.makedirs(routed_dir, exist_ok=True)
            return os.path.join(routed_dir, file_name)
        except Exception as exc:
            print(f"[TradeLogRoute] fallback_to_legacy file={file_name} err={exc}")
            return legacy_path

    def _ensure_legacy_trade_log_link(self, legacy_path: str, routed_path: str) -> None:
        if os.path.abspath(legacy_path) == os.path.abspath(routed_path):
            return

        try:
            os.makedirs(os.path.dirname(routed_path), exist_ok=True)

            if os.path.islink(legacy_path):
                current_target = os.path.realpath(legacy_path)
                desired_target = os.path.realpath(routed_path)
                if current_target == desired_target:
                    return
                os.unlink(legacy_path)
            elif os.path.exists(legacy_path):
                if not os.path.exists(routed_path):
                    shutil.move(legacy_path, routed_path)
                else:
                    # Preserve pre-existing legacy data by appending it once.
                    with open(legacy_path, "rb") as src, open(routed_path, "ab") as dst:
                        shutil.copyfileobj(src, dst)
                    os.remove(legacy_path)

            if os.path.lexists(legacy_path):
                os.unlink(legacy_path)

            rel_target = os.path.relpath(routed_path, os.path.dirname(legacy_path))
            os.symlink(rel_target, legacy_path)
        except Exception as exc:
            print(f"[TradeLogRoute] legacy_link_failed path={legacy_path} err={exc}")

    def _auth_event_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return auth_events_path(self.project_root, day=day)

    def _token_status(self) -> Dict[str, Any]:
        path = self.token_path if os.path.isabs(self.token_path) else os.path.join(self.project_root, self.token_path)
        status: Dict[str, Any] = {
            "token_path": path,
            "exists": os.path.exists(path),
            "size_bytes": 0,
            "age_seconds": None,
            "expires_at": "",
            "expires_in_seconds": None,
        }
        if not status["exists"]:
            return status

        try:
            st = os.stat(path)
            status["size_bytes"] = int(st.st_size)
            status["age_seconds"] = max((datetime.now(timezone.utc).timestamp() - float(st.st_mtime)), 0.0)
        except Exception:
            pass

        try:
            with open(path, "r", encoding="utf-8") as f:
                token_obj = json.load(f)
            if isinstance(token_obj, dict):
                expiry_sources = [token_obj]
                nested = token_obj.get("token")
                if isinstance(nested, dict):
                    expiry_sources.insert(0, nested)

                exp_value: Any = ""
                for source in expiry_sources:
                    for k in ("expires_at", "expiresAt", "expires", "expires_time"):
                        raw = source.get(k)
                        if raw not in (None, ""):
                            exp_value = raw
                            break
                    if exp_value not in (None, ""):
                        break

                if exp_value not in (None, ""):
                    status["expires_at"] = str(exp_value)
                    try:
                        if isinstance(exp_value, (int, float)):
                            exp_ts = float(exp_value)
                        else:
                            norm = str(exp_value).strip().replace("Z", "+00:00")
                            exp_dt = datetime.fromisoformat(norm)
                            if exp_dt.tzinfo is None:
                                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                            exp_ts = exp_dt.astimezone(timezone.utc).timestamp()
                        status["expires_in_seconds"] = float(exp_ts - datetime.now(timezone.utc).timestamp())
                    except Exception:
                        pass
        except Exception:
            pass

        return status

    def _log_auth_event(
        self,
        *,
        event: str,
        status: str,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp_utc": now_utc_iso(),
            "event": event,
            "status": status,
            "reason": reason,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "callback_url": self.callback_url,
            "requested_browser": os.getenv("SCHWAB_AUTH_REQUESTED_BROWSER", "").strip(),
            "token_status": self._token_status(),
        }
        corr = current_correlation()
        if corr.get("run_id"):
            payload["run_id"] = corr["run_id"]
        if corr.get("iter_id"):
            payload["iter_id"] = corr["iter_id"]
        if details:
            payload["details"] = details
        safe_append_channel_event(
            self._auth_event_path(),
            payload,
            project_root=self.project_root,
            source="base_trader.authenticate",
            channel="auth",
            schema="auth",
        )

    def authenticate(self):
        """Performs Schwab OAuth handshake."""
        print("Starting Handshake with Schwab...")

        max_token_age_raw = os.getenv("SCHWAB_MAX_TOKEN_AGE_SECONDS", "0").strip().lower()
        if max_token_age_raw in {"", "none", "null"}:
            max_token_age = None
        else:
            try:
                max_token_age = max(float(max_token_age_raw), 0.0)
            except Exception:
                max_token_age = 0.0

        interactive = os.getenv("SCHWAB_AUTH_INTERACTIVE", "0").strip() == "1"
        callback_timeout = float(os.getenv("SCHWAB_AUTH_CALLBACK_TIMEOUT_SECONDS", "300"))
        requested_browser = os.getenv("SCHWAB_AUTH_REQUESTED_BROWSER", "").strip() or None

        self._log_auth_event(
            event="auth_start",
            status="started",
            details={
                "interactive": bool(interactive),
                "callback_timeout_seconds": float(callback_timeout),
                "requested_browser": requested_browser,
                "max_token_age_seconds": max_token_age,
            },
        )

        try:
            self.client = easy_client(
                api_key=self.api_key,
                app_secret=self.app_secret,
                callback_url=self.callback_url,
                token_path=self.token_path,
                max_token_age=max_token_age,
                callback_timeout=callback_timeout,
                interactive=interactive,
                requested_browser=requested_browser,
            )
            self._log_auth_event(
                event="auth_success",
                status="ok",
                details={
                    "interactive": bool(interactive),
                    "callback_timeout_seconds": float(callback_timeout),
                    "requested_browser": requested_browser,
                    "max_token_age_seconds": max_token_age,
                },
            )
            print("Handshake Successful.")
            return self.client
        except Exception as exc:
            self._log_auth_event(
                event="auth_error",
                status="error",
                reason=f"{type(exc).__name__}:{exc}",
                details={
                    "interactive": bool(interactive),
                    "callback_timeout_seconds": float(callback_timeout),
                    "requested_browser": requested_browser,
                    "max_token_age_seconds": max_token_age,
                },
            )
            raise

    def get_market_data(self, symbol: str):
        """Placeholder for standardized price/history fetching."""
        _ = symbol
        return None

    def _is_trade_action(self, action: str) -> bool:
        return action.strip().upper() in {
            "BUY",
            "SELL",
            "SELL_SHORT",
            "BUY_TO_COVER",
            "BUY_TO_OPEN",
            "BUY_TO_CLOSE",
            "SELL_TO_OPEN",
            "SELL_TO_CLOSE",
        }

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _paper_signed_quantity(self, action: str, quantity: float) -> float:
        qty = max(self._as_float(quantity, 0.0), 0.0)
        side = action.strip().upper()
        if side in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
            return qty
        if side in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
            return -qty
        return 0.0

    def _paper_fill_price(self, *, features: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        for key in ("fill_price", "execution_price", "price", "mark_price", "last_price"):
            val = self._as_float(metadata.get(key), 0.0)
            if val > 0.0:
                return val
        return max(self._as_float(features.get("last_price"), 0.0), 0.0)

    def _paper_execution_model_inputs(
        self,
        *,
        symbol: str,
        features: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        symbol_key = str(symbol or "").strip().upper()
        vol = max(
            self._as_float(features.get("volatility_1m"), 0.0),
            self._as_float(features.get("vol_30m"), 0.0) / 3.0,
            self._as_float(metadata.get("volatility_1m"), 0.0),
        )
        spread = max(
            self._as_float(metadata.get("spread_bps"), 0.0),
            self._as_float(features.get("spread_bps"), 0.0),
        )
        if spread <= 0.0:
            spread = 16.0 if self.shadow_domain == "crypto" else 8.0
            spread += min(max(vol * 2500.0, 0.0), 25.0)
            if symbol_key in {"MSTR", "SMCI", "COIN", "TSLA", "SOXL", "SOXS"}:
                spread *= 1.25

        if ZoneInfo is not None:
            now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
            if now_et.hour < 9 or now_et.hour >= 16:
                spread *= 1.4

        latency_ms = max(
            self._as_float(metadata.get("latency_ms"), 0.0),
            self._as_float(features.get("latency_ms"), 0.0),
        )
        if latency_ms <= 0.0:
            qd = max(self._as_float(features.get("queue_depth"), 0.0), self._as_float(metadata.get("queue_depth"), 0.0))
            latency_ms = 120.0 + min(qd, 1000.0) * 0.05

        bid_size = max(self._as_float(metadata.get("bid_size"), 0.0), self._as_float(features.get("bid_size"), 0.0), 250.0)
        ask_size = max(self._as_float(metadata.get("ask_size"), 0.0), self._as_float(features.get("ask_size"), 0.0), 250.0)

        return {
            "spread_bps": float(max(spread, 0.5)),
            "volatility_1m": float(max(vol, 0.0)),
            "latency_ms": float(min(max(latency_ms, 40.0), 1200.0)),
            "bid_size": float(bid_size),
            "ask_size": float(ask_size),
        }

    def _paper_pnl_fields(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        features: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        symbol_key = str(symbol).upper()
        position = self._paper_positions.get(symbol_key, {"qty": 0.0, "avg_price": 0.0, "mark_price": 0.0})
        prev_qty = self._as_float(position.get("qty"), 0.0)
        prev_avg = self._as_float(position.get("avg_price"), 0.0)

        fill_price = self._paper_fill_price(features=features, metadata=metadata)
        mark_price = max(
            self._as_float(metadata.get("mark_price"), 0.0),
            self._as_float(features.get("last_price"), 0.0),
            fill_price,
        )
        signed_qty = self._paper_signed_quantity(action, quantity)

        realized_delta = 0.0
        new_qty = prev_qty
        new_avg = prev_avg

        if signed_qty != 0.0 and fill_price > 0.0:
            if prev_qty == 0.0 or (prev_qty > 0.0 and signed_qty > 0.0) or (prev_qty < 0.0 and signed_qty < 0.0):
                total_abs = abs(prev_qty) + abs(signed_qty)
                new_qty = prev_qty + signed_qty
                if total_abs > 0.0 and new_qty != 0.0:
                    new_avg = ((abs(prev_qty) * prev_avg) + (abs(signed_qty) * fill_price)) / total_abs
                else:
                    new_avg = 0.0
            else:
                closing_qty = min(abs(prev_qty), abs(signed_qty))
                if prev_qty > 0.0:
                    realized_delta = (fill_price - prev_avg) * closing_qty
                else:
                    realized_delta = (prev_avg - fill_price) * closing_qty

                residual_abs = abs(signed_qty) - closing_qty
                if residual_abs > 0.0:
                    new_qty = (1.0 if signed_qty > 0.0 else -1.0) * residual_abs
                    new_avg = fill_price
                else:
                    new_qty = prev_qty + signed_qty
                    if new_qty == 0.0:
                        new_avg = 0.0
                    else:
                        new_avg = prev_avg

        if mark_price <= 0.0:
            mark_price = fill_price

        self._paper_realized_total += realized_delta
        position = {
            "qty": float(new_qty),
            "avg_price": float(new_avg),
            "mark_price": float(mark_price),
        }
        self._paper_positions[symbol_key] = position

        unrealized_symbol = float(new_qty) * (float(mark_price) - float(new_avg))
        unrealized_total = 0.0
        for row in self._paper_positions.values():
            qty_i = self._as_float(row.get("qty"), 0.0)
            avg_i = self._as_float(row.get("avg_price"), 0.0)
            mark_i = self._as_float(row.get("mark_price"), 0.0)
            if mark_i <= 0.0:
                mark_i = avg_i
            unrealized_total += qty_i * (mark_i - avg_i)

        return {
            "fill_price": float(fill_price),
            "mark_price": float(mark_price),
            "position_qty": float(new_qty),
            "position_avg_price": float(new_avg),
            "realized": float(realized_delta),
            "unrealized": float(unrealized_symbol),
            "realized_pnl": float(realized_delta),
            "unrealized_pnl": float(unrealized_symbol),
            "realized_pnl_total": float(self._paper_realized_total),
            "unrealized_pnl_total": float(unrealized_total),
        }

    def _explanation_log_paths(self) -> tuple[str, str]:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        jsonl_path, text_path = decision_explanations_paths(self.project_root, self.mode_label, day=day)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        return jsonl_path, text_path

    def _paper_bridge_paths(self) -> tuple[str, str]:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        os.makedirs(self.paper_bridge_log_dir, exist_ok=True)
        daily_jsonl = os.path.join(self.paper_bridge_log_dir, f"paper_bridge_orders_{day}.jsonl")
        latest_json = os.path.join(self.paper_bridge_log_dir, "latest_order.json")
        return daily_jsonl, latest_json

    def _bridge_paper_order(self, *, paper_order: Dict[str, Any], result_status: str) -> Dict[str, Any]:
        bridge_result: Dict[str, Any] = {
            "enabled": self.paper_bridge_enabled,
            "mode": self.paper_bridge_mode,
            "webhook_configured": bool(self.paper_bridge_url),
            "jsonl_written": False,
            "webhook_sent": False,
            "webhook_status_code": 0,
            "error": "",
        }
        if not self.paper_bridge_enabled:
            return bridge_result

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "bridge_source": self.paper_bridge_source,
            "bridge_mode_label": self.mode_label,
            "status": result_status,
            "symbol": str(paper_order.get("symbol", "")),
            "action": str(paper_order.get("action", "")),
            "quantity": float(paper_order.get("quantity", 0.0) or 0.0),
            "model_score": float(paper_order.get("model_score", 0.0) or 0.0),
            "threshold": float(paper_order.get("threshold", 0.0) or 0.0),
            "strategy": str(paper_order.get("strategy", "")),
            "metadata": paper_order.get("metadata", {}) if isinstance(paper_order.get("metadata"), dict) else {},
            "fill_price": float(paper_order.get("fill_price", 0.0) or 0.0),
            "mark_price": float(paper_order.get("mark_price", 0.0) or 0.0),
            "position_qty": float(paper_order.get("position_qty", 0.0) or 0.0),
            "position_avg_price": float(paper_order.get("position_avg_price", 0.0) or 0.0),
            "realized": float(paper_order.get("realized", 0.0) or 0.0),
            "unrealized": float(paper_order.get("unrealized", 0.0) or 0.0),
            "realized_pnl": float(paper_order.get("realized_pnl", 0.0) or 0.0),
            "unrealized_pnl": float(paper_order.get("unrealized_pnl", 0.0) or 0.0),
            "realized_pnl_total": float(paper_order.get("realized_pnl_total", 0.0) or 0.0),
            "unrealized_pnl_total": float(paper_order.get("unrealized_pnl_total", 0.0) or 0.0),
            "decision_id": str(paper_order.get("decision_id", "") or ""),
            "parent_decision_id": str(paper_order.get("parent_decision_id", "") or ""),
            "run_id": str(paper_order.get("run_id", "") or ""),
            "iter_id": str(paper_order.get("iter_id", "") or ""),
        }

        try:
            if self.paper_bridge_mode in {"jsonl", "both"}:
                daily_jsonl, latest_json = self._paper_bridge_paths()
                self._record_jsonl(daily_jsonl, payload)
                safe_write_json_atomic(
                    latest_json,
                    payload,
                    project_root=self.project_root,
                    source="base_trader.paper_bridge_latest_order",
                )
                bridge_result["jsonl_written"] = True

            if self.paper_bridge_mode in {"webhook", "both"}:
                if not self.paper_bridge_url:
                    if not self._paper_bridge_warned_missing_url:
                        print("[PaperBridge] webhook mode enabled but PAPER_BROKER_BRIDGE_URL is empty")
                        self._paper_bridge_warned_missing_url = True
                    bridge_result["error"] = "missing_webhook_url"
                    return bridge_result

                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                req = urllib.request.Request(
                    url=self.paper_bridge_url,
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.paper_bridge_timeout_seconds) as resp:
                    status_code = int(getattr(resp, "status", 0) or resp.getcode() or 0)
                bridge_result["webhook_status_code"] = status_code
                bridge_result["webhook_sent"] = 200 <= status_code < 300
                if not bridge_result["webhook_sent"]:
                    bridge_result["error"] = f"webhook_non_2xx:{status_code}"
        except Exception as exc:
            bridge_result["error"] = str(exc)

        return bridge_result

    def _emit_decision_explanation(
        self,
        *,
        status: str,
        decision_entry: Dict[str, Any],
        safety: Optional[Dict[str, Any]] = None,
    ) -> None:
        gates = decision_entry.get("gates", {})
        gate_summary = ", ".join(
            f"{k}={'PASS' if bool(v) else 'FAIL'}" for k, v in gates.items()
        ) or "none"

        reasons = decision_entry.get("reasons", [])
        reasons_summary = " | ".join(str(r) for r in reasons) if reasons else "none"

        safety_summary = ""
        if safety:
            safety_summary = (
                " safety="
                f"market_data_only={int(bool(safety.get('market_data_only')))}"
                f",execution_enabled={int(bool(safety.get('execution_enabled')))}"
            )

        line = (
            "[Decision] "
            f"mode={self.mode_label} "
            f"status={status} "
            f"symbol={decision_entry.get('symbol')} "
            f"action={decision_entry.get('action')} "
            f"score={float(decision_entry.get('model_score', 0.0)):.4f} "
            f"threshold={float(decision_entry.get('threshold', 0.0)):.4f} "
            f"reasons={reasons_summary} "
            f"gates={gate_summary}"
            f"{safety_summary}"
        )
        print(line)

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode_label,
            "status": status,
            "symbol": decision_entry.get("symbol"),
            "action": decision_entry.get("action"),
            "quantity": decision_entry.get("quantity"),
            "strategy": decision_entry.get("strategy"),
            "decision_id": decision_entry.get("decision_id", ""),
            "parent_decision_id": decision_entry.get("parent_decision_id", ""),
            "run_id": decision_entry.get("run_id", ""),
            "iter_id": decision_entry.get("iter_id", ""),
            "model_score": float(decision_entry.get("model_score", 0.0)),
            "threshold": float(decision_entry.get("threshold", 0.0)),
            "reasons": reasons,
            "gates": gates,
            "features": decision_entry.get("features", {}),
            "safety": safety or {},
            "metadata": decision_entry.get("metadata", {}),
        }

        jsonl_path, text_path = self._explanation_log_paths()
        safe_append_channel_event(
            jsonl_path,
            payload,
            project_root=self.project_root,
            source="base_trader.decision_explanation",
            channel="decision",
            schema="decision",
        )
        try:
            with open(text_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            safe_append_jsonl(
                self._auth_event_path(),
                {
                    "timestamp_utc": now_utc_iso(),
                    "event": "decision_text_write_error",
                    "status": "error",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "mode": self.mode,
                    "mode_label": self.mode_label,
                },
                project_root=self.project_root,
                source="base_trader.decision_explanation",
            )

    def _execution_guard_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return execution_guard_path(self.project_root, self.mode, day=day)

    def _log_live_guard_event(
        self,
        *,
        event: str,
        status: str,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        corr = current_correlation()
        payload: Dict[str, Any] = {
            "timestamp_utc": now_utc_iso(),
            "event": str(event or "").strip(),
            "status": str(status or "").strip(),
            "reason": str(reason or "").strip(),
            "mode": self.mode,
            "mode_label": self.mode_label,
            "account_hash_configured": bool(self.live_account_hash),
        }
        if corr.get("run_id"):
            payload["run_id"] = corr["run_id"]
        if corr.get("iter_id"):
            payload["iter_id"] = corr["iter_id"]
        if details:
            payload["details"] = details

        safe_append_channel_event(
            self._execution_guard_path(),
            payload,
            project_root=self.project_root,
            source="base_trader.live_guard",
            channel="execution_guard",
            schema="execution_guard",
        )

    def _live_softguard_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return live_softguard_path(self.project_root, day=day)

    def _log_softguard_event(
        self,
        *,
        event: str,
        status: str,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        corr = current_correlation()
        payload: Dict[str, Any] = {
            "timestamp_utc": now_utc_iso(),
            "event": str(event or "").strip(),
            "status": str(status or "").strip(),
            "reason": str(reason or "").strip(),
            "mode": self.mode,
            "mode_label": self.mode_label,
            "global_halt": bool(self._global_trading_halt_enabled()),
            "operator_stop": bool(self._operator_stop_enabled()),
        }
        if corr.get("run_id"):
            payload["run_id"] = corr["run_id"]
        if corr.get("iter_id"):
            payload["iter_id"] = corr["iter_id"]
        if details:
            payload["details"] = details

        safe_append_channel_event(
            self._live_softguard_path(),
            payload,
            project_root=self.project_root,
            source="base_trader.live_softguard",
            channel="softguard",
            schema="softguard",
        )

    def _global_trading_halt_enabled(self) -> bool:
        env_halt = os.getenv("GLOBAL_TRADING_HALT", "0").strip().lower() in {"1", "true", "yes", "on"}
        return env_halt or os.path.exists(self.global_halt_flag_path)

    def _operator_stop_enabled(self) -> bool:
        env_stop = os.getenv("OPERATOR_STOP", "0").strip().lower() in {"1", "true", "yes", "on"}
        return env_stop or bool(self.operator_stop_flag_path and os.path.exists(self.operator_stop_flag_path))

    def _engage_global_halt(self, *, reason: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "timestamp_utc": now_utc_iso(),
            "reason": str(reason or "softguard"),
            "details": details or {},
            "source": "base_trader.softguard",
        }
        try:
            ok = write_halt_flag_atomic(
                self.global_halt_flag_path,
                payload,
                project_root=self.project_root,
                source="base_trader.softguard",
            )
            if not ok:
                raise RuntimeError(f"halt_flag_write_failed:{self.global_halt_flag_path}")
            self._log_softguard_event(
                event="global_halt_set",
                status="ok",
                reason=str(reason or "softguard"),
                details=details,
            )
            return {"ok": True, "path": self.global_halt_flag_path, "payload": payload}
        except Exception as exc:
            err = f"{type(exc).__name__}:{exc}"
            self._log_softguard_event(
                event="global_halt_set",
                status="error",
                reason=err,
                details={"requested_reason": str(reason or "softguard")},
            )
            return {"ok": False, "error": err, "path": self.global_halt_flag_path}

    def _auto_cancel_open_orders_if_due(self, *, reason: str) -> Dict[str, Any]:
        now_ts = time.time()
        cooldown = float(self.live_softguard_auto_cancel_cooldown_seconds)
        if cooldown > 0.0 and self._softguard_last_auto_cancel_ts > 0.0:
            if (now_ts - self._softguard_last_auto_cancel_ts) < cooldown:
                remaining = round(cooldown - (now_ts - self._softguard_last_auto_cancel_ts), 6)
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "cooldown",
                    "remaining_seconds": max(remaining, 0.0),
                }

        out = self.cancel_all_live_open_orders()
        self._softguard_last_auto_cancel_ts = now_ts
        self._log_softguard_event(
            event="auto_cancel_open_orders",
            status=("ok" if bool(out.get("ok", False)) else "error"),
            reason=str(reason or "softguard"),
            details=out,
        )
        return out

    def _maybe_run_emergency_liquidation(self, *, trigger: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.live_softguard_emergency_liquidation_enabled:
            return {"ok": True, "skipped": True, "reason": "disabled"}

        now_ts = time.time()
        cooldown = float(self.live_softguard_emergency_liquidation_cooldown_seconds)
        if cooldown > 0.0 and self._softguard_last_emergency_liq_ts > 0.0:
            if (now_ts - self._softguard_last_emergency_liq_ts) < cooldown:
                remaining = round(cooldown - (now_ts - self._softguard_last_emergency_liq_ts), 6)
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "cooldown",
                    "remaining_seconds": max(remaining, 0.0),
                }

        cancel_out = self.cancel_all_live_open_orders()
        liq_out = self.emergency_liquidate_all_positions(dry_run=False)
        self._softguard_last_emergency_liq_ts = now_ts

        payload = {
            "ok": bool(cancel_out.get("ok", False)) and bool(liq_out.get("ok", False)),
            "trigger": str(trigger or "softguard"),
            "cancel": cancel_out,
            "liquidation": liq_out,
            "details": details or {},
        }
        self._log_softguard_event(
            event="emergency_liquidation",
            status=("ok" if payload["ok"] else "error"),
            reason=str(trigger or "softguard"),
            details=payload,
        )
        return payload

    def _as_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _coerce_json_obj_or_list(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if hasattr(value, "json"):
            try:
                payload = value.json()
                if isinstance(payload, (dict, list)):
                    return payload
            except Exception:
                return {}
        if isinstance(value, str) and value.strip():
            try:
                payload = json.loads(value)
                if isinstance(payload, (dict, list)):
                    return payload
            except Exception:
                return {}
        return {}

    def _coerce_json_payload(self, value: Any) -> Dict[str, Any]:
        payload = self._coerce_json_obj_or_list(value)
        if isinstance(payload, dict):
            return payload
        return {}

    def _reference_price(self, *, features: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        for key in ("limit_price", "fill_price", "execution_price", "price", "mark_price", "last_price"):
            val = self._as_float(metadata.get(key), 0.0)
            if val > 0.0:
                return val
        for key in ("last_price", "mark_price", "close_price", "entry_price"):
            val = self._as_float(features.get(key), 0.0)
            if val > 0.0:
                return val
        return 0.0

    def _first_price_from_sources(self, *, metadata: Dict[str, Any], features: Dict[str, Any], keys: Tuple[str, ...]) -> float:
        for key in keys:
            val = self._as_float(metadata.get(key), 0.0)
            if val > 0.0:
                return val
        for key in keys:
            val = self._as_float(features.get(key), 0.0)
            if val > 0.0:
                return val
        return 0.0

    def _intended_live_execution_price(
        self,
        *,
        action: str,
        limit_price: float,
        reference_price: float,
        metadata: Dict[str, Any],
        features: Dict[str, Any],
    ) -> float:
        limit = max(self._as_float(limit_price, 0.0), 0.0)
        if limit > 0.0:
            return limit

        signed_one = self._paper_signed_quantity(action, 1.0)
        if signed_one > 0.0:
            price = self._first_price_from_sources(
                metadata=metadata,
                features=features,
                keys=("ask_price", "ask", "best_ask", "offer_price", "offer", "mark_price", "last_price", "price", "close_price"),
            )
            if price > 0.0:
                return price
        elif signed_one < 0.0:
            price = self._first_price_from_sources(
                metadata=metadata,
                features=features,
                keys=("bid_price", "bid", "best_bid", "mark_price", "last_price", "price", "close_price"),
            )
            if price > 0.0:
                return price

        return max(self._as_float(reference_price, 0.0), 0.0)

    def _status_code_from_error_text(self, value: str) -> int:
        text = str(value or "")
        match = re.search(r"http_status_(\d{3})", text)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return 0
        return 0

    def _is_retryable_api_error(self, *, status_code: int = 0, error_text: str = "") -> bool:
        code = int(status_code or 0)
        if code > 0:
            return code in self.live_api_retryable_status_codes

        code_from_text = self._status_code_from_error_text(error_text)
        if code_from_text > 0:
            return code_from_text in self.live_api_retryable_status_codes

        lowered = str(error_text or "").lower()
        retryable_fragments = (
            "timeout",
            "timed out",
            "temporary",
            "temporarily",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection error",
            "disconnected",
            "service unavailable",
            "rate limit",
            "too many requests",
            "bad gateway",
            "gateway timeout",
        )
        return any(fragment in lowered for fragment in retryable_fragments)

    def _retry_delay_seconds(self, attempt_number: int) -> float:
        base = float(self.live_api_retry_backoff_seconds)
        if base <= 0.0:
            return 0.0

        exponent = max(int(attempt_number) - 1, 0)
        delay = base * (float(self.live_api_retry_backoff_multiplier) ** exponent)
        max_delay = float(self.live_api_retry_max_backoff_seconds)
        if max_delay > 0.0:
            delay = min(delay, max_delay)
        jitter = float(self.live_api_retry_jitter_seconds)
        if jitter > 0.0:
            delay += random.uniform(0.0, jitter)
        return max(delay, 0.0)

    def _extract_order_id(self, response: Any) -> str:
        payload = self._coerce_json_payload(response)
        for key in ("order_id", "orderId", "id"):
            val = payload.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()

        headers = getattr(response, "headers", None)
        if headers is not None:
            for key in ("Location", "location"):
                try:
                    location = headers.get(key)
                except Exception:
                    location = None
                if not location:
                    continue
                tail = str(location).rstrip("/").split("/")[-1]
                if tail:
                    return tail

        return ""

    def _invoke_client_candidates(
        self,
        *,
        operation: str,
        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            out = {
                "ok": False,
                "operation": operation,
                "error": "client_not_authenticated",
            }
            self._log_live_guard_event(event=operation, status="error", reason="client_not_authenticated", details=context)
            return out

        if not self.live_guard.allow_api_call("broker_api"):
            out = {
                "ok": False,
                "operation": operation,
                "error": "api_circuit_open",
                "cooldown_seconds": self.live_risk_config.api_cooldown_seconds,
                "fail_limit": self.live_risk_config.api_fail_limit,
            }
            self._log_live_guard_event(event=operation, status="blocked", reason="api_circuit_open", details=context)
            self._log_softguard_event(
                event="api_failure_guard",
                status="blocked",
                reason="api_circuit_open",
                details={"operation": operation, **(context or {})},
            )
            if self.live_softguard_auto_halt_on_api_circuit:
                self._engage_global_halt(
                    reason="softguard_api_circuit_open",
                    details={"operation": operation, **(context or {})},
                )
            return out

        signature_errors: List[str] = []
        method_seen = False
        max_attempts = max(int(self.live_api_retry_attempts), 1)
        attempt_failures: List[Dict[str, Any]] = []
        final_failure: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            attempt_retryable_failure: Optional[Dict[str, Any]] = None

            for method_name, args, kwargs in candidates:
                fn = getattr(self.client, method_name, None)
                if not callable(fn):
                    continue
                method_seen = True

                started = time.time()
                try:
                    response = fn(*args, **kwargs)
                    status_code = self._as_int(getattr(response, "status_code", 0), 0)
                    if status_code >= 400:
                        raise RuntimeError(f"http_status_{status_code}")

                    self.live_guard.record_api_success("broker_api")
                    latency_ms = round((time.time() - started) * 1000.0, 3)
                    payload = {
                        "ok": True,
                        "operation": operation,
                        "method": method_name,
                        "latency_ms": latency_ms,
                        "status_code": status_code,
                        "response": response,
                        "response_payload": self._coerce_json_payload(response),
                        "attempt": attempt,
                        "attempts_made": attempt,
                        "max_attempts": max_attempts,
                    }
                    order_id = self._extract_order_id(response)
                    if order_id:
                        payload["order_id"] = order_id

                    self._log_live_guard_event(
                        event=operation,
                        status="ok",
                        details={
                            "method": method_name,
                            "latency_ms": latency_ms,
                            "status_code": status_code,
                            "attempt": attempt,
                            "attempts_made": attempt,
                            "max_attempts": max_attempts,
                            **(context or {}),
                        },
                    )
                    return payload
                except TypeError as exc:
                    signature_errors.append(f"{method_name}:{exc}")
                    continue
                except Exception as exc:
                    latency_ms = round((time.time() - started) * 1000.0, 3)
                    err = f"{type(exc).__name__}:{exc}"
                    status_code = self._status_code_from_error_text(err)
                    retryable = self._is_retryable_api_error(status_code=status_code, error_text=err)
                    failure = {
                        "method": method_name,
                        "error": err,
                        "latency_ms": latency_ms,
                        "retryable": bool(retryable),
                        "status_code": int(status_code),
                        "attempt": attempt,
                    }
                    attempt_failures.append(failure)
                    self._log_live_guard_event(
                        event=operation,
                        status="error",
                        reason=err,
                        details={
                            "method": method_name,
                            "latency_ms": latency_ms,
                            "retryable": bool(retryable),
                            "status_code": int(status_code),
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            **(context or {}),
                        },
                    )
                    if retryable:
                        attempt_retryable_failure = failure
                        continue

                    final_failure = failure
                    break

            if final_failure is not None:
                break

            if attempt_retryable_failure is None:
                break

            if attempt >= max_attempts:
                final_failure = attempt_retryable_failure
                break

            delay_seconds = self._retry_delay_seconds(attempt)
            self._log_softguard_event(
                event="api_retry",
                status="retrying",
                reason=str(attempt_retryable_failure.get("error", "retryable_api_error")),
                details={
                    "operation": operation,
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "delay_seconds": float(delay_seconds),
                    "method": str(attempt_retryable_failure.get("method", "")),
                    "status_code": int(attempt_retryable_failure.get("status_code", 0) or 0),
                    **(context or {}),
                },
            )
            if delay_seconds > 0.0:
                time.sleep(delay_seconds)

        if method_seen and signature_errors and (not attempt_failures):
            reason = "method_signature_mismatch"
            details = {"errors": signature_errors, **(context or {})}
            self._log_live_guard_event(event=operation, status="error", reason=reason, details=details)
            return {
                "ok": False,
                "operation": operation,
                "error": reason,
                "details": details,
                "attempts_made": 0,
                "max_attempts": max_attempts,
            }

        if not method_seen:
            reason = "method_not_available"
            details = {"candidate_methods": [name for name, _, _ in candidates], **(context or {})}
            self._log_live_guard_event(event=operation, status="error", reason=reason, details=details)
            return {
                "ok": False,
                "operation": operation,
                "error": reason,
                "details": details,
                "attempts_made": 0,
                "max_attempts": max_attempts,
            }

        if final_failure is None and attempt_failures:
            final_failure = attempt_failures[-1]

        if final_failure is None:
            reason = "unknown_client_error"
            details = context or {}
            self._log_live_guard_event(event=operation, status="error", reason=reason, details=details)
            return {
                "ok": False,
                "operation": operation,
                "error": reason,
                "details": details,
                "attempts_made": 0,
                "max_attempts": max_attempts,
            }

        opened = self.live_guard.record_api_failure("broker_api")
        attempts_made = int(final_failure.get("attempt", len(attempt_failures)) or len(attempt_failures) or 1)

        self._log_live_guard_event(
            event=operation,
            status="error",
            reason=str(final_failure.get("error", "api_call_failed")),
            details={
                "method": str(final_failure.get("method", "")),
                "latency_ms": float(final_failure.get("latency_ms", 0.0) or 0.0),
                "retryable": bool(final_failure.get("retryable", False)),
                "status_code": int(final_failure.get("status_code", 0) or 0),
                "attempt": int(final_failure.get("attempt", attempts_made) or attempts_made),
                "attempts_made": attempts_made,
                "max_attempts": max_attempts,
                "circuit_opened": bool(opened),
                **(context or {}),
            },
        )
        if opened and self.live_softguard_auto_halt_on_api_circuit:
            self._engage_global_halt(
                reason="softguard_api_circuit_opened",
                details={
                    "operation": operation,
                    "attempts_made": attempts_made,
                    "max_attempts": max_attempts,
                    **(context or {}),
                },
            )

        return {
            "ok": False,
            "operation": operation,
            "method": str(final_failure.get("method", "")),
            "error": str(final_failure.get("error", "api_call_failed")),
            "latency_ms": float(final_failure.get("latency_ms", 0.0) or 0.0),
            "status_code": int(final_failure.get("status_code", 0) or 0),
            "retryable": bool(final_failure.get("retryable", False)),
            "attempts_made": attempts_made,
            "max_attempts": max_attempts,
            "circuit_opened": bool(opened),
            "details": {
                "failures": attempt_failures[-max_attempts:],
            },
        }

    def _order_instruction(self, action: str) -> str:
        side = str(action or "").strip().upper()
        mapping = {
            "BUY": "BUY",
            "SELL": "SELL",
            "SELL_SHORT": "SELL_SHORT",
            "BUY_TO_COVER": "BUY_TO_COVER",
            "BUY_TO_OPEN": "BUY_TO_OPEN",
            "BUY_TO_CLOSE": "BUY_TO_CLOSE",
            "SELL_TO_OPEN": "SELL_TO_OPEN",
            "SELL_TO_CLOSE": "SELL_TO_CLOSE",
        }
        if side in mapping:
            return mapping[side]
        raise ValueError(f"unsupported_order_action:{side}")

    def _build_live_order_spec(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float = 0.0,
        asset_type: str = "EQUITY",
    ) -> Dict[str, Any]:
        qty = max(self._as_float(quantity, 0.0), 0.0)
        if qty <= 0.0:
            raise ValueError("quantity_must_be_positive")

        instruction = self._order_instruction(action)
        limit = self._as_float(limit_price, 0.0)

        order_type = "MARKET"
        out: Dict[str, Any] = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": qty,
                    "instrument": {
                        "symbol": str(symbol).upper(),
                        "assetType": str(asset_type or "EQUITY").upper(),
                    },
                }
            ],
        }
        if limit > 0.0:
            out["orderType"] = "LIMIT"
            out["price"] = round(limit, 6)
        return out

    def _live_place_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        order_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("place_order", (self.live_account_hash, order_spec), {}))
        candidates.append(("place_order", (order_spec,), {}))

        out = self._invoke_client_candidates(
            operation="place_order",
            candidates=candidates,
            context={"symbol": str(symbol).upper(), "action": str(action).upper(), "quantity": float(quantity)},
        )
        return out

    def modify_live_order(
        self,
        *,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float = 0.0,
        asset_type: str = "EQUITY",
    ) -> Dict[str, Any]:
        if self._operator_stop_enabled() or self._global_trading_halt_enabled():
            reason = "operator_stop" if self._operator_stop_enabled() else "global_trading_halt"
            self._log_softguard_event(
                event="modify_order_blocked",
                status="blocked",
                reason=reason,
                details={"order_id": str(order_id or ""), "symbol": str(symbol).upper()},
            )
            return {"ok": False, "error": reason}

        oid = str(order_id or "").strip()
        if not oid:
            return {"ok": False, "error": "missing_order_id"}

        spec = self._build_live_order_spec(
            symbol=symbol,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            asset_type=asset_type,
        )

        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("replace_order", (self.live_account_hash, oid, spec), {}))
        candidates.append(("replace_order", (oid, spec), {}))

        out = self._invoke_client_candidates(
            operation="replace_order",
            candidates=candidates,
            context={"order_id": oid, "symbol": str(symbol).upper(), "action": str(action).upper(), "quantity": float(quantity)},
        )
        if out.get("ok"):
            self.live_guard.register_open_order(order_id=oid, symbol=symbol, action=action, quantity=quantity)
        return out

    def cancel_live_order(self, *, order_id: str) -> Dict[str, Any]:
        oid = str(order_id or "").strip()
        if not oid:
            return {"ok": False, "error": "missing_order_id"}

        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("cancel_order", (self.live_account_hash, oid), {}))
        candidates.append(("cancel_order", (oid,), {}))

        out = self._invoke_client_candidates(
            operation="cancel_order",
            candidates=candidates,
            context={"order_id": oid},
        )
        if out.get("ok"):
            self.live_guard.close_open_order(oid)
        return out

    def _live_fetch_order(self, order_id: str) -> Dict[str, Any]:
        oid = str(order_id or "").strip()
        if not oid:
            return {"ok": False, "error": "missing_order_id"}

        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("get_order", (self.live_account_hash, oid), {}))
        candidates.append(("get_order", (oid,), {}))

        out = self._invoke_client_candidates(
            operation="get_order",
            candidates=candidates,
            context={"order_id": oid},
        )
        if not out.get("ok"):
            return out

        payload = self._coerce_json_payload(out.get("response"))
        out["order_payload"] = payload
        return out

    def _order_status(self, payload: Dict[str, Any]) -> str:
        for key in ("status", "orderStatus", "state"):
            raw = payload.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip().upper()
        return "UNKNOWN"

    def _extract_fill_action(self, payload: Dict[str, Any]) -> str:
        for key in ("instruction", "action", "side"):
            raw = payload.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip().upper()

        legs = payload.get("orderLegCollection")
        if isinstance(legs, list) and legs:
            leg0 = legs[0] if isinstance(legs[0], dict) else {}
            raw = leg0.get("instruction")
            if raw is not None and str(raw).strip():
                return str(raw).strip().upper()

        return "BUY"

    def _filled_qty_price(self, payload: Dict[str, Any]) -> Tuple[float, float]:
        qty = 0.0
        price = 0.0

        for q_key in ("filledQuantity", "filled_quantity", "quantityFilled", "filledQty"):
            q_val = self._as_float(payload.get(q_key), 0.0)
            if q_val > 0.0:
                qty = q_val
                break

        for p_key in ("fillPrice", "averageFillPrice", "price", "execution_price"):
            p_val = self._as_float(payload.get(p_key), 0.0)
            if p_val > 0.0:
                price = p_val
                break

        return qty, price

    def refresh_live_fills(self, *, order_id: Optional[str] = None) -> Dict[str, Any]:
        order_ids = [str(order_id).strip()] if (order_id and str(order_id).strip()) else self.live_guard.open_order_ids()
        rows: List[Dict[str, Any]] = []

        for oid in order_ids:
            fetch = self._live_fetch_order(oid)
            if not fetch.get("ok"):
                rows.append({"order_id": oid, "status": "ORDER_FETCH_FAILED", "error": fetch.get("error", "unknown")})
                continue

            payload = fetch.get("order_payload") if isinstance(fetch.get("order_payload"), dict) else {}
            status = self._order_status(payload)
            symbol = str(payload.get("symbol", "")).strip().upper()
            if not symbol:
                legs = payload.get("orderLegCollection")
                if isinstance(legs, list) and legs and isinstance(legs[0], dict):
                    inst = legs[0].get("instrument") if isinstance(legs[0].get("instrument"), dict) else {}
                    symbol = str(inst.get("symbol", "")).strip().upper()

            if status in {"FILLED", "EXECUTED"}:
                qty, price = self._filled_qty_price(payload)
                action = self._extract_fill_action(payload)
                fill_state = None
                if qty > 0.0 and price > 0.0 and symbol:
                    fill_state = self.live_guard.record_fill(symbol=symbol, action=action, quantity=qty, fill_price=price)
                self.live_guard.close_open_order(oid)
                rows.append(
                    {
                        "order_id": oid,
                        "status": status,
                        "symbol": symbol,
                        "filled_qty": float(qty),
                        "fill_price": float(price),
                        "fill_state": fill_state,
                    }
                )
            elif status in {"CANCELED", "REJECTED", "EXPIRED"}:
                self.live_guard.close_open_order(oid)
                rows.append({"order_id": oid, "status": status, "symbol": symbol})
            else:
                rows.append({"order_id": oid, "status": status, "symbol": symbol})

        return {
            "status": "ok",
            "processed": len(rows),
            "fills": rows,
        }

    def _iter_account_payload_nodes(self, payload: Any):
        if isinstance(payload, dict):
            yield payload
            sec = payload.get("securitiesAccount") if isinstance(payload.get("securitiesAccount"), dict) else None
            if isinstance(sec, dict):
                yield sec
            accounts = payload.get("accounts")
            if isinstance(accounts, list):
                for account in accounts:
                    if not isinstance(account, dict):
                        continue
                    yield account
                    sec2 = account.get("securitiesAccount") if isinstance(account.get("securitiesAccount"), dict) else None
                    if isinstance(sec2, dict):
                        yield sec2
        elif isinstance(payload, list):
            for account in payload:
                if not isinstance(account, dict):
                    continue
                yield account
                sec3 = account.get("securitiesAccount") if isinstance(account.get("securitiesAccount"), dict) else None
                if isinstance(sec3, dict):
                    yield sec3

    def _extract_all_positions_from_payload(self, payload: Any) -> List[Dict[str, Any]]:
        by_symbol: Dict[str, Dict[str, Any]] = {}

        for node in self._iter_account_payload_nodes(payload):
            positions = node.get("positions") if isinstance(node.get("positions"), list) else []
            for row in positions:
                if not isinstance(row, dict):
                    continue
                inst = row.get("instrument") if isinstance(row.get("instrument"), dict) else {}
                symbol = str(inst.get("symbol", "")).strip().upper()
                if not symbol:
                    continue
                long_qty = self._as_float(row.get("longQuantity"), 0.0)
                short_qty = self._as_float(row.get("shortQuantity"), 0.0)
                if "netQuantity" in row:
                    qty = self._as_float(row.get("netQuantity"), 0.0)
                elif "quantity" in row:
                    qty = self._as_float(row.get("quantity"), 0.0)
                else:
                    qty = long_qty - short_qty

                if abs(qty) <= 0.0:
                    continue

                prior = by_symbol.get(symbol)
                if prior is None:
                    by_symbol[symbol] = {
                        "symbol": symbol,
                        "quantity": float(qty),
                        "asset_type": str(inst.get("assetType", "EQUITY") or "EQUITY").upper(),
                    }
                else:
                    prior["quantity"] = float(prior.get("quantity", 0.0) or 0.0) + float(qty)

        out: List[Dict[str, Any]] = []
        for symbol in sorted(by_symbol.keys()):
            row = by_symbol[symbol]
            if abs(float(row.get("quantity", 0.0) or 0.0)) <= 0.0:
                continue
            out.append(row)
        return out

    def _extract_open_order_ids_from_payload(self, payload: Any) -> List[str]:
        open_status = {
            "QUEUED",
            "WORKING",
            "OPEN",
            "PENDING_ACTIVATION",
            "PENDING_ACKNOWLEDGEMENT",
            "AWAITING_PARENT_ORDER",
            "AWAITING_CONDITION",
            "ACCEPTED",
        }

        order_ids: set[str] = set()

        def _scan_order_list(rows: Any) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                status = self._order_status(row)
                oid = str(row.get("orderId", row.get("id", ""))).strip()
                if oid and status in open_status:
                    order_ids.add(oid)

        for node in self._iter_account_payload_nodes(payload):
            _scan_order_list(node.get("orderStrategies"))
            _scan_order_list(node.get("orders"))

        return sorted(order_ids)

    def _live_fetch_accounts_payload(self) -> Dict[str, Any]:
        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("get_account", (self.live_account_hash,), {}))
        candidates.append(("get_accounts", tuple(), {}))
        candidates.append(("get_account", tuple(), {}))

        out = self._invoke_client_candidates(
            operation="get_accounts_snapshot",
            candidates=candidates,
            context={},
        )
        if not out.get("ok"):
            return out

        payload = self._coerce_json_obj_or_list(out.get("response"))
        return {
            "ok": True,
            "payload": payload,
            "operation": "get_accounts_snapshot",
        }

    def cancel_all_live_open_orders(self, *, max_orders: int = 200) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "error": "client_not_authenticated", "canceled": [], "failed": []}

        ids: set[str] = set(self.live_guard.open_order_ids())
        fetched = self._live_fetch_accounts_payload()
        if fetched.get("ok"):
            payload = fetched.get("payload")
            for oid in self._extract_open_order_ids_from_payload(payload):
                if oid:
                    ids.add(str(oid))

        selected = sorted(ids)[: max(int(max_orders), 1)]
        canceled: List[str] = []
        failed: List[Dict[str, Any]] = []

        for oid in selected:
            out = self.cancel_live_order(order_id=oid)
            if out.get("ok"):
                canceled.append(oid)
            else:
                failed.append({"order_id": oid, "error": out.get("error", "cancel_failed")})

        result = {
            "ok": len(failed) == 0,
            "requested": len(selected),
            "canceled": canceled,
            "failed": failed,
            "source_open_order_ids": selected,
        }
        return result

    def emergency_liquidate_all_positions(self, *, max_positions: int = 200, dry_run: bool = True) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "error": "client_not_authenticated", "orders": []}

        fetched = self._live_fetch_accounts_payload()
        if not fetched.get("ok"):
            return {
                "ok": False,
                "error": fetched.get("error", "accounts_fetch_failed"),
                "orders": [],
            }

        payload = fetched.get("payload")
        positions = self._extract_all_positions_from_payload(payload)
        selected = positions[: max(int(max_positions), 1)]
        orders: List[Dict[str, Any]] = []

        for pos in selected:
            symbol = str(pos.get("symbol", "")).strip().upper()
            qty = abs(self._as_float(pos.get("quantity"), 0.0))
            signed_qty = self._as_float(pos.get("quantity"), 0.0)
            if (not symbol) or qty <= 0.0:
                continue

            action = "SELL" if signed_qty > 0.0 else "BUY_TO_COVER"
            asset_type = str(pos.get("asset_type", "EQUITY") or "EQUITY").upper()

            if dry_run:
                orders.append(
                    {
                        "symbol": symbol,
                        "quantity": float(qty),
                        "action": action,
                        "asset_type": asset_type,
                        "status": "DRY_RUN",
                    }
                )
                continue

            try:
                spec = self._build_live_order_spec(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    limit_price=0.0,
                    asset_type=asset_type,
                )
                out = self._live_place_order(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    order_spec=spec,
                )
                orders.append(
                    {
                        "symbol": symbol,
                        "quantity": float(qty),
                        "action": action,
                        "asset_type": asset_type,
                        "status": ("SUBMITTED" if out.get("ok") else "SUBMIT_FAILED"),
                        "order_id": str(out.get("order_id", "") or ""),
                        "error": out.get("error", "") if not out.get("ok") else "",
                    }
                )
            except Exception as exc:
                orders.append(
                    {
                        "symbol": symbol,
                        "quantity": float(qty),
                        "action": action,
                        "asset_type": asset_type,
                        "status": "EXCEPTION",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )

        failed = [o for o in orders if str(o.get("status", "")).upper() not in {"SUBMITTED", "DRY_RUN"}]
        return {
            "ok": len(failed) == 0,
            "dry_run": bool(dry_run),
            "positions_considered": len(selected),
            "orders": orders,
            "failed": failed,
        }

    def _extract_position_qty_from_payload(self, payload: Dict[str, Any], symbol: str) -> Optional[float]:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return None

        def _from_positions(positions: Any) -> Optional[float]:
            if not isinstance(positions, list):
                return None
            for row in positions:
                if not isinstance(row, dict):
                    continue
                inst = row.get("instrument") if isinstance(row.get("instrument"), dict) else {}
                sym = str(inst.get("symbol", "")).strip().upper()
                if sym != symbol_key:
                    continue
                for key in ("longQuantity", "quantity", "netQuantity"):
                    if key in row:
                        return self._as_float(row.get(key), 0.0)
            return None

        qty = _from_positions(payload.get("positions"))
        if qty is not None:
            return qty

        sec = payload.get("securitiesAccount") if isinstance(payload.get("securitiesAccount"), dict) else {}
        qty = _from_positions(sec.get("positions"))
        if qty is not None:
            return qty

        accounts = payload.get("accounts")
        if isinstance(accounts, list):
            for account in accounts:
                if not isinstance(account, dict):
                    continue
                sec2 = account.get("securitiesAccount") if isinstance(account.get("securitiesAccount"), dict) else account
                qty = _from_positions(sec2.get("positions"))
                if qty is not None:
                    return qty

        return None

    def _live_fetch_broker_position(self, *, symbol: str) -> Dict[str, Any]:
        candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        if self.live_account_hash:
            candidates.append(("get_account", (self.live_account_hash,), {}))
        candidates.append(("get_account", tuple(), {}))
        candidates.append(("get_accounts", tuple(), {}))

        out = self._invoke_client_candidates(
            operation="get_positions",
            candidates=candidates,
            context={"symbol": str(symbol).upper()},
        )
        if not out.get("ok"):
            return out

        payload = self._coerce_json_payload(out.get("response"))
        qty = self._extract_position_qty_from_payload(payload, symbol)
        if qty is None:
            return {
                "ok": False,
                "error": "position_not_found",
                "symbol": str(symbol).upper(),
                "payload": payload,
            }
        return {
            "ok": True,
            "symbol": str(symbol).upper(),
            "broker_qty": float(qty),
            "payload": payload,
        }

    def confirm_live_position_state(self, *, symbol: str) -> Dict[str, Any]:
        fetched = self._live_fetch_broker_position(symbol=symbol)
        if not fetched.get("ok"):
            self._log_live_guard_event(
                event="position_reconcile",
                status="error",
                reason=str(fetched.get("error", "position_fetch_failed")),
                details={"symbol": str(symbol).upper()},
            )
            return fetched

        manual_tolerance = (
            self.live_manual_trade_qty_tolerance
            if self.live_manual_trade_awareness_enabled
            else self.live_position_reconcile_tolerance
        )
        reconcile = self.live_guard.reconcile_broker_position(
            symbol=symbol,
            broker_qty=self._as_float(fetched.get("broker_qty"), 0.0),
            tolerance=self.live_position_reconcile_tolerance,
            manual_adjustment_tolerance=manual_tolerance,
        )

        reconcile_status = str(reconcile.get("status", "mismatch"))
        if reconcile_status == "match":
            status = "ok"
            reason = "ok"
        elif reconcile_status == "manual_adjustment_detected" and self.live_manual_trade_awareness_enabled:
            status = "manual_adjustment"
            reason = "manual_adjustment_detected"
        else:
            status = "mismatch"
            reason = "position_mismatch"

        self._log_live_guard_event(
            event="position_reconcile",
            status=status,
            reason=reason,
            details=reconcile,
        )
        return reconcile

    def _pre_trade_reconcile_before_order(self, *, symbol: str) -> Dict[str, Any]:
        fetched = self._live_fetch_broker_position(symbol=symbol)
        if not fetched.get("ok"):
            reason = str(fetched.get("error", "position_fetch_failed"))
            details = {
                "symbol": str(symbol).upper(),
                "error": reason,
                "block_on_error": bool(self.live_pretrade_reconcile_block_on_error),
            }
            self._log_live_guard_event(
                event="pre_trade_position_reconcile",
                status="error",
                reason=reason,
                details=details,
            )
            return {
                "ok": (not bool(self.live_pretrade_reconcile_block_on_error)),
                "gate": "pre_trade_position_reconcile_error",
                "reason": reason,
                "details": details,
            }

        broker_qty = self._as_float(fetched.get("broker_qty"), 0.0)
        manual_tolerance = (
            self.live_manual_trade_qty_tolerance
            if self.live_manual_trade_awareness_enabled
            else self.live_position_reconcile_tolerance
        )
        reconcile = self.live_guard.reconcile_broker_position(
            symbol=symbol,
            broker_qty=broker_qty,
            tolerance=self.live_position_reconcile_tolerance,
            manual_adjustment_tolerance=manual_tolerance,
        )
        mismatch = not bool(reconcile.get("ok", False))
        manual_adjustment = bool(reconcile.get("manual_adjustment_detected", False)) and bool(self.live_manual_trade_awareness_enabled)

        synced_local = False
        if mismatch:
            if manual_adjustment and self.live_manual_trade_auto_sync_local:
                self.live_guard.set_local_position(symbol=symbol, quantity=broker_qty)
                synced_local = True
            elif (not manual_adjustment) and self.live_pretrade_reconcile_sync_local:
                self.live_guard.set_local_position(symbol=symbol, quantity=broker_qty)
                synced_local = True

        details = {
            **reconcile,
            "manual_trade_awareness_enabled": bool(self.live_manual_trade_awareness_enabled),
            "manual_trade_auto_sync_local": bool(self.live_manual_trade_auto_sync_local),
            "synced_local_position": bool(synced_local),
            "block_on_mismatch": bool(self.live_pretrade_reconcile_block_on_mismatch),
        }
        if synced_local:
            details["local_qty_after_sync"] = self.live_guard.local_position_qty(symbol)

        if not mismatch:
            status = "ok"
            reason = "ok"
            blocked = False
        elif manual_adjustment:
            status = "manual_adjustment"
            reason = "manual_adjustment_detected"
            blocked = False
        else:
            status = "mismatch"
            reason = "position_mismatch"
            blocked = bool(self.live_pretrade_reconcile_block_on_mismatch)

        self._log_live_guard_event(
            event="pre_trade_position_reconcile",
            status=status,
            reason=reason,
            details=details,
        )

        return {
            "ok": (not blocked),
            "gate": "pre_trade_position_reconcile",
            "reason": reason,
            "details": details,
        }

    def execute_decision(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        model_score: float,
        threshold: float,
        features: Dict[str, Any],
        gates: Dict[str, bool],
        reasons: list[str],
        strategy: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Routes decisions to shadow, paper, or live execution with full audit logging."""
        md = dict(metadata or {})
        corr = current_correlation()
        if corr.get("run_id") and (not str(md.get("run_id") or "").strip()):
            md["run_id"] = corr["run_id"]
        if corr.get("iter_id") and (not str(md.get("iter_id") or "").strip()):
            md["iter_id"] = corr["iter_id"]
        if not str(md.get("decision_id") or "").strip():
            md["decision_id"] = str(uuid.uuid4())
        if not str(md.get("parent_decision_id") or "").strip():
            snap = str(md.get("snapshot_id") or "").strip()
            if snap:
                md["parent_decision_id"] = f"{snap}:root"

        decision_entry = self.decision_logger.log_decision(
            symbol=symbol,
            action=action,
            model_score=model_score,
            threshold=threshold,
            quantity=quantity,
            features=features,
            gates=gates,
            reasons=reasons,
            strategy=strategy,
            metadata={"mode": self.mode, **md},
        )

        safety: Optional[Dict[str, Any]] = None

        if decision_entry["decision"] == "BLOCK":
            status = "BLOCKED"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
            }
        elif self._is_trade_action(action) and (self.market_data_only or not self.execution_enabled):
            # Hard safety lock: keeps the bot in data-only behavior.
            safety = {
                "market_data_only": self.market_data_only,
                "execution_enabled": self.execution_enabled,
            }
            status = "DATA_ONLY_BLOCKED"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
                "safety": safety,
            }
        elif self.mode == "shadow":
            status = "SHADOW_ONLY"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
            }
        elif self.mode == "paper":
            paper_metadata = dict(md)
            ref_price = self._reference_price(features=features, metadata=paper_metadata)
            limit_price = self._as_float(paper_metadata.get("limit_price"), 0.0)
            intended_price = self._intended_live_execution_price(
                action=action,
                limit_price=limit_price,
                reference_price=ref_price,
                metadata=paper_metadata,
                features=features,
            )

            if self._is_trade_action(action):
                paper_guard = self.live_guard.pre_trade_check(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    reference_price=ref_price,
                    intended_price=intended_price,
                )
                if not paper_guard.ok:
                    status = "PAPER_GUARD_BLOCKED"
                    guard_payload = {
                        "gate": paper_guard.gate,
                        "reason": paper_guard.reason,
                        "details": paper_guard.details,
                    }
                    self._log_live_guard_event(
                        event="pre_trade_check",
                        status="blocked",
                        reason=paper_guard.reason,
                        details={
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "quantity": float(quantity),
                            **guard_payload,
                        },
                    )
                    result = {
                        "status": status,
                        "mode": self.mode,
                        "decision": decision_entry,
                        "live_guard_decision": guard_payload,
                        "live_guard": self.live_guard.snapshot(),
                    }
                    self._emit_decision_explanation(
                        status=status,
                        decision_entry=decision_entry,
                        safety=safety,
                    )
                    return result

                self.live_guard.mark_trade_submitted(symbol=symbol)

            paper_pnl = self._paper_pnl_fields(
                symbol=symbol,
                action=action,
                quantity=quantity,
                features=features,
                metadata=paper_metadata,
            )

            expected_fill: Dict[str, float] = {}
            model_inputs: Dict[str, float] = {}
            if self._is_trade_action(action) and ref_price > 0.0:
                model_inputs = self._paper_execution_model_inputs(
                    symbol=symbol,
                    features=features,
                    metadata=paper_metadata,
                )
                expected_fill = self.live_guard.model_expected_fill(
                    action=action,
                    reference_price=ref_price,
                    quantity=quantity,
                    spread_bps=float(model_inputs.get("spread_bps", 8.0)),
                    volatility_1m=float(model_inputs.get("volatility_1m", 0.0)),
                    latency_ms=float(model_inputs.get("latency_ms", 120.0)),
                    bid_size=float(model_inputs.get("bid_size", 1000.0)),
                    ask_size=float(model_inputs.get("ask_size", 1000.0)),
                )

            paper = self._record_jsonl(
                self.paper_log_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": self.mode,
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "model_score": float(model_score),
                    "threshold": float(threshold),
                    "strategy": strategy,
                    "decision_id": str(decision_entry.get("decision_id", "") or ""),
                    "parent_decision_id": str(decision_entry.get("parent_decision_id", "") or ""),
                    "run_id": str(decision_entry.get("run_id", "") or ""),
                    "iter_id": str(decision_entry.get("iter_id", "") or ""),
                    "metadata": paper_metadata,
                    "reference_price": float(ref_price),
                    "intended_price": float(intended_price),
                    "expected_fill_price": float(expected_fill.get("expected_fill_price", 0.0) or 0.0),
                    "expected_slippage_bps": float(expected_fill.get("expected_slippage_bps", 0.0) or 0.0),
                    "expected_impact_bps": float(expected_fill.get("impact_bps", 0.0) or 0.0),
                    "model_spread_bps": float(model_inputs.get("spread_bps", 0.0) or 0.0),
                    "model_latency_ms": float(model_inputs.get("latency_ms", 0.0) or 0.0),
                    "model_bid_size": float(model_inputs.get("bid_size", 0.0) or 0.0),
                    "model_ask_size": float(model_inputs.get("ask_size", 0.0) or 0.0),
                    **paper_pnl,
                },
            )

            fill_state: Dict[str, Any] = {}
            position_reconcile: Dict[str, Any] = {}
            lifecycle_reconcile: Dict[str, Any] = {}
            if self._is_trade_action(action):
                order_id = str(decision_entry.get("decision_id", "") or str(uuid.uuid4()))
                self.live_guard.register_open_order(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                )
                fill_state = self.live_guard.record_fill(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    fill_price=self._as_float(paper_pnl.get("fill_price"), 0.0),
                    expected_fill_price=self._as_float(expected_fill.get("expected_fill_price"), 0.0),
                    reference_price=ref_price,
                )
                self.live_guard.close_open_order(order_id)

                broker_qty = self.live_guard.local_position_qty(symbol)
                position_reconcile = self.live_guard.reconcile_broker_position(
                    symbol=symbol,
                    broker_qty=broker_qty,
                    tolerance=self.live_position_reconcile_tolerance,
                )
                lifecycle_reconcile = self.live_guard.reconcile_order_lifecycle(
                    broker_open_orders=[],
                    position_tolerance=self.live_position_reconcile_tolerance,
                )
                lifecycle_ok = bool(lifecycle_reconcile.get("ok", False))
                self._log_live_guard_event(
                    event="paper_order_lifecycle_reconcile",
                    status=("ok" if lifecycle_ok else "mismatch"),
                    reason=("ok" if lifecycle_ok else "paper_order_lifecycle_mismatch"),
                    details={
                        "symbol": str(symbol).upper(),
                        "decision_id": str(decision_entry.get("decision_id", "") or ""),
                        "position_reconcile": position_reconcile,
                        "order_lifecycle_reconcile": lifecycle_reconcile,
                    },
                )

            status = "PAPER_EXECUTED"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
                "paper_order": paper,
                "paper_fill_model": fill_state,
                "position_reconcile": position_reconcile,
                "order_lifecycle_reconcile": lifecycle_reconcile,
                "live_guard": self.live_guard.snapshot(),
            }
            if self._is_trade_action(action):
                bridge = self._bridge_paper_order(paper_order=paper, result_status=status)
                result["paper_bridge"] = bridge
        else:
            # live mode
            if self.client is None:
                raise RuntimeError("Cannot execute live order without authentication")

            if not self._is_trade_action(action):
                status = "LIVE_NO_TRADE_ACTION"
                result = {
                    "status": status,
                    "mode": self.mode,
                    "decision": decision_entry,
                    "live_guard": self.live_guard.snapshot(),
                }
            else:
                operator_stop = self._operator_stop_enabled()
                global_halt = self._global_trading_halt_enabled()
                if operator_stop or global_halt:
                    reason = "operator_stop" if operator_stop else "global_trading_halt"
                    status = "LIVE_OPERATOR_STOP_BLOCKED" if operator_stop else "LIVE_GLOBAL_HALT_BLOCKED"
                    auto_cancel = None
                    if operator_stop and self.live_softguard_auto_cancel_on_operator_stop:
                        auto_cancel = self._auto_cancel_open_orders_if_due(reason=reason)
                    elif global_halt and self.live_softguard_auto_cancel_on_global_halt:
                        auto_cancel = self._auto_cancel_open_orders_if_due(reason=reason)
                    block_payload = {
                        "gate": reason,
                        "reason": reason,
                        "details": {
                            "symbol": str(symbol).upper(),
                            "operator_stop": bool(operator_stop),
                            "global_halt": bool(global_halt),
                            "auto_cancel": auto_cancel or {},
                        },
                    }
                    self._log_softguard_event(
                        event="operator_override" if operator_stop else "global_halt_guard",
                        status="blocked",
                        reason=reason,
                        details=block_payload["details"],
                    )
                    result = {
                        "status": status,
                        "mode": self.mode,
                        "decision": decision_entry,
                        "live_guard_decision": block_payload,
                        "live_guard": self.live_guard.snapshot(),
                    }
                    self._emit_decision_explanation(
                        status=status,
                        decision_entry=decision_entry,
                        safety=safety,
                    )
                    return result

                if self.live_pretrade_reconcile_required:
                    pre_reconcile = self._pre_trade_reconcile_before_order(symbol=symbol)
                    if not bool(pre_reconcile.get("ok", False)):
                        status = "LIVE_GUARD_BLOCKED"
                        guard_payload = {
                            "gate": str(pre_reconcile.get("gate", "pre_trade_position_reconcile")),
                            "reason": str(pre_reconcile.get("reason", "pre_trade_position_reconcile_failed")),
                            "details": pre_reconcile.get("details", {}),
                        }
                        pre_reason = str(guard_payload.get("reason", ""))
                        self._log_softguard_event(
                            event=("position_mismatch_guard" if pre_reason == "position_mismatch" else "position_reconcile_guard"),
                            status="blocked",
                            reason=pre_reason,
                            details={
                                "symbol": str(symbol).upper(),
                                **(guard_payload.get("details", {}) if isinstance(guard_payload.get("details"), dict) else {}),
                            },
                        )
                        if pre_reason == "position_mismatch" and self.live_softguard_auto_halt_on_position_mismatch:
                            guard_payload["auto_halt"] = self._engage_global_halt(
                                reason="softguard_position_mismatch",
                                details={
                                    "symbol": str(symbol).upper(),
                                    "guard": guard_payload,
                                },
                            )
                            if self.live_softguard_auto_cancel_on_global_halt:
                                guard_payload["auto_cancel"] = self._auto_cancel_open_orders_if_due(reason="position_mismatch")
                            guard_payload["emergency_liquidation"] = self._maybe_run_emergency_liquidation(
                                trigger="position_mismatch_pretrade",
                                details={"symbol": str(symbol).upper()},
                            )

                        result = {
                            "status": status,
                            "mode": self.mode,
                            "decision": decision_entry,
                            "live_guard_decision": guard_payload,
                            "live_guard": self.live_guard.snapshot(),
                        }
                        self._emit_decision_explanation(
                            status=status,
                            decision_entry=decision_entry,
                            safety=safety,
                        )
                        return result

                ref_price = self._reference_price(features=features, metadata=md)
                limit_price = self._as_float(md.get("limit_price"), 0.0)
                intended_price = self._intended_live_execution_price(
                    action=action,
                    limit_price=limit_price,
                    reference_price=ref_price,
                    metadata=md,
                    features=features,
                )
                guard_decision = self.live_guard.pre_trade_check(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    reference_price=ref_price,
                    intended_price=intended_price,
                )

                if not guard_decision.ok:
                    status = "LIVE_GUARD_BLOCKED"
                    guard_payload = {
                        "gate": guard_decision.gate,
                        "reason": guard_decision.reason,
                        "details": guard_decision.details,
                    }
                    self._log_live_guard_event(
                        event="pre_trade_check",
                        status="blocked",
                        reason=guard_decision.reason,
                        details={
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "quantity": float(quantity),
                            **guard_payload,
                        },
                    )

                    guard_event = "pre_trade_guard"
                    gate_name = str(guard_decision.gate or "")
                    if gate_name in {"position_limit", "order_notional_limit", "open_order_limit_total", "open_order_limit_symbol", "daily_loss_cap"}:
                        guard_event = "risk_limit_breach"
                    elif gate_name == "slippage_limit":
                        guard_event = "slippage_guard"
                    elif gate_name.startswith("trade_throttle"):
                        guard_event = "throttle_limit"
                    elif gate_name == "api_circuit_breaker":
                        guard_event = "api_failure_guard"

                    self._log_softguard_event(
                        event=guard_event,
                        status="blocked",
                        reason=str(guard_decision.reason),
                        details={
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "quantity": float(quantity),
                            **guard_payload,
                        },
                    )

                    if gate_name == "api_circuit_breaker" and self.live_softguard_auto_halt_on_api_circuit:
                        guard_payload["auto_halt"] = self._engage_global_halt(
                            reason="softguard_api_circuit_breaker",
                            details={
                                "symbol": str(symbol).upper(),
                                "action": str(action).upper(),
                                "quantity": float(quantity),
                                "guard": guard_payload,
                            },
                        )
                        if self.live_softguard_auto_cancel_on_global_halt:
                            guard_payload["auto_cancel"] = self._auto_cancel_open_orders_if_due(reason="api_circuit_breaker")

                    if gate_name == "daily_loss_cap":
                        guard_payload["emergency_liquidation"] = self._maybe_run_emergency_liquidation(
                            trigger="daily_loss_cap",
                            details={
                                "symbol": str(symbol).upper(),
                                "action": str(action).upper(),
                            },
                        )

                    result = {
                        "status": status,
                        "mode": self.mode,
                        "decision": decision_entry,
                        "live_guard_decision": guard_payload,
                        "live_guard": self.live_guard.snapshot(),
                    }
                else:
                    asset_type = str(md.get("asset_type") or "EQUITY").strip().upper() or "EQUITY"
                    order_spec = self._build_live_order_spec(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        limit_price=limit_price,
                        asset_type=asset_type,
                    )
                    place = self._live_place_order(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        order_spec=order_spec,
                    )

                    if not place.get("ok"):
                        status = "LIVE_ORDER_SUBMIT_FAILED"
                        result = {
                            "status": status,
                            "mode": self.mode,
                            "decision": decision_entry,
                            "live_order": {
                                "symbol": str(symbol).upper(),
                                "action": str(action).upper(),
                                "quantity": float(quantity),
                                "order_spec": order_spec,
                                "error": place.get("error", "submit_failed"),
                                "details": place.get("details", {}),
                            },
                            "live_guard": self.live_guard.snapshot(),
                        }
                    else:
                        self.live_guard.mark_trade_submitted(symbol=symbol)
                        order_id = str(place.get("order_id", "") or "").strip()
                        if order_id:
                            self.live_guard.register_open_order(
                                order_id=order_id,
                                symbol=symbol,
                                action=action,
                                quantity=quantity,
                            )

                        fills = (
                            self.refresh_live_fills(order_id=order_id)
                            if order_id
                            else {"status": "SKIPPED_NO_ORDER_ID", "processed": 0, "fills": []}
                        )
                        reconcile = self.confirm_live_position_state(symbol=symbol)

                        status = "LIVE_ORDER_SUBMITTED" if order_id else "LIVE_ORDER_ACK_NO_ID"
                        result = {
                            "status": status,
                            "mode": self.mode,
                            "decision": decision_entry,
                            "live_order": {
                                "order_id": order_id,
                                "symbol": str(symbol).upper(),
                                "action": str(action).upper(),
                                "quantity": float(quantity),
                                "order_spec": order_spec,
                                "api": {
                                    "method": place.get("method", ""),
                                    "latency_ms": place.get("latency_ms", 0.0),
                                    "status_code": place.get("status_code", 0),
                                },
                            },
                            "fills": fills,
                            "position_reconcile": reconcile,
                            "live_guard": self.live_guard.snapshot(),
                        }

        self._emit_decision_explanation(
            status=status,
            decision_entry=decision_entry,
            safety=safety,
        )
        return result

    def _record_jsonl(self, path: str, row: Dict[str, Any]) -> Dict[str, Any]:
        safe_append_jsonl(
            path,
            row,
            project_root=self.project_root,
            source="base_trader.record_jsonl",
        )
        return row
