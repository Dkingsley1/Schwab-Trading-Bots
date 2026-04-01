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
from math import gcd

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from typing import Any, Dict, List, Optional, Tuple

from schwab.auth import easy_client

from core.decision_logger import DecisionLogger
from core.derivatives_features import _days_to_expiry, _extract_option_rows, _option_row_strike, _option_side
from core.live_execution_controls import LiveExecutionGuard, LiveRiskConfig
from core.path_registry import auth_events_path, decision_explanations_paths, execution_guard_path, live_softguard_path

from core.accountability import current_correlation, now_utc_iso, safe_append_jsonl, safe_append_channel_event, safe_write_json_atomic
from core.halt_flags import write_halt_flag_atomic


_FUTURES_MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
_FUTURES_CODE_TO_MONTH = {v: k for k, v in _FUTURES_MONTH_CODES.items()}
_QUARTERLY_FUTURES_ROOTS = {
    "ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K", "ZB", "ZN", "ZF", "ZT", "6E", "6J", "6B",
}
_FUTURES_CONTRACT_MULTIPLIERS = {
    "ES": 50.0,
    "MES": 5.0,
    "NQ": 20.0,
    "MNQ": 2.0,
    "YM": 5.0,
    "MYM": 0.5,
    "RTY": 50.0,
    "M2K": 5.0,
    "CL": 1000.0,
    "MCL": 100.0,
    "GC": 100.0,
    "MGC": 10.0,
    "SI": 5000.0,
    "SIL": 1000.0,
    "HG": 25000.0,
    "NG": 10000.0,
    "RB": 42000.0,
    "HO": 42000.0,
    "ZB": 1000.0,
    "ZN": 1000.0,
    "ZF": 1000.0,
    "ZT": 2000.0,
    "6E": 125000.0,
    "6J": 12500000.0,
    "6B": 62500.0,
}
_FUTURES_CONTRACT_RE = re.compile(r"^/?([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d{1,4})$")


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
        self.live_account_hash_auto_discover = os.getenv("SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER", "1").strip() == "1"
        self.live_accounts_snapshot_allow_global_fallback = (
            os.getenv("LIVE_ACCOUNTS_SNAPSHOT_ALLOW_GLOBAL_FALLBACK", "0").strip() == "1"
        )
        self._live_account_hash_last_refresh_ts = 0.0
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
        self.live_accounts_snapshot_soft_fail_grace = max(
            int(os.getenv("LIVE_ACCOUNTS_SNAPSHOT_SOFT_FAIL_GRACE", "3")),
            1,
        )
        self._accounts_snapshot_soft_fail_streak = 0
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

    def _extract_account_hash_rows(self, payload: Any) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not isinstance(payload, list):
            return rows
        for item in payload:
            if not isinstance(item, dict):
                continue
            account_hash = str(item.get("hashValue") or item.get("account_hash") or "").strip()
            if not account_hash:
                continue
            rows.append(
                {
                    "account_hash": account_hash,
                    "account_number": str(item.get("accountNumber") or item.get("account_number") or "").strip(),
                }
            )
        return rows

    def _discover_live_account_hash(self, *, force: bool = False) -> str:
        if self.client is None:
            return str(self.live_account_hash or "").strip()
        if not self.live_account_hash_auto_discover:
            return str(self.live_account_hash or "").strip()
        if self.live_account_hash and not force:
            return str(self.live_account_hash).strip()

        now_ts = time.time()
        if not force and (now_ts - float(self._live_account_hash_last_refresh_ts)) < 30.0:
            return str(self.live_account_hash or "").strip()
        self._live_account_hash_last_refresh_ts = now_ts

        fn = getattr(self.client, "get_account_numbers", None)
        if not callable(fn):
            return str(self.live_account_hash or "").strip()

        try:
            response = fn()
            status_code = self._as_int(getattr(response, "status_code", 0), 0)
            if status_code >= 400:
                raise RuntimeError(f"http_status_{status_code}")
            payload = self._coerce_json_obj_or_list(response)
            rows = self._extract_account_hash_rows(payload)
            if rows:
                self.live_account_hash = str(rows[0].get("account_hash") or "").strip()
        except Exception:
            pass

        return str(self.live_account_hash or "").strip()

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
            if not self.live_account_hash:
                self._discover_live_account_hash(force=True)
            self._log_auth_event(
                event="auth_success",
                status="ok",
                details={
                    "interactive": bool(interactive),
                    "callback_timeout_seconds": float(callback_timeout),
                    "requested_browser": requested_browser,
                    "max_token_age_seconds": max_token_age,
                    "account_hash_configured": bool(self.live_account_hash),
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
            "CLOSE",
            "ROLL",
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

                    if operation == "get_accounts_snapshot":
                        self._accounts_snapshot_soft_fail_streak = 0
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

        attempts_made = int(final_failure.get("attempt", len(attempt_failures)) or len(attempt_failures) or 1)
        if operation == "get_accounts_snapshot":
            self._accounts_snapshot_soft_fail_streak += 1
            soft_streak = int(self._accounts_snapshot_soft_fail_streak)
            soft_grace = int(self.live_accounts_snapshot_soft_fail_grace)
            if soft_streak <= soft_grace:
                details = {
                    "method": str(final_failure.get("method", "")),
                    "latency_ms": float(final_failure.get("latency_ms", 0.0) or 0.0),
                    "retryable": bool(final_failure.get("retryable", False)),
                    "status_code": int(final_failure.get("status_code", 0) or 0),
                    "attempt": int(final_failure.get("attempt", attempts_made) or attempts_made),
                    "attempts_made": attempts_made,
                    "max_attempts": max_attempts,
                    "soft_fail_streak": soft_streak,
                    "soft_fail_grace": soft_grace,
                    **(context or {}),
                }
                self._log_live_guard_event(
                    event=operation,
                    status="warn",
                    reason=str(final_failure.get("error", "accounts_snapshot_transient_failure")),
                    details=details,
                )
                self._log_softguard_event(
                    event="accounts_snapshot_transient_failure",
                    status="warn",
                    reason=str(final_failure.get("error", "accounts_snapshot_transient_failure")),
                    details={
                        "operation": operation,
                        "attempts_made": attempts_made,
                        "max_attempts": max_attempts,
                        "soft_fail_streak": soft_streak,
                        "soft_fail_grace": soft_grace,
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
                    "circuit_opened": False,
                    "soft_failure": True,
                    "soft_fail_streak": soft_streak,
                    "soft_fail_grace": soft_grace,
                    "details": {
                        "failures": attempt_failures[-max_attempts:],
                    },
                }

        opened = self.live_guard.record_api_failure("broker_api")

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

    def _quote_client_candidates(self, *, symbol: str) -> List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]:
        symbol_value = str(symbol or "").strip().upper()
        return [
            ("get_quote", (symbol_value,), {}),
            ("quote", (symbol_value,), {}),
            ("get_quotes", ((symbol_value,),), {}),
            ("get_quotes", ([symbol_value],), {}),
            ("quotes", ((symbol_value,),), {}),
            ("quotes", ([symbol_value],), {}),
        ]

    def _extract_quote_payload(self, raw: Any, symbol: str) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        sym = str(symbol or "").strip().upper()
        if sym in raw and isinstance(raw[sym], dict):
            return raw[sym]

        normalized = re.sub(r"[^A-Z0-9]", "", sym)
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            if key.upper() == sym:
                return value
            if re.sub(r"[^A-Z0-9]", "", key.upper()) == normalized:
                return value

        dict_children = [v for v in raw.values() if isinstance(v, dict)]
        if len(dict_children) == 1:
            return dict_children[0]
        return {}

    def _quote_field(self, payload: Dict[str, Any], *keys: str) -> Any:
        if not isinstance(payload, dict):
            return None

        containers: List[Dict[str, Any]] = [payload]
        for nested in ("quote", "regular", "reference", "extended", "fundamental"):
            child = payload.get(nested)
            if isinstance(child, dict):
                containers.append(child)

        for container in containers:
            for key in keys:
                if key in container and container.get(key) is not None:
                    return container.get(key)
        return None

    def _fetch_live_quote(self, *, symbol: str) -> Dict[str, Any]:
        out = self._invoke_client_candidates(
            operation="get_quote",
            candidates=self._quote_client_candidates(symbol=symbol),
            context={"symbol": str(symbol).upper()},
        )
        if not out.get("ok"):
            return out

        payload = self._coerce_json_obj_or_list(out.get("response"))
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "operation": "get_quote",
                "error": "invalid_quote_payload",
                "details": {"symbol": str(symbol).upper()},
            }

        quote_payload = self._extract_quote_payload(payload, symbol)
        out["payload"] = payload
        out["quote_payload"] = quote_payload
        return out

    def _option_chain_client_candidates(self, *, symbol: str, strike_count: int) -> List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]:
        symbol_value = str(symbol or "").strip().upper()
        arg_sets = [
            ((symbol_value,), {"strike_count": strike_count, "include_quotes": True}),
            ((symbol_value,), {"strike_count": strike_count}),
            ((symbol_value,), {"include_quotes": True}),
            ((symbol_value,), {}),
            ((), {"symbol": symbol_value, "strike_count": strike_count, "include_quotes": True}),
            ((), {"symbol": symbol_value, "strike_count": strike_count}),
            ((), {"symbol": symbol_value, "include_quotes": True}),
            ((), {"symbol": symbol_value}),
        ]
        out: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        for method_name in (
            "get_option_chain",
            "get_options_chain",
            "option_chain",
            "options_chain",
            "get_option_chain_for_symbol",
        ):
            for args, kwargs in arg_sets:
                out.append((method_name, args, dict(kwargs)))
        return out

    def _fetch_live_option_chain(self, *, symbol: str) -> Dict[str, Any]:
        strike_count = max(int(os.getenv("SCHWAB_OPTIONS_CHAIN_STRIKE_COUNT", "18") or 18), 4)
        out = self._invoke_client_candidates(
            operation="get_option_chain",
            candidates=self._option_chain_client_candidates(symbol=symbol, strike_count=strike_count),
            context={"symbol": str(symbol).upper(), "strike_count": strike_count},
        )
        if not out.get("ok"):
            return out

        payload = self._coerce_json_obj_or_list(out.get("response"))
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "operation": "get_option_chain",
                "error": "invalid_option_chain_payload",
                "details": {"symbol": str(symbol).upper()},
            }

        out["payload"] = payload
        return out

    def _option_contract_symbol_from_row(self, row: Dict[str, Any]) -> str:
        for key in ("symbol", "optionSymbol", "option_symbol"):
            raw = row.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip().upper()
        return ""

    def _option_quote_from_row(self, row: Dict[str, Any], *, instruction: str) -> float:
        bid = max(self._as_float(row.get("bidPrice"), 0.0), self._as_float(row.get("bid"), 0.0))
        ask = max(self._as_float(row.get("askPrice"), 0.0), self._as_float(row.get("ask"), 0.0))
        mark = max(self._as_float(row.get("mark"), 0.0), self._as_float(row.get("markPrice"), 0.0))
        last = max(self._as_float(row.get("last"), 0.0), self._as_float(row.get("lastPrice"), 0.0))
        if instruction in {"BUY_TO_OPEN", "BUY_TO_CLOSE", "BUY", "BUY_TO_COVER"}:
            return max(ask, mark, last, bid, 0.0)
        return max(bid, mark, last, ask, 0.0)

    def _option_leg_instruction(self, *, overall_action: str, leg_side: str) -> str:
        side = str(leg_side or overall_action or "").strip().upper()
        overall = str(overall_action or "").strip().upper()
        close_reverse = {
            "BUY_TO_OPEN": "SELL_TO_CLOSE",
            "SELL_TO_OPEN": "BUY_TO_CLOSE",
            "BUY": "SELL",
            "SELL": "BUY_TO_COVER",
        }
        if overall in {"CLOSE", "BUY_TO_CLOSE", "SELL_TO_CLOSE"}:
            side = close_reverse.get(side, side)
        return self._order_instruction(side)

    def _pick_option_chain_contract(
        self,
        *,
        symbol: str,
        payload: Dict[str, Any],
        leg: Dict[str, Any],
        overall_action: str,
        now_ts: float,
    ) -> Dict[str, Any]:
        option_type = str(leg.get("type") or leg.get("option_type") or "").strip().upper()
        target_strike = max(self._as_float(leg.get("strike"), 0.0), 0.0)
        target_expiry_days = max(self._as_float(leg.get("expiry_days"), 0.0), 0.0)
        requested_qty = max(int(round(self._as_float(leg.get("quantity"), 1.0))), 1)
        if option_type not in {"CALL", "PUT"}:
            raise ValueError(f"unsupported_option_type:{option_type or 'UNKNOWN'}")

        instruction = self._option_leg_instruction(overall_action=overall_action, leg_side=str(leg.get("side") or ""))
        best_match: Optional[Dict[str, Any]] = None
        best_score = float("inf")

        for row in _extract_option_rows(payload, now_ts=now_ts):
            if _option_side(row) != option_type:
                continue

            contract_symbol = self._option_contract_symbol_from_row(row)
            if not contract_symbol:
                continue

            strike_value = _option_row_strike(row, None)
            expiry_value = _days_to_expiry(
                row.get("daysToExpiration") if row.get("daysToExpiration") is not None else row.get("expirationDate"),
                now_ts=now_ts,
            )
            if strike_value <= 0.0 or expiry_value is None:
                continue

            strike_gap = abs(strike_value - target_strike)
            expiry_gap = abs(float(expiry_value) - target_expiry_days)
            target_strike_denom = max(target_strike, 1.0)
            target_expiry_denom = max(target_expiry_days, 1.0)
            quote_value = self._option_quote_from_row(row, instruction=instruction)

            bid = max(self._as_float(row.get("bidPrice"), 0.0), self._as_float(row.get("bid"), 0.0))
            ask = max(self._as_float(row.get("askPrice"), 0.0), self._as_float(row.get("ask"), 0.0))
            spread_penalty = 0.0
            mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else 0.0
            if mid > 0.0 and ask >= bid:
                spread_penalty = min((ask - bid) / mid, 1.0)
            elif quote_value <= 0.0:
                spread_penalty = 1.0

            score = (
                7.0 * (strike_gap / target_strike_denom)
                + 3.0 * (expiry_gap / target_expiry_denom)
                + 0.5 * spread_penalty
            )
            if score >= best_score:
                continue

            best_score = score
            best_match = {
                "contract_symbol": contract_symbol,
                "instruction": instruction,
                "option_type": option_type,
                "target_strike": float(target_strike),
                "resolved_strike": float(strike_value),
                "strike_gap": float(strike_gap),
                "target_expiry_days": float(target_expiry_days),
                "resolved_expiry_days": float(expiry_value),
                "expiry_gap": float(expiry_gap),
                "quantity": int(requested_qty),
                "quote": float(quote_value),
                "row": dict(row),
            }

        if best_match is None:
            raise ValueError(f"option_contract_not_found:{str(symbol).upper()}:{option_type}:{target_strike:.2f}:{target_expiry_days:.1f}")

        max_strike_gap = max(target_strike * 0.12, 3.0)
        max_expiry_gap = max(target_expiry_days * 0.75, 14.0)
        if best_match["strike_gap"] > max_strike_gap or best_match["expiry_gap"] > max_expiry_gap:
            raise ValueError(
                "option_contract_resolution_too_wide:"
                f"{best_match['contract_symbol']}:strike_gap={best_match['strike_gap']:.2f}:expiry_gap={best_match['expiry_gap']:.2f}"
            )
        return best_match

    def _strategy_unit_quantity(self, quantities: List[int]) -> int:
        unit_qty = 0
        for raw_qty in quantities:
            qty = max(int(raw_qty), 1)
            unit_qty = qty if unit_qty == 0 else gcd(unit_qty, qty)
        return max(unit_qty, 1)

    def _options_complex_strategy_type(self, *, options_style: str, legs: List[Dict[str, Any]], action: str = "") -> str:
        style = str(options_style or "").strip().upper()
        if str(action or "").strip().upper() == "ROLL":
            if style in {
                "BULL_PUT_CREDIT_SPREAD",
                "BEAR_CALL_CREDIT_SPREAD",
                "BULL_CALL_DEBIT_SPREAD",
                "BEAR_PUT_DEBIT_SPREAD",
            }:
                return "VERTICAL_ROLL"
            return "CUSTOM"
        if len(legs) <= 1:
            return "NONE"

        mapping = {
            "PROTECTIVE_COLLAR": "COLLAR_SYNTHETIC",
            "EVENT_VOL_STRADDLE": "STRADDLE",
            "EVENT_VOL_STRANGLE": "STRANGLE",
            "POOR_MANS_COVERED_CALL": "DIAGONAL",
            "BULL_PUT_CREDIT_SPREAD": "VERTICAL",
            "BEAR_CALL_CREDIT_SPREAD": "VERTICAL",
            "BULL_CALL_DEBIT_SPREAD": "VERTICAL",
            "BEAR_PUT_DEBIT_SPREAD": "VERTICAL",
            "CALL_CALENDAR_SPREAD": "CALENDAR",
            "PUT_CALENDAR_SPREAD": "CALENDAR",
            "DIAGONAL_CALENDAR_SPREAD": "DIAGONAL",
            "IRON_CONDOR": "IRON_CONDOR",
            "IRON_BUTTERFLY": "CUSTOM",
            "BROKEN_WING_BUTTERFLY": "CUSTOM",
            "RISK_REVERSAL_BULLISH": "CUSTOM",
            "RISK_REVERSAL_BEARISH": "CUSTOM",
        }
        return mapping.get(style, "CUSTOM")

    def _options_roll_leg_specs(self, *, options_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_legs = options_plan.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            return []
        current_dte = max(int(round(self._as_float(options_plan.get("dte_days"), 0.0))), 0)
        target_dte = max(int(round(self._as_float(options_plan.get("roll_target_dte_days"), 0.0))), 0)
        if current_dte <= 0:
            current_dte = max(int(round(max(self._as_float(leg.get("expiry_days"), 0.0), 0.0))) for leg in raw_legs if isinstance(leg, dict))
        if target_dte <= 0:
            target_dte = current_dte + max(int(os.getenv("OPTIONS_ROLL_FORWARD_DAYS", "21") or 21), 7)
        delta_dte = max(target_dte - max(current_dte, 1), 7)

        close_legs: List[Dict[str, Any]] = []
        open_legs: List[Dict[str, Any]] = []
        for raw_leg in raw_legs:
            if not isinstance(raw_leg, dict):
                continue
            close_leg = dict(raw_leg)
            close_leg["_execution_mode"] = "CLOSE"
            close_legs.append(close_leg)

            open_leg = dict(raw_leg)
            open_leg["expiry_days"] = max(int(round(self._as_float(raw_leg.get("expiry_days"), current_dte))) + delta_dte, target_dte)
            open_leg["_execution_mode"] = "OPEN"
            open_legs.append(open_leg)
        return close_legs + open_legs

    def _futures_month_cycle(self, root_symbol: str) -> List[int]:
        root = str(root_symbol or "").strip().upper()
        if root in _QUARTERLY_FUTURES_ROOTS:
            return [3, 6, 9, 12]
        return list(range(1, 13))

    def _normalize_futures_root_symbol(self, symbol: str) -> str:
        raw = str(symbol or "").strip().upper()
        if not raw:
            return ""
        if raw.startswith("/"):
            raw = raw[1:]
        if raw.endswith("=F"):
            raw = raw[:-2]
        match = _FUTURES_CONTRACT_RE.match(raw)
        if match:
            return match.group(1)
        return raw

    def _parse_futures_contract_symbol(self, symbol: str) -> Optional[Tuple[str, int, int]]:
        raw = str(symbol or "").strip().upper()
        match = _FUTURES_CONTRACT_RE.match(raw)
        if not match:
            return None
        root, month_code, year_text = match.groups()
        month = _FUTURES_CODE_TO_MONTH.get(month_code)
        if month is None:
            return None
        year_value = int(year_text)
        if year_value < 100:
            year_value += 2000
        return root, month, year_value

    def _futures_contract_multiplier(self, root_symbol: str) -> float:
        root = self._normalize_futures_root_symbol(root_symbol)
        return max(float(_FUTURES_CONTRACT_MULTIPLIERS.get(root, float(os.getenv("FUTURES_CONTRACT_MULTIPLIER_DEFAULT", "50") or 50.0))), 1.0)

    def _advance_futures_cycle(self, *, month: int, year: int, cycle: List[int], offset_contracts: int) -> Tuple[int, int]:
        target_month = int(month)
        target_year = int(year)
        cycle_sorted = list(sorted(int(x) for x in cycle if 1 <= int(x) <= 12)) or list(range(1, 13))
        current_index = cycle_sorted.index(target_month) if target_month in cycle_sorted else 0
        remaining = max(int(offset_contracts), 0)
        while remaining > 0:
            current_index += 1
            if current_index >= len(cycle_sorted):
                current_index = 0
                target_year += 1
            remaining -= 1
        return cycle_sorted[current_index], target_year

    def _candidate_futures_symbols(self, *, root_symbol: str, month: int, year: int, prefer_slash: bool) -> List[str]:
        root = self._normalize_futures_root_symbol(root_symbol)
        code = _FUTURES_MONTH_CODES[int(month)]
        yy = str(year)[-2:]
        y = str(year)[-1:]
        variants = [f"{root}{code}{yy}", f"{root}{code}{y}", f"{root}{code}{year}"]
        out: List[str] = []
        for base in variants:
            if prefer_slash:
                out.append(f"/{base}")
            out.append(base)
            if not prefer_slash:
                out.append(f"/{base}")
        seen: set[str] = set()
        uniq: List[str] = []
        for candidate in out:
            if candidate in seen:
                continue
            seen.add(candidate)
            uniq.append(candidate)
        return uniq

    def _futures_active_symbol_hints(self, payload: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        for key in (
            "futureActiveSymbol",
            "activeSymbol",
            "activeContractSymbol",
            "futureSymbol",
            "symbol",
        ):
            raw = self._quote_field(payload, key)
            if raw is None or not str(raw).strip():
                continue
            hints.append(str(raw).strip().upper())
        seen: set[str] = set()
        uniq: List[str] = []
        for hint in hints:
            if hint in seen:
                continue
            seen.add(hint)
            uniq.append(hint)
        return uniq

    def _resolve_live_futures_contract_symbol(self, *, symbol: str, month_offset: int) -> Dict[str, Any]:
        root_symbol = self._normalize_futures_root_symbol(symbol)
        if not root_symbol:
            raise ValueError("missing_futures_root_symbol")

        prefer_slash = str(symbol or "").strip().startswith("/")
        parsed = self._parse_futures_contract_symbol(symbol)
        if parsed is not None:
            base_root, base_month, base_year = parsed
            cycle = self._futures_month_cycle(base_root)
            target_month, target_year = self._advance_futures_cycle(
                month=base_month,
                year=base_year,
                cycle=cycle,
                offset_contracts=max(int(month_offset), 0),
            )
            for candidate in self._candidate_futures_symbols(
                root_symbol=base_root,
                month=target_month,
                year=target_year,
                prefer_slash=prefer_slash,
            ):
                quote = self._fetch_live_quote(symbol=candidate)
                if quote.get("ok") and isinstance(quote.get("quote_payload"), dict) and quote.get("quote_payload"):
                    return {
                        "contract_symbol": candidate,
                        "quote_payload": quote.get("quote_payload"),
                        "method": str(quote.get("method", "")),
                    }

        root_quote = self._fetch_live_quote(symbol=symbol)
        if root_quote.get("ok"):
            quote_payload = root_quote.get("quote_payload") if isinstance(root_quote.get("quote_payload"), dict) else {}
            for hint in self._futures_active_symbol_hints(quote_payload):
                parsed_hint = self._parse_futures_contract_symbol(hint)
                if parsed_hint is None:
                    continue
                hint_root, hint_month, hint_year = parsed_hint
                cycle = self._futures_month_cycle(hint_root)
                target_month, target_year = self._advance_futures_cycle(
                    month=hint_month,
                    year=hint_year,
                    cycle=cycle,
                    offset_contracts=max(int(month_offset), 0),
                )
                for candidate in self._candidate_futures_symbols(
                    root_symbol=hint_root,
                    month=target_month,
                    year=target_year,
                    prefer_slash=hint.startswith("/"),
                ):
                    quote = self._fetch_live_quote(symbol=candidate)
                    if quote.get("ok") and isinstance(quote.get("quote_payload"), dict) and quote.get("quote_payload"):
                        return {
                            "contract_symbol": candidate,
                            "quote_payload": quote.get("quote_payload"),
                            "method": str(quote.get("method", "")),
                        }

        now_utc = datetime.now(timezone.utc)
        cycle = self._futures_month_cycle(root_symbol)
        candidate_month = cycle[0]
        candidate_year = now_utc.year
        for cycle_month in cycle:
            if cycle_month >= now_utc.month:
                candidate_month = cycle_month
                break
        else:
            candidate_month = cycle[0]
            candidate_year += 1

        target_month, target_year = self._advance_futures_cycle(
            month=candidate_month,
            year=candidate_year,
            cycle=cycle,
            offset_contracts=max(int(month_offset), 0),
        )
        for candidate in self._candidate_futures_symbols(
            root_symbol=root_symbol,
            month=target_month,
            year=target_year,
            prefer_slash=prefer_slash,
        ):
            quote = self._fetch_live_quote(symbol=candidate)
            if quote.get("ok") and isinstance(quote.get("quote_payload"), dict) and quote.get("quote_payload"):
                return {
                    "contract_symbol": candidate,
                    "quote_payload": quote.get("quote_payload"),
                    "method": str(quote.get("method", "")),
                }

        raise ValueError(f"futures_contract_not_found:{str(symbol).upper()}:offset={int(month_offset)}")

    def _build_live_futures_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float,
        futures_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan_action = str(action or "").strip().upper()
        raw_legs = futures_plan.get("roll_legs") if plan_action == "ROLL" else futures_plan.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            raise ValueError("missing_futures_plan_legs")

        order_legs: List[Dict[str, Any]] = []
        resolved_legs: List[Dict[str, Any]] = []
        reference_price = 0.0
        contract_multiplier = self._futures_contract_multiplier(symbol)
        signed_price_total = 0.0
        final_quantities: List[int] = []

        for raw_leg in raw_legs:
            if not isinstance(raw_leg, dict):
                continue
            side = self._order_instruction(str(raw_leg.get("side") or plan_action or "BUY"))
            resolved = self._resolve_live_futures_contract_symbol(
                symbol=str(symbol),
                month_offset=max(int(raw_leg.get("month_offset", 0) or 0), 0),
            )
            contract_symbol = str(resolved.get("contract_symbol", "") or "").upper()
            if not contract_symbol:
                raise ValueError("resolved_futures_contract_missing_symbol")
            quote_payload = resolved.get("quote_payload") if isinstance(resolved.get("quote_payload"), dict) else {}
            quote_ref = max(
                self._as_float(self._quote_field(quote_payload, "lastPrice", "mark", "markPrice", "closePrice"), 0.0),
                0.0,
            )
            if quote_ref > 0.0:
                reference_price = max(reference_price, quote_ref)
            leg_qty = max(int(round(self._as_float(raw_leg.get("quantity"), quantity))), 1)
            signed_leg_price = quote_ref if side.startswith("BUY") else -quote_ref
            signed_price_total += signed_leg_price * leg_qty
            final_quantities.append(leg_qty)
            order_legs.append(
                {
                    "instruction": side,
                    "quantity": float(leg_qty),
                    "instrument": {
                        "symbol": contract_symbol,
                        "assetType": "FUTURE",
                    },
                }
            )
            resolved_legs.append(
                {
                    "contract_symbol": contract_symbol,
                    "instruction": side,
                    "quantity": leg_qty,
                    "month_offset": int(raw_leg.get("month_offset", 0) or 0),
                }
            )

        if not order_legs:
            raise ValueError("empty_futures_order_legs")

        price_value = max(self._as_float(limit_price, 0.0), 0.0)
        order_spec: Dict[str, Any] = {
            "orderType": "MARKET" if price_value <= 0.0 else "LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": order_legs,
        }
        if price_value > 0.0:
            order_spec["price"] = round(price_value, 6)

        strategy_units = self._strategy_unit_quantity(final_quantities)
        estimated_unit_price = abs(signed_price_total) / max(float(strategy_units), 1.0)
        reference_value = float(price_value if price_value > 0.0 else (estimated_unit_price if estimated_unit_price > 0.0 else reference_price))

        return {
            "order_spec": order_spec,
            "reference_price": reference_value,
            "intended_price": reference_value,
            "notional_multiplier": float(contract_multiplier),
            "details": {
                "futures_style": str(futures_plan.get("futures_style", "")),
                "strategy_family": str(futures_plan.get("strategy_family", "")),
                "resolved_legs": resolved_legs,
                "estimated_unit_price": float(estimated_unit_price),
            },
        }

    def _build_live_options_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float,
        options_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_legs = options_plan.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            raise ValueError("missing_options_plan_legs")

        plan_action = str(action or "").strip().upper()

        fetch = self._fetch_live_option_chain(symbol=symbol)
        if not fetch.get("ok"):
            raise RuntimeError(str(fetch.get("error", "option_chain_unavailable")))

        payload = fetch.get("payload") if isinstance(fetch.get("payload"), dict) else {}
        now_ts = time.time()

        requested_qty = max(int(round(self._as_float(quantity, 0.0))), 0)
        plan_contracts = max(int(round(self._as_float(options_plan.get("contracts"), 0.0))), 0)
        scale = 1
        if plan_contracts > 0 and requested_qty > 0:
            raw_scale = requested_qty / max(plan_contracts, 1)
            rounded_scale = int(round(raw_scale))
            if abs(raw_scale - rounded_scale) > 1e-6:
                raise ValueError("non_integer_options_quantity_scale")
            scale = max(rounded_scale, 1)
        elif requested_qty > 0 and plan_contracts <= 0:
            scale = max(requested_qty, 1)

        leg_specs: List[Dict[str, Any]]
        if plan_action == "ROLL":
            leg_specs = self._options_roll_leg_specs(options_plan=options_plan)
            if not leg_specs:
                raise ValueError("options_roll_leg_specs_missing")
        else:
            leg_specs = [dict(leg) for leg in raw_legs if isinstance(leg, dict)]

        order_legs: List[Dict[str, Any]] = []
        resolved_legs: List[Dict[str, Any]] = []
        signed_price_total = 0.0
        final_quantities: List[int] = []
        for raw_leg in leg_specs:
            execution_mode = str(raw_leg.get("_execution_mode", "") or "").strip().upper()
            overall_action = "CLOSE" if execution_mode == "CLOSE" else action

            resolved = self._pick_option_chain_contract(
                symbol=symbol,
                payload=payload,
                leg=raw_leg,
                overall_action=overall_action,
                now_ts=now_ts,
            )
            final_qty = max(int(resolved["quantity"]) * scale, 1)
            instruction = str(resolved["instruction"])
            quote_value = max(float(resolved["quote"]), 0.0)
            signed_leg_price = quote_value if instruction.startswith("BUY") else -quote_value
            signed_price_total += signed_leg_price * final_qty
            final_quantities.append(final_qty)

            order_legs.append(
                {
                    "instruction": instruction,
                    "quantity": float(final_qty),
                    "instrument": {
                        "symbol": str(resolved["contract_symbol"]),
                        "assetType": "OPTION",
                    },
                }
            )
            resolved_legs.append(
                {
                    "contract_symbol": str(resolved["contract_symbol"]),
                    "instruction": instruction,
                    "quantity": int(final_qty),
                    "quote": float(quote_value),
                    "resolved_strike": float(resolved["resolved_strike"]),
                    "resolved_expiry_days": float(resolved["resolved_expiry_days"]),
                    "execution_mode": execution_mode or ("OPEN" if plan_action == "ROLL" else "TRADE"),
                }
            )

        if not order_legs:
            raise ValueError("empty_options_order_legs")

        strategy_units = self._strategy_unit_quantity(final_quantities)
        estimated_unit_price = abs(signed_price_total) / max(float(strategy_units), 1.0)
        explicit_limit = max(self._as_float(limit_price, 0.0), 0.0)

        if len(order_legs) == 1:
            effective_price = explicit_limit if explicit_limit > 0.0 else estimated_unit_price
            order_type = "LIMIT" if effective_price > 0.0 else "MARKET"
            price_value = effective_price
        else:
            effective_price = explicit_limit if explicit_limit > 0.0 else estimated_unit_price
            if abs(signed_price_total) <= 0.005 * max(float(strategy_units), 1.0):
                order_type = "NET_ZERO"
                price_value = 0.0
            elif signed_price_total > 0.0:
                order_type = "NET_DEBIT"
                price_value = effective_price
            else:
                order_type = "NET_CREDIT"
                price_value = effective_price
            if order_type != "NET_ZERO" and price_value <= 0.0:
                raise ValueError("options_net_price_unavailable")

        order_spec: Dict[str, Any] = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": order_legs,
        }

        complex_type = self._options_complex_strategy_type(
            options_style=str(options_plan.get("options_style", "")),
            legs=order_legs,
            action=plan_action,
        )
        if complex_type != "NONE":
            order_spec["complexOrderStrategyType"] = complex_type

        if price_value > 0.0 and order_type != "MARKET":
            order_spec["price"] = round(price_value, 2)

        return {
            "order_spec": order_spec,
            "reference_price": float(explicit_limit if explicit_limit > 0.0 else estimated_unit_price),
            "intended_price": float(explicit_limit if explicit_limit > 0.0 else estimated_unit_price),
            "notional_multiplier": 100.0,
            "details": {
                "option_chain_method": str(fetch.get("method", "")),
                "options_style": str(options_plan.get("options_style", "")),
                "strategy_family": str(options_plan.get("strategy_family", "")),
                "resolved_legs": resolved_legs,
                "estimated_unit_price": float(estimated_unit_price),
            },
        }

    def _build_live_single_order_spec(
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

    def _build_live_order_spec(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float = 0.0,
        asset_type: str = "EQUITY",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        md = metadata if isinstance(metadata, dict) else {}
        options_plan = md.get("options_plan") if isinstance(md.get("options_plan"), dict) else {}
        futures_plan = md.get("futures_plan") if isinstance(md.get("futures_plan"), dict) else {}
        if isinstance(options_plan, dict) and options_plan.get("legs"):
            return self._build_live_options_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                limit_price=limit_price,
                options_plan=options_plan,
            )["order_spec"]
        if isinstance(futures_plan, dict) and futures_plan.get("legs"):
            return self._build_live_futures_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                limit_price=limit_price,
                futures_plan=futures_plan,
            )["order_spec"]
        return self._build_live_single_order_spec(
            symbol=symbol,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            asset_type=asset_type,
        )

    def _prepare_live_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: float,
        limit_price: float = 0.0,
        asset_type: str = "EQUITY",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        md = metadata if isinstance(metadata, dict) else {}
        options_plan = md.get("options_plan") if isinstance(md.get("options_plan"), dict) else {}
        futures_plan = md.get("futures_plan") if isinstance(md.get("futures_plan"), dict) else {}

        if isinstance(options_plan, dict) and options_plan.get("legs"):
            try:
                preview = self._build_live_options_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    limit_price=limit_price,
                    options_plan=options_plan,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"options_order_prep_failed:{type(exc).__name__}:{exc}",
                    "details": {
                        "options_style": str(options_plan.get("options_style", "")),
                        "strategy_family": str(options_plan.get("strategy_family", "")),
                    },
                }
            return {"ok": True, **preview}

        if isinstance(futures_plan, dict) and futures_plan.get("legs"):
            try:
                preview = self._build_live_futures_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    limit_price=limit_price,
                    futures_plan=futures_plan,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"futures_order_prep_failed:{type(exc).__name__}:{exc}",
                    "details": {
                        "futures_style": str(futures_plan.get("futures_style", "")),
                        "strategy_family": str(futures_plan.get("strategy_family", "")),
                    },
                }
            return {"ok": True, **preview}

        return {
            "ok": True,
            "order_spec": self._build_live_single_order_spec(
                symbol=symbol,
                action=action,
                quantity=quantity,
                limit_price=limit_price,
                asset_type=asset_type,
            ),
            "reference_price": 0.0,
            "intended_price": 0.0,
            "notional_multiplier": 1.0,
            "details": {},
        }

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
        if not self.live_account_hash:
            self._discover_live_account_hash(force=True)

        def _snapshot_candidates() -> List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]:
            candidates: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
            if self.live_account_hash:
                candidates.append(("get_account", (self.live_account_hash,), {}))
            if (not candidates) or self.live_accounts_snapshot_allow_global_fallback:
                candidates.append(("get_accounts", tuple(), {}))
                candidates.append(("get_account", tuple(), {}))
            return candidates

        out = self._invoke_client_candidates(
            operation="get_accounts_snapshot",
            candidates=_snapshot_candidates(),
            context={"account_hash_configured": bool(self.live_account_hash)},
        )
        if (not out.get("ok")) and self.live_account_hash:
            status_code = self._as_int(out.get("status_code", 0), 0)
            if status_code in {401, 403, 404}:
                previous_hash = str(self.live_account_hash)
                self.live_account_hash = ""
                refreshed_hash = self._discover_live_account_hash(force=True)
                if refreshed_hash and refreshed_hash != previous_hash:
                    out = self._invoke_client_candidates(
                        operation="get_accounts_snapshot",
                        candidates=_snapshot_candidates(),
                        context={
                            "account_hash_configured": bool(self.live_account_hash),
                            "account_hash_refreshed": True,
                        },
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
                asset_type = str(md.get("asset_type") or "EQUITY").strip().upper() or "EQUITY"
                prepared_order = self._prepare_live_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    limit_price=limit_price,
                    asset_type=asset_type,
                    metadata=md,
                )
                if not prepared_order.get("ok"):
                    unsupported = bool(prepared_order.get("unsupported", False))
                    status = "LIVE_UNSUPPORTED_DERIVATIVES_ORDER" if unsupported else "LIVE_ORDER_PREP_FAILED"
                    reason = str(prepared_order.get("error", "order_prep_failed"))
                    self._log_softguard_event(
                        event="derivatives_execution_guard" if unsupported else "order_prep_failed",
                        status="blocked" if unsupported else "error",
                        reason=reason,
                        details={
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "quantity": float(quantity),
                            **(prepared_order.get("details", {}) if isinstance(prepared_order.get("details"), dict) else {}),
                        },
                    )
                    result = {
                        "status": status,
                        "mode": self.mode,
                        "decision": decision_entry,
                        "live_order": {
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "quantity": float(quantity),
                            "error": reason,
                            "details": prepared_order.get("details", {}),
                        },
                        "live_guard": self.live_guard.snapshot(),
                    }
                    self._emit_decision_explanation(
                        status=status,
                        decision_entry=decision_entry,
                        safety=safety,
                    )
                    return result

                guard_reference_price = float(prepared_order.get("reference_price", 0.0) or 0.0)
                if guard_reference_price <= 0.0:
                    guard_reference_price = ref_price
                guard_intended_price = float(prepared_order.get("intended_price", 0.0) or 0.0)
                if guard_intended_price <= 0.0:
                    guard_intended_price = intended_price
                notional_multiplier = max(float(prepared_order.get("notional_multiplier", 1.0) or 1.0), 1.0)
                guard_decision = self.live_guard.pre_trade_check(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    reference_price=guard_reference_price,
                    intended_price=guard_intended_price,
                    notional_multiplier=notional_multiplier,
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
                    order_spec = prepared_order.get("order_spec") if isinstance(prepared_order.get("order_spec"), dict) else {}
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
                                "details": {
                                    **(prepared_order.get("details", {}) if isinstance(prepared_order.get("details"), dict) else {}),
                                    **(place.get("details", {}) if isinstance(place.get("details"), dict) else {}),
                                },
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
                                "details": prepared_order.get("details", {}),
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
