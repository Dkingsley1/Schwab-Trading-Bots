# --- THE GENETIC CODE ---
# This file handles plumbing so strategy modules can focus on signal generation.

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from schwab.auth import easy_client

from core.decision_logger import DecisionLogger


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
        self.paper_log_path = os.path.join(self.project_root, f"paper_trades_{self.mode_label}.jsonl")
        self.live_log_path = os.path.join(self.project_root, f"live_orders_{self.mode_label}.jsonl")
        self.paper_bridge_log_dir = os.path.join(self.project_root, "exports", "paper_broker_bridge", self.mode_label)
        self.decision_logger = DecisionLogger(self.project_root, subdir=os.path.join("decisions", self.mode_label))

    def authenticate(self):
        """Performs Schwab OAuth handshake."""
        print("Starting Handshake with Schwab...")
        self.client = easy_client(
            api_key=self.api_key,
            app_secret=self.app_secret,
            callback_url=self.callback_url,
            token_path=self.token_path,
        )
        print("Handshake Successful.")
        return self.client

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

    def _explanation_log_paths(self) -> tuple[str, str]:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        base_dir = os.path.join(self.project_root, "decision_explanations", self.mode_label)
        os.makedirs(base_dir, exist_ok=True)

        jsonl_path = os.path.join(base_dir, f"decision_explanations_{day}.jsonl")
        text_path = os.path.join(base_dir, "latest_decisions.log")
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
        }

        try:
            if self.paper_bridge_mode in {"jsonl", "both"}:
                daily_jsonl, latest_json = self._paper_bridge_paths()
                self._record_jsonl(daily_jsonl, payload)
                with open(latest_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
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
            "model_score": float(decision_entry.get("model_score", 0.0)),
            "threshold": float(decision_entry.get("threshold", 0.0)),
            "reasons": reasons,
            "gates": gates,
            "features": decision_entry.get("features", {}),
            "safety": safety or {},
            "metadata": decision_entry.get("metadata", {}),
        }

        jsonl_path, text_path = self._explanation_log_paths()
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        with open(text_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

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
            metadata={"mode": self.mode, **(metadata or {})},
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
                    "metadata": metadata or {},
                },
            )
            status = "PAPER_EXECUTED"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
                "paper_order": paper,
            }
            if self._is_trade_action(action):
                bridge = self._bridge_paper_order(paper_order=paper, result_status=status)
                result["paper_bridge"] = bridge
        else:
            # live mode
            if self.client is None:
                raise RuntimeError("Cannot execute live order without authentication")

            live_payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "strategy": strategy,
                "note": "Live order routing placeholder; implement Schwab order endpoint here.",
            }
            self._record_jsonl(self.live_log_path, live_payload)

            status = "LIVE_READY_NOOP"
            result = {
                "status": status,
                "mode": self.mode,
                "decision": decision_entry,
                "live_payload": live_payload,
            }

        self._emit_decision_explanation(
            status=status,
            decision_entry=decision_entry,
            safety=safety,
        )
        return result

    def _record_jsonl(self, path: str, row: Dict[str, Any]) -> Dict[str, Any]:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return row
