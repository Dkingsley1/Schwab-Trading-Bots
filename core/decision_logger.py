import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.accountability import current_correlation, safe_append_channel_event
from core.path_registry import decision_log_path


_MINIMAL_FEATURE_KEYS = (
    "last_price",
    "price",
    "pct_from_close",
    "mom_5m",
    "mom_15m",
    "vol_30m",
    "volatility_1m",
    "spread_bps",
    "bid_size",
    "ask_size",
    "range_pos",
    "tradeability_score",
    "market_micro_tradeability_score_norm",
    "market_micro_order_flow_imbalance_norm",
    "execution_fitness_norm",
    "allocation_confidence_norm",
    "cross_bot_conflict_norm",
    "quant_model_resource_pressure_norm",
    "paper_profitability_master_profit_score_norm",
    "paper_profitability_master_drag_norm",
    "paper_profitability_grandmaster_profit_score_norm",
    "paper_profitability_grandmaster_drag_norm",
    "grand_master_vote",
    "active_sub_bots",
    "sized_qty",
)
_ESSENTIAL_FEATURE_SUFFIXES = (
    "_edge_norm",
    "_risk_norm",
    "_confidence_norm",
    "_quality_norm",
    "_freshness_norm",
    "_readiness_norm",
    "_pressure_norm",
    "_signal_signed",
)
_ESSENTIAL_FEATURE_PREFIXES = (
    "ctx_",
    "execution_",
    "market_micro_",
    "paper_profitability_",
    "quant_strategy_",
)
_ESSENTIAL_FULL_LAYERS = {
    "grand_master",
    "master_bot",
    "options_master",
    "master_options",
    "futures_master",
    "master_futures",
    "sub_bot_paper_mirror",
    "options_sub_bot_paper_mirror",
    "futures_sub_bot_paper_mirror",
}


def normalize_decision_feature_mode(raw: Any) -> str:
    mode = str(raw or "full").strip().lower()
    return mode if mode in {"full", "essential", "minimal"} else "full"


def compact_decision_features(
    features: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    mode: str = "full",
    preserve_primary_layers: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Bound repeated feature payloads while retaining point-in-time join evidence."""
    source = dict(features or {})
    md = dict(metadata or {})
    requested_mode = normalize_decision_feature_mode(mode)
    layer = str(md.get("layer") or "").strip().lower()
    effective_mode = requested_mode
    if requested_mode == "essential" and preserve_primary_layers and layer in _ESSENTIAL_FULL_LAYERS:
        effective_mode = "full"

    snapshot_id = str(md.get("snapshot_id") or "").strip()
    if effective_mode == "full":
        return source, {
            "schema_version": 1,
            "requested_mode": requested_mode,
            "effective_mode": "full",
            "source_feature_count": len(source),
            "retained_feature_count": len(source),
            "omitted_feature_count": 0,
            "feature_snapshot_id": snapshot_id,
            "lossless": True,
        }

    limit = 40 if effective_mode == "minimal" else 128
    selected: Dict[str, Any] = {}
    for key in _MINIMAL_FEATURE_KEYS:
        if key in source:
            selected[key] = source[key]

    if effective_mode == "essential" and len(selected) < limit:
        ranked_keys = sorted(
            (
                str(key)
                for key, value in source.items()
                if str(key) not in selected
                and isinstance(value, (bool, int, float))
                and (
                    str(key).startswith(_ESSENTIAL_FEATURE_PREFIXES)
                    or str(key).endswith(_ESSENTIAL_FEATURE_SUFFIXES)
                )
            )
        )
        for key in ranked_keys:
            selected[key] = source[key]
            if len(selected) >= limit:
                break

    omitted = max(len(source) - len(selected), 0)
    return selected, {
        "schema_version": 1,
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "source_feature_count": len(source),
        "retained_feature_count": len(selected),
        "omitted_feature_count": omitted,
        "feature_snapshot_id": snapshot_id,
        "lossless": False,
        "lossless_source": "primary_decision_with_matching_snapshot_id" if snapshot_id else "",
        "join_keys": ["feature_snapshot_id", "symbol", "timestamp_utc"],
    }


class DecisionLogger:
    """Writes model/risk decision audits for every trade candidate."""

    def __init__(self, project_root: Optional[str] = None, subdir: str = "decisions"):
        if project_root is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.project_root = project_root
        self.subdir = str(subdir or "decisions")
        self.log_dir = os.path.join(project_root, self.subdir)
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return decision_log_path(self.project_root, self.subdir, day=day)

    def log_decision(
        self,
        *,
        symbol: str,
        action: str,
        model_score: float,
        threshold: float,
        quantity: float,
        features: Dict[str, Any],
        gates: Dict[str, bool],
        reasons: List[str],
        strategy: str = "default",
        order_type: str = "market",
        metadata: Optional[Dict[str, Any]] = None,
        feature_mode: str = "full",
    ) -> Dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        allow_trade = all(bool(v) for v in gates.values())

        md = dict(metadata or {})
        corr = current_correlation()

        run_id = str(md.get("run_id") or corr.get("run_id") or "").strip()
        iter_id = str(md.get("iter_id") or corr.get("iter_id") or "").strip()
        decision_id = str(md.get("decision_id") or uuid.uuid4())
        parent_decision_id = str(md.get("parent_decision_id") or "").strip()

        md["decision_id"] = decision_id
        md["parent_decision_id"] = parent_decision_id
        if run_id:
            md["run_id"] = run_id
        if iter_id:
            md["iter_id"] = iter_id

        logged_features, feature_contract = compact_decision_features(
            features,
            metadata=md,
            mode=feature_mode,
        )

        entry = {
            "timestamp_utc": ts,
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "model_score": float(model_score),
            "threshold": float(threshold),
            "decision": "EXECUTE" if allow_trade else "BLOCK",
            "decision_id": decision_id,
            "parent_decision_id": parent_decision_id,
            "parent_message_id": parent_decision_id,
            "run_id": run_id,
            "iter_id": iter_id,
            "features": logged_features,
            "feature_compaction_contract": feature_contract,
            "gates": gates,
            "reasons": reasons,
            "metadata": md,
        }

        self._append_jsonl(entry)
        return entry

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        path = self._log_path()
        safe_append_channel_event(
            path,
            payload,
            project_root=self.project_root,
            source="decision_logger",
            channel="decision",
            schema="decision",
        )

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        path = self._log_path()
        if not os.path.exists(path):
            return []

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if limit <= 0:
            return rows
        return rows[-limit:]
