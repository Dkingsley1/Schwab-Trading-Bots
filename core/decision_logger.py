import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.accountability import current_correlation, safe_append_channel_event
from core.path_registry import decision_log_path


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
            "features": features,
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
