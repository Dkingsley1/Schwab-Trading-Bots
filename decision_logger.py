import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DecisionLogger:
    """Writes model/risk decision audits for every trade candidate."""

    def __init__(self, project_root: Optional[str] = None, subdir: str = "decisions"):
        if project_root is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.project_root = project_root
        self.log_dir = os.path.join(project_root, subdir)
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"trade_decisions_{day}.jsonl")

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
            "features": features,
            "gates": gates,
            "reasons": reasons,
            "metadata": metadata or {},
        }

        self._append_jsonl(entry)
        return entry

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        path = self._log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

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
