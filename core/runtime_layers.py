import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


class StateCache:
    def __init__(self, default_ttl_seconds: float = 2.0) -> None:
        self.default_ttl_seconds = max(float(default_ttl_seconds), 0.0)
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        row = self._store.get(key)
        if not row:
            return None
        expires_at, value = row
        if now >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(float(ttl_seconds), 0.0)
        self._store[key] = (time.time() + ttl, value)


class CircuitBreaker:
    def __init__(self, fail_limit: int = 5, cooldown_seconds: int = 120) -> None:
        self.fail_limit = max(int(fail_limit), 1)
        self.cooldown_seconds = max(int(cooldown_seconds), 1)
        self._failures: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        return time.time() >= self._open_until.get(key, 0.0)

    def record_success(self, key: str) -> None:
        self._failures[key] = 0

    def record_failure(self, key: str) -> bool:
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self.fail_limit:
            self._open_until[key] = time.time() + self.cooldown_seconds
            self._failures[key] = 0
            return True
        return False


@dataclass
class BackpressureStatus:
    overloaded: bool
    loop_seconds: float
    ratio_vs_interval: float


class BackpressureController:
    def __init__(self, overload_ratio: float = 1.5) -> None:
        self.overload_ratio = max(float(overload_ratio), 1.0)

    def evaluate(self, loop_seconds: float, interval_seconds: int) -> BackpressureStatus:
        interval = max(float(interval_seconds), 1.0)
        ratio = float(loop_seconds) / interval
        return BackpressureStatus(overloaded=ratio >= self.overload_ratio, loop_seconds=float(loop_seconds), ratio_vs_interval=ratio)


class TelemetryEmitter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def emit(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


class CheckpointStore:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self, payload: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)


def config_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class CanaryRollout:
    def __init__(self, max_weight: float = 0.08) -> None:
        self.max_weight = min(max(float(max_weight), 0.0), 1.0)

    def apply(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows

        clipped = []
        for r in rows:
            out = dict(r)
            if bool(out.get("promoted", False)):
                out["weight"] = min(float(out.get("weight", 0.0)), self.max_weight)
            clipped.append(out)

        total = sum(max(float(x.get("weight", 0.0)), 0.0) for x in clipped)
        if total > 0:
            for x in clipped:
                x["weight"] = max(float(x.get("weight", 0.0)), 0.0) / total
        return clipped
