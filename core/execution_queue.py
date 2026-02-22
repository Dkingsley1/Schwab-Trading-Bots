from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional


@dataclass
class OrderRequest:
    symbol: str
    action: str
    quantity: float
    priority: int
    metadata: Dict[str, Any]


class ExecutionQueue:
    def __init__(self, max_depth: int = 4000):
        self.max_depth = max(int(max_depth), 10)
        self._heap: List[tuple[int, int, OrderRequest]] = []
        self._counter = 0

    def enqueue(self, req: OrderRequest) -> bool:
        if len(self._heap) >= self.max_depth:
            return False
        self._counter += 1
        heappush(self._heap, (-int(req.priority), self._counter, req))
        return True

    def pop(self) -> Optional[OrderRequest]:
        if not self._heap:
            return None
        _, _, req = heappop(self._heap)
        return req

    def cancel_replace(self, *, symbol: str, new_req: OrderRequest) -> bool:
        kept: List[tuple[int, int, OrderRequest]] = []
        replaced = False
        while self._heap:
            item = heappop(self._heap)
            req = item[2]
            if req.symbol == symbol and not replaced:
                replaced = True
                continue
            kept.append(item)
        for item in kept:
            heappush(self._heap, item)
        return self.enqueue(new_req)

    def size(self) -> int:
        return len(self._heap)
