"""Optional JSONL decoding accelerator; retain stdlib compatibility on rejection."""

from __future__ import annotations

import json
from typing import Any

try:
    from msgspec.json import decode as _decode
except Exception:
    _decode = None


def loads(text: str) -> Any:
    if _decode is not None and isinstance(text, str):
        try:
            return _decode(text)
        except Exception:
            # Legacy JSONL can contain NaN, Infinity or escaped lone surrogates.
            pass
    return json.loads(text)
