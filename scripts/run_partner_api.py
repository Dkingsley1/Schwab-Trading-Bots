#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.licensing_api import build_partner_api


def main() -> int:
    host = os.getenv("LICENSING_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(float(os.getenv("LICENSING_API_PORT", "8787") or 8787))
    except Exception:
        port = 8787
    app = build_partner_api()
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
