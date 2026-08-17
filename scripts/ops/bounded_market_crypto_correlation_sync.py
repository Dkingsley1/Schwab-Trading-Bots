#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "market_crypto_correlation_sync_latest.json"
PAYLOAD_PATH = PROJECT_ROOT / "exports" / "external_context" / "market_crypto_correlation_latest.json"


def _json_from_stdout(stdout: str) -> dict:
    for line in reversed([row.strip() for row in str(stdout or "").splitlines() if row.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_timestamp(path: Path, payload: dict) -> float:
    for key in ("timestamp_utc", "generated_utc", "updated_at_utc", "updated_at", "created_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            continue
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _timeout_payload(*, outer_timeout_seconds: int, proc: subprocess.Popen[str]) -> dict:
    now = datetime.now(timezone.utc)
    existing_payload = _load_json(PAYLOAD_PATH)
    observed_ts = _payload_timestamp(PAYLOAD_PATH, existing_payload) if existing_payload else 0.0
    age_seconds = max(time.time() - observed_ts, 0.0) if observed_ts > 0.0 else None
    max_fallback_age = max(
        float(os.getenv("MARKET_CRYPTO_CORRELATION_TIMEOUT_FALLBACK_MAX_AGE_SECONDS", "43200") or 43200),
        0.0,
    )
    fallback_ready = bool(existing_payload and age_seconds is not None and age_seconds <= max_fallback_age)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": bool(fallback_ready),
        "status": "degraded",
        "timeout": True,
        "outer_timeout_seconds": int(outer_timeout_seconds),
        "error": "bounded_market_crypto_correlation_sync_timeout",
        "fallback_used": bool(fallback_ready),
        "partial_data": bool(fallback_ready),
        "fallback_payload_path": str(PAYLOAD_PATH),
        "fallback_payload_age_seconds": round(float(age_seconds), 3) if age_seconds is not None else None,
        "fallback_max_age_seconds": round(float(max_fallback_age), 3),
        "sources": {
            "last_known_market_crypto_correlation_payload": {
                "ok": bool(fallback_ready),
                "contract_participates": True,
                "age_seconds": round(float(age_seconds), 3) if age_seconds is not None else None,
                "path": str(PAYLOAD_PATH),
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run market/crypto correlation sync with a hard outer timeout.")
    parser.add_argument("--outer-timeout-seconds", type=int, default=100)
    args, passthrough = parser.parse_known_args()

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "collect_market_crypto_correlation_context.py"), *passthrough]
    if "--json" not in passthrough:
        cmd.append("--json")
    timeout_seconds = max(int(args.outer_timeout_seconds), 5)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        payload = _timeout_payload(outer_timeout_seconds=timeout_seconds, proc=proc)
        HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=True))
        return 0 if bool(payload.get("ok", False)) else 124
    if stderr:
        sys.stderr.write(stderr)
    payload = _json_from_stdout(stdout)
    if payload:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print((stdout or "").strip())
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
