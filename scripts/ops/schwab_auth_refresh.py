#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "token.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_auth_refresh_latest.json"


def _token_status(path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "token_path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "age_seconds": None,
        "expires_at": "",
        "expires_in_seconds": None,
    }
    if not path.exists():
        return status

    try:
        st = path.stat()
        status["size_bytes"] = int(st.st_size)
        status["age_seconds"] = max(datetime.now(timezone.utc).timestamp() - float(st.st_mtime), 0.0)
    except Exception:
        return status

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status

    expires_at = str(payload.get("expires_at") or "").strip()
    status["expires_at"] = expires_at
    if expires_at:
        try:
            expires_epoch = float(expires_at)
            status["expires_in_seconds"] = expires_epoch - datetime.now(timezone.utc).timestamp()
        except Exception:
            status["expires_in_seconds"] = None
    return status


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Schwab OAuth refresh.")
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--callback-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--requested-browser", default="")
    parser.add_argument("--skip-account-probe", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("SCHWAB_API_KEY", "").strip()
    app_secret = os.getenv("SCHWAB_SECRET", "").strip()
    callback_url = (
        os.getenv("SCHWAB_CALLBACK_URL", "").strip()
        or os.getenv("SCHWAB_REDIRECT", "").strip()
        or "https://127.0.0.1:8182"
    )
    token_path = Path(args.token_path).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()

    before = _token_status(token_path)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "interactive": True,
        "token_before": before,
        "token_after": {},
        "callback_url": callback_url,
        "requested_browser": args.requested_browser or None,
        "account_probe_ok": None,
        "account_probe_status_code": None,
        "reason": "",
    }

    invalid = {"", "YOUR_KEY_HERE", "YOUR_SECRET_HERE", "YOUR_REAL_KEY", "YOUR_REAL_SECRET", "<real_key>", "<real_secret>"}
    if api_key in invalid or app_secret in invalid:
        payload["reason"] = "missing_credentials"
        payload["token_after"] = before
        _write_json(out_path, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            print("schwab_auth_refresh ok=0 reason=missing_credentials")
        return 2

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    os.environ["SCHWAB_AUTH_INTERACTIVE"] = "1"
    os.environ["SCHWAB_MAX_TOKEN_AGE_SECONDS"] = "0"
    os.environ["SCHWAB_AUTH_CALLBACK_TIMEOUT_SECONDS"] = str(max(float(args.callback_timeout_seconds), 5.0))
    if args.requested_browser:
        os.environ["SCHWAB_AUTH_REQUESTED_BROWSER"] = args.requested_browser

    try:
        from core.base_trader import BaseTrader

        trader = BaseTrader(api_key=api_key, app_secret=app_secret, callback_url=callback_url, mode="shadow")
        trader.token_path = str(token_path)
        client = trader.authenticate()
        payload["ok"] = True
        payload["reason"] = "auth_success"

        if not args.skip_account_probe:
            try:
                resp = client.get_account_numbers()
                status_code = int(getattr(resp, "status_code", 0) or 0)
                payload["account_probe_status_code"] = status_code
                payload["account_probe_ok"] = 200 <= status_code < 300
                if not payload["account_probe_ok"]:
                    payload["ok"] = False
                    payload["reason"] = f"account_probe_failed:{status_code}"
            except Exception as exc:
                payload["ok"] = False
                payload["account_probe_ok"] = False
                payload["reason"] = f"account_probe_error:{type(exc).__name__}:{exc}"
    except Exception as exc:
        payload["ok"] = False
        payload["reason"] = f"auth_error:{type(exc).__name__}:{exc}"

    payload["token_after"] = _token_status(token_path)
    _write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(
            "schwab_auth_refresh "
            f"ok={int(bool(payload['ok']))} "
            f"reason={payload['reason']}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
