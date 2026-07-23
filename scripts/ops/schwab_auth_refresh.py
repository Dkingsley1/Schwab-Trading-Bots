#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.brokers.schwab.common import (
    credentials_ready,
    schwab_credentials_from_env,
    token_needs_refresh as common_token_needs_refresh,
    token_status as common_token_status,
)

DEFAULT_TOKEN_PATH = PROJECT_ROOT / "token.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_auth_refresh_latest.json"
BROWSER_APP_ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "google-chrome": "Google Chrome",
    "chromium": "Chromium",
    "safari": "Safari",
    "firefox": "Firefox",
    "edge": "Microsoft Edge",
    "microsoft edge": "Microsoft Edge",
    "microsoft-edge": "Microsoft Edge",
    "msedge": "Microsoft Edge",
}


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _browser_auth_disabled() -> bool:
    return _env_flag("SCHWAB_AUTH_BROWSER_DISABLED", "0") or not _env_flag("SCHWAB_AUTH_ALLOW_BROWSER_OPEN", "1")


def _token_status(path: Path) -> Dict[str, Any]:
    return common_token_status(path)


def _token_needs_refresh(status: Dict[str, Any], min_expires_seconds: float) -> tuple[bool, str]:
    return common_token_needs_refresh(
        status,
        min_expires_seconds=min_expires_seconds,
        ready_reason="token_ready",
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_browser_app_name(requested_browser: str | None) -> str | None:
    raw = str(requested_browser or "").strip()
    if not raw:
        return None
    return BROWSER_APP_ALIASES.get(raw.lower(), raw)


def _open_url_via_applescript(url: str, requested_browser: str | None) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "not_macos"

    app_name = _normalize_browser_app_name(requested_browser)
    if not app_name:
        return False, "missing_app_name"

    escaped_url = json.dumps(str(url))
    script = (
        f'tell application "{app_name}"\n'
        "activate\n"
        f"open location {escaped_url}\n"
        "end tell\n"
    )
    proc = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, f"applescript_app:{app_name}"
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    return False, stderr or stdout or f"exit_{proc.returncode}"


def _open_url_via_macos(url: str, requested_browser: str | None) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "not_macos"

    ok, method = _open_url_via_applescript(url, requested_browser)
    if ok:
        return True, method

    commands = []
    app_name = _normalize_browser_app_name(requested_browser)
    if app_name:
        commands.append(["/usr/bin/open", "-a", app_name, url])
    commands.append(["/usr/bin/open", url])

    last_error = ""
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            method = f"open_app:{app_name}" if len(command) >= 4 else "open_default"
            return True, method
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        last_error = stderr or stdout or f"exit_{proc.returncode}"
    return False, last_error or "open_failed"


class _SchwabAuthBrowserController:
    def __init__(self, requested_browser: str | None, primary_controller: Any = None) -> None:
        self.requested_browser = requested_browser
        self.primary_controller = primary_controller

    def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
        ok, _ = _open_url_via_macos(url, self.requested_browser)
        if ok:
            browser_label = _normalize_browser_app_name(self.requested_browser) or "default browser"
            print(f"Opening Schwab authorization page in {browser_label}...")
            return True

        if self.primary_controller is not None:
            try:
                return bool(self.primary_controller.open(url, new=new, autoraise=autoraise))
            except Exception:
                return False
        return False


def _install_schwab_browser_fallback() -> bool:
    try:
        schwab_auth = importlib.import_module("schwab.auth")
    except Exception:
        return False

    webbrowser_mod = getattr(schwab_auth, "webbrowser", None)
    original_get = getattr(webbrowser_mod, "get", None)
    if webbrowser_mod is None or original_get is None:
        return False
    if getattr(original_get, "_schwab_auth_refresh_patched", False):
        return True

    def _patched_get(requested_browser: str | None = None):
        primary_controller = None
        try:
            primary_controller = original_get(requested_browser)
        except Exception:
            primary_controller = None
        return _SchwabAuthBrowserController(requested_browser, primary_controller=primary_controller)

    _patched_get._schwab_auth_refresh_patched = True  # type: ignore[attr-defined]
    webbrowser_mod.get = _patched_get
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Schwab OAuth refresh.")
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--callback-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--min-expires-seconds",
        type=float,
        default=float(os.getenv("SCHWAB_AUTH_MIN_EXPIRES_SECONDS", os.getenv("PREMARKET_TOKEN_MIN_EXPIRES_SECONDS", "1500"))),
    )
    parser.add_argument("--requested-browser", default=os.getenv("SCHWAB_AUTH_REQUESTED_BROWSER", "chrome"))
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=_browser_auth_disabled(),
        help="Do not open a browser; return a blocked status if interactive OAuth would be required.",
    )
    parser.add_argument(
        "--prompt-before-browser",
        action="store_true",
        help="Require an extra ENTER press before opening the browser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Open the Schwab browser auth flow even when the token already satisfies the expiry floor.",
    )
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
    requested_browser = str(args.requested_browser or "").strip()
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "interactive": True,
        "token_before": before,
        "token_after": {},
        "callback_url": callback_url,
        "requested_browser": requested_browser or None,
        "requested_browser_resolved": _normalize_browser_app_name(requested_browser),
        "prompt_before_browser": bool(args.prompt_before_browser),
        "force_auth": bool(args.force),
        "skipped": False,
        "account_probe_ok": None,
        "account_probe_status_code": None,
        "reason": "",
    }

    refresh_needed_before, refresh_reason_before = _token_needs_refresh(
        before,
        min_expires_seconds=max(float(args.min_expires_seconds), 0.0),
    )
    payload["refresh_needed_before"] = bool(refresh_needed_before)
    payload["refresh_reason_before"] = refresh_reason_before
    if not args.force and not refresh_needed_before:
        payload["ok"] = True
        payload["skipped"] = True
        payload["reason"] = "token_already_ready"
        payload["token_after"] = before
        payload["refresh_needed_after"] = False
        payload["refresh_reason_after"] = refresh_reason_before
        _write_json(out_path, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            print("schwab_auth_refresh ok=1 reason=token_already_ready")
        return 0

    if not credentials_ready(schwab_credentials_from_env()):
        payload["reason"] = "missing_credentials"
        payload["token_after"] = before
        _write_json(out_path, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            print("schwab_auth_refresh ok=0 reason=missing_credentials")
        return 2

    if args.no_browser:
        payload["reason"] = "browser_disabled_token_refresh_required"
        payload["token_after"] = before
        payload["refresh_needed_after"] = bool(refresh_needed_before)
        payload["refresh_reason_after"] = refresh_reason_before
        _write_json(out_path, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            print("schwab_auth_refresh ok=0 reason=browser_disabled_token_refresh_required")
        return 3

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    _install_schwab_browser_fallback()

    # In schwab-py, "interactive" only controls whether it pauses for an
    # extra ENTER press before opening the browser. For this explicit
    # browser-based auth command, immediate launch is the least surprising
    # default, while still allowing the old pause behavior when requested.
    os.environ["SCHWAB_AUTH_INTERACTIVE"] = "1" if args.prompt_before_browser else "0"
    # Force the interactive flow to mint a genuinely new token instead of silently
    # accepting an already-near-expiry token from disk.
    os.environ["SCHWAB_MAX_TOKEN_AGE_SECONDS"] = os.getenv("SCHWAB_INTERACTIVE_FORCE_MAX_TOKEN_AGE_SECONDS", "1")
    os.environ["SCHWAB_AUTH_CALLBACK_TIMEOUT_SECONDS"] = str(max(float(args.callback_timeout_seconds), 5.0))
    if requested_browser:
        os.environ["SCHWAB_AUTH_REQUESTED_BROWSER"] = requested_browser

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
    refresh_needed_after, refresh_reason_after = _token_needs_refresh(
        payload["token_after"],
        min_expires_seconds=max(float(args.min_expires_seconds), 0.0),
    )
    payload["refresh_needed_after"] = bool(refresh_needed_after)
    payload["refresh_reason_after"] = refresh_reason_after
    if payload["ok"] and refresh_needed_after:
        payload["ok"] = False
        payload["reason"] = f"token_not_ready_after_auth:{refresh_reason_after}"
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
