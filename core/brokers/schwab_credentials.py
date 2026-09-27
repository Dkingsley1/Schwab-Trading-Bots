from __future__ import annotations

import getpass
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from core.brokers.models import BrokerCredentials


DEFAULT_REDIRECT = "https://127.0.0.1:8182"
DEFAULT_SERVICES = {
    "api_key": "schwab_trading_bot/SCHWAB_API_KEY",
    "app_secret": "schwab_trading_bot/SCHWAB_SECRET",
    "callback_url": "schwab_trading_bot/SCHWAB_REDIRECT",
}
INVALID_VALUES = {
    "",
    "YOUR_KEY_HERE",
    "YOUR_SECRET_HERE",
    "YOUR_REAL_KEY",
    "YOUR_REAL_SECRET",
    "<real_key>",
    "<real_secret>",
}

KeychainReader = Callable[[str, str], str]

MANAGED_RUNTIME_REQUIRED_ENV = "SCHWAB_MANAGED_RUNTIME_REQUIRED"
MANAGED_RUNTIME_ATTESTED_ENV = "SCHWAB_MANAGED_RUNTIME_ATTESTED"
MANAGED_RUNTIME_SOURCE_ENV = "SCHWAB_MANAGED_RUNTIME_SOURCE"
MANAGED_RUNTIME_SOURCES = {
    "load_runtime_env",
    "opsctl",
    "managed_launcher",
}


def _enabled(value: object, *, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def credential_value_ready(value: object) -> bool:
    return str(value or "").strip() not in INVALID_VALUES


def managed_schwab_runtime_status(
    env: Mapping[str, str] | None = None,
    *,
    require_by_default: bool = False,
) -> dict[str, object]:
    """Return a redacted attestation for the Schwab credential execution path."""
    values = env if env is not None else os.environ
    required = _enabled(
        values.get(MANAGED_RUNTIME_REQUIRED_ENV, "1" if require_by_default else "0"),
        default=require_by_default,
    )
    attested = _enabled(
        values.get(MANAGED_RUNTIME_ATTESTED_ENV, "0"),
        default=False,
    )
    source = str(values.get(MANAGED_RUNTIME_SOURCE_ENV, "") or "").strip().lower()
    source_valid = source in MANAGED_RUNTIME_SOURCES
    ready = bool((not required) or (attested and source_valid))
    return {
        "required": required,
        "attested": attested,
        "source": source,
        "source_valid": source_valid,
        "ready": ready,
        "secret_material_present": False,
    }


def enforce_managed_schwab_runtime(
    env: Mapping[str, str] | None = None,
    *,
    require_by_default: bool = False,
) -> dict[str, object]:
    status = managed_schwab_runtime_status(
        env,
        require_by_default=require_by_default,
    )
    if not bool(status["ready"]):
        raise RuntimeError("schwab_managed_runtime_attestation_required")
    return status


def _read_keychain_secret(service: str, account: str) -> str:
    security = Path("/usr/bin/security")
    if sys.platform != "darwin" or not security.exists() or not service:
        return ""
    try:
        proc = subprocess.run(
            [
                str(security),
                "find-generic-password",
                "-a",
                account,
                "-s",
                service,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def resolve_schwab_credentials(
    env: Mapping[str, str] | None = None,
    *,
    read_keychain: KeychainReader = _read_keychain_secret,
) -> BrokerCredentials:
    """Resolve Schwab credentials without requiring secrets in process argv/env."""
    values = env if env is not None else os.environ
    api_key = str(values.get("SCHWAB_API_KEY", "") or "").strip()
    app_secret = str(values.get("SCHWAB_SECRET", "") or "").strip()
    callback_url = str(
        values.get("SCHWAB_CALLBACK_URL", "")
        or values.get("SCHWAB_REDIRECT", "")
        or ""
    ).strip()

    keychain_enabled = _enabled(
        values.get("SCHWAB_KEYCHAIN_RUNTIME_RESOLUTION_ENABLED", "0"),
        default=False,
    )
    if keychain_enabled:
        account = str(values.get("SCHWAB_KEYCHAIN_ACCOUNT", "") or "").strip() or getpass.getuser()
        services = {
            "api_key": str(
                values.get("SCHWAB_API_KEY_KEYCHAIN_SERVICE", DEFAULT_SERVICES["api_key"])
                or DEFAULT_SERVICES["api_key"]
            ).strip(),
            "app_secret": str(
                values.get("SCHWAB_SECRET_KEYCHAIN_SERVICE", DEFAULT_SERVICES["app_secret"])
                or DEFAULT_SERVICES["app_secret"]
            ).strip(),
            "callback_url": str(
                values.get("SCHWAB_REDIRECT_KEYCHAIN_SERVICE", DEFAULT_SERVICES["callback_url"])
                or DEFAULT_SERVICES["callback_url"]
            ).strip(),
        }
        if not credential_value_ready(api_key):
            api_key = str(read_keychain(services["api_key"], account) or "").strip()
        if not credential_value_ready(app_secret):
            app_secret = str(read_keychain(services["app_secret"], account) or "").strip()
        if not callback_url:
            callback_url = str(read_keychain(services["callback_url"], account) or "").strip()

    return BrokerCredentials(
        api_key=api_key or "YOUR_KEY_HERE",
        app_secret=app_secret or "YOUR_SECRET_HERE",
        callback_url=callback_url or DEFAULT_REDIRECT,
    )


def schwab_credentials_ready(credentials: BrokerCredentials) -> bool:
    return bool(
        credential_value_ready(credentials.api_key)
        and credential_value_ready(credentials.app_secret)
    )
