#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / "config" / ".env.secrets.local"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_credentials_setup_latest.json"
DEFAULT_API_KEY_SERVICE = "schwab_trading_bot/SCHWAB_API_KEY"
DEFAULT_SECRET_SERVICE = "schwab_trading_bot/SCHWAB_SECRET"
DEFAULT_REDIRECT_SERVICE = "schwab_trading_bot/SCHWAB_REDIRECT"
DEFAULT_REDIRECT = "https://127.0.0.1:8182"

INVALID_VALUES = {
    "",
    "YOUR_KEY_HERE",
    "YOUR_SECRET_HERE",
    "YOUR_REAL_KEY",
    "YOUR_REAL_SECRET",
    "<real_key>",
    "<real_secret>",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _placeholder_or_empty(value: Any) -> bool:
    text = str(value or "").strip()
    return text in INVALID_VALUES


def _account() -> str:
    return os.getenv("SCHWAB_KEYCHAIN_ACCOUNT", "").strip() or getpass.getuser()


def _service_names() -> dict[str, str]:
    return {
        "api_key": os.getenv("SCHWAB_API_KEY_KEYCHAIN_SERVICE", DEFAULT_API_KEY_SERVICE).strip() or DEFAULT_API_KEY_SERVICE,
        "secret": os.getenv("SCHWAB_SECRET_KEYCHAIN_SERVICE", DEFAULT_SECRET_SERVICE).strip() or DEFAULT_SECRET_SERVICE,
        "redirect": os.getenv("SCHWAB_REDIRECT_KEYCHAIN_SERVICE", DEFAULT_REDIRECT_SERVICE).strip() or DEFAULT_REDIRECT_SERVICE,
    }


def _security_bin() -> Path:
    return Path("/usr/bin/security")


def _read_keychain_secret(service: str, account: str) -> str:
    if sys.platform != "darwin" or not _security_bin().exists():
        return ""
    proc = subprocess.run(
        [str(_security_bin()), "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _write_keychain_secret(service: str, account: str, value: str) -> tuple[bool, str]:
    if sys.platform != "darwin" or not _security_bin().exists():
        return False, "macos_keychain_unavailable"
    proc = subprocess.run(
        [str(_security_bin()), "add-generic-password", "-a", account, "-s", service, "-w", value, "-U"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, "stored"
    return False, (proc.stderr or proc.stdout or f"security_exit_{proc.returncode}").strip()


def _quote_env(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _write_env_file(path: Path, values: dict[str, str], *, force: bool) -> tuple[bool, str]:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_keys = {
        line.split("=", 1)[0].strip()
        for line in existing.splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    protected = [key for key in values if key in existing_keys and not force]
    if protected:
        return False, "env_file_keys_exist_use_force:" + ",".join(sorted(protected))

    kept_lines = [
        line
        for line in existing.splitlines()
        if not (line.strip() and not line.strip().startswith("#") and "=" in line and line.split("=", 1)[0].strip() in values)
    ]
    if kept_lines and kept_lines[-1].strip():
        kept_lines.append("")
    if not kept_lines:
        kept_lines = ["# Local Schwab secrets. This file is ignored by git and loaded by opsctl."]
    for key, value in values.items():
        kept_lines.append(f"{key}={_quote_env(value)}")
    content = "\n".join(kept_lines).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return True, "stored"


def _read_env_file_presence(path: Path) -> dict[str, bool]:
    values = {"api_key": False, "secret": False, "redirect": False}
    if not path.exists():
        return values
    key_map = {
        "SCHWAB_API_KEY": "api_key",
        "SCHWAB_SECRET": "secret",
        "SCHWAB_REDIRECT": "redirect",
        "SCHWAB_CALLBACK_URL": "redirect",
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        target = key_map.get(key.strip())
        if not target:
            continue
        stripped = value.strip().strip("'").strip('"')
        if not _placeholder_or_empty(stripped):
            values[target] = True
    return values


def _credential_sources(
    *,
    env: dict[str, str] | None = None,
    account: str | None = None,
    services: dict[str, str] | None = None,
    env_file: Path | None = None,
    read_keychain: Callable[[str, str], str] = _read_keychain_secret,
) -> dict[str, Any]:
    env = env if isinstance(env, dict) else os.environ
    account = account or _account()
    services = services if isinstance(services, dict) else _service_names()
    keychain_disabled = str(env.get("SCHWAB_KEYCHAIN_FALLBACK_ENABLED", "")).strip().lower() in {"0", "false", "no", "off"}
    keychain_enabled = not keychain_disabled
    keychain_present: dict[str, bool] = {"api_key": False, "secret": False, "redirect": False}
    if keychain_enabled:
        for name, service in services.items():
            keychain_present[name] = not _placeholder_or_empty(read_keychain(service, account))
    env_present = {
        "api_key": not _placeholder_or_empty(env.get("SCHWAB_API_KEY", "")),
        "secret": not _placeholder_or_empty(env.get("SCHWAB_SECRET", "")),
        "redirect": not _placeholder_or_empty(env.get("SCHWAB_CALLBACK_URL", "") or env.get("SCHWAB_REDIRECT", "")),
    }
    env_file_present = _read_env_file_presence(env_file) if env_file is not None else {"api_key": False, "secret": False, "redirect": False}
    ready = bool(
        (env_present["api_key"] or keychain_present["api_key"] or env_file_present["api_key"])
        and (env_present["secret"] or keychain_present["secret"] or env_file_present["secret"])
    )
    return {
        "ready": ready,
        "env_present": env_present,
        "env_file_present": env_file_present,
        "keychain_present": keychain_present,
        "keychain_enabled": bool(keychain_enabled),
        "account": account,
        "services": services,
    }


def _prompt_secret(label: str, secret_prompt_fn: Callable[[str], str]) -> str:
    value = secret_prompt_fn(f"{label}: ").strip()
    if _placeholder_or_empty(value):
        raise ValueError(f"{label} was empty or placeholder")
    return value


def build_payload(
    *,
    interactive: bool,
    store: str,
    force: bool,
    env_file: Path,
    out_path: Path,
    input_fn: Callable[[str], str] = input,
    secret_prompt_fn: Callable[[str], str] = getpass.getpass,
    read_keychain: Callable[[str, str], str] = _read_keychain_secret,
    write_keychain: Callable[[str, str, str], tuple[bool, str]] = _write_keychain_secret,
) -> dict[str, Any]:
    account = _account()
    services = _service_names()
    before = _credential_sources(account=account, services=services, env_file=env_file, read_keychain=read_keychain)
    result: dict[str, Any] = {
        "store": store,
        "interactive": bool(interactive),
        "changed": False,
        "write_results": [],
    }

    if interactive:
        api_key = _prompt_secret("Schwab API key", secret_prompt_fn)
        secret = _prompt_secret("Schwab app secret", secret_prompt_fn)
        redirect_prompt = f"Schwab redirect/callback URL [{DEFAULT_REDIRECT}]: "
        redirect = input_fn(redirect_prompt).strip() or DEFAULT_REDIRECT
        if _placeholder_or_empty(redirect):
            redirect = DEFAULT_REDIRECT

        if store == "keychain":
            for name, value in (("api_key", api_key), ("secret", secret), ("redirect", redirect)):
                ok, message = write_keychain(services[name], account, value)
                result["write_results"].append({"target": name, "ok": bool(ok), "message": message})
            result["changed"] = all(bool(row.get("ok", False)) for row in result["write_results"])
        else:
            ok, message = _write_env_file(
                env_file,
                {
                    "SCHWAB_API_KEY": api_key,
                    "SCHWAB_SECRET": secret,
                    "SCHWAB_REDIRECT": redirect,
                    "SCHWAB_CALLBACK_URL": redirect,
                },
                force=force,
            )
            result["write_results"].append({"target": str(env_file), "ok": bool(ok), "message": message})
            result["changed"] = bool(ok)

    after = _credential_sources(account=account, services=services, env_file=env_file, read_keychain=read_keychain)
    status = "ready" if after["ready"] else "missing_credentials"
    if result["write_results"] and not all(bool(row.get("ok", False)) for row in result["write_results"]):
        status = "write_failed"
    payload = {
        "timestamp_utc": _iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "credential_sources_before": before,
        "credential_sources_after": after,
        "result": result,
        "policy": {
            "credential_values_logged": False,
            "preferred_store": "macos_keychain",
            "fallback_store": "config/.env.secrets.local",
            "browser_launch": "not_performed_by_credential_setup",
            "headless_browser": "never",
        },
        "recommended_next_commands": [
            "./scripts/ops/opsctl.sh token-refresh-interactive --force --prompt-before-browser --json",
            "./scripts/ops/opsctl.sh schwab-auth-supervisor --apply --json",
        ],
        "artifact_paths": {
            "json": str(out_path),
            "env_file": str(env_file),
        },
    }
    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    default_store = "keychain" if sys.platform == "darwin" else "env-file"
    parser = argparse.ArgumentParser(description="Safely enter or check local Schwab API credentials without logging secret values.")
    parser.add_argument("--interactive", action="store_true", help="Prompt locally for Schwab API key, app secret, and redirect URL.")
    parser.add_argument("--check", action="store_true", help="Only report whether credentials are available through env or Keychain.")
    parser.add_argument("--store", choices=["keychain", "env-file"], default=default_store)
    parser.add_argument("--force", action="store_true", help="Allow replacing existing keys in the env-file fallback.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    interactive = bool(args.interactive)
    if not interactive and not args.check:
        args.check = True

    payload = build_payload(
        interactive=interactive,
        store=str(args.store),
        force=bool(args.force),
        env_file=Path(args.env_file).expanduser(),
        out_path=Path(args.out_file).expanduser(),
    )
    _write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_credentials_setup "
            f"status={payload.get('overall_status', '')} "
            f"store={args.store} "
            f"ready={int(bool(payload.get('ok', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
