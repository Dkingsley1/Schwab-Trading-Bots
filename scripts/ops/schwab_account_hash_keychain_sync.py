#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import getpass
import hmac
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.accountability import safe_write_json_atomic
from scripts.brokers.schwab.common import build_schwab_trader, fetch_account_rows

DEFAULT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "schwab_account_hash_keychain_sync_latest.json"
)
KeychainWriter = Callable[[str, str, str], tuple[bool, str]]
KeychainReader = Callable[[str, str], str]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _account_tail(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-4:] if len(digits) >= 4 else ""


def _keychain_account() -> str:
    return os.getenv("SCHWAB_KEYCHAIN_ACCOUNT", "").strip() or getpass.getuser()


def _service_name(env_name: str) -> str:
    override = os.getenv(f"{env_name}_KEYCHAIN_SERVICE", "").strip()
    return override or f"schwab_trading_bot/{env_name}"


def _security_bin() -> Path:
    return Path("/usr/bin/security")


def _write_keychain_secret(service: str, account: str, value: str) -> tuple[bool, str]:
    if sys.platform != "darwin" or not _security_bin().exists():
        return False, "macos_keychain_unavailable"
    proc = subprocess.run(
        [
            str(_security_bin()),
            "add-generic-password",
            "-a",
            account,
            "-s",
            service,
            "-w",
            value,
            "-U",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        (True, "stored")
        if proc.returncode == 0
        else (False, f"security_exit_{proc.returncode}")
    )


def _read_keychain_secret(service: str, account: str) -> str:
    if sys.platform != "darwin" or not _security_bin().exists():
        return ""
    proc = subprocess.run(
        [
            str(_security_bin()),
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
    )
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _build_sync_plan(
    *,
    account_rows: list[Mapping[str, Any]],
    aliases: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    alias_rows = (
        aliases.get("schwab_accounts")
        if isinstance(aliases.get("schwab_accounts"), Mapping)
        else {}
    )
    slots = registry.get("account_slots")
    slots = slots if isinstance(slots, list) else []
    slot_by_policy = {
        str(row.get("account_policy_key") or "").strip(): dict(row)
        for row in slots
        if isinstance(row, Mapping) and str(row.get("account_policy_key") or "").strip()
    }

    bindings: list[dict[str, Any]] = []
    blockers: list[str] = []
    seen_tails: set[str] = set()
    for raw in account_rows:
        tail = _account_tail(raw.get("account_number"))
        account_reference = str(raw.get("account_hash") or "").strip()
        if not tail or not account_reference:
            blockers.append("broker_account_row_missing_tail_or_hash")
            continue
        if tail in seen_tails:
            blockers.append(f"duplicate_broker_account_tail:{tail}")
            continue
        seen_tails.add(tail)
        alias = alias_rows.get(f"tail:{tail}")
        alias = dict(alias) if isinstance(alias, Mapping) else {}
        policy_key = str(alias.get("account_policy_key") or "").strip()
        slot = slot_by_policy.get(policy_key, {})
        if not alias or not bool(alias.get("operator_verified", False)):
            blockers.append(f"operator_verified_account_alias_missing:{tail}")
            continue
        if not slot:
            blockers.append(f"account_policy_registry_slot_missing:{policy_key}")
            continue
        env_names = [
            str(name or "").strip()
            for name in slot.get("env_names", [])
            if str(name or "").strip().endswith("_HASH")
            and str(name or "").strip() != "SCHWAB_ACCOUNT_HASH"
        ]
        if not env_names:
            blockers.append(f"account_hash_runtime_binding_missing:{policy_key}")
            continue
        bindings.append(
            {
                "account_number_tail": tail,
                "account_policy_key": policy_key,
                "canary_candidate": bool(slot.get("canary_candidate", False)),
                "env_names": list(dict.fromkeys(env_names)),
                "account_reference": account_reference,
            }
        )

    candidate_bindings = [row for row in bindings if row["canary_candidate"]]
    if len(candidate_bindings) != 1:
        blockers.append(
            f"designated_canary_account_binding_count_invalid:{len(candidate_bindings)}"
        )
    if not account_rows:
        blockers.append("no_connected_schwab_accounts_discovered")
    return {
        "ready": not blockers,
        "bindings": bindings,
        "blockers": list(dict.fromkeys(blockers)),
        "connected_account_count": len(account_rows),
        "mapped_account_count": len(bindings),
        "candidate_binding_count": len(candidate_bindings),
    }


def _apply_sync_plan(
    plan: Mapping[str, Any],
    *,
    account: str,
    write_keychain: KeychainWriter = _write_keychain_secret,
    read_keychain: KeychainReader = _read_keychain_secret,
) -> dict[str, Any]:
    if not bool(plan.get("ready", False)):
        return {
            "ok": False,
            "write_results": [],
            "stored_binding_count": 0,
        }
    write_results: list[dict[str, Any]] = []
    for binding in plan.get("bindings", []):
        if not isinstance(binding, Mapping):
            continue
        account_reference = str(binding.get("account_reference") or "").strip()
        for env_name in binding.get("env_names", []):
            variable = str(env_name or "").strip()
            service = _service_name(variable)
            stored, message = write_keychain(service, account, account_reference)
            verified = bool(
                stored
                and hmac.compare_digest(
                    str(read_keychain(service, account) or "").strip(),
                    account_reference,
                )
            )
            write_results.append(
                {
                    "account_number_tail": str(
                        binding.get("account_number_tail") or ""
                    ),
                    "account_policy_key": str(binding.get("account_policy_key") or ""),
                    "canary_candidate": bool(binding.get("canary_candidate", False)),
                    "runtime_variable": variable,
                    "keychain_service": service,
                    "stored": bool(stored),
                    "verified": verified,
                    "message": str(message or ""),
                    "raw_account_hash_emitted": False,
                }
            )
    return {
        "ok": bool(plan.get("ready", False))
        and bool(write_results)
        and all(row["verified"] for row in write_results),
        "write_results": write_results,
        "stored_binding_count": sum(bool(row["verified"]) for row in write_results),
    }


def sync_account_hashes(
    project_root: Path,
    *,
    quiet_auth: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    output = out_path or DEFAULT_OUT_PATH
    old_execution = os.environ.get("ALLOW_ORDER_EXECUTION")
    old_market_only = os.environ.get("MARKET_DATA_ONLY")
    os.environ["ALLOW_ORDER_EXECUTION"] = "0"
    os.environ["MARKET_DATA_ONLY"] = "1"
    try:
        trader = build_schwab_trader(
            project_root,
            mode="shadow",
            missing_credentials_message=(
                "Schwab credentials are required for account hash Keychain sync"
            ),
        )
        if quiet_auth:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):
                    trader.authenticate()
        else:
            trader.authenticate()
        account_rows = fetch_account_rows(trader.client)
        plan = _build_sync_plan(
            account_rows=account_rows,
            aliases=_load_json(project_root / "config" / "account_aliases.json"),
            registry=_load_json(
                project_root / "config" / "account_policy_registry.json"
            ),
        )
        applied = _apply_sync_plan(plan, account=_keychain_account())
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": bool(applied.get("ok", False)),
            "connected_account_count": plan["connected_account_count"],
            "mapped_account_count": plan["mapped_account_count"],
            "candidate_binding_count": plan["candidate_binding_count"],
            "stored_binding_count": applied["stored_binding_count"],
            "blockers": plan["blockers"],
            "bindings": applied["write_results"],
            "raw_account_hashes_persisted_to_files": False,
            "live_execution_authority": False,
            "policy": (
                "only operator-verified account aliases may bind broker account hashes "
                "to owner Keychain runtime variables"
            ),
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "blockers": [f"account_hash_keychain_sync_failed:{type(exc).__name__}"],
            "raw_account_hashes_persisted_to_files": False,
            "live_execution_authority": False,
        }
    finally:
        if old_execution is None:
            os.environ.pop("ALLOW_ORDER_EXECUTION", None)
        else:
            os.environ["ALLOW_ORDER_EXECUTION"] = old_execution
        if old_market_only is None:
            os.environ.pop("MARKET_DATA_ONLY", None)
        else:
            os.environ["MARKET_DATA_ONLY"] = old_market_only

    safe_write_json_atomic(
        str(output),
        payload,
        project_root=str(project_root),
        source="schwab_account_hash_keychain_sync",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover operator-verified Schwab account hashes and store them in "
            "the macOS Keychain without granting live execution."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--show-auth", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else None
    payload = sync_account_hashes(
        root,
        quiet_auth=not bool(args.show_auth),
        out_path=out_path,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_account_hash_keychain_sync "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"mapped={payload.get('mapped_account_count', 0)} "
            f"stored={payload.get('stored_binding_count', 0)} "
            f"blockers={','.join(payload.get('blockers', [])) or 'none'}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
