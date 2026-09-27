"""Read-only Coinbase Advanced account access, separate from public collectors."""

from __future__ import annotations

import base64
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

import requests

HOST = "api.coinbase.com"
PERMISSIONS_PATH = "/api/v3/brokerage/key_permissions"
ACCOUNTS_PATH = "/api/v3/brokerage/accounts"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class CoinbaseAccountError(Exception):
    """Only fixed, nonsecret reason codes may cross the operator boundary."""


@dataclass(frozen=True, repr=False)
class CoinbaseAccountCredentials:
    name: str = field(repr=False)
    private_key: str = field(repr=False)

    @classmethod
    def from_payload(cls, payload: Any) -> CoinbaseAccountCredentials:
        if not isinstance(payload, dict):
            raise CoinbaseAccountError("invalid_key_file")
        name, pem = payload.get("name"), payload.get("privateKey")
        if not isinstance(name, str) or len(name) > 512:
            raise CoinbaseAccountError("invalid_key_name")
        name = name.strip()
        try:
            is_uuid = str(UUID(name)) == name.lower()
        except ValueError:
            is_uuid = False
        if not is_uuid and not re.fullmatch(
            r"organizations/[A-Za-z0-9_-]+/apiKeys/[A-Za-z0-9_-]+", name
        ):
            raise CoinbaseAccountError("invalid_key_name")
        if not isinstance(pem, str) or len(pem) > 16384:
            raise CoinbaseAccountError("invalid_private_key")
        _signing_key(pem)
        return cls(name, pem)

    def payload(self) -> dict[str, str]:
        return {"name": self.name, "privateKey": self.private_key}


def _signing_key(secret: str):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519
    from joserfc.jwk import ECKey, OKPKey

    try:
        value = secret.strip().replace("\\n", "\n").replace("\r", "\n")
        if value.startswith("-----BEGIN"):
            for kind in ("EC PRIVATE KEY", "PRIVATE KEY"):
                begin, end = f"-----BEGIN {kind}-----", f"-----END {kind}-----"
                if value.startswith(begin) and value.endswith(end):
                    body = "".join(value[len(begin) : -len(end)].split())
                    value = f"{begin}\n{body}\n{end}\n"
                    break
            key = serialization.load_pem_private_key(value.encode(), password=None)
        else:
            raw = base64.b64decode("".join(value.split()), validate=True)
            if len(raw) not in (32, 64):
                raise ValueError()
            key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
            if len(raw) == 64 and key.public_key().public_bytes_raw() != raw[32:]:
                raise ValueError()
        if isinstance(key, ec.EllipticCurvePrivateKey) and isinstance(
            key.curve, ec.SECP256R1
        ):
            return ECKey.import_key(key), "ES256"
        if isinstance(key, ed25519.Ed25519PrivateKey):
            return OKPKey.import_key(key), "EdDSA"
    except Exception:
        raise CoinbaseAccountError("invalid_or_unsupported_private_key") from None
    raise CoinbaseAccountError("invalid_or_unsupported_private_key")


def read_only_permissions(payload: dict[str, Any]) -> dict[str, bool]:
    # Key capabilities are evidence, not authority for this GET-only client.
    required = ("can_view", "can_trade", "can_transfer")
    if not isinstance(payload, dict) or any(
        type(payload.get(name)) is not bool for name in required
    ):
        raise CoinbaseAccountError("read_access_without_transfer_required")
    if payload["can_view"] is not True or payload["can_transfer"] is not False:
        raise CoinbaseAccountError("read_access_without_transfer_required")
    if "can_receive" in payload and payload["can_receive"] is not False:
        raise CoinbaseAccountError("read_access_without_transfer_required")
    return {
        name: payload[name] for name in (*required, "can_receive") if name in payload
    }


def coinbase_jws_registry():
    from joserfc.jws import JWSRegistry
    from joserfc.registry import HeaderParameter

    return JWSRegistry(
        header_registry={
            "nonce": HeaderParameter("Coinbase request nonce", "str", required=True)
        },
        algorithms=["ES256", "EdDSA"],
    )


def _account(row: Any) -> dict[str, str]:
    if not isinstance(row, dict):
        raise CoinbaseAccountError("invalid_account_response")
    account_id, currency = row.get("uuid"), row.get("currency")
    if not isinstance(account_id, str) or not re.fullmatch(
        r"[A-Za-z0-9-]{1,128}", account_id
    ):
        raise CoinbaseAccountError("invalid_account_response")
    if not isinstance(currency, str) or not re.fullmatch(
        r"[A-Z0-9._-]{1,32}", currency
    ):
        raise CoinbaseAccountError("invalid_account_response")
    result = {"account_id": account_id, "currency": currency}
    for source, target in (("available_balance", "available"), ("hold", "hold")):
        balance = row.get(source)
        if not isinstance(balance, dict) or balance.get("currency") != currency:
            raise CoinbaseAccountError("invalid_account_balance")
        value = balance.get("value")
        try:
            if not isinstance(value, str) or len(value) > 100:
                raise ValueError()
            amount = Decimal(value)
            if not amount.is_finite() or amount < 0:
                raise ValueError()
        except (InvalidOperation, ValueError):
            raise CoinbaseAccountError("invalid_account_balance") from None
        result[target] = value
    return result


class CoinbaseReadOnlyClient:
    def __init__(self, credentials: CoinbaseAccountCredentials, *, session=None):
        self._credentials = credentials
        self._key, self._algorithm = _signing_key(credentials.private_key)
        self._session = session if session is not None else requests.Session()
        # Ignore ambient proxies/netrc and never follow redirects with account auth.
        self._session.trust_env = False

    def close(self) -> None:
        self._session.close()

    def _get(self, path: str, *, deadline: float, params=None) -> dict[str, Any]:
        from joserfc import jwt

        if path not in {PERMISSIONS_PATH, ACCOUNTS_PATH}:
            raise CoinbaseAccountError("endpoint_not_allowed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CoinbaseAccountError("request_budget_exhausted")
        now = int(time.time())
        token = jwt.encode(
            {
                "alg": self._algorithm,
                "kid": self._credentials.name,
                "nonce": secrets.token_hex(),
            },
            {
                "sub": self._credentials.name,
                "iss": "cdp",
                "nbf": now,
                "exp": now + 120,
                "uri": f"GET {HOST}{path}",
            },
            self._key,
            registry=coinbase_jws_registry(),
        )
        try:
            with self._session.get(
                f"https://{HOST}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=(min(5.0, remaining), min(8.0, remaining)),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    # Provider bodies and exception strings may contain identifiers.
                    raise CoinbaseAccountError(
                        f"coinbase_http_{int(response.status_code)}"
                    )
                content = bytearray()
                for chunk in response.iter_content(chunk_size=16384):
                    if time.monotonic() > deadline:
                        raise CoinbaseAccountError("request_budget_exhausted")
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise CoinbaseAccountError("response_too_large")
                payload = json.loads(content)
        except requests.RequestException:
            raise CoinbaseAccountError("coinbase_network_error") from None
        except (ValueError, UnicodeError):
            raise CoinbaseAccountError("invalid_coinbase_json") from None
        if not isinstance(payload, dict):
            raise CoinbaseAccountError("invalid_coinbase_response")
        return payload

    def snapshot(self) -> dict[str, Any]:
        deadline = time.monotonic() + 45.0
        permissions = read_only_permissions(
            self._get(PERMISSIONS_PATH, deadline=deadline)
        )
        rows, seen_accounts, seen_cursors = [], set(), set()
        params: dict[str, Any] = {"limit": 250}
        for _ in range(10):
            page = self._get(ACCOUNTS_PATH, deadline=deadline, params=params)
            accounts = page.get("accounts")
            if (
                not isinstance(accounts, list)
                or len(accounts) > 250
                or type(page.get("has_next")) is not bool
            ):
                raise CoinbaseAccountError("invalid_accounts_page")
            for row in accounts:
                normalized = _account(row)
                if normalized["account_id"] in seen_accounts:
                    raise CoinbaseAccountError("duplicate_account_in_pages")
                seen_accounts.add(normalized["account_id"])
                rows.append(normalized)
            if not page["has_next"]:
                return {
                    "permissions": permissions,
                    "accounts": rows,
                    "complete": True,
                    "scope": "key_permissioned_portfolio",
                    "valuation_included": False,
                }
            cursor = page.get("cursor")
            if (
                not isinstance(cursor, str)
                or not cursor
                or len(cursor) > 2048
                or cursor in seen_cursors
            ):
                raise CoinbaseAccountError("invalid_pagination_cursor")
            seen_cursors.add(cursor)
            params = {"limit": 250, "cursor": cursor}
        raise CoinbaseAccountError("account_page_budget_exhausted")
