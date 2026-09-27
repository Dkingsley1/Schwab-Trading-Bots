import base64
import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone

import pytest
import requests
from joserfc import jwt
from joserfc.jwk import ECKey

from core import coinbase_account as account
from core.brokers.coinbase import CoinbaseBrokerAdapter
from scripts.ops import coinbase_account_link as link


@pytest.fixture
def credentials():
    key = ECKey.generate_key()
    return account.CoinbaseAccountCredentials.from_payload(
        {
            "name": "organizations/test-org/apiKeys/test-key",
            "privateKey": key.as_pem(private=True).decode(),
        }
    )


def row(account_id="account-one", currency="BTC"):
    return {
        "uuid": account_id,
        "currency": currency,
        "available_balance": {"value": "1.23000000", "currency": currency},
        "hold": {"value": "0.01", "currency": currency},
        "name": "private-wallet-name",
    }


def permissions():
    return {"can_view": True, "can_trade": False, "can_transfer": False}


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield json.dumps(self.payload).encode()


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def client_for(credentials, pages):
    session = Session([Response(permissions()), *[Response(page) for page in pages]])
    return account.CoinbaseReadOnlyClient(credentials, session=session), session


def test_signed_gets_and_complete_pagination(credentials):
    client, session = client_for(
        credentials,
        [
            {"accounts": [row()], "has_next": True, "cursor": "next"},
            {"accounts": [row("account-two", "USD")], "has_next": False},
        ],
    )
    snapshot = client.snapshot()
    assert snapshot["complete"] is True
    assert len(snapshot["accounts"]) == 2
    assert snapshot["accounts"][0]["available"] == "1.23000000"
    assert "private-wallet-name" not in json.dumps(snapshot)
    assert session.trust_env is False
    tokens = []
    for url, kwargs in session.calls:
        assert url.startswith("https://api.coinbase.com/api/v3/brokerage/")
        assert kwargs["allow_redirects"] is False
        token = kwargs["headers"]["Authorization"].split(" ")[1]
        decoded = jwt.decode(
            token,
            ECKey.import_key(credentials.private_key),
            registry=account.coinbase_jws_registry(),
        )
        assert decoded.claims["uri"] == f"GET {url.removeprefix('https://')}"
        assert decoded.claims["exp"] - decoded.claims["nbf"] == 120
        assert decoded.claims["iss"] == "cdp"
        assert decoded.header["kid"] == credentials.name
        tokens.append(token)
    assert len(set(tokens)) == 3
    assert session.calls[2][1]["params"]["cursor"] == "next"
    client.close()
    assert session.closed


@pytest.mark.parametrize(
    "patch",
    [
        {"can_view": False},
        {"can_view": 1},
        {"can_trade": 0},
        {"can_transfer": True},
        {"can_transfer": "false"},
        {"can_receive": True},
        {"can_trade": None},
    ],
)
def test_permissions_fail_before_holdings_fetch(credentials, patch):
    session = Session([Response({**permissions(), **patch})])
    client = account.CoinbaseReadOnlyClient(credentials, session=session)
    with pytest.raises(
        account.CoinbaseAccountError, match="read_access_without_transfer_required"
    ):
        client.snapshot()
    assert len(session.calls) == 1


def test_missing_permission_is_not_false():
    with pytest.raises(account.CoinbaseAccountError):
        account.read_only_permissions({"can_view": True, "can_trade": False})


@pytest.mark.parametrize(
    "page,reason",
    [
        ({"accounts": [], "has_next": True}, "invalid_pagination_cursor"),
        ({"accounts": [], "has_next": "false"}, "invalid_accounts_page"),
        ({"accounts": [row(), row()], "has_next": False}, "duplicate_account_in_pages"),
        ({"accounts": "bad", "has_next": False}, "invalid_accounts_page"),
    ],
)
def test_bad_pages_are_not_complete(credentials, page, reason):
    client, _ = client_for(credentials, [page])
    with pytest.raises(account.CoinbaseAccountError, match=reason):
        client.snapshot()


def test_repeated_cursor_rejected(credentials):
    client, _ = client_for(
        credentials,
        [
            {"accounts": [], "has_next": True, "cursor": "same"},
            {"accounts": [], "has_next": True, "cursor": "same"},
        ],
    )
    with pytest.raises(account.CoinbaseAccountError, match="invalid_pagination_cursor"):
        client.snapshot()


def test_page_budget_rejects_partial_snapshot(credentials):
    client, _ = client_for(
        credentials,
        [{"accounts": [], "has_next": True, "cursor": str(n)} for n in range(10)],
    )
    with pytest.raises(
        account.CoinbaseAccountError, match="account_page_budget_exhausted"
    ):
        client.snapshot()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-0.1", 1, None])
def test_invalid_balance_rejected(value):
    item = row()
    item["available_balance"]["value"] = value
    with pytest.raises(account.CoinbaseAccountError, match="invalid_account_balance"):
        account._account(item)


def test_currency_mismatch_rejected():
    item = row()
    item["hold"]["currency"] = "USD"
    with pytest.raises(account.CoinbaseAccountError):
        account._account(item)


@pytest.mark.parametrize("code", [301, 401, 403, 429, 500])
def test_http_failures_suppress_provider_text(credentials, code):
    session = Session([Response({"secret": credentials.private_key}, status_code=code)])
    client = account.CoinbaseReadOnlyClient(credentials, session=session)
    with pytest.raises(account.CoinbaseAccountError, match=f"^coinbase_http_{code}$"):
        client.snapshot()


def test_network_error_is_redacted(credentials):
    class BrokenSession(Session):
        def get(self, *args, **kwargs):
            raise requests.ConnectionError(credentials.private_key)

    client = account.CoinbaseReadOnlyClient(credentials, session=BrokenSession([]))
    with pytest.raises(account.CoinbaseAccountError, match="^coinbase_network_error$"):
        client.snapshot()


def test_endpoint_and_time_budgets(credentials):
    client, session = client_for(credentials, [])
    with pytest.raises(account.CoinbaseAccountError, match="endpoint_not_allowed"):
        client._get("/api/v3/brokerage/orders", deadline=time.monotonic() + 5)
    with pytest.raises(account.CoinbaseAccountError, match="request_budget_exhausted"):
        client._get(account.ACCOUNTS_PATH, deadline=0)
    assert not session.calls


def test_response_size_budget(credentials, monkeypatch):
    client, _ = client_for(credentials, [])
    monkeypatch.setattr(account, "MAX_RESPONSE_BYTES", 5)
    with pytest.raises(account.CoinbaseAccountError, match="response_too_large"):
        client.snapshot()


def test_bad_key_types_rejected(credentials):
    for payload in (
        [],
        {},
        {"name": credentials.name, "privateKey": "not-a-private-key"},
    ):
        with pytest.raises(account.CoinbaseAccountError):
            account.CoinbaseAccountCredentials.from_payload(payload)
    key = ECKey.generate_key("P-384")
    with pytest.raises(
        account.CoinbaseAccountError, match="invalid_or_unsupported_private_key"
    ):
        account.CoinbaseAccountCredentials.from_payload(
            {"name": credentials.name, "privateKey": key.as_pem(private=True).decode()}
        )
    assert credentials.private_key not in repr(credentials)


@pytest.mark.parametrize("encoding", ["seed", "expanded", "pem", "escaped_pem"])
def test_ed25519_uuid_signing_and_trade_capability_remains_read_only(encoding):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from joserfc.jwk import OKPKey

    key = ed25519.Ed25519PrivateKey.generate()
    if encoding in {"seed", "expanded"}:
        raw = key.private_bytes_raw()
        if encoding == "expanded":
            raw += key.public_key().public_bytes_raw()
        secret = base64.b64encode(raw).decode()
    else:
        secret = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        if encoding == "escaped_pem":
            secret = secret.replace("\n", "\\n")
    credentials = account.CoinbaseAccountCredentials.from_payload(
        {"name": "12345678-1234-1234-1234-123456789abc", "privateKey": secret}
    )
    session = Session(
        [
            Response({**permissions(), "can_trade": True}),
            Response({"accounts": [row()], "has_next": False}),
        ]
    )
    client = account.CoinbaseReadOnlyClient(credentials, session=session)
    snapshot = client.snapshot()
    assert snapshot["permissions"]["can_trade"] is True
    assert snapshot["complete"] is True
    for url, kwargs in session.calls:
        token = kwargs["headers"]["Authorization"].split(" ")[1]
        decoded = jwt.decode(
            token, OKPKey.import_key(key), registry=account.coinbase_jws_registry()
        )
        assert decoded.header["alg"] == "EdDSA"
        assert decoded.claims["sub"] == credentials.name
        assert decoded.claims["uri"] == f"GET {url.removeprefix('https://')}"
    with pytest.raises(account.CoinbaseAccountError, match="endpoint_not_allowed"):
        client._get("/api/v3/brokerage/orders", deadline=time.monotonic() + 5)
    assert len(session.calls) == 2
    assert CoinbaseBrokerAdapter.capabilities.supports_live_execution is False


def test_expanded_ed25519_public_half_must_match():
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.generate()
    secret = base64.b64encode(key.private_bytes_raw() + b"x" * 32).decode()
    with pytest.raises(
        account.CoinbaseAccountError, match="invalid_or_unsupported_private_key"
    ):
        account.CoinbaseAccountCredentials.from_payload(
            {"name": "12345678-1234-1234-1234-123456789abc", "privateKey": secret}
        )


@pytest.mark.parametrize(
    "name", ["coinbase_trading_bot", "{12345678-1234-1234-1234-123456789abc}", ""]
)
def test_friendly_names_are_not_key_ids(credentials, name):
    with pytest.raises(account.CoinbaseAccountError, match="invalid_key_name"):
        account.CoinbaseAccountCredentials.from_payload(
            {"name": name, "privateKey": credentials.private_key}
        )


class GoodClient:
    def __init__(self, credentials):
        pass

    def snapshot(self):
        return {
            "permissions": permissions(),
            "accounts": [account._account(row())],
            "complete": True,
            "scope": "key_permissioned_portfolio",
        }

    def close(self):
        pass


class BadClient(GoodClient):
    def snapshot(self):
        raise account.CoinbaseAccountError("view_only_key_required")


def download(tmp_path, credentials):
    path = tmp_path.resolve() / "download.json"
    path.write_text(json.dumps(credentials.payload()))
    path.chmod(0o644)
    return path


def test_import_is_private_and_status_is_separate(tmp_path, credentials):
    path = download(tmp_path, credentials)
    state = tmp_path.resolve() / "private"
    result = link.sync_account(
        state_dir=state, key_file=path, client_factory=GoodClient
    )
    assert result["connected"] is True
    assert result["account_count"] == 1
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    for file in [path, state / "connection.json", state / "status.json"]:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    for output in [
        result,
        link.status(state),
        json.loads((state / "status.json").read_text()),
    ]:
        assert credentials.private_key not in json.dumps(output)
        assert credentials.name not in json.dumps(output)
        assert "account-one" not in json.dumps(output)
        assert "1.23000000" not in json.dumps(output)
        assert output["live_execution_allowed"] is False
        assert output["transfers_allowed"] is False
    assert link.status(state)["connected"] is True
    assert (
        link.sync_account(state_dir=state, client_factory=GoodClient)["connected"]
        is True
    )


def test_failed_import_does_not_store_credentials(tmp_path, credentials):
    state = tmp_path.resolve() / "private"
    result = link.sync_account(
        state_dir=state,
        key_file=download(tmp_path, credentials),
        client_factory=BadClient,
    )
    assert result["connected"] is False
    assert not (state / "connection.json").exists()
    assert link.status(state)["overall_status"] == "blocked"


def test_failed_refresh_and_replace_preserve_previous_snapshot(tmp_path, credentials):
    state = tmp_path.resolve() / "private"
    path = download(tmp_path, credentials)
    link.sync_account(state_dir=state, key_file=path, client_factory=GoodClient)
    original = (state / "connection.json").read_bytes()
    result = link.sync_account(
        state_dir=state, key_file=path, client_factory=GoodClient
    )
    assert result["reason"] == "already_configured_use_replace"
    result = link.sync_account(state_dir=state, client_factory=BadClient)
    assert result["connected"] is False
    assert (state / "connection.json").read_bytes() == original
    assert link.status(state)["overall_status"] == "blocked"
    assert link.sync_account(
        state_dir=state, key_file=path, replace=True, client_factory=GoodClient
    )["connected"]


@pytest.mark.parametrize("offset", [-301, 30])
def test_status_stale_or_future_is_not_connected(tmp_path, credentials, offset):
    state = tmp_path.resolve() / "private"
    link.sync_account(
        state_dir=state,
        key_file=download(tmp_path, credentials),
        client_factory=GoodClient,
    )
    receipt = link._read_json(state / "status.json")
    receipt["verified_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=offset)
    ).isoformat()
    link._write_json(state / "status.json", receipt)
    result = link.status(state)
    assert result["overall_status"] == "stale"
    assert result["connected"] is False


def test_status_missing_connection_or_changed_generation_not_ready(
    tmp_path, credentials
):
    state = tmp_path.resolve() / "private"
    link.sync_account(
        state_dir=state,
        key_file=download(tmp_path, credentials),
        client_factory=GoodClient,
    )
    connection = link._read_json(state / "connection.json")
    connection["verified_at"] = "different"
    link._write_json(state / "connection.json", connection)
    assert not link.status(state)["connected"]
    (state / "connection.json").unlink()
    assert not link.status(state)["connected"]


def test_world_readable_saved_state_rejected(tmp_path, credentials):
    state = tmp_path.resolve() / "private"
    link.sync_account(
        state_dir=state,
        key_file=download(tmp_path, credentials),
        client_factory=GoodClient,
    )
    (state / "connection.json").chmod(0o644)
    assert not link.status(state)["connected"]
    result = link.sync_account(state_dir=state, client_factory=GoodClient)
    assert result["reason"] == "unsafe_file_permissions"


def test_protected_and_symlink_paths_rejected_before_read(tmp_path):
    with pytest.raises(account.CoinbaseAccountError, match="protected_path"):
        link._read_json(link.Path("/Volumes/VIDEO/key.json"))
    symlink = tmp_path.resolve() / "alias"
    symlink.symlink_to("/Volumes/VIDEO")
    with pytest.raises(account.CoinbaseAccountError, match="symlink_path_rejected"):
        link._read_json(symlink / "key.json")


def test_lock_prevents_interleaved_import(tmp_path, credentials):
    state = tmp_path.resolve() / "private"
    link._private_dir(state)
    lock_fd = os.open(state / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        link.fcntl.flock(lock_fd, link.fcntl.LOCK_EX | link.fcntl.LOCK_NB)
        result = link.sync_account(
            state_dir=state,
            key_file=download(tmp_path, credentials),
            client_factory=GoodClient,
        )
        assert result["reason"] == "account_sync_already_running"
        assert not (state / "connection.json").exists()
        assert not (state / "status.json").exists()
    finally:
        os.close(lock_fd)


def test_public_adapter_retains_no_account_or_order_authority():
    capabilities = CoinbaseBrokerAdapter.capabilities
    assert not capabilities.requires_auth
    assert not capabilities.supports_live_execution
    assert not capabilities.supports_order_place
    assert not capabilities.supports_account_snapshot
