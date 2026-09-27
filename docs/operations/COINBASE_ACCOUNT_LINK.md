# Coinbase Read-Only Account Connection

This is the documented auth flow for linking the owner's Coinbase / Coinbase
Advanced account. It is not Coinbase Wallet, Coinbase Exchange, or an OAuth
integration for other people's accounts. Public crypto collection and training
do not require this connection.

## Create And Link

1. Sign in yourself at [Coinbase Secret API Keys](https://portal.cdp.coinbase.com/api-keys/secret).
2. Use an Ed25519 or ECDSA (P-256 / ES256) key for the intended portfolio.
   View-only is preferred. An existing View + Trade key can be used for this
   GET-only connection; Trade is reported as a key capability, not platform
   execution authority. Disable Transfer and Receive if offered. Set an IP allowlist when
   the host has a stable known egress IP; do not guess an address.
3. Download the API key JSON locally. Do not paste the secret, password, JWT,
   recovery phrase, or downloaded file contents into chat, shell arguments, or logs.
4. Run the import with only the path:

```bash
./scripts/ops/opsctl.sh coinbase-account --link-key-file "$HOME/Downloads/cdp_api_key.json" --json
```

The importer accepts a bare UUID Key ID or an organizations/.../apiKeys/... key
name, not the friendly display name. It supports base64 Ed25519 secrets (32-byte
seed or validated 64-byte seed/public pair) and Ed25519 or P-256 PEM private keys,
including escaped newlines. It restricts the downloaded file to mode 0600,
validates the private key, asks Coinbase for its actual permissions, and fetches
every account page within bounded budgets. Missing or nonboolean permissions,
disabled View, enabled Transfer/Receive, malformed responses, redirects, and
incomplete pagination fail closed. A Trade-enabled key retains that authority
outside this platform; this importer cannot revoke provider-side permissions.
An existing connection is never replaced unless `--replace` is explicitly added.

## Storage And Refresh

The importer stores one atomic credential-and-snapshot generation in
`~/Library/Application Support/SchwabTradingPlatform/coinbase-readonly/connection.json`.
The directory is owner-only (0700), files are owner-only (0600), and no credentials
enter repository config, shell environment, report artifacts, or command output.
This is local file storage, **not Keychain encryption**. The downloaded original
is not deleted; after a successful import, remove that duplicate yourself using
your normal secret-handling process. Do not include this directory in shared exports.

```bash
./scripts/ops/opsctl.sh coinbase-account --json
./scripts/ops/opsctl.sh coinbase-account --status --json
./scripts/ops/opsctl.sh coinbase-api-health --json
```

The first command rechecks permissions and refreshes holdings on demand. The
second reads only local last-check evidence; it never contacts Coinbase and goes
stale after five minutes. A failed refresh preserves the old private snapshot but
marks the connection blocked. The public API health report includes the separate
personal-account status; its own `ready` still describes public market data.

Snapshots contain available and held currency quantities, not USD valuations,
settled-cash attestations, order history, or derivatives positions. Coverage is
limited to the API key's permissioned portfolio, not necessarily the entire
Coinbase profile. No background refresh service is installed by this command.

There are only two allowed authenticated GET endpoints: key permissions and
accounts. This connection cannot place/cancel orders, transfer funds, switch the
Schwab account, grant portfolio allocation authority, qualify a model, or enable
live money. Existing public-data broker capabilities remain unchanged.

Revoke the key in Coinbase to remove its authority. For rotation, create a new
View-only Ed25519 or ECDSA key and import it with `--replace`; only a successful verification
can replace the old local connection. Failed verification has no retry or write
authority at Coinbase.

## Sources

- [Coinbase API key authentication](https://docs.cdp.coinbase.com/coinbase-app/authentication-authorization/api-key-authentication)
- [Key permissions](https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/data-api/get-api-key-permissions)
- [Account pagination and portfolio scope](https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/accounts/list-accounts)
