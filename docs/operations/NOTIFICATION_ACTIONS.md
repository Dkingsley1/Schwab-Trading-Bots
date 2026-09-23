# Notification Click Actions

Owner: `scripts/ops/mac_notification_watch.py`. These actions apply to newly
delivered Mac desktop alerts. Existing delivered alerts and forwarded iMessages
cannot acquire a new local action. Nothing automatically logs in or mutates the
trading platform merely because a notification is delivered.

| Alert | Click Destination |
| --- | --- |
| Schwab auth expired, blocked, or nearing expiry | Terminal-owned supervised Schwab sign-in, opening Chrome with a live OAuth callback listener |
| Restart storms, missing sleeves, lid/sleep events | Current process watchdog report |
| Storage disconnect | Storage mount guard report |
| Swap/memory pressure | Swap pressure governor report |
| Creative-app contention | Creative cotenant guard report |
| Tripwire | Incident timeline |
| Global halt | Kill-switch report, or the halt marker if the report is absent |
| Global halt cleared | Halt recovery report |
| Incident auto-halt | Incident auto-halt report |
| Critical preflight | Preflight report |
| Covered calls and other named critical alerts | Exact source alert when present; otherwise incident review/timeline |
| System intelligence | Current handoff report |

All non-auth actions are file views. No click places/cancels orders, clears
halts, deletes data, changes risk limits or restarts the platform. Report text
never becomes an executable command. Auth uses a fixed local launcher with a
kernel lock, rejects an occupied callback port, and preserves live-money locks.
The localhost callback belongs to this Mac, not the user's phone.

## Native Transport

The watcher checks the user-local `~/Applications/terminal-notifier.app` and
the two existing Homebrew executable locations. September 23 installation used
the official terminal-notifier 3.0.0 source archive, verified against Homebrew's
SHA-256 `10dea2da3a698e0a5119400c0500ff82ce14ddb140e7fea0ce68bf57719620ed`,
built for arm64 and verified with `codesign --verify --strict`. No startup
prompt, login settings or default notification recipients were changed.

macOS must allow notifications for terminal-notifier. The watcher uses its
non-waiting click mode; no worker waits for each notification. Delivery has a
10-second bound and falls back to the existing AppleScript notification on
native failure. Fallback receipts explicitly set `click_action_available=false`.
Installed transport is not proof of notification permission or a successful
human click. Source code and action mappings are tested separately from that
desktop verification. Event deduplication and repeat cooldowns are unchanged.

## Verification

Run from a normal Mac Terminal:

```bash
/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/opsctl.sh notify-test --disable-imessage
```

Allow notifications if prompted, then click the test alert. It should open
`process_watchdog_latest.json`, not start or stop anything. Inspect
`governance/health/mac_notification_watch_state.json` for transport installation
and the latest actual delivery result. This permission/click check must succeed
before declaring the desktop interaction fully verified.
