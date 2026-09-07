# Schwab Account Capability Truth

The account truth layer keeps three independent axes for every Schwab account:

1. Registration and tax wrapper, such as taxable brokerage or Roth IRA.
2. Trading access, such as cash, limited margin, full margin, or portfolio margin.
3. Current broker state, including balances, calls, restrictions, positions, and interest.

Schwab's provider `type=MARGIN` value is preserved exactly, but it is not treated as
proof that borrowing, short stock, interest-bearing debit, or live execution is allowed.
Limited margin is modeled as a trading feature with `borrowing_allowed=false` and
`margin_interest_possible=false`.

## Sources Of Truth

- Provider facts come from the fresh Schwab account snapshot. Safe primitive account,
  balance, position, and instrument fields are retained with a field inventory.
- Operator classifications come from the local ignored files
  `config/account_aliases.json` and `config/account_policy_registry.json`.
- Tracked example files document the schema without exposing account identifiers.
- Raw account numbers, hashes, tokens, secrets, and CUSIPs are not emitted by the
  normalized account-study artifact.

## Debit And Option Semantics

- `marginBalance` remains a provider field. A negative value is not automatically
  labeled as debt, borrowing, or an option close cost.
- `accruedInterest`, account calls, and operator-confirmed borrowing access are reported
  separately. Ambiguous negative balances require broker UI confirmation.
- Short-option close marks are estimated separately from current short-option market
  value and are paired with covered-versus-uncovered collateral checks.

## Canary Contract

Account truth can block but cannot grant live execution. A canary candidate must have a
complete operator classification, a policy binding, borrowing disabled, a cash-only
budget, an allowed route, a positive cap, sufficient cash proxy, no broker call, no
closing-only restriction, and no uncovered short option. Passing this preflight still
leaves `live_execution_authority=false`; independent live-readiness and operator gates
remain mandatory.

The primary artifacts are produced by:

```bash
./scripts/ops/opsctl.sh account-position-study --json
./scripts/ops/opsctl.sh account-policy-context --json
```
