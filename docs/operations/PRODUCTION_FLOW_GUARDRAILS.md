# Production Flow Guardrails

This contract keeps the soak and future production-grade paper/live-canary flows from depending on branch mutations, stale latest artifacts, or ad hoc operator state.

## Enforced Contracts

1. Runtime commands write to `governance/`, `runtime/`, `exports/`, `logs/`, or temp roots. Canonical source updates require explicit operator intent.
2. `master_bot_registry.json` is protected during normal runtime. `scripts/run_master_bot.py` writes `governance/health/master_bot_registry_candidate_latest.json` by default and only updates source with `--allow-source-registry-write`.
3. Generated showcase docs are GitHub Actions artifacts. The refresh workflow no longer pushes generated commits back into the same branch.
4. Stale `latest` ticker artifacts cannot silently shrink the Schwab symbol universe. Static fallback and sentinel symbols are regression checked.
5. Deployment profile boundaries are explicit in `config/deployment_profiles.json`: `local_mac_soak`, `ci`, `paper_prod`, and `live_canary`.
6. Self-healing is observe-first, dry-run-first, rate-limited, audited, rollback-aware, and forbidden from editing canonical source files.
7. Credential entry is not an interactive shell flow. Token lease monitoring and broker auth incident artifacts are required.
8. Promotion gates depend on versioned snapshots with freshness contracts. Unknown, missing, or stale gate state blocks promotion.
9. CI runs a production smoke pass plus a protected-source mutation guard.
10. Infrastructure bots must keep live-canary money blocked until `live_canary_readiness_contract_latest.json` proves no raw D-grade posture, no paper-trading dropouts, no auth/token surprises, no runtime source mutation, clean CI, clean storage pressure, and fresh promotion/paper gates for the sustained window.
11. `production_quality_control_latest.json` turns live-canary blockers into deterministic, safe, ordered repair lanes. It has no live-execution authority and delegates execution only through the infrabot governor exact allowlist.

## Commands

Run the production contract locally:

```bash
python scripts/ops/production_flow_smoke.py --json
```

Publish the live-canary hardening bar for infrastructure bots:

```bash
./scripts/ops/opsctl.sh live-canary-readiness --apply --json
```

Publish the production-quality repair contract:

```bash
./scripts/ops/opsctl.sh production-quality --apply --refresh-contract --json
```

Check that protected source files were not changed by runtime or CI steps:

```bash
python scripts/ops/source_mutation_guard.py --check-clean --json
```

Perform an intentional source registry update:

```bash
python scripts/run_master_bot.py --allow-source-registry-write
```

Without that flag, master bot refreshes are candidate-only and leave the tracked registry untouched.
