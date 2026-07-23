# Production Flow Guardrails

This contract keeps the soak and future production-grade paper/live-canary flows from depending on branch mutations, stale latest artifacts, or ad hoc operator state.

## Enforced Contracts

1. Runtime commands write to `governance/`, `runtime/`, `exports/`, `logs/`, or temp roots. Canonical source updates require explicit operator intent.
2. `master_bot_registry.json` is protected during normal runtime. `scripts/run_master_bot.py` writes `governance/health/master_bot_registry_candidate_latest.json` by default and only updates source with `--allow-source-registry-write`.
3. Paper-ramp roster promotion is candidate-only by default. `scripts/ops/paper_400_ramp_control.py --apply --promote-roster` writes `governance/health/paper_400_ramp_registry_candidate_latest.json` unless `--allow-source-registry-write` or `PAPER_400_RAMP_ALLOW_SOURCE_REGISTRY_WRITE=1` is explicitly set.
4. Runtime-throttle registry stabilization is candidate-only by default. `scripts/ops/runtime_throttle_control.py --apply` writes `governance/health/runtime_throttle_registry_candidate_latest.json` unless `--allow-source-registry-write` or `RUNTIME_THROTTLE_ALLOW_SOURCE_REGISTRY_WRITE=1` is explicitly set.
5. Generated showcase docs are GitHub Actions artifacts. The refresh workflow no longer pushes generated commits back into the same branch.
6. Stale `latest` ticker artifacts cannot silently shrink the Schwab symbol universe. Static fallback and sentinel symbols are regression checked.
7. Deployment profile boundaries are explicit in `config/deployment_profiles.json`: `local_mac_soak`, `ci`, `paper_prod`, and `live_canary`.
8. Self-healing is observe-first, dry-run-first, rate-limited, audited, rollback-aware, and forbidden from editing canonical source files.
9. Credential entry is not an interactive shell flow. Token lease monitoring and broker auth incident artifacts are required.
10. Promotion gates depend on versioned snapshots with freshness contracts. Unknown, missing, or stale gate state blocks promotion.
11. CI runs a production smoke pass plus a protected-source mutation guard.
12. Infrastructure bots must keep live-canary money blocked until `live_canary_readiness_contract_latest.json` proves no raw D-grade posture, no paper-trading dropouts, no auth/token surprises, no runtime source mutation, clean CI, clean storage pressure, and fresh promotion/paper gates for the sustained window.
13. `production_quality_control_latest.json` turns live-canary blockers into deterministic, safe, ordered repair lanes. It has no live-execution authority and delegates execution only through the infrabot governor exact allowlist.
14. `production_quality_slo_guard_latest.json` keeps state across checks so repeated production-quality lane failures become warnings or breaches instead of isolated snapshots. Breached lanes require bounded escalation, not unlimited repair loops.
15. `production_hardening_watch_latest.json` is the scheduled control loop for unattended soak hardening. It consumes published live-canary, production-quality, SLO, and governor contracts without refreshing source-shaped registry inputs; safe repair execution is opt-in and remains gated by the governor exact allowlist.

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

Track recurring production-quality lane degradation:

```bash
./scripts/ops/opsctl.sh production-quality-slo --apply --refresh-quality --json
```

Run the continuous hardening watch once:

```bash
./scripts/ops/opsctl.sh production-hardening-watch --apply --json
```

Publish a candidate-only paper-ramp roster promotion:

```bash
./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json
```

Publish candidate-only runtime-throttle registry stabilization:

```bash
./scripts/ops/opsctl.sh runtime-throttle --apply --json
```

Install only the production hardening launchd watch:

```bash
./scripts/install_production_hardening_watch_launchd.sh
```

Check that protected source files were not changed by runtime or CI steps:

```bash
python scripts/ops/source_mutation_guard.py --check-clean --json
```

Perform an intentional source registry update:

```bash
python scripts/run_master_bot.py --allow-source-registry-write
# or, for an intentional paper-ramp roster source update:
PAPER_400_RAMP_ALLOW_SOURCE_REGISTRY_WRITE=1 ./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json
# or, for an intentional runtime-throttle source update:
RUNTIME_THROTTLE_ALLOW_SOURCE_REGISTRY_WRITE=1 ./scripts/ops/opsctl.sh runtime-throttle --apply --json
```

Without that flag, master bot refreshes are candidate-only and leave the tracked registry untouched.
