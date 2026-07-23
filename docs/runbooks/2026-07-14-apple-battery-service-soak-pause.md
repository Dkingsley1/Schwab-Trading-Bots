# 2026-07-14 Apple Battery Service Soak Pause

## Summary

The Mac is scheduled for Apple battery service on July 14, 2026 at approximately 2:40 PM America/New_York. Apple advised that data loss is possible as a standard hardware-service risk.

This is a planned external hardware maintenance event. It should not be classified as a trading-system degradation, unexplained outage, watchdog failure, Schwab auth failure, paper-trading failure, or data-collection failure.

## Documented Offline Window

- Offline/service pause start: `2026-07-14T14:40:00-04:00` / `2026-07-14T18:40:00Z`.
- Segment 2 acceptance/system ready again: `2026-07-22T20:53:55-04:00` / `2026-07-23T00:53:55Z`.
- Recorded downtime for soak accounting: `8 days, 6 hours, 13 minutes, 55 seconds`.
- Treatment: planned external hardware maintenance between soak segments, not a system failure.

## Soak Accounting

- Segment 1: pre-service soak evidence remains valid until operator shutdown for Apple battery service.
- Maintenance pause: expected interruption caused by physical Mac service.
- Segment 2: post-service soak begins only after restore-readiness and runtime health checks pass.

The 30-day soak should be reviewed as segmented evidence rather than erased. Continuous unattended runtime is expected to break during service, but the reason is known, planned, and external to the trading system.

## Pre-Service Backup Evidence

- Primary backup: `/Volumes/VIDEO/SchwabTradingBot_PreService_20260714T141912Z`
- Primary size: `218G`
- Primary manifest count: `57,344` files
- Secondary essentials mirror: `/Volumes/LaCie/SchwabTradingBot_PreService_Essentials_20260714T141912Z`
- Secondary size: `9.7G`
- Secondary manifest count: `17,192` files
- Full `local_fallback_storage` included in primary backup: source `164G`, backup `165G`
- Primary critical SHA256 checks: OK
- Secondary critical SHA256 checks: OK
- Git bundle verification: OK, complete history
- Daily state snapshot: OK, 5 files checked, 0 missing

## Pre-Service Runtime Evidence

- Unattended soak readiness: `ready`, `A+`, target `30` days
- Live readiness smoke: `ready`, score `100.0`
- Live submit enabled: `0`
- Expected mode: paper trading and collection only, no live orders

## Post-Service Acceptance Checklist

1. Confirm `/Users/dankingsley/PycharmProjects/schwab_trading_bot` exists on internal storage.
2. Mount `/Volumes/BOT_LOGS` and verify external storage routing.
3. Run `./scripts/ops/opsctl.sh soak-readiness --target-days 30`.
4. Run `python scripts/live_readiness_smoke.py` using the project Python runtime.
5. Verify Schwab auth lease and token readiness.
6. Verify paper trading lanes and data collection are running.
7. Keep live order submission disabled unless a separate supervised live approval is explicitly granted.

## Segment 2 Acceptance

- Accepted at `2026-07-22T20:53:55-04:00` / `2026-07-23T00:53:55Z`.
- Unattended soak readiness: `ready`, `A+`, score `100.0`, safe unattended `true`.
- Storage, runtime loops, and alerting sections: `A+`.
- External storage route: certified `external`, `3/3` routes verified, `0` split-brain conflicts.
- Paper execution truth: `ready`, `A+`, score `97.0`.
- Runtime paper regression guard: `ready`, paper ramp `armed`, failed guards `0`.
- Process watchdog: `ready`, all-sleeves fanout live with `22/22` jobs running.
- Segment 2 source event: `governance/maintenance_events/20260722_apple_battery_service_segment_2_acceptance.json`.

## Governance Artifacts

- Event JSON: `governance/maintenance_events/20260714_apple_battery_service_soak_pause.json`
- Latest planned hardware maintenance pointer: `governance/maintenance_events/latest_planned_hardware_maintenance.json`
- Latest segment 2 acceptance pointer: `governance/maintenance_events/latest_segment_2_acceptance.json`
- Event log: `governance/maintenance_events/events.jsonl`
