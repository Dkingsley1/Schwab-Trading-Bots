# 2026-08-20 macOS Software Update Restart

## Classification

This was planned host software maintenance initiated by the operator. It is not an unexplained outage, a trading-system degradation, or a trading-system failure.

## Window

- Host shutdown: `2026-08-20T07:50:00-04:00` (`2026-08-20T11:50:00Z`)
- Host boot: `2026-08-20T07:53:00-04:00` (`2026-08-20T11:53:00Z`)
- Post-restart acceptance: `2026-08-20T12:19:40-04:00` (`2026-08-20T16:19:40Z`)
- Excluded active-runtime interval: `4 hours, 29 minutes, 40 seconds`

The acceptance endpoint is intentionally later than boot. Active-runtime evidence resumes only after the compact settlement pass verifies halt state, Schwab auth, runtime throttle, process health, and paper-fleet readiness.

## Soak Treatment

- Preserve all valid pre-update segmented soak history.
- Do not reset the candidate clock solely because the host underwent this planned restart.
- Do not count the maintenance interval as degradation or a system failure.
- Do not credit the offline interval as active runtime.
- Resume active-runtime credit after the acceptance checks pass.
- Account for any separately reviewed source change through the production-candidate generation process, not through this maintenance event.

## Acceptance Evidence

`governance/health/post_restart_settlement_latest.json` reported `ready` at `2026-08-20T16:19:40.180246+00:00`: global halt clear, Schwab auth lease healthy, runtime throttle ready, zero restart storms, zero process alerts, and the paper fleet ready. Live order submission remained disabled.
