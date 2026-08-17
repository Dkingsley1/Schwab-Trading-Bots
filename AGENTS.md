# Codex Project Guardrails

These instructions are the project-level guardrails for Codex work in this repository. Read them before making code, documentation, Git, or GitHub changes.

## Source Of Truth

- Start every non-trivial change by checking `docs/architecture/SOURCE_OF_TRUTH.md`.
- Treat generated artifacts as evidence, not ownership. Change the owning source first, then regenerate artifacts.
- Keep `README.md`, `COMMANDS.md`, and architecture docs aligned with the implemented system.
- Run `./scripts/ops/opsctl.sh codex-project-guard --json` before committing or publishing Codex-authored changes.

## Scope Discipline

- Let the newest user request set the scope.
- Keep unrelated dirty work untouched, even when it appears in `git status`.
- Stage explicit paths only. Do not use `git add -A` for mixed worktrees.
- Do not publish unrelated files just because they are already modified.
- When a requested topic is declared separate, keep it out of the current README, PR, and commit.

## Current Separate Domains

- Logic, audio, 96 kHz, sample-rate, and standalone app runtime work is separate from the Schwab trading-system README and source-of-truth work unless the user explicitly asks to join them.

## Apple Silicon Runtime Policy

- Treat the trading system as performance-core-primary on Apple Silicon. Core market loops, single-writer drains, sleeve launchers, and time-sensitive collectors should use the detected performance-core budget as the primary runway.
- On the current M1 Max class host, the intended primary budget is 8 performance cores. Efficiency cores are low-priority spillover/support space for telemetry, reports, maintenance, and other non-critical work.
- Do not pretend macOS provides portable hard CPU pinning for these Python and shell loops. Express this policy through `scripts/ops/apple_silicon_profile.py`, runtime env overrides, worker caps, nice/QoS behavior, and foreground-app governors.
- When Logic Pro, Final Cut Pro, Music/iTunes, or other foreground work is active, protect normal Mac use first and allow the computer-task, memory-efficiency, and operator-mode overrides to shrink worker counts even when performance cores are available.
- Keep `/Volumes/VIDEO` off-limits while applying storage, cleanup, runtime, or CPU allocation policy.

## Forbidden Volumes

- `/Volumes/VIDEO` is user-owned exFAT media storage and is off-limits to Codex and infrabot cleanup. Do not inspect, index, mount, unmount, repair, prune, move, delete, chmod, chown, quarantine, or run maintenance commands against it. If a request might involve `/Volumes/VIDEO`, stop and ask the user first.
- The assigned reminder bot is `video_volume_boundary_guard`, surfaced through `codex-project-guard`.

## Safety Rules

- Do not run destructive Git commands unless the user explicitly asks.
- Do not change credentials, tokens, or secret-bearing files except through the documented auth flows.
- Prefer repo-local commands through `scripts/ops/opsctl.sh` when a command exists.
- Before a GitHub update, verify the README and architecture docs do not mention an intentionally separate domain.

## Regression Guardrails

- Treat regression guards as first-class safety surfaces, not report-only scripts.
- Prefer bounded repair loops with per-surface retry budgets over shared blanket retries.
- Route heavy storage, retrain, and PDF/report refresh work through quiet-hours or cold-lane windows when live lanes are protected.
- Keep tenant-facing licensing/API health notifications deduplicated by guard surface and severity.
- Run `./scripts/ops/opsctl.sh grade-regression-guard --json` and `./scripts/ops/opsctl.sh grade-regression-autopilot --json` after changing guardrail contracts.
