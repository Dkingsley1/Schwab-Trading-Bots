# Codex Project Guardrails

These instructions are the project-level guardrails for Codex work in this repository.

## Source Of Truth

- Start every non-trivial change by checking `docs/architecture/SOURCE_OF_TRUTH.md`.
- Treat generated artifacts as evidence, not ownership. Change the owning source first, then regenerate artifacts.
- Keep `README.md`, `COMMANDS.md`, and architecture docs aligned with the implemented system.

## Scope Discipline

- Let the newest user request set the scope.
- Keep unrelated dirty work untouched, even when it appears in `git status`.
- Stage explicit paths only. Do not use broad staging in mixed worktrees.
- Do not publish unrelated files just because they are already modified.
- When a requested topic is declared separate, keep it out of the current README, PR, and commit.

## Current Separate Domains

- Logic, audio, 96 kHz, sample-rate, and standalone app runtime work is separate from the Schwab trading-system README and source-of-truth work unless explicitly joined.

## Safety Rules

- Do not run destructive Git commands unless explicitly requested.
- Do not change credentials, tokens, or secret-bearing files except through documented auth flows.
- Prefer repo-local commands through `scripts/ops/opsctl.sh` when a command exists.
- Before a GitHub update, verify the README and architecture docs do not mention an intentionally separate domain.
