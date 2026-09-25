# Native Command Audit

The existing `command_validity` lifecycle job runs every 600 seconds when the
platform is on and the scheduler admits it. It invokes `command-validity
--safe-audit --timeout-sec 30 --summary-json`; no separate daemon or Codex automation is
required. Scheduled deferrals remain visible through lifecycle receipts.

Every entry in the generated COMMANDS.md gets a record in
`governance/health/command_surface_audit_latest.json`. Compact status lives in
`command_validity_latest.json`; detailed evidence is rewritten only when its
content changes, not every timer tick. The audit checks documented
dispatch routes, referenced implementation files, Python/shell syntax, source
hashes, inventory drift, declared purpose, and repeated snippets. Source checks
are cached only within a pass, so source changes are rechecked on the next pass.

`static_pass` is not functional certification. Exact argument validity, genuine
usefulness and real-world side effects require isolated tests or supervised
evidence. Missing evidence stays explicit. External tools and dynamic shell
routes are not silently declared working. Duplicate commands are review items,
not permission to delete an intentional alternative workflow.

The audit never runs the documented snippets, imports their Python modules,
clears halts, changes risk limits, deletes data, restarts services, or submits
orders. The only subprocesses in this mode are fixed shell syntax checks.
Changing COMMANDS.md cannot grant execution authority. Curated inventory fixes
belong in `commands_hygiene_bot.py`; the scheduled audit does not repin source
contracts or rewrite a frozen release candidate.

Adaptive controls may select already-approved runtime profiles. Infrabots can
refresh evidence and invoke approved repair owners. Changing contract meaning,
protected storage scope, trading authority, or source acceptance still requires
a reviewed change and regression tests. Automatic recurring operation does not
mean unlimited CPU, I/O, restarts, or authority.
