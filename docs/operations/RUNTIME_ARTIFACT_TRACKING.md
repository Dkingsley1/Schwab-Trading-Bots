# Runtime Artifact Tracking

Generated operator reports and registry backups are local evidence, not release
source. The public repository tracks reviewed, payload-free inventory snapshots
under `docs/releases/`; it does not publish account/position report payloads or
machine-specific archive destinations.

## September 16, 2026 Review

`docs/releases/2026-09-16-runtime-artifact-inventory.json` accounts for all 361
previously untracked entries: 289 reports, 21 compressed registry backups, and
51 backup lookup links. It records relative paths, byte counts, SHA-256 hashes,
hash scope, and explicit dispositions. The reports had no byte-identical
duplicates and no references by basename in the tracked source at review time.
That reference check alone does not establish that history is disposable.

One empty, unopened report was removed. The other 288 reports retain unique
historical observations and remain local, including one non-JSON terminal
capture with a `.json` suffix. All 72 backup entries remain at their original
paths. No backup payload or archive destination was deleted or traversed.

For ordinary reports, a digest covers file bytes. For compressed backups, it
covers compressed bytes only, not decompression or restore validity. For links,
it covers the link text only; neither target existence nor archive integrity is
asserted. A retained report may change after this point-in-time inventory.

## Boundaries

- `.gitignore` excludes root `work/*.json`, `work/*.log`, the named writer sample,
  and generated `backups/master_bot_registry_before_*.json.gz` entries. Source,
  tests, configuration, and the inventory itself remain tracked.
- Before removing any other artifact, verify its owner, retention requirements,
  active readers/writers, and either its emptiness or a durable exact duplicate.
  A stale label, absent code reference, or healthy newer report is not sufficient.
- Preserve backup lookup links. Existing deep-cold and retention owners govern
  archive lifecycle; this inventory grants no new cleanup or volume authority.
- Do not commit raw reports, credentials, account identifiers, local absolute
  paths, or link destinations to the public repository. A secret-pattern scan
  alone is not permission to publish financial or operational payloads.
- This inventory is not the immutable release manifest. The native
  `release-freeze --write-release-manifest` owner still requires a clean,
  upstream-synchronized source tree and an active freeze. Runtime inventory
  does not certify restore, health, profitability, or trading readiness.
