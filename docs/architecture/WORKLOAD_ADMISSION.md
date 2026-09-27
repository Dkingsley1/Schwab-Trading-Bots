# Workload-Specific Resource Admission

The native runtime-smooth owner publishes `workload_admission` in
`governance/health/runtime_throttle_control_latest.json`. Its existing
20-second schedule is unchanged. Source age, actual scheduler spacing, admission,
execution completion, backlog clearance and trading readiness remain separate.

## Admission Matrix

The canonical policy is `core/workload_admission.py`. Only hosts with at least
eight logical CPUs and matching current CPU observations receive extra headroom.
CPU percentages count one core as 100%, not the whole machine.

| Workload | Maximum Load Per Logical CPU | Minimum Free Memory | Local Disk Floor | Clear Observation Span | Execution Owner |
| --- | --- | --- | --- | --- | --- |
| Capacity probe | 0.85 | 35% | 16 GiB | 60 seconds | Classification only; cannot launch work |
| Observer | 1.00 | 25% | 16 GiB | Immediate | Fixed infrastructure assessment, 60-second supervisor |
| Storage recovery | 0.85 | 25% | 16 GiB | Immediate | Existing compression owner, one 25%-of-one-core worker, 240-second adaptive ceiling |
| Maintenance | 0.85 | 30% | 64 GiB | 60 seconds | Four allowlisted repair slots, 180-second supervisor |
| Training canary | 0.85 | 35% | 64 GiB | 60 seconds | Existing training owner and its deadlines; extra CPU admission capped at one target |
| Bulk | 0.62 | 40% | 125 GiB | 180 seconds | Diagnostic eligibility only; no new bulk launch path |

Both one- and five-minute load must fit. Each class also has a saturation
ceiling. Ready underlying resource, runtime and memory producers must be
timezone-aware, nonfuture and at most 90 seconds old. Thermal/power warnings,
protective holds, creative activity/cooldown, explicit support-pause advice,
memory pressure, invalid counters, swap above 8 GiB or throttled pages withhold
extra admission.

Aggregate CPU is capped at 65% of logical capacity. Foreground, system, support,
protected and unknown CPU categories have separate capacity-scaled ceilings.
Original 35%/50% activity flags remain visible; sustained whole-host clearance
can classify overlapping activity as bounded work rather than host saturation.
The new autonomic path never widens P-core workers, permits at most one training
target and caps collector reopening at 28%, subject to existing storage gates.
Known ingestion, reserve and reliability observers are attributed to bot-owned
observation, not unknown external load; their CPU remains counted.

Recovery uses independent producer time. Repeated publication of the same
sample cannot earn clearance. Missing observations, gaps over 90 seconds or
adverse readings reset the affected class immediately. An aggregate protect
band alone cannot starve observation or lossless recovery when measured limits
and explicit pause checks are clear. It still blocks extra ordinary maintenance,
training and bulk admission.

## Native Supervision

`run_guarded_maintenance.sh` now uses the guard's `--execute` mode. A stable
kernel lock spans admission, the child and cleanup. The directory receipt names
the living supervisor; age alone cannot steal an active or unknown-owner lock.
Direct legacy begin/end calls cannot obtain the adaptive relaxation.

The fixed non-apply observer has its own single-flight pool and
`maintenance_observer_latest.json` receipt, so unrelated long maintenance
cannot starve it or have its execution receipt overwritten by an observation.
The assessment owner's own singleton still excludes concurrent apply against
the same infrastructure owner.

The four maintenance slots are infrastructure autofix, system drift, section
grade and grade regression. Their overnight/macro restrictions remain intact.
With a fresh lease, default cooldown is at least five minutes and three times
the previous run duration, bounded by the original cooldown. Explicit per-slot
intervals are respected. Existing wrappers keep their niceness/background policy.
Adaptive jobs use single-threaded numerical-library settings, not hard CPU pinning.

The existing infrastructure launcher first runs a fixed assessment without
`--apply`. This observation has a separate 60-second budget, at most five
seconds of default jitter and no overnight/SQL-writer exclusion. Macro and
operator holds still apply. Expensive repairs then request their own admission.

The supervisor rechecks adaptive admission, current load, live disk space and
operator/maintenance holds every five seconds. Revocation or timeout triggers
the existing owned-process-tree termination helpers and bounded cleanup.
Normal completion, failure, interruption, timeout and resource revocation are
distinct receipts. Failed load readings cannot become zero load.

## Preserved Authority

No SQL writer reserve, queue threshold, current-day/latest-backup protection,
retention requirement, restored-byte verification, training eligibility,
promotion proof, broker reconciliation or live-execution gate is relaxed.
Compression still holds its shared storage lock, checks scratch capacity,
revalidates each source and fully restores/hashes it before replacing it.
The writer independently measures disk space again before admitting a drain.
The existing 15-minute pressure-recovery owner also invokes lifecycle backup
compression in bounded 32-file/180-second batches with a 15-minute cooldown.
When its outer load gate uses extra workload headroom, it runs only the two
verified compression owners and reserve recheck, never the broader telemetry,
offload or database-maintenance paths. The normal retention owner remains
active and shares the same storage lock and backup exclusions.
These are native platform controls, not Codex automations.
