"""Dependency-aware directions for existing infrastructure owners, not a new scheduler."""

from datetime import datetime, timezone

GROUPS = (
    (
        "runtime",
        "runtime_throttle_control",
        (),
        (
            "process_lane_ownership",
            "launchd_job_health",
            "stuck_report_pdf_browser_jobs",
        ),
        "Contain excess workers and repair ownership before increasing work.",
    ),
    (
        "storage",
        "soak_self_healing_control",
        ("runtime",),
        ("external_drive_route_health", "stateful_storage_regression"),
        "Recover verified capacity without deleting active or protected evidence.",
    ),
    (
        "provider",
        "provider_access_guard",
        (),
        ("schwab_auth_supervisor", "coinbase_api_health"),
        "Preserve provider cooldowns; escalate interactive authentication to the operator.",
    ),
    (
        "ingestion",
        "storage_backpressure_autopilot",
        ("storage", "runtime"),
        ("sql_ingestion_lag_and_backlog",),
        "Drain through the existing single writer only after capacity admission.",
    ),
    (
        "integrity",
        "storage_disaster_recovery",
        ("storage",),
        ("autonomous_recovery_drills",),
        "Verify restore and recovery evidence; distinguish a receipt from an actual drill.",
    ),
    (
        "commands",
        "command_validity_bot",
        (),
        ("command_docs_vs_opsctl_routes",),
        "Audit command usefulness and dispatch without executing operator-gated commands.",
    ),
    (
        "evidence",
        "runtime_artifact_refresh",
        ("storage", "ingestion", "runtime"),
        (
            "one_numbers_original_coverage",
            "governance_artifact_freshness",
            "operator_cockpit_readiness",
            "point_in_time_replay",
            "cold_lane_research_factory",
        ),
        "Refresh producer-owned evidence in dependency order; never renew its source time.",
    ),
    (
        "assurance",
        "master_infrastructure_supervisor",
        (),
        ("child_repair_bot_outcomes", "self_auditing_infra_bots"),
        "Audit failed/no-progress repairs, escalate unknowns and preserve operator boundaries.",
    ),
)

# Only these exact commands may be dispatched by the operations master.
ACTIONS = {
    "runtime": (("runtime-throttle", "--apply", "--json"), 45),
    "storage": (
        (
            "soak-self-heal",
            "--storage-recovery-only",
            "--quick-storage-recovery",
            "--apply",
            "--json",
        ),
        100,
    ),
    "ingestion": (("ingestion-storage-control", "--json"), 30),
    "commands": (("command-validity", "--safe-audit", "--summary-json"), 60),
}

RESPONSIBILITIES = {
    "inventory": "Map every observed check to a subgroup and accountable owner; escalate unmapped checks.",
    "triage": "Prioritize containment, capacity, provider access, ingestion, integrity, then evidence and research.",
    "dependencies": "Hold downstream repairs behind unresolved upstream needs; keep independent diagnostics available.",
    "resource_admission": "Respect fresh workload leases, storage reserves, power state and operator/maintenance/global holds.",
    "work_ownership": "Delegate to existing owners and their locks; never create a second writer, repair daemon or scheduler.",
    "bounded_dispatch": "At most two allowlisted owner calls in a 150-second work window; retain per-owner cooldown across invocations.",
    "recovery_verification": "Command exit success is not repair proof; completion requires fresh owner evidence and the declared success criterion.",
    "anti_oscillation": "Do not retry a dispatched owner for ten minutes, including failed and uncertain attempts; do not clear its retry debt.",
    "freshness": "Reject missing, future, corrupt and expired control observations; a new wrapper does not refresh its inputs.",
    "audit": "Record selection, deferral, attempts, before-state and evidence requirements without storing secrets.",
    "escalation": "Identify operator, capacity, provider and qualification needs that automation cannot honestly clear.",
    "release_handoff": "Present reviewed-source needs to the operator; never accept, publish, arm, attest or promote a release automatically.",
    "economics": "Separate operational health, definition completeness and economic evidence; never force a trade or manufacture profitability.",
    "self_limits": "No credentials, orders, halt clearance, reserve reduction, contract rewriting or protected-volume access.",
}

VERIFICATION = {
    "runtime": "Fresh resource, process-ownership and restart observations; no duplicate worker or unexplained restart storm.",
    "storage": "Owner-verified reclaimed bytes, source integrity and measured reserve headroom; a completed cleanup is not capacity clearance.",
    "provider": "Relevant successful provider requests after cooldown; account authentication alone cannot prove market-data recovery.",
    "ingestion": "Fresh backlog delta, oldest pending age and committed writer progress; starting a drain is not completion.",
    "integrity": "Fresh verified backup/restore and database integrity receipts tied to the actual source and destination.",
    "commands": "Safe command audit with explicit route coverage and unresolved operator-gated probes; documentation is not execution proof.",
    "evidence": "Producer-owned artifacts with original input timestamps, lineage and coverage inside their individual freshness contracts.",
    "assurance": "Fresh subgroup results, explicit unresolved needs and failed-attempt history; master execution alone cannot clear system health.",
}


def build_directions(checks, controls, *, now=None):
    now = now or datetime.now(timezone.utc)
    by_name = {
        row["name"]: row for row in checks if isinstance(row, dict) and row.get("name")
    }
    covered = {check for _, _, _, names, _ in GROUPS for check in names}
    source_ready = bool(controls.get("fresh"))
    storage_admitted = source_ready and controls.get("storage_pressure") is False
    runtime_admitted = source_ready and controls.get("maintenance_admitted") is True
    ingestion_ready = (
        by_name.get("sql_ingestion_lag_and_backlog", {}).get("status") == "ready"
    )
    dependencies = {
        "storage": storage_admitted,
        "runtime": runtime_admitted,
        "ingestion": ingestion_ready,
    }
    directives = []
    for priority, (name, owner, depends_on, names, mission) in enumerate(GROUPS, 1):
        issues = [key for key in names if by_name.get(key, {}).get("status") != "ready"]
        if name == "storage" and not storage_admitted:
            issues.append("local_storage_pressure_or_unknown")
        if name == "runtime" and not runtime_admitted:
            issues.append("resource_admission_or_unknown")
        if name == "provider" and controls.get("provider_cooldown"):
            issues.append("provider_cooldown_active")
        missing = [key for key in depends_on if not dependencies.get(key, False)]
        # Recovery admission is purpose-specific: it need not wait for normal
        # maintenance admission, otherwise storage pressure creates a deadlock.
        if name == "storage":
            missing = (
                []
                if source_ready and controls.get("storage_recovery_admitted")
                else ["storage_recovery_admission"]
            )
        if name == "runtime":
            missing = [] if source_ready else ["fresh_control_observations"]
        if controls.get("operator_hold"):
            missing.append("operator_or_maintenance_hold")
        action = ACTIONS.get(name)
        directives.append(
            {
                "subgroup": name,
                "owner": owner,
                "priority": priority,
                "mission": mission,
                "issues": issues,
                "depends_on": list(depends_on),
                "blocked_by": missing if issues else [],
                "directive": (
                    "observe"
                    if not issues
                    else (
                        "defer"
                        if missing
                        else "delegate" if action else "owner_followup"
                    )
                ),
                "command": (
                    list(action[0]) if action and issues and not missing else None
                ),
                "timeout_seconds": action[1] if action else None,
                "success_requires": VERIFICATION[name],
                "verification_mode": "subsequent_owner_observation; no_command_exit_credit",
                "scope": "coordinator_dispatch_only; independent_owners_keep_existing_admission",
            }
        )
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "role": "operations_master_infrabot",
        "status": (
            "held"
            if controls.get("operator_hold")
            else (
                "needs_attention"
                if set(by_name) - covered or any(row["issues"] for row in directives)
                else "ready"
            )
        ),
        "scheduler": "existing_master_infrastructure_supervisor",
        "responsibilities": RESPONSIBILITIES,
        "controls": controls,
        "directives": directives,
        "unmapped_checks": sorted(set(by_name) - covered),
        "escalations": [
            {
                "subgroup": row["subgroup"],
                "owner": row["owner"],
                "needs": row["blocked_by"] or row["issues"],
                "verification_required": row["success_requires"],
            }
            for row in directives
            if row["directive"] in {"defer", "owner_followup"}
        ],
        "dispatch_limits": {
            "max_attempts_per_pass": 2,
            "work_seconds": 150,
            "owner_cooldown_seconds": 600,
        },
        "authority": {
            "order": False,
            "attestation": False,
            "source_acceptance": False,
            "halt_clearance": False,
            "policy_relaxation": False,
            "new_scheduler": False,
        },
        "completion_credit": False,
    }
