# System Responsibility And Authority Contracts

The platform keeps its specialized bots, redundant observers, and layered controls, but it gives every mutable state domain one logical writer. The executable catalog is `config/system_role_contracts_v1.json`; `core/system_role_contracts.py` validates it and enforces component actions at runtime.

## Operating Planes

| Plane | Roles | Responsibility boundary |
| --- | --- | --- |
| Data | `data_collector`, `context_processor` | Acquire and normalize attributable observations. Never decide or trade. |
| Decision | `strategy_bot`, `sleeve_manager`, `master_coordinator`, `grand_master_coordinator` | Produce, aggregate, and rank hypotheses and order intentions. Never submit orders or grant authority. |
| Control | `risk_governance`, `promotion_governance` | Veto unsafe intentions and evaluate candidate evidence. Never originate signals or self-approve live execution. |
| Execution | `paper_execution_gateway`, `live_execution_gateway` | Be the only paper and live order writers. Consume already-approved intentions. |
| Truth | `truth_reconciliation` | Append and reconcile broker, account, position, fill, and execution facts. Never rewrite outcomes. |
| Operations | `infrastructure_maintainer`, `observability_reporter`, `training_research`, `evaluation_auditor` | Repair, report, train, and evaluate within bounded authority. Never alter trading logic as an operational repair. |

The canonical flow is:

```text
collect -> normalize -> score -> sleeve intent -> master rank -> global coordination
        -> risk veto/approval -> paper gateway or separately gated live gateway -> truth reconciliation
```

Risk governance, live execution, and truth reconciliation are non-bypassable roles. The Grand Master may coordinate policy but cannot grant itself or another component execution authority.

## Required Role Definition

Every role declares all of the following fields:

- Purpose, tier, and escalation owner.
- Allowed inputs and owned outputs.
- Write authority and explicit execution authority.
- Allowed and forbidden actions.
- Triggers and evidence outputs.
- Freshness SLO and failure behavior.
- Resource profile and maximum parallelism.

The contract also defines shared taxonomies for execution modes, freshness classes, failure classes, resource profiles, lifecycle states, severity, action classes, and configuration precedence. Safety flags and candidate-bound state outrank operator overrides, policy files, environment defaults, and code defaults.

## State Ownership

Each mutable domain declares one `writer_component_id`, one required action, its resource patterns, and permitted reader roles. Validation fails when a domain is missing an owner, a component claims an action outside its role, a resource has multiple writers, a control surface points to the wrong source, or a registry role is unmapped.

Redundant implementations may still observe the same state. For example, both watchdog implementations are observers behind the logical `process_restart_controller`. A shared action lease serializes the actual restart so redundancy does not become split-brain mutation.

Sensitive actions currently have exclusive owners:

| Action | Logical owner |
| --- | --- |
| `paper_submit` | `paper_execution_gateway` |
| `live_submit` | `live_execution_gateway` |
| `veto_trade` | `risk_governance_controller` |
| `record_truth` | `execution_truth_ledger` |
| `restart_process` | `process_restart_controller` |
| `manage_storage` | `storage_lifecycle_controller` |
| `publish_dashboard` | `observability_reporter` |
| `promote_candidate` | `production_candidate_controller` |
| `write_candidate_state` | `production_candidate_controller` |

Paper submission, live submission, process restart, promotion, and candidate-state writes also use single-flight file leases. Unknown, ambiguous, or misrouted actions fail closed.

## Runtime And Soak Integration

The execution lane checks its role at startup and acquires the appropriate paper or live lease around the actual execution call. The live order firewall independently verifies that the live gateway owns `live_submit`. Both process supervisors verify restart authority, and the process watchdog rechecks target liveness under the shared restart lease before spawning.

The control publishes `governance/health/system_role_contract_latest.json`. It is required by the unattended soak, always-on reliability sentinel, runtime dashboard, daily verifier, and system self-model. The artifact reports role, component, domain, control-binding, exclusive-action, registry-coverage, conflict, and content-receipt evidence.

Run:

```bash
./scripts/ops/opsctl.sh system-role-contract --json
./scripts/ops/opsctl.sh system-role-contract \
  --component live_execution_gateway \
  --action live_submit \
  --state-domain live_order_submission \
  --json
```

An `A+` means the responsibility catalog is structurally complete, source-backed, fully mapped, and conflict-free. It does not prove profitability, satisfy elapsed-time evidence, or unlock live execution.

## Change Rules

When adding a bot, service, writer, or control surface:

1. Reuse an existing role when its purpose and authority match; add a new role only for a genuinely new responsibility.
2. Register each concrete component and its source files.
3. Assign every new mutable artifact to one state domain and one writer.
4. Declare readers separately from writers.
5. Add high-impact mutations to the exclusive action map and lease table.
6. Route failures to an escalation owner; never grant self-escalation.
7. Refresh the role artifact and run the focused contract tests before accepting a new production candidate.

The permanent safety contract forbids automatic live-authority grants, automatic promotion authority, truth rewriting, self-escalation, and profitability guarantees.
