#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrabot_gap_roster_latest.json"


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _health(project_root: Path, name: str) -> dict[str, Any]:
    return _as_dict(load_json(project_root / "governance" / "health" / name))


def _opsctl(project_root: Path, *args: str) -> list[str]:
    return [str(project_root / "scripts" / "ops" / "opsctl.sh"), *args]


def _script(project_root: Path, rel: str, *args: str) -> list[str]:
    return [str(Path(sys.executable)), str(project_root / rel), *args]


def _contains(values: Any, needle: str) -> bool:
    return any(needle in str(value) for value in _as_list(values))


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True

    payload: dict[str, Any] = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
        "payload": payload,
    }


def _bot(
    *,
    bot_id: str,
    title: str,
    layer: str,
    responsibility: str,
    needs_action: bool,
    evidence: list[str],
    command: list[str],
    stop_when: str,
    severity: str = "medium",
    risk_level: str = "low",
    cadence: str = "on_degradation",
    integration_targets: list[str] | None = None,
) -> dict[str, Any]:
    clean_evidence = ordered_unique([str(item) for item in evidence if str(item).strip()])
    return {
        "id": bot_id,
        "name": bot_id,
        "title": title,
        "layer": layer,
        "assigned": True,
        "active": bool(needs_action),
        "needs_action": bool(needs_action),
        "state": "triggered" if needs_action else "watching",
        "severity": severity if needs_action else "normal",
        "risk_level": risk_level,
        "cadence": cadence,
        "responsibility": responsibility,
        "evidence": clean_evidence,
        "command": list(command),
        "stop_when": stop_when,
        "integration_targets": integration_targets
        or [
            "system_cleanliness_infrabot",
            "operator_cockpit",
            "system_needs_intelligence",
            "master_infrastructure_supervisor",
        ],
        "authority_boundary": "advisory_and_safe_repair_only_no_live_execution_authority",
    }


def _writer_lock_handoff(project_root: Path, writer: dict[str, Any]) -> dict[str, Any]:
    state = _as_dict(writer.get("writer_state_before"))
    summary = _as_dict(writer.get("summary"))
    needed = bool(state.get("complete_lock_handoff_needed") or summary.get("completed_writer_lock_handoff_needed"))
    return _bot(
        bot_id="writer_lock_handoff_infrabot",
        title="Writer Lock Handoff Infrabot",
        layer="writer_backlog",
        responsibility="Release completed SQL writer lock handoffs after the grace window when the writer is complete and no child writer is alive.",
        needs_action=needed,
        severity="high",
        evidence=[
            f"writer_overall_status={writer.get('overall_status', '')}",
            f"writer_step={state.get('current_step') or state.get('effective_current_step') or ''}",
            f"complete_lock_handoff_needed={needed}",
            f"child_writer_active={bool(state.get('child_writer_active', False))}",
        ],
        command=_opsctl(project_root, "writer-cycle-coordinator", "--apply", "--skip-drain", "--skip-maintenance", "--json"),
        stop_when="writer lock is released or writer_state_before.complete_lock_handoff_needed is false.",
    )


def _health_truth(project_root: Path, cockpit: dict[str, Any], artifact_slo: dict[str, Any], runtime_refresh: dict[str, Any]) -> dict[str, Any]:
    hardening = _as_dict(cockpit.get("hardening_scorecard"))
    recommended = _as_list(cockpit.get("recommended_actions"))
    stale_count = _safe_int(artifact_slo.get("stale_count") or _as_dict(artifact_slo.get("summary")).get("stale_count"), 0)
    refresh_status = str(runtime_refresh.get("overall_status") or "").lower()
    needs_action = (
        _contains(recommended, "health_gates_stale")
        or stale_count > 0
        or bool(hardening) and not bool(hardening.get("self_auditing_bots_current", True))
        or refresh_status in {"degraded", "blocked"}
    )
    return _bot(
        bot_id="health_truth_reconciler_infrabot",
        title="Health Truth Reconciler Infrabot",
        layer="truth_reconciliation",
        responsibility="Refresh stale health surfaces and reconcile cockpit claims against current storage, writer, runtime, and source artifacts.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"cockpit_status={cockpit.get('overall_status', '')}",
            f"health_gates_stale={_contains(recommended, 'health_gates_stale')}",
            f"artifact_stale_count={stale_count}",
            f"self_auditing_bots_current={hardening.get('self_auditing_bots_current', '')}",
            f"runtime_refresh_status={runtime_refresh.get('overall_status', '')}",
        ],
        command=_opsctl(project_root, "dashboard-refresh", "--json"),
        stop_when="operator cockpit, artifact freshness, and fast health agree on current ready/degraded state.",
    )


def _provider_cross_verification(project_root: Path, provider: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    provider_status = str(provider.get("overall_status") or "").lower()
    source_status = str(source.get("overall_status") or "").lower()
    source_overall = _as_dict(source.get("overall"))
    unverified = _as_list(source_overall.get("unverified_sources")) or _as_list(source.get("unverified_artifacts"))
    required_failures = _as_list(provider.get("required_failures"))
    soft_failures = _as_list(provider.get("soft_failures"))
    needs_action = provider_status not in {"ready", "ok"} or source_status in {"degraded", "blocked"} or bool(unverified)
    return _bot(
        bot_id="provider_cross_verification_infrabot",
        title="Provider Cross-Verification Infrabot",
        layer="collectors_sources",
        responsibility="Keep required and optional provider context cross-verified so decisions are not leaning on single-provider freshness.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"provider_status={provider.get('overall_status', '')}",
            f"source_status={source.get('overall_status', '')}",
            f"required_failure_count={len(required_failures)}",
            f"soft_failure_count={len(soft_failures)}",
            f"unverified_count={len(unverified)}",
        ],
        command=_opsctl(project_root, "source-verification-refresh", "--apply", "--json"),
        stop_when="provider mesh is ready and source verification has no required failures or unverified sources.",
    )


def _paper_feedback(project_root: Path, paper_profitability: dict[str, Any], paper_performance: dict[str, Any]) -> dict[str, Any]:
    grade = str(paper_profitability.get("profitability_grade") or "").upper()
    status = str(paper_profitability.get("overall_status") or "").lower()
    perf_status = str(paper_performance.get("overall_status") or "").lower()
    loss_causes = _as_list(paper_profitability.get("top_loss_causes"))
    needs_action = grade in {"D", "F", "C"} or status in {"protective_tightening", "degraded", "blocked"} or perf_status in {"stale", "degraded", "blocked"}
    return _bot(
        bot_id="paper_feedback_repair_infrabot",
        title="Paper Feedback Repair Infrabot",
        layer="paper_feedback",
        responsibility="Turn losing paper sleeve evidence into deweights, confirmation requirements, label repairs, and report freshness checks.",
        needs_action=needs_action,
        severity="high",
        evidence=[
            f"profitability_grade={grade}",
            f"profitability_status={paper_profitability.get('overall_status', '')}",
            f"paper_performance_status={paper_performance.get('overall_status', '')}",
            f"top_loss_cause_count={len(loss_causes)}",
        ],
        command=_opsctl(project_root, "paper-profitability-control", "--apply", "--json"),
        stop_when="paper profitability grade reaches at least B or active strategy controls are no longer tightening.",
    )


def _promotion_replay(project_root: Path, cleanliness: dict[str, Any], promotion_quality: dict[str, Any]) -> dict[str, Any]:
    blocked_layers = _as_list(cleanliness.get("blocked_layers"))
    status = str(promotion_quality.get("overall_status") or "").lower()
    ok = bool(promotion_quality.get("ok", False))
    needs_action = "promotion_replay" in blocked_layers or status in {"blocked", "degraded"} or (promotion_quality and not ok)
    return _bot(
        bot_id="promotion_replay_gate_infrabot",
        title="Promotion Replay Gate Infrabot",
        layer="promotion_replay",
        responsibility="Keep promotions held until replay hash, golden replay, lineage packets, and quality gates are clean.",
        needs_action=needs_action,
        severity="high",
        evidence=[
            f"cleanliness_blocked_layers={','.join(str(x) for x in blocked_layers)}",
            f"promotion_quality_status={promotion_quality.get('overall_status', '')}",
            f"promotion_quality_ok={ok}",
        ],
        command=_opsctl(project_root, "system-cleanliness-autopilot", "--apply", "--json"),
        stop_when="promotion_replay is no longer blocked and promotion_quality_gate reports ok.",
    )


def _bot_data_labeling(project_root: Path, bot_needs: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(bot_needs.get("need_counts"))
    collect_more = _safe_int(counts.get("collect_more_data"), 0)
    quality_retrain = _safe_int(counts.get("targeted_quality_retrain"), 0)
    topoff = _safe_int(counts.get("top_off_walk_forward_runs"), 0)
    collect_first = _safe_int(intake.get("collect_first_count"), 0)
    needs_action = collect_more > 0 or quality_retrain > 0 or topoff > 0 or collect_first > 0
    return _bot(
        bot_id="bot_data_labeling_targeter_infrabot",
        title="Bot Data Labeling Targeter Infrabot",
        layer="bot_data_quality",
        responsibility="Route sample-starved and weak bots into exact data, side-label, precision, or walk-forward topoff work before training.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"collect_more_data={collect_more}",
            f"targeted_quality_retrain={quality_retrain}",
            f"top_off_walk_forward_runs={topoff}",
            f"collect_first_count={collect_first}",
        ],
        command=_opsctl(project_root, "training-data-intake", "--apply", "--focus-limit", "160", "--json"),
        stop_when="bot-needs has no collect-more-data floor breaches and trainable candidates are not blocked by label quality.",
    )


def _recovery_drill(project_root: Path, sleeves: dict[str, Any], cockpit: dict[str, Any]) -> dict[str, Any]:
    surfaces = _as_dict(cockpit.get("surfaces"))
    rolling = str(_as_dict(surfaces.get("rolling_restart_controller")).get("status") or "").lower()
    blackstart = str(_as_dict(surfaces.get("blackstart_recovery")).get("status") or "").lower()
    chaos = str(_as_dict(surfaces.get("chaos_drill_coordinator")).get("status") or "").lower()
    sleeve_status = str(sleeves.get("overall_status") or "").lower()
    needs_action = sleeve_status not in {"ready", "ok"} or rolling in {"blocked", "degraded"} or blackstart in {"blocked", "degraded"} or chaos in {"blocked", "degraded"}
    return _bot(
        bot_id="recovery_drill_infrabot",
        title="Recovery Drill Infrabot",
        layer="recovery_hygiene",
        responsibility="Exercise read-only restart, blackstart, and sleeve recovery checks so launcher-down notices do not become restart storms.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"sleeve_launcher_status={sleeves.get('overall_status', '')}",
            f"rolling_restart_status={rolling}",
            f"blackstart_status={blackstart}",
            f"chaos_drill_status={chaos}",
        ],
        command=_opsctl(project_root, "post-restart-settle", "--apply", "--json"),
        stop_when="sleeve launcher is ready and rolling restart, blackstart, and chaos drill surfaces are not blocked.",
    )


def _self_audit_freshness(project_root: Path, cockpit: dict[str, Any], self_model: dict[str, Any], master_infra: dict[str, Any]) -> dict[str, Any]:
    hardening = _as_dict(cockpit.get("hardening_scorecard"))
    master_status = str(master_infra.get("overall_status") or "").lower()
    self_status = str(self_model.get("overall_status") or "").lower()
    needs_action = (
        not bool(hardening.get("self_auditing_bots_current", True))
        or self_status in {"degraded", "blocked"}
        or master_status in {"degraded", "blocked"}
    )
    return _bot(
        bot_id="self_audit_freshness_infrabot",
        title="Self-Audit Freshness Infrabot",
        layer="self_awareness",
        responsibility="Keep the self-model, master infrastructure supervisor, and what-do-I-need surfaces fresh and internally consistent.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"self_auditing_bots_current={hardening.get('self_auditing_bots_current', '')}",
            f"system_self_model_status={self_model.get('overall_status', '')}",
            f"master_infra_status={master_infra.get('overall_status', '')}",
        ],
        command=_opsctl(project_root, "master-infra-supervisor", "--json"),
        stop_when="self-auditing bots are current and master infrastructure supervisor is ready.",
    )


def _cotenant_headroom(project_root: Path, pressure: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    host_score = _safe_float(pressure.get("host_saturation_score"), 0.0)
    compute = str(pressure.get("compute_pressure_level") or "").lower()
    mem = str(pressure.get("memory_pressure_level") or "").lower()
    memory_status = str(memory.get("overall_status") or "").lower()
    needs_action = host_score >= 50.0 or compute in {"elevated", "high"} or mem in {"elevated", "high"} or memory_status in {"degraded", "blocked"}
    return _bot(
        bot_id="cotenant_headroom_guard_infrabot",
        title="Co-Tenant Headroom Guard Infrabot",
        layer="host_resource_governance",
        responsibility="Keep backlog, collectors, training, MLX, Logic/FCP/Music, and foreground app use inside one shared headroom policy.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"host_saturation_score={host_score:.2f}",
            f"compute_pressure_level={compute}",
            f"memory_pressure_level={mem}",
            f"memory_efficiency_status={memory.get('overall_status', '')}",
        ],
        command=_opsctl(project_root, "pressure-relief", "--apply", "--json"),
        stop_when="host saturation is below 50 and compute/memory pressure are normal while foreground-app governor remains active.",
    )


def _protected_volume_boundary(project_root: Path, host: dict[str, Any], codex_guard: dict[str, Any]) -> dict[str, Any]:
    body_map = _as_dict(host.get("body_map"))
    storage = _as_dict(body_map.get("storage_layout"))
    protected = [str(item) for item in _as_list(storage.get("protected_volumes")) if str(item).strip()]
    adapters = _as_dict(host.get("adapters"))
    protected_adapter = _as_dict(adapters.get("protected_storage"))
    denylist = [str(item) for item in _as_list(protected_adapter.get("denylist")) if str(item).strip()]
    codex_boundary = _as_dict(codex_guard.get("workspace_boundary"))
    blocked_volume = str(codex_boundary.get("blocked_volume") or "")
    video_ready = "/Volumes/VIDEO" in set(protected + denylist) or blocked_volume == "/Volumes/VIDEO"
    needs_action = not video_ready
    return _bot(
        bot_id="protected_volume_boundary_infrabot",
        title="Protected Volume Boundary Infrabot",
        layer="storage_safety",
        responsibility="Keep protected external volumes, especially /Volumes/VIDEO, denylisted from cleanup, pruning, benchmarks, and automated writes.",
        needs_action=needs_action,
        severity="critical",
        risk_level="none",
        evidence=[
            f"video_in_host_contract={('/Volumes/VIDEO' in protected)}",
            f"video_in_os_adapter_denylist={('/Volumes/VIDEO' in denylist)}",
            f"codex_blocked_volume={blocked_volume}",
        ],
        command=_opsctl(project_root, "host-capability", "--json"),
        stop_when="/Volumes/VIDEO appears in host capability protected volumes and protected-storage denylist.",
    )


def _training_batch_readiness(project_root: Path, training_runtime: dict[str, Any]) -> dict[str, Any]:
    launch = _as_dict(training_runtime.get("training_launch_contract"))
    host = _as_dict(training_runtime.get("host_training_headroom_gate"))
    launch_blockers = _as_list(launch.get("launch_blockers"))
    prep_blockers = _as_list(launch.get("prep_blockers"))
    host_batch_cap = _safe_int(host.get("batch_cap"), 0)
    memory_batch10_safe = bool(host.get("batch10_training_safe", False))
    memory_batch20_safe = bool(host.get("batch20_training_safe", False))
    status = str(training_runtime.get("overall_status") or "").lower()
    needs_action = (
        status in {"constrained", "blocked", "degraded"}
        or bool(launch_blockers)
        or bool(prep_blockers)
        or host_batch_cap < 10
        or not memory_batch10_safe
    )
    return _bot(
        bot_id="training_batch_readiness_infrabot",
        title="Training Batch Readiness Infrabot",
        layer="training_readiness",
        responsibility="Keep batch-10 and batch-20 training readiness honest by checking snapshot freshness, memory soak, writer idleness, and governor caps.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"training_runtime_status={training_runtime.get('overall_status', '')}",
            f"launch_allowed={bool(launch.get('launch_allowed', False))}",
            f"host_batch_cap={host_batch_cap}",
            f"batch10_training_safe={memory_batch10_safe}",
            f"batch20_training_safe={memory_batch20_safe}",
            f"launch_blockers={','.join(str(x) for x in launch_blockers)}",
        ],
        command=_opsctl(project_root, "training-runtime-control", "--limit", "20", "--json"),
        stop_when="training runtime is ready, batch10 is safe, and launch blockers are empty.",
    )


def _paper_execution_queue(project_root: Path, paper_backlog: dict[str, Any], reconciliation: dict[str, Any]) -> dict[str, Any]:
    pending = _safe_int(paper_backlog.get("pending_rows_after"), 0)
    backlog_ok = bool(paper_backlog.get("ok", False)) if paper_backlog else True
    reconciliation_ok = bool(reconciliation.get("ok", False)) if reconciliation else True
    status = str(reconciliation.get("overall_status") or "").lower()
    needs_action = pending > 0 or not backlog_ok or not reconciliation_ok or status in {"degraded", "blocked", "stale"}
    return _bot(
        bot_id="paper_execution_queue_reconciler_infrabot",
        title="Paper Execution Queue Reconciler Infrabot",
        layer="paper_execution",
        responsibility="Keep paper execution intents, calibration, and reconciliation from silently lagging behind paper sleeve decisions.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"pending_rows_after={pending}",
            f"paper_backlog_ok={backlog_ok}",
            f"paper_reconciliation_ok={reconciliation_ok}",
            f"paper_reconciliation_status={reconciliation.get('overall_status', '')}",
        ],
        command=_opsctl(project_root, "paper-calibration", "--json"),
        stop_when="paper execution backlog is zero and reconciliation/calibration reports are ok.",
    )


def _duplicate_alpha_compression(project_root: Path, bot_quality: dict[str, Any], training_quality: dict[str, Any]) -> dict[str, Any]:
    quality_status = str(bot_quality.get("overall_status") or "").lower()
    targeted = _as_dict(training_quality.get("targeted_actions"))
    probation = _as_list(targeted.get("quality_probation_bot_ids"))
    recommended = _as_list(bot_quality.get("recommended_actions"))
    needs_action = quality_status in {"needs_work", "degraded", "blocked"} or bool(probation)
    return _bot(
        bot_id="duplicate_alpha_compression_infrabot",
        title="Duplicate Alpha Compression Infrabot",
        layer="bot_quality",
        responsibility="Keep duplicate/overlapping alpha, teacher curation, and quality probation queues from bloating the bot population.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"bot_quality_status={bot_quality.get('overall_status', '')}",
            f"quality_probation_count={len(probation)}",
            f"recommended_action_count={len(recommended)}",
        ],
        command=_opsctl(project_root, "bot-quality-autopilot", "--apply", "--json"),
        stop_when="bot quality is ready and quality probation/duplicate-alpha queues are no longer growing.",
    )


def _livefeed_mirror_continuity(project_root: Path, livefeed: dict[str, Any], heavy_view: dict[str, Any]) -> dict[str, Any]:
    status = str(livefeed.get("status") or "").lower()
    alive = bool(livefeed.get("alive", False))
    heavy_mode = str(heavy_view.get("mode") or "").lower()
    idle = _safe_int(livefeed.get("idle_heartbeat_seconds"), 0)
    needs_action = (not alive) or status in {"stopped", "down", "paused_runtime_pressure", "error"} or heavy_mode in {"expired_or_closed", "stopped"}
    return _bot(
        bot_id="livefeed_mirror_continuity_infrabot",
        title="Livefeed Mirror Continuity Infrabot",
        layer="observability_feed",
        responsibility="Keep the local livefeed mirror flowing with bounded tails, TTL awareness, and pressure-aware refreshes.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"livefeed_status={livefeed.get('status', '')}",
            f"livefeed_alive={alive}",
            f"livefeed_idle_heartbeat_seconds={idle}",
            f"heavy_view_mode={heavy_view.get('mode', '')}",
        ],
        command=_opsctl(project_root, "livefeed-refresh", "--dry-run"),
        stop_when="livefeed is alive and not paused/stopped while heavy view is not expired.",
    )


def _auth_lease_preflight(project_root: Path, auth_lease: dict[str, Any], auth_supervisor: dict[str, Any]) -> dict[str, Any]:
    lease_budget = _as_dict(auth_lease.get("lease_budget"))
    token = _as_dict(auth_supervisor.get("token"))
    lease_status = str(auth_lease.get("overall_status") or "").lower()
    supervisor_status = str(auth_supervisor.get("overall_status") or "").lower()
    expires_in = _safe_float(lease_budget.get("expires_in_seconds") or token.get("expires_in_seconds"), 0.0)
    min_lease = _safe_float(lease_budget.get("min_lease_seconds"), 1200.0)
    refresh_needed = bool(token.get("refresh_needed", False))
    needs_action = lease_status not in {"ready", "ok"} or supervisor_status not in {"ready", "ok"} or refresh_needed or expires_in < min_lease
    return _bot(
        bot_id="auth_lease_preflight_infrabot",
        title="Auth Lease Preflight Infrabot",
        layer="broker_auth",
        responsibility="Watch Schwab auth lease, token age, callback helpers, and premarket token guard before collectors or sleeves need credentials.",
        needs_action=needs_action,
        severity="high" if expires_in < 600.0 or refresh_needed else "medium",
        evidence=[
            f"auth_lease_status={auth_lease.get('overall_status', '')}",
            f"auth_supervisor_status={auth_supervisor.get('overall_status', '')}",
            f"expires_in_seconds={expires_in:.1f}",
            f"refresh_needed={refresh_needed}",
        ],
        command=_opsctl(project_root, "schwab-auth-supervisor", "--json"),
        stop_when="auth lease is healthy, token is ready, and expires_in_seconds is above the minimum lease budget.",
    )


def _market_explanation_evidence(project_root: Path, explainer: dict[str, Any]) -> dict[str, Any]:
    status = str(explainer.get("overall_status") or "").lower()
    confidence = _safe_float(explainer.get("primary_confidence"), 0.0)
    symbol_events = _safe_int(explainer.get("symbol_evidence_count"), 0)
    unknowns = _as_list(explainer.get("unknowns"))
    coverage = _as_dict(explainer.get("source_coverage"))
    missing_coverage = [name for name, value in coverage.items() if not bool(value)]
    needs_action = status in {"thin", "degraded", "blocked"} or symbol_events <= 0 or confidence < 0.7 or bool(missing_coverage)
    return _bot(
        bot_id="market_explanation_evidence_infrabot",
        title="Market Explanation Evidence Infrabot",
        layer="decision_explainability",
        responsibility="Backfill symbol-specific and cross-source evidence so market-move explanations are decision-row backed instead of only context backed.",
        needs_action=needs_action,
        severity="medium",
        evidence=[
            f"market_explainer_status={explainer.get('overall_status', '')}",
            f"symbol={explainer.get('symbol', '')}",
            f"symbol_evidence_count={symbol_events}",
            f"primary_confidence={confidence:.2f}",
            f"missing_source_coverage={','.join(missing_coverage)}",
            f"unknown_count={len(unknowns)}",
        ],
        command=_opsctl(project_root, "decision-intelligence", "--json"),
        stop_when="market explainer is ready, confidence is at least 0.70, and symbol-specific evidence is present.",
    )


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, timeout_sec: int = 300) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    writer = _health(project_root, "writer_cycle_coordinator_latest.json")
    cockpit = _health(project_root, "operator_cockpit_latest.json")
    artifact_slo = _health(project_root, "artifact_freshness_slo_latest.json")
    runtime_refresh = _health(project_root, "runtime_artifact_refresh_latest.json")
    provider = _health(project_root, "provider_mesh_latest.json")
    source = _health(project_root, "source_verification_latest.json")
    paper_profitability = _health(project_root, "paper_profitability_control_latest.json")
    paper_performance = _health(project_root, "paper_performance_latest.json")
    cleanliness = _health(project_root, "system_cleanliness_infrabot_latest.json")
    promotion_quality = _health(project_root, "promotion_quality_gate_latest.json")
    bot_needs = _health(project_root, "bot_needs_intelligence_latest.json")
    data_intake = _health(project_root, "training_data_intake_expansion_latest.json")
    sleeves = _health(project_root, "all_sleeves_launcher_latest.json")
    self_model = _health(project_root, "system_self_model_latest.json")
    master_infra = _health(project_root, "master_infrastructure_supervisor_latest.json")
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    host = _health(project_root, "host_capability_contract_latest.json")
    codex_guard = _health(project_root, "codex_project_guard_latest.json")
    training_runtime = _health(project_root, "training_runtime_control_latest.json")
    paper_execution_backlog = _health(project_root, "paper_execution_backlog_relief_latest.json")
    paper_reconciliation = _health(project_root, "paper_reconciliation_slo_latest.json")
    bot_quality = _health(project_root, "bot_quality_autopilot_latest.json")
    training_quality = _health(project_root, "training_quality_control_latest.json")
    livefeed = _health(project_root, "livefeed_local_latest.json")
    livefeed_heavy = _health(project_root, "live_feed_heavy_view_latest.json")
    auth_lease = _health(project_root, "auth_lease_manager_latest.json")
    auth_supervisor = _health(project_root, "schwab_auth_supervisor_latest.json")
    market_explainer = _health(project_root, "market_move_explainer_latest.json")

    bots = [
        _writer_lock_handoff(project_root, writer),
        _health_truth(project_root, cockpit, artifact_slo, runtime_refresh),
        _provider_cross_verification(project_root, provider, source),
        _paper_feedback(project_root, paper_profitability, paper_performance),
        _promotion_replay(project_root, cleanliness, promotion_quality),
        _bot_data_labeling(project_root, bot_needs, data_intake),
        _recovery_drill(project_root, sleeves, cockpit),
        _self_audit_freshness(project_root, cockpit, self_model, master_infra),
        _cotenant_headroom(project_root, pressure, memory),
        _protected_volume_boundary(project_root, host, codex_guard),
        _training_batch_readiness(project_root, training_runtime),
        _paper_execution_queue(project_root, paper_execution_backlog, paper_reconciliation),
        _duplicate_alpha_compression(project_root, bot_quality, training_quality),
        _livefeed_mirror_continuity(project_root, livefeed, livefeed_heavy),
        _auth_lease_preflight(project_root, auth_lease, auth_supervisor),
        _market_explanation_evidence(project_root, market_explainer),
    ]

    active = [bot for bot in bots if bool(bot.get("needs_action"))]
    critical = [bot for bot in active if str(bot.get("severity")) == "critical"]
    high = [bot for bot in active if str(bot.get("severity")) == "high"]

    attempts: list[dict[str, Any]] = []
    if apply:
        for bot in active:
            cmd = list(bot.get("command") or [])
            if not cmd:
                continue
            attempts.append({**_run_json(cmd, cwd=project_root, timeout_sec=timeout_sec), "name": bot.get("name"), "layer": bot.get("layer")})

    failed_attempts = [row for row in attempts if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}]
    overall_status = "ready"
    if failed_attempts or critical:
        overall_status = "blocked"
    elif active:
        overall_status = "needs_action"

    recommended_actions = ordered_unique(
        [
            "run the writer lock handoff infrabot before heavier drain waves" if any(bot["id"] == "writer_lock_handoff_infrabot" for bot in active) else "",
            "refresh stale health truth before trusting cockpit degradation grades" if any(bot["id"] == "health_truth_reconciler_infrabot" for bot in active) else "",
            "cross-verify optional providers before letting context lanes influence decisions" if any(bot["id"] == "provider_cross_verification_infrabot" for bot in active) else "",
            "keep paper profitability controls in protective tightening until loss causes clear" if any(bot["id"] == "paper_feedback_repair_infrabot" for bot in active) else "",
            "hold promotion until replay lineage and promotion packets are clean" if any(bot["id"] == "promotion_replay_gate_infrabot" for bot in active) else "",
            "route collect-first bots through data intake and label repairs before broad training" if any(bot["id"] == "bot_data_labeling_targeter_infrabot" for bot in active) else "",
            "keep recovery drills read-only and live execution blocked" if any(bot["id"] == "recovery_drill_infrabot" for bot in active) else "",
            "refresh self-audit and master infrastructure supervisor before expansion" if any(bot["id"] == "self_audit_freshness_infrabot" for bot in active) else "",
            "apply pressure relief when host saturation crosses the co-tenant headroom line" if any(bot["id"] == "cotenant_headroom_guard_infrabot" for bot in active) else "",
            "preserve /Volumes/VIDEO as a never-write, never-prune protected volume" if any(bot["id"] == "protected_volume_boundary_infrabot" for bot in active) else "",
            "keep training in prep/micro-canary until batch readiness gates actually clear" if any(bot["id"] == "training_batch_readiness_infrabot" for bot in active) else "",
            "refresh paper execution calibration when execution queue or reconciliation evidence lags" if any(bot["id"] == "paper_execution_queue_reconciler_infrabot" for bot in active) else "",
            "compress duplicate alpha and quality-probation debt before expanding the bot roster" if any(bot["id"] == "duplicate_alpha_compression_infrabot" for bot in active) else "",
            "keep livefeed mirrors pressure-aware so observability stays flowing without fighting the computer" if any(bot["id"] == "livefeed_mirror_continuity_infrabot" for bot in active) else "",
            "watch Schwab auth lease before market windows instead of waiting for sleeve failures" if any(bot["id"] == "auth_lease_preflight_infrabot" for bot in active) else "",
            "collect symbol-specific evidence before trusting thin market-move explanations" if any(bot["id"] == "market_explanation_evidence_infrabot" for bot in active) else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "bot_count": len(bots),
        "active_count": len(active),
        "critical_count": len(critical),
        "high_count": len(high),
        "assigned_infrabots": [str(bot.get("id")) for bot in bots],
        "active_infrabots": [str(bot.get("id")) for bot in active],
        "infrabots": bots,
        "attempts": [
            {
                "name": row.get("name"),
                "layer": row.get("layer"),
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "integration_contract": {
            "owner_bot": "infrabot_gap_roster",
            "supervised_by": [
                "system_cleanliness_infrabot",
                "infrastructure_autofix_bot",
                "operator_cockpit",
                "master_infrastructure_supervisor",
            ],
            "safe_apply_supported": True,
            "destructive_actions_operator_gated": True,
            "live_execution_authority": False,
            "protected_volume_denylist": ["/Volumes/VIDEO"],
            "policy": "assign exact infrabots to current operational gaps without enabling live execution or broad retrains",
        },
        "source_artifacts": {
            "writer": str(health_root / "writer_cycle_coordinator_latest.json"),
            "cockpit": str(health_root / "operator_cockpit_latest.json"),
            "provider_mesh": str(health_root / "provider_mesh_latest.json"),
            "paper_profitability": str(health_root / "paper_profitability_control_latest.json"),
            "bot_needs": str(health_root / "bot_needs_intelligence_latest.json"),
            "host_capability": str(health_root / "host_capability_contract_latest.json"),
            "training_runtime": str(health_root / "training_runtime_control_latest.json"),
            "paper_execution_backlog": str(health_root / "paper_execution_backlog_relief_latest.json"),
            "livefeed": str(health_root / "livefeed_local_latest.json"),
            "auth_lease": str(health_root / "auth_lease_manager_latest.json"),
            "market_explainer": str(health_root / "market_move_explainer_latest.json"),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign infrabots to current system gaps.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "infrabot_gap_roster "
            f"overall_status={payload.get('overall_status')} "
            f"active={payload.get('active_count')}/{payload.get('bot_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_action"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
