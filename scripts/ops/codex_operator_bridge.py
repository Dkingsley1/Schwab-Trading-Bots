#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "codex_operator_bridge_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "codex_operator_bridge_latest.md"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _age_seconds(raw_timestamp: Any, *, now: datetime | None = None) -> float | None:
    stamp = str(raw_timestamp or "").strip()
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return (current - parsed.astimezone(timezone.utc)).total_seconds()
    except Exception:
        return None


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return default


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _compact_command(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _top_named(rows: list[Any], *, name_key: str = "name", value_key: str = "executions", limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "name": str(row.get(name_key) or row.get("profile") or row.get("symbol") or row.get("strategy") or ""),
                value_key: row.get(value_key),
            }
        )
        if len(out) >= limit:
            break
    return out


def _paper_trade_snapshot(paper: dict[str, Any]) -> dict[str, Any]:
    day = paper.get("day") if isinstance(paper.get("day"), dict) else {}
    week = paper.get("week") if isinstance(paper.get("week"), dict) else {}
    sleeves = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    current_sleeves = [row for row in sleeves if bool(row.get("current_day_available", False))]
    weakest = sorted(
        sleeves,
        key=lambda row: _safe_float(row.get("ending_net_pnl_total"), 0.0),
    )[:6]
    strongest = sorted(
        sleeves,
        key=lambda row: _safe_float(row.get("ending_net_pnl_total"), 0.0),
        reverse=True,
    )[:6]
    return {
        "status": "ready" if bool(day.get("available", False) or week.get("available", False)) else "missing",
        "day": {
            "day_utc": str(day.get("day_utc") or ""),
            "available": bool(day.get("available", False)),
            "executions": _safe_int(day.get("executions"), 0),
            "buy_count": _safe_int(day.get("buy_count"), 0),
            "sell_count": _safe_int(day.get("sell_count"), 0),
            "unique_symbols": _safe_int(day.get("unique_symbols"), 0),
            "ending_net_pnl_total": round(_safe_float(day.get("ending_net_pnl_total"), 0.0), 6),
            "change_vs_previous_day": round(_safe_float(day.get("change_vs_previous_day"), 0.0), 6),
            "top_profiles": _top_named(_as_list(day.get("top_profiles")), limit=5),
            "top_symbols": _top_named(_as_list(day.get("top_symbols")), limit=5),
            "top_strategies": _top_named(_as_list(day.get("top_strategies")), limit=5),
        },
        "week": {
            "week_start_day_utc": str(week.get("week_start_day_utc") or ""),
            "week_end_day_utc": str(week.get("week_end_day_utc") or ""),
            "available": bool(week.get("available", False)),
            "executions": _safe_int(week.get("executions"), 0),
            "ending_net_pnl_total": round(_safe_float(week.get("ending_net_pnl_total"), 0.0), 6),
            "week_to_date_change": round(_safe_float(week.get("week_to_date_change"), 0.0), 6),
            "rolling_change": round(_safe_float(week.get("rolling_change"), 0.0), 6),
            "top_profiles": _top_named(_as_list(week.get("top_profiles")), limit=5),
            "top_symbols": _top_named(_as_list(week.get("top_symbols")), limit=5),
            "top_strategies": _top_named(_as_list(week.get("top_strategies")), limit=5),
        },
        "sleeves": {
            "total_reported": len(sleeves),
            "current_day_count": len(current_sleeves),
            "strongest": [
                {
                    "profile": str(row.get("profile") or ""),
                    "day_utc": str(row.get("day_utc") or ""),
                    "executions": _safe_int(row.get("executions"), 0),
                    "ending_net_pnl_total": round(_safe_float(row.get("ending_net_pnl_total"), 0.0), 6),
                    "win_rate": row.get("win_rate"),
                    "data_status": str(row.get("data_status") or ""),
                }
                for row in strongest
            ],
            "weakest": [
                {
                    "profile": str(row.get("profile") or ""),
                    "day_utc": str(row.get("day_utc") or ""),
                    "executions": _safe_int(row.get("executions"), 0),
                    "ending_net_pnl_total": round(_safe_float(row.get("ending_net_pnl_total"), 0.0), 6),
                    "win_rate": row.get("win_rate"),
                    "data_status": str(row.get("data_status") or ""),
                    "top_loss_causes": _top_named(_as_list(row.get("top_loss_causes")), name_key="cause", value_key="loss_total", limit=3),
                }
                for row in weakest
            ],
        },
    }


def _training_snapshot(training: dict[str, Any]) -> dict[str, Any]:
    contract = training.get("training_launch_contract") if isinstance(training.get("training_launch_contract"), dict) else {}
    host_gate = training.get("host_training_headroom_gate") if isinstance(training.get("host_training_headroom_gate"), dict) else {}
    drain = training.get("pretraining_drain_buffer") if isinstance(training.get("pretraining_drain_buffer"), dict) else {}
    drain_writer = drain.get("writer") if isinstance(drain.get("writer"), dict) else {}
    quality = training.get("training_quality") if isinstance(training.get("training_quality"), dict) else {}
    bot_needs = training.get("bot_needs") if isinstance(training.get("bot_needs"), dict) else {}
    next_stage = host_gate.get("next_reentry_stage") if isinstance(host_gate.get("next_reentry_stage"), dict) else {}
    return {
        "status": _status(training),
        "quality_score": _safe_float(quality.get("training_quality_score"), _safe_float(contract.get("training_quality_score"), 0.0)),
        "mode": str(contract.get("mode") or ""),
        "launch_allowed": bool(contract.get("launch_allowed", False)),
        "launch_blockers": [str(item) for item in _as_list(contract.get("launch_blockers"))],
        "requested_batch_size": _safe_int(contract.get("requested_batch_size"), 0),
        "recommended_batch_size": _safe_int(contract.get("recommended_batch_size"), 0),
        "recommended_retrain_command": _compact_command(contract.get("recommended_retrain_command")),
        "recommended_prep_commands": [_compact_command(cmd) for cmd in _as_list(contract.get("recommended_prep_commands")) if _compact_command(cmd)],
        "host_gate": {
            "status": str(host_gate.get("status") or ""),
            "safe_for_training": bool(host_gate.get("safe_for_training", False)),
            "batch_cap": _safe_int(host_gate.get("batch_cap"), 0),
            "memory_status": str(host_gate.get("memory_status") or ""),
            "memory_decision": str(host_gate.get("memory_decision") or ""),
            "next_reentry_stage": {
                "stage": str(next_stage.get("stage") or ""),
                "profile": str(next_stage.get("profile") or ""),
                "allowed_now": bool(next_stage.get("allowed_now", False)),
                "max_parallel_trainings": _safe_int(next_stage.get("max_parallel_trainings"), 0),
            },
        },
        "pretraining_drain_buffer": {
            "status": str(drain.get("status") or ""),
            "safe_to_launch_now": bool(drain.get("safe_to_launch_now", False)),
            "launch_blocker": str(drain.get("launch_blocker") or ""),
            "batch_cap": _safe_int(drain.get("batch_cap"), 0),
            "pending_lines": _safe_int(drain.get("pending_lines"), 0),
            "oldest_pending_age_seconds": round(_safe_float(drain.get("oldest_pending_age_seconds"), 0.0), 3),
            "writer": {
                "active": bool(drain_writer.get("active", False)),
                "running": bool(drain_writer.get("running", False)),
                "status": str(drain_writer.get("status") or ""),
                "current_step": str(drain_writer.get("current_step") or ""),
                "completed_shard_count": _safe_int(drain_writer.get("completed_shard_count"), 0),
                "planned_shard_count": _safe_int(drain_writer.get("planned_shard_count"), 0),
                "progress_age_minutes": round(_safe_float(drain_writer.get("progress_age_minutes"), 0.0), 3),
                "cycle_age_minutes": round(_safe_float(drain_writer.get("cycle_age_minutes"), 0.0), 3),
            },
        },
        "bot_needs": {
            "status": _status(bot_needs),
            "training_topoff_candidates": _safe_int(bot_needs.get("training_topoff_candidates"), 0),
            "need_counts": bot_needs.get("need_counts") if isinstance(bot_needs.get("need_counts"), dict) else {},
        },
    }


def _writer_snapshot(writer: dict[str, Any]) -> dict[str, Any]:
    writer_state = writer.get("writer_state_before") if isinstance(writer.get("writer_state_before"), dict) else {}
    if not writer_state:
        writer_state = writer.get("writer_state_after_wait") if isinstance(writer.get("writer_state_after_wait"), dict) else {}
    summary = writer.get("summary") if isinstance(writer.get("summary"), dict) else {}
    effect = writer.get("drain_effectiveness") if isinstance(writer.get("drain_effectiveness"), dict) else {}
    completed = _safe_int(writer_state.get("completed_shard_count"), 0)
    planned = _safe_int(writer_state.get("planned_shard_count"), 0)
    return {
        "status": _status(writer),
        "active": bool(writer_state.get("active", summary.get("writer_active_after_wait", False))),
        "current_step": str(writer_state.get("current_step") or summary.get("writer_current_step") or ""),
        "completed_shard_count": completed,
        "planned_shard_count": planned,
        "remaining_shard_count": max(planned - completed, 0),
        "progress_age_minutes": round(_safe_float(writer_state.get("progress_age_minutes"), 0.0), 3),
        "cycle_age_minutes": round(_safe_float(writer_state.get("cycle_age_minutes"), 0.0), 3),
        "pending_after": _safe_int(effect.get("pending_after"), 0),
        "core_pending_after": _safe_int(effect.get("core_pending_after"), 0),
        "oldest_pending_age_after_seconds": round(_safe_float(effect.get("oldest_pending_age_after_seconds"), 0.0), 3),
        "recommended_actions": [str(item) for item in _as_list(writer.get("recommended_actions"))[:6]],
    }


def _with_training_writer_view(writer: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    drain = training.get("pretraining_drain_buffer") if isinstance(training.get("pretraining_drain_buffer"), dict) else {}
    drain_writer = drain.get("writer") if isinstance(drain.get("writer"), dict) else {}
    if not drain_writer:
        return writer
    merged = dict(writer)
    merged["raw_active"] = bool(writer.get("active", False))
    merged["training_gate_status"] = str(drain.get("status") or "")
    merged["active"] = bool(drain_writer.get("active", writer.get("active", False)))
    merged["running"] = bool(drain_writer.get("running", False))
    merged["current_step"] = str(drain_writer.get("current_step") or writer.get("current_step") or "")
    merged["completed_shard_count"] = _safe_int(drain_writer.get("completed_shard_count"), _safe_int(writer.get("completed_shard_count"), 0))
    merged["planned_shard_count"] = _safe_int(drain_writer.get("planned_shard_count"), _safe_int(writer.get("planned_shard_count"), 0))
    merged["remaining_shard_count"] = max(_safe_int(merged.get("planned_shard_count"), 0) - _safe_int(merged.get("completed_shard_count"), 0), 0)
    merged["progress_age_minutes"] = round(_safe_float(drain_writer.get("progress_age_minutes"), _safe_float(writer.get("progress_age_minutes"), 0.0)), 3)
    merged["cycle_age_minutes"] = round(_safe_float(drain_writer.get("cycle_age_minutes"), _safe_float(writer.get("cycle_age_minutes"), 0.0)), 3)
    if str(drain.get("status") or "") == "clear" and bool(drain.get("safe_to_launch_now", False)):
        merged["status"] = "clear"
    return merged


def _memory_snapshot(memory: dict[str, Any]) -> dict[str, Any]:
    classification = memory.get("classification") if isinstance(memory.get("classification"), dict) else {}
    reopen = memory.get("reopen_gate") if isinstance(memory.get("reopen_gate"), dict) else {}
    snapshot = memory.get("snapshot") if isinstance(memory.get("snapshot"), dict) else {}
    return {
        "status": _status(memory),
        "classification": str(classification.get("status") or ""),
        "decision": str(classification.get("decision") or ""),
        "safe_for_training": bool(reopen.get("safe_for_training", False)),
        "training_batch_cap": _safe_int(reopen.get("training_batch_cap"), 0),
        "consecutive_memory_clear_samples": _safe_int(reopen.get("consecutive_memory_clear_samples"), 0),
        "memory_clear_required_samples_for_training": _safe_int(reopen.get("memory_clear_required_samples_for_training"), 0),
        "swap_used_gb": round(_safe_float(snapshot.get("swap_used_gb"), 0.0), 3),
        "compressed_store_gb": round(_safe_float(snapshot.get("compressed_store_gb"), 0.0), 3),
        "compressed_pressure_gb": round(_safe_float(snapshot.get("compressed_pressure_gb"), 0.0), 3),
        "pages_throttled": _safe_int(snapshot.get("pages_throttled"), 0),
    }


def _runtime_snapshot(runtime: dict[str, Any]) -> dict[str, Any]:
    governor = runtime.get("runtime_saturation_governor_v2") if isinstance(runtime.get("runtime_saturation_governor_v2"), dict) else {}
    training_policy = governor.get("training_policy") if isinstance(governor.get("training_policy"), dict) else {}
    paper_policy = runtime.get("paper_execution_policy") if isinstance(runtime.get("paper_execution_policy"), dict) else {}
    return {
        "status": _status(runtime),
        "throttle_profile": str(runtime.get("throttle_profile") or ""),
        "host_saturation_score": round(_safe_float(runtime.get("host_saturation_score"), 0.0), 3),
        "compute_pressure_level": str(runtime.get("compute_pressure_level") or ""),
        "memory_pressure_level": str(runtime.get("memory_pressure_level") or ""),
        "training_policy": {
            "mode": str(training_policy.get("mode") or ""),
            "training_paused": bool(training_policy.get("training_paused", False)),
            "max_parallel_trainings": _safe_int(training_policy.get("max_parallel_trainings"), 0),
            "batch10_allowed": bool(training_policy.get("batch10_allowed", False)),
            "batch20_allowed": bool(training_policy.get("batch20_allowed", False)),
        },
        "paper_execution": {
            "allowed": bool(paper_policy.get("paper_execution_allowed", paper_policy.get("ok", False))),
            "paused": bool(paper_policy.get("pause_paper_execution", False)),
            "reason": str(paper_policy.get("reason") or ""),
            "stage": str(paper_policy.get("stage") or ""),
        },
    }


def _livefeed_snapshot(livefeed: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _status(livefeed),
        "alive": bool(livefeed.get("alive", False)),
        "source": str(livefeed.get("source") or ""),
        "heavy": _safe_int(livefeed.get("heavy"), 0),
        "idle_heartbeat_seconds": _safe_int(livefeed.get("idle_heartbeat_seconds"), 0),
        "pause_reason": str(livefeed.get("pause_reason") or ""),
        "last_line_utc": str(livefeed.get("last_line_utc") or ""),
    }


def _mac_watch_status(mac_watch: dict[str, Any]) -> str:
    explicit = _status(mac_watch)
    if explicit != "missing":
        return explicit
    if not mac_watch:
        return "missing"
    age = _age_seconds(mac_watch.get("timestamp_utc"))
    if age is not None and age <= _safe_float(mac_watch.get("max_alert_age_seconds"), 900.0):
        return "ready"
    if bool(mac_watch.get("imessage_enabled", False)) or bool(mac_watch.get("imessage_recipient_configured", False)):
        return "advisory"
    return "missing"


def _last_mac_sent(mac_watch: dict[str, Any]) -> str | None:
    sent_at = mac_watch.get("last_sent_at")
    if isinstance(sent_at, dict) and sent_at:
        values = [str(value) for value in sent_at.values() if str(value).strip()]
        return max(values) if values else None
    delivery = mac_watch.get("last_delivery") if isinstance(mac_watch.get("last_delivery"), dict) else {}
    if delivery:
        return str(mac_watch.get("timestamp_utc") or "") or None
    return None


def _last_mac_error(mac_watch: dict[str, Any]) -> str | None:
    delivery = mac_watch.get("last_delivery") if isinstance(mac_watch.get("last_delivery"), dict) else {}
    errors: list[str] = []
    for payload in delivery.values():
        if isinstance(payload, dict):
            stderr = str(payload.get("stderr") or "").strip()
            returncode = _safe_int(payload.get("returncode"), 0)
            if stderr:
                errors.append(stderr)
            elif returncode != 0:
                errors.append(f"returncode={returncode}")
    return "; ".join(errors) if errors else None


def _notification_snapshot(notifications: dict[str, Any], remote: dict[str, Any], mac_watch: dict[str, Any]) -> dict[str, Any]:
    return {
        "ladder_status": _status(notifications),
        "remote_status": _status(remote),
        "mac_watch_status": _mac_watch_status(mac_watch),
        "mac_watch_imessage_enabled": bool(mac_watch.get("imessage_enabled", False)),
        "mac_watch_imessage_recipient_configured": bool(mac_watch.get("imessage_recipient_configured", False)),
        "last_sent": notifications.get("last_sent") or remote.get("last_sent") or _last_mac_sent(mac_watch),
        "last_error": notifications.get("last_error") or remote.get("last_error") or _last_mac_error(mac_watch),
    }


def _safe_commands(
    training: dict[str, Any],
    writer: dict[str, Any],
    memory: dict[str, Any],
    runtime: dict[str, Any],
) -> list[list[str]]:
    commands: list[list[str]] = []
    train_command = _compact_command(training.get("recommended_retrain_command"))
    if train_command:
        commands.append(train_command)
    for cmd in _as_list(training.get("recommended_prep_commands")):
        compact = _compact_command(cmd)
        if compact:
            commands.append(compact)
    for cmd in (
        ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
        ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
        ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
        ["./scripts/ops/opsctl.sh", "codex-operator-bridge", "--json"],
    ):
        commands.append(cmd)
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for command in commands:
        key_parts = list(command)
        if key_parts and key_parts[0].endswith("opsctl.sh"):
            key_parts[0] = "opsctl.sh"
        key = tuple(key_parts)
        if key in seen:
            continue
        seen.add(key)
        unique.append(command)
    return unique[:10]


def _attention_packet(
    *,
    paper: dict[str, Any],
    training: dict[str, Any],
    writer: dict[str, Any],
    memory: dict[str, Any],
    runtime: dict[str, Any],
    livefeed: dict[str, Any],
    notifications: dict[str, Any],
) -> dict[str, Any]:
    needs: list[str] = []
    trade_notes: list[str] = []
    blockers: list[str] = []
    do_not_do = [
        "do_not_launch_training_when_training_launch_contract_launch_allowed_false",
        "do_not_start_parallel_sqlite_writers",
        "do_not_disable_paper_trade_lock_or_enable_live_execution_from_this_bridge",
        "do_not_treat_paper_pnl_as_broker_realized_cash_without_broker_confirmation",
    ]

    day = paper.get("day") if isinstance(paper.get("day"), dict) else {}
    if _safe_float(day.get("change_vs_previous_day"), 0.0) < 0:
        needs.append("explain_negative_paper_day_and_check_loss_causes")
        trade_notes.append("paper_day_negative")
    if _safe_int(day.get("executions"), 0) > 0:
        trade_notes.append("paper_execution_flow_active")

    if not bool(training.get("launch_allowed", False)):
        blockers.extend([str(item) for item in _as_list(training.get("launch_blockers"))])
        needs.append("watch_training_gate_and_only_launch_from_recommended_command")
    elif _safe_int(training.get("recommended_batch_size"), 0) >= 30:
        needs.append("stage_large_batch_training_now")
    elif _safe_int(training.get("recommended_batch_size"), 0) > 0:
        needs.append("stage_training_at_guarded_batch_size")

    if bool(writer.get("active", False)):
        needs.append("watch_single_writer_catchup_before_heavy_work")
    if not bool(memory.get("safe_for_training", False)):
        needs.append("wait_for_memory_reopen_samples_before_training")
    if not bool(livefeed.get("alive", False)):
        needs.append("refresh_livefeed_before_using_operator_view")
    if str(notifications.get("ladder_status") or "") not in {"ready", "advisory"}:
        needs.append("inspect_notification_ladder")
    if str(runtime.get("status") or "") in {"blocked", "critical", "degraded"}:
        needs.append("inspect_runtime_pressure_before_more_work")

    return {
        "needs_codex": list(dict.fromkeys(needs)),
        "trade_notes": list(dict.fromkeys(trade_notes)),
        "active_blockers": list(dict.fromkeys(blockers)),
        "safe_next_commands": _safe_commands(training, writer, memory, runtime),
        "do_not_do": do_not_do,
        "communication_contract": {
            "delivery_channel": "artifact_handoff",
            "primary_json": str(DEFAULT_OUT_PATH),
            "primary_markdown": str(DEFAULT_MARKDOWN_PATH),
            "style": "compact_delta_first_exact_commands",
            "proactive_delivery_to_codex": True,
        },
    }


def _overall_status(sections: dict[str, dict[str, Any]], attention: dict[str, Any]) -> str:
    blockers = attention.get("active_blockers") if isinstance(attention.get("active_blockers"), list) else []
    if any(str(item).strip() for item in blockers):
        return "advisory"
    statuses = {str(section.get("status") or "") for section in sections.values()}
    if "blocked" in statuses or "critical" in statuses:
        return "blocked"
    if "degraded" in statuses or "needs_work" in statuses:
        return "degraded"
    return "ready"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health = project_root / "governance" / "health"
    paper = _paper_trade_snapshot(_load_json(health / "paper_performance_latest.json"))
    training = _training_snapshot(_load_json(health / "training_runtime_control_latest.json"))
    writer = _with_training_writer_view(_writer_snapshot(_load_json(health / "writer_cycle_coordinator_latest.json")), training)
    memory = _memory_snapshot(_load_json(health / "memory_pressure_intelligence_latest.json"))
    runtime = _runtime_snapshot(_load_json(health / "runtime_throttle_control_latest.json"))
    livefeed = _livefeed_snapshot(_load_json(health / "livefeed_local_latest.json"))
    notifications = _notification_snapshot(
        _load_json(health / "notification_escalation_ladder_latest.json"),
        _load_json(health / "remote_alert_control_latest.json"),
        _load_json(health / "mac_notification_watch_state.json"),
    )
    sections = {
        "paper_trading": paper,
        "training": training,
        "writer": writer,
        "memory": memory,
        "runtime": runtime,
        "livefeed": livefeed,
        "notifications": notifications,
    }
    attention = _attention_packet(
        paper=paper,
        training=training,
        writer=writer,
        memory=memory,
        runtime=runtime,
        livefeed=livefeed,
        notifications=notifications,
    )
    summary_lines = [
        f"paper day pnl={paper['day']['ending_net_pnl_total']} change={paper['day']['change_vs_previous_day']} executions={paper['day']['executions']}",
        f"training launch_allowed={training['launch_allowed']} recommended_batch={training['recommended_batch_size']} blockers={','.join(training['launch_blockers']) or 'none'}",
        f"writer active={writer['active']} shards={writer['completed_shard_count']}/{writer['planned_shard_count']} pending={writer['pending_after']}",
        f"memory class={memory['classification']} safe_for_training={memory['safe_for_training']} clear_samples={memory['consecutive_memory_clear_samples']}/{memory['memory_clear_required_samples_for_training']}",
        f"livefeed status={livefeed['status']} alive={livefeed['alive']} source={livefeed['source']}",
    ]
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": _overall_status(sections, attention) in {"ready", "advisory"},
        "overall_status": _overall_status(sections, attention),
        "bridge_version": "codex_operator_bridge_v1",
        "summary_lines": summary_lines,
        "attention_packet": attention,
        "sections": sections,
        "source_files": {
            "paper_performance": str(health / "paper_performance_latest.json"),
            "training_runtime_control": str(health / "training_runtime_control_latest.json"),
            "writer_cycle_coordinator": str(health / "writer_cycle_coordinator_latest.json"),
            "memory_pressure_intelligence": str(health / "memory_pressure_intelligence_latest.json"),
            "runtime_throttle": str(health / "runtime_throttle_control_latest.json"),
            "livefeed": str(health / "livefeed_local_latest.json"),
            "notification_escalation_ladder": str(health / "notification_escalation_ladder_latest.json"),
        },
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    attention = payload.get("attention_packet") if isinstance(payload.get("attention_packet"), dict) else {}
    sections = payload.get("sections") if isinstance(payload.get("sections"), dict) else {}
    paper = sections.get("paper_trading") if isinstance(sections.get("paper_trading"), dict) else {}
    training = sections.get("training") if isinstance(sections.get("training"), dict) else {}
    writer = sections.get("writer") if isinstance(sections.get("writer"), dict) else {}
    memory = sections.get("memory") if isinstance(sections.get("memory"), dict) else {}
    lines = [
        "# Codex Operator Bridge",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        "",
        "## Digest",
        "",
    ]
    for line in payload.get("summary_lines") if isinstance(payload.get("summary_lines"), list) else []:
        lines.append(f"- {line}")
    lines.extend(["", "## Needs Codex", ""])
    needs = attention.get("needs_codex") if isinstance(attention.get("needs_codex"), list) else []
    if needs:
        for item in needs:
            lines.append(f"- `{item}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "## Safe Commands", ""])
    for command in attention.get("safe_next_commands") if isinstance(attention.get("safe_next_commands"), list) else []:
        if isinstance(command, list) and command:
            lines.append(f"- `{' '.join(str(part) for part in command)}`")
    lines.extend(
        [
            "",
            "## Paper Trading",
            "",
            f"- Day PnL: `{((paper.get('day') or {}).get('ending_net_pnl_total') if isinstance(paper.get('day'), dict) else '')}`",
            f"- Day change: `{((paper.get('day') or {}).get('change_vs_previous_day') if isinstance(paper.get('day'), dict) else '')}`",
            f"- Day executions: `{((paper.get('day') or {}).get('executions') if isinstance(paper.get('day'), dict) else '')}`",
            f"- Week executions: `{((paper.get('week') or {}).get('executions') if isinstance(paper.get('week'), dict) else '')}`",
            "",
            "## Training Gate",
            "",
            f"- Launch allowed: `{training.get('launch_allowed', '')}`",
            f"- Recommended batch: `{training.get('recommended_batch_size', '')}`",
            f"- Blockers: `{', '.join(training.get('launch_blockers') or []) if isinstance(training.get('launch_blockers'), list) else ''}`",
            "",
            "## Writer And Memory",
            "",
            f"- Writer shards: `{writer.get('completed_shard_count', '')}/{writer.get('planned_shard_count', '')}`",
            f"- Writer pending: `{writer.get('pending_after', '')}`",
            f"- Memory classification: `{memory.get('classification', '')}`",
            f"- Memory clear samples: `{memory.get('consecutive_memory_clear_samples', '')}/{memory.get('memory_clear_required_samples_for_training', '')}`",
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in attention.get("do_not_do") if isinstance(attention.get("do_not_do"), list) else []:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, out_path: Path, markdown_path: Path) -> None:
    _write_json(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Codex/operator communication bridge packet.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    write_outputs(
        payload,
        out_path=Path(args.out_file).expanduser(),
        markdown_path=Path(args.markdown_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "codex_operator_bridge "
            f"status={payload['overall_status']} "
            f"needs={len(payload['attention_packet']['needs_codex'])} "
            f"safe_commands={len(payload['attention_packet']['safe_next_commands'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
