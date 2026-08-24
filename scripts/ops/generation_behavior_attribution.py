#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
    from scripts.ops.production_excellence_control import verify_candidate_event_chain
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload
    from .production_excellence_control import verify_candidate_event_chain


DEFAULT_EVENT_PATH = Path("governance/evidence/production_candidate_events.jsonl")
DEFAULT_SOAK_PATH = Path(
    "governance/health/continuous_soak_integrity_control_latest.json"
)
DEFAULT_PERFORMANCE_PATH = Path("governance/health/paper_performance_latest.json")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path(
    "governance/research/generation_behavior_attribution_latest.json"
)
DEFAULT_MARKDOWN_PATH = Path(
    "exports/reports/operator/generation_behavior_attribution_latest.md"
)
ACCEPTED_EVENT_TYPES = frozenset(
    {"candidate_change_accepted", "candidate_chain_recovery_anchor"}
)
GENERATION_PATTERN = re.compile(r"-g(?P<generation>\d+)$")
DAY_PATTERN = re.compile(r"master_control_(?P<day>20\d{6})\.jsonl(?:\.gz)?$")
DIRECTIONAL_ACTIONS = frozenset({"BUY", "SELL"})


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _read_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain = verify_candidate_event_chain(path)
    if not bool(chain.get("ok", False)):
        return [], chain
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return [], {**chain, "ok": False, "errors": ["candidate_event_log_unreadable"]}
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("event_type") or "") not in ACCEPTED_EVENT_TYPES:
            continue
        if _utc(row.get("timestamp_utc")) is None:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: _utc(row.get("timestamp_utc"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    return rows, chain


def _candidate_windows(
    events: Sequence[Mapping[str, Any]], *, now: datetime
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_candidate: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(events):
        event = dict(raw)
        candidate_id = str(event.get("candidate_id") or "").strip()
        started = _utc(event.get("timestamp_utc"))
        ended = (
            _utc(events[index + 1].get("timestamp_utc"))
            if index + 1 < len(events)
            else now
        )
        if not candidate_id or started is None or ended is None or ended < started:
            continue
        row = {
            "candidate_id": candidate_id,
            "generation": _safe_int(event.get("generation"), 0),
            "event_type": str(event.get("event_type") or ""),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_wall_clock_hours": round(
                (ended - started).total_seconds() / 3600.0, 6
            ),
            "change_reason": str(event.get("change_reason") or ""),
            "changed_scopes": sorted(
                {
                    str(item)
                    for item in _as_list(event.get("changed_scopes"))
                    if str(item)
                }
            ),
            "git_head": str(event.get("git_head") or ""),
            "candidate_sha256": str(event.get("overall_sha256") or ""),
        }
        rows.append(row)
        by_candidate[candidate_id] = row
    return rows, by_candidate


def _generation_from_candidate_id(candidate_id: str) -> int:
    match = GENERATION_PATTERN.search(str(candidate_id or ""))
    return int(match.group("generation")) if match else 0


def _row_candidate_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    metadata = _as_dict(row.get("metadata"))
    binding = _as_dict(row.get("candidate_binding"))
    candidate_id = str(
        row.get("production_candidate_id")
        or metadata.get("production_candidate_id")
        or binding.get("observed_candidate_id")
        or ""
    ).strip()
    generation = _safe_int(
        metadata.get("production_candidate_generation"),
        _safe_int(
            binding.get("generation"), _generation_from_candidate_id(candidate_id)
        ),
    )
    return candidate_id, generation


def _source_roots(project_root: Path) -> list[tuple[int, Path, str]]:
    rows: list[tuple[int, Path, str]] = [
        (0, project_root / "governance", "project_hot_governance"),
        (
            1,
            project_root / "local_fallback_storage" / "governance",
            "project_local_fallback",
        ),
    ]
    configured = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    if configured:
        external = Path(configured).expanduser()
    else:
        mount = Path(
            os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")
        ).expanduser()
        project_dir = str(
            os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")
            or "schwab_trading_bot"
        ).strip()
        external = mount / project_dir
    rows.extend(
        [
            (
                2,
                external / "local_fallback_storage" / "governance",
                "external_local_fallback",
            ),
            (3, external / "data" / "stale_stage", "external_stale_stage"),
            (
                4,
                external / "cold_archive" / "deep_cold" / "stale_stage",
                "external_deep_cold",
            ),
        ]
    )
    deduped: list[tuple[int, Path, str]] = []
    seen: set[str] = set()
    for priority, path, label in rows:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((priority, path, label))
    return deduped


def _decision_files(
    project_root: Path, *, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    start_day = start.strftime("%Y%m%d")
    end_day = end.strftime("%Y%m%d")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for priority, source_root, label in _source_roots(project_root):
        if not source_root.is_dir():
            continue
        try:
            candidates = source_root.rglob("master_control_*.jsonl*")
        except OSError:
            continue
        for path in candidates:
            if not path.is_file() or "shadow" not in "/".join(path.parts).lower():
                continue
            match = DAY_PATTERN.search(path.name)
            if not match:
                continue
            day = match.group("day")
            if day < start_day or day > end_day:
                continue
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            files.append(
                {
                    "path": path,
                    "priority": priority,
                    "source_class": label,
                    "day": day,
                }
            )
    files.sort(key=lambda row: (int(row["priority"]), str(row["path"])))
    return files


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(row, dict):
                    yield row
    except (OSError, EOFError, gzip.BadGzipFile):
        return


def _row_identity(row: Mapping[str, Any]) -> str:
    message_id = str(row.get("message_id") or "").strip()
    if message_id:
        return f"message:{message_id}"
    candidate_id, generation = _row_candidate_identity(row)
    material = {
        "candidate_id": candidate_id,
        "generation": generation,
        "timestamp_utc": str(row.get("timestamp_utc") or ""),
        "profile": str(row.get("shadow_profile") or row.get("profile") or ""),
        "symbol": str(row.get("symbol") or ""),
        "snapshot_id": str(row.get("snapshot_id") or ""),
        "action": str(row.get("action") or row.get("master_action") or ""),
    }
    return f"content:{_canonical_hash(material)}"


def _new_aggregate() -> dict[str, Any]:
    return {
        "decision_count": 0,
        "identity_bound_decision_count": 0,
        "legacy_window_associated_decision_count": 0,
        "timestamps": [],
        "minute_buckets": set(),
        "profiles": set(),
        "symbols": set(),
        "source_files": set(),
        "source_classes": Counter(),
        "actions": Counter(),
        "intent_actions": Counter(),
        "dispositions": Counter(),
        "blocking_stages": Counter(),
        "guard_categories": Counter(),
        "guard_block_count": 0,
        "route_receipt_present_count": 0,
        "route_receipt_valid_count": 0,
        "route_quality_count": 0,
        "route_quality_sum": 0.0,
        "freshness_present_count": 0,
        "freshness_pass_count": 0,
        "latency_slo_present_count": 0,
        "latency_slo_pass_count": 0,
        "latency_ms_count": 0,
        "latency_ms_sum": 0.0,
        "utility_count": 0,
        "utility_sum": 0.0,
        "source_quality_count": 0,
        "source_quality_sum": 0.0,
        "lane_kill_switch_count": 0,
        "global_kill_switch_count": 0,
    }


def _mean(total: float, count: int) -> float | None:
    return round(total / count, 6) if count > 0 else None


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def _counter_rows(counter: Counter[str], *, limit: int = 12) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count} for value, count in counter.most_common(limit)
    ]


def _finalize_aggregate(raw: Mapping[str, Any]) -> dict[str, Any]:
    decision_count = _safe_int(raw.get("decision_count"), 0)
    actions = Counter(raw.get("actions") or {})
    intents = Counter(raw.get("intent_actions") or {})
    timestamps = [
        value for value in raw.get("timestamps", []) if isinstance(value, datetime)
    ]
    directional_intents = sum(intents[action] for action in DIRECTIONAL_ACTIONS)
    final_directional = sum(actions[action] for action in DIRECTIONAL_ACTIONS)
    identity_bound_count = _safe_int(raw.get("identity_bound_decision_count"), 0)
    legacy_count = _safe_int(raw.get("legacy_window_associated_decision_count"), 0)
    attribution_tier = (
        "identity_bound"
        if identity_bound_count and not legacy_count
        else (
            "legacy_window_association"
            if legacy_count and not identity_bound_count
            else (
                "mixed_identity_and_legacy_window_association"
                if identity_bound_count and legacy_count
                else "no_observations"
            )
        )
    )
    return {
        "decision_count": decision_count,
        "identity_bound_decision_count": identity_bound_count,
        "legacy_window_associated_decision_count": legacy_count,
        "attribution_tier": attribution_tier,
        "first_decision_utc": min(timestamps).isoformat() if timestamps else "",
        "last_decision_utc": max(timestamps).isoformat() if timestamps else "",
        "observed_decision_days": len(
            {value.date().isoformat() for value in timestamps}
        ),
        "decision_observation_hours": round(
            len(raw.get("minute_buckets", set())) / 60.0, 6
        ),
        "profile_count": len(raw.get("profiles", set())),
        "profiles": sorted(raw.get("profiles", set())),
        "symbol_count": len(raw.get("symbols", set())),
        "source_file_count": len(raw.get("source_files", set())),
        "source_class_counts": dict(
            sorted(Counter(raw.get("source_classes") or {}).items())
        ),
        "action_counts": dict(sorted(actions.items())),
        "intent_action_counts": dict(sorted(intents.items())),
        "directional_intent_rate": _rate(directional_intents, decision_count),
        "final_directional_rate": _rate(final_directional, decision_count),
        "intent_conversion_rate": _rate(final_directional, directional_intents),
        "guard_block_rate": _rate(
            _safe_int(raw.get("guard_block_count"), 0), decision_count
        ),
        "no_edge_hold_rate": _rate(
            Counter(raw.get("dispositions") or {}).get("no_edge_hold", 0),
            decision_count,
        ),
        "top_dispositions": _counter_rows(Counter(raw.get("dispositions") or {})),
        "top_blocking_stages": _counter_rows(Counter(raw.get("blocking_stages") or {})),
        "top_guard_categories": _counter_rows(
            Counter(raw.get("guard_categories") or {})
        ),
        "ingestion_route_receipt_valid_rate": _rate(
            _safe_int(raw.get("route_receipt_valid_count"), 0),
            _safe_int(raw.get("route_receipt_present_count"), 0),
        ),
        "ingestion_route_quality_mean": _mean(
            _safe_float(raw.get("route_quality_sum"), 0.0),
            _safe_int(raw.get("route_quality_count"), 0),
        ),
        "feature_freshness_pass_rate": _rate(
            _safe_int(raw.get("freshness_pass_count"), 0),
            _safe_int(raw.get("freshness_present_count"), 0),
        ),
        "latency_slo_pass_rate": _rate(
            _safe_int(raw.get("latency_slo_pass_count"), 0),
            _safe_int(raw.get("latency_slo_present_count"), 0),
        ),
        "master_latency_ms_mean": _mean(
            _safe_float(raw.get("latency_ms_sum"), 0.0),
            _safe_int(raw.get("latency_ms_count"), 0),
        ),
        "decision_utility_mean": _mean(
            _safe_float(raw.get("utility_sum"), 0.0),
            _safe_int(raw.get("utility_count"), 0),
        ),
        "source_quality_mean": _mean(
            _safe_float(raw.get("source_quality_sum"), 0.0),
            _safe_int(raw.get("source_quality_count"), 0),
        ),
        "lane_kill_switch_rate": _rate(
            _safe_int(raw.get("lane_kill_switch_count"), 0), decision_count
        ),
        "global_kill_switch_rate": _rate(
            _safe_int(raw.get("global_kill_switch_count"), 0), decision_count
        ),
    }


def _scan_decisions(
    files: Sequence[Mapping[str, Any]],
    *,
    windows: Mapping[str, Mapping[str, Any]],
    allow_legacy_window_association: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = defaultdict(_new_aggregate)
    seen: set[str] = set()
    counters = Counter()
    timeline = sorted(
        (
            started,
            ended,
            str(window.get("candidate_id") or ""),
            _safe_int(window.get("generation"), 0),
        )
        for window in windows.values()
        if (started := _utc(window.get("started_utc"))) is not None
        and (ended := _utc(window.get("ended_utc"))) is not None
        and started <= ended
    )
    timeline_starts = [row[0] for row in timeline]
    profile_aggregates: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for file_row in files:
        path = Path(file_row["path"])
        source_class = str(file_row.get("source_class") or "unknown")
        counters["files_opened"] += 1
        for row in _iter_jsonl(path):
            counters["records_read"] += 1
            if str(
                row.get("channel") or "decision"
            ).lower() != "decision" and not row.get("master_action"):
                counters["non_decision_records_skipped"] += 1
                continue
            timestamp = _utc(row.get("timestamp_utc"))
            if timestamp is None:
                counters["timestamp_missing_or_invalid_records"] += 1
                continue
            candidate_id, generation = _row_candidate_identity(row)
            window = _as_dict(windows.get(candidate_id))
            attribution_tier = "identity_bound"
            if not candidate_id or not window:
                if candidate_id or not allow_legacy_window_association:
                    counters["unbound_or_unknown_candidate_records"] += 1
                    continue
                timeline_index = bisect.bisect_right(timeline_starts, timestamp) - 1
                if timeline_index < 0:
                    counters["unbound_or_unknown_candidate_records"] += 1
                    continue
                (
                    inferred_started,
                    inferred_ended,
                    inferred_candidate_id,
                    inferred_generation,
                ) = timeline[timeline_index]
                if not (inferred_started <= timestamp < inferred_ended):
                    counters["unbound_or_unknown_candidate_records"] += 1
                    continue
                candidate_id = inferred_candidate_id
                generation = inferred_generation
                window = _as_dict(windows.get(candidate_id))
                attribution_tier = "legacy_window_associated"
            elif generation != _safe_int(window.get("generation"), 0):
                counters["candidate_generation_mismatch_records"] += 1
                continue
            started = _utc(window.get("started_utc"))
            ended = _utc(window.get("ended_utc"))
            if started is None or ended is None or not (started <= timestamp < ended):
                counters["outside_candidate_window_records"] += 1
                continue
            identity = _row_identity(row)
            if identity in seen:
                counters["duplicate_records_suppressed"] += 1
                continue
            seen.add(identity)
            counters["records_accepted"] += 1
            counters[f"{attribution_tier}_records_accepted"] += 1
            aggregate = aggregates[candidate_id]
            aggregate["decision_count"] += 1
            aggregate[f"{attribution_tier}_decision_count"] += 1
            aggregate["timestamps"].append(timestamp)
            aggregate["minute_buckets"].add(
                timestamp.replace(second=0, microsecond=0).isoformat()
            )
            profile = (
                str(row.get("shadow_profile") or row.get("profile") or "default")
                .strip()
                .lower()
                or "default"
            )
            symbol = str(row.get("symbol") or "").strip().upper()
            aggregate["profiles"].add(profile)
            if symbol:
                aggregate["symbols"].add(symbol)
            aggregate["source_files"].add(str(path))
            aggregate["source_classes"][source_class] += 1
            action = (
                str(row.get("action") or row.get("master_action") or "UNKNOWN")
                .strip()
                .upper()
                or "UNKNOWN"
            )
            intent = (
                str(row.get("master_intent_action") or action).strip().upper()
                or "UNKNOWN"
            )
            disposition = (
                str(row.get("decision_disposition") or "unknown").strip().lower()
                or "unknown"
            )
            stage = (
                str(row.get("decision_blocking_stage") or "none").strip().lower()
                or "none"
            )
            aggregate["actions"][action] += 1
            aggregate["intent_actions"][intent] += 1
            aggregate["dispositions"][disposition] += 1
            aggregate["blocking_stages"][stage] += 1
            for category in _as_list(row.get("decision_guard_categories")):
                if str(category):
                    aggregate["guard_categories"][str(category)] += 1
            guard_blocked = bool(
                row.get("master_guard_blocked_intent", False)
                or row.get("decision_guard_reasons")
                or row.get("decision_guard_categories")
            )
            aggregate["guard_block_count"] += int(guard_blocked)
            route_receipt = row.get(
                "institutional_decision_flow_ingestion_route_receipt_valid"
            )
            if route_receipt is not None:
                aggregate["route_receipt_present_count"] += 1
                aggregate["route_receipt_valid_count"] += int(bool(route_receipt))
            route_quality = row.get(
                "institutional_decision_flow_ingestion_route_quality_norm"
            )
            if route_quality is not None:
                aggregate["route_quality_count"] += 1
                aggregate["route_quality_sum"] += _safe_float(route_quality, 0.0)
            freshness = _as_dict(row.get("feature_freshness"))
            if freshness:
                aggregate["freshness_present_count"] += 1
                aggregate["freshness_pass_count"] += int(
                    bool(freshness.get("ok", False))
                )
            slo = _as_dict(row.get("master_latency_slo"))
            if slo:
                aggregate["latency_slo_present_count"] += 1
                aggregate["latency_slo_pass_count"] += int(bool(slo.get("ok", False)))
                if slo.get("elapsed_ms") is not None:
                    aggregate["latency_ms_count"] += 1
                    aggregate["latency_ms_sum"] += _safe_float(
                        slo.get("elapsed_ms"), 0.0
                    )
            utility = row.get("institutional_decision_flow_utility_norm")
            if utility is not None:
                aggregate["utility_count"] += 1
                aggregate["utility_sum"] += _safe_float(utility, 0.0)
            quality = row.get("source_quality_score")
            if quality is not None:
                aggregate["source_quality_count"] += 1
                aggregate["source_quality_sum"] += _safe_float(quality, 0.0)
            breakers = _as_dict(row.get("circuit_breakers"))
            aggregate["lane_kill_switch_count"] += int(
                bool(breakers.get("lane_kill_switch_active", False))
            )
            aggregate["global_kill_switch_count"] += int(
                bool(breakers.get("kill_switch_active", False))
            )
            profile_key = f"{candidate_id}:{profile}"
            profile_aggregates[profile_key]["actions"][action] += 1
            profile_aggregates[profile_key]["dispositions"][disposition] += 1

    finalized = {
        candidate_id: _finalize_aggregate(raw)
        for candidate_id, raw in aggregates.items()
    }
    for candidate_id, row in finalized.items():
        profile_rows: list[dict[str, Any]] = []
        for profile in row.get("profiles", []):
            key = f"{candidate_id}:{profile}"
            actions = profile_aggregates[key]["actions"]
            dispositions = profile_aggregates[key]["dispositions"]
            count = sum(actions.values())
            profile_rows.append(
                {
                    "profile": profile,
                    "decision_count": count,
                    "directional_action_rate": _rate(
                        sum(actions[action] for action in DIRECTIONAL_ACTIONS), count
                    ),
                    "no_edge_hold_rate": _rate(
                        dispositions.get("no_edge_hold", 0), count
                    ),
                    "action_counts": dict(sorted(actions.items())),
                }
            )
        row["profile_behavior"] = sorted(
            profile_rows,
            key=lambda item: (
                -_safe_int(item.get("decision_count"), 0),
                str(item.get("profile") or ""),
            ),
        )
    return finalized, dict(sorted(counters.items()))


def _paper_flows(performance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contract = _as_dict(performance.get("developmental_generation_flows"))
    return {
        str(row.get("candidate_id") or ""): dict(row)
        for row in _as_list(contract.get("generation_flows"))
        if isinstance(row, Mapping) and str(row.get("candidate_id") or "")
    }


def _paper_summary(flow: Mapping[str, Any]) -> dict[str, Any]:
    sample_count = _safe_int(flow.get("sample_count"), 0)
    post_cost = _safe_float(flow.get("post_cost_pnl_delta_total"), 0.0)
    return {
        "sample_count": sample_count,
        "observed_days": _safe_int(flow.get("observed_days"), 0),
        "post_cost_pnl_delta_total": round(post_cost, 6) if flow else None,
        "post_cost_pnl_per_sample": (
            round(post_cost / sample_count, 6) if sample_count else None
        ),
        "execution_cost_total": (
            round(_safe_float(flow.get("execution_cost_total"), 0.0), 6)
            if flow
            else None
        ),
        "generation_identity_consistent": bool(
            flow.get("candidate_generation_consistent", False)
        ),
        "developmental_only": True,
        "promotion_grade_eligible": False,
    }


def _generation_rows(
    windows: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    flows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        candidate_id = str(window.get("candidate_id") or "")
        behavior = dict(
            decisions.get(candidate_id) or _finalize_aggregate(_new_aggregate())
        )
        wall_hours = _safe_float(window.get("elapsed_wall_clock_hours"), 0.0)
        decision_count = _safe_int(behavior.get("decision_count"), 0)
        behavior["decisions_per_wall_clock_hour"] = (
            round(decision_count / wall_hours, 6) if wall_hours > 0.0 else None
        )
        observation_hours = _safe_float(behavior.get("decision_observation_hours"), 0.0)
        behavior["decisions_per_observation_hour"] = (
            round(decision_count / observation_hours, 6)
            if observation_hours > 0.0
            else None
        )
        attribution_tier = str(behavior.get("attribution_tier") or "no_observations")
        paper_sample_count = _safe_int(
            _as_dict(flows.get(candidate_id)).get("sample_count"), 0
        )
        if decision_count > 0:
            behavior_label = (
                "identity_bound_behavior"
                if attribution_tier == "identity_bound"
                else (
                    "legacy_window_associated_behavior"
                    if attribution_tier == "legacy_window_association"
                    else "mixed_identity_and_legacy_window_associated_behavior"
                )
            )
        else:
            behavior_label = "no_behavior_observations"
        rows.append(
            {
                **dict(window),
                "behavior": behavior,
                "paper_outcomes": _paper_summary(_as_dict(flows.get(candidate_id))),
                "evidence_status": (
                    f"{behavior_label}_and_candidate_bound_paper_outcomes"
                    if decision_count > 0 and paper_sample_count > 0
                    else (
                        behavior_label
                        if decision_count > 0
                        else (
                            "paper_outcomes_only"
                            if paper_sample_count > 0
                            else "no_identity_bound_observations"
                        )
                    )
                ),
                "counts_toward_cumulative_segmented_soak": True,
                "counts_toward_current_clean_720_hour_window": False,
                "association_is_causal_proof": False,
            }
        )
    return rows


COMPARISON_METRICS = (
    ("decision_count", "behavior"),
    ("decision_observation_hours", "behavior"),
    ("decisions_per_wall_clock_hour", "behavior"),
    ("decisions_per_observation_hour", "behavior"),
    ("profile_count", "behavior"),
    ("symbol_count", "behavior"),
    ("directional_intent_rate", "behavior"),
    ("final_directional_rate", "behavior"),
    ("intent_conversion_rate", "behavior"),
    ("guard_block_rate", "behavior"),
    ("no_edge_hold_rate", "behavior"),
    ("ingestion_route_receipt_valid_rate", "behavior"),
    ("ingestion_route_quality_mean", "behavior"),
    ("feature_freshness_pass_rate", "behavior"),
    ("latency_slo_pass_rate", "behavior"),
    ("master_latency_ms_mean", "behavior"),
    ("decision_utility_mean", "behavior"),
    ("source_quality_mean", "behavior"),
    ("lane_kill_switch_rate", "behavior"),
    ("global_kill_switch_rate", "behavior"),
    ("sample_count", "paper_outcomes"),
    ("post_cost_pnl_delta_total", "paper_outcomes"),
    ("post_cost_pnl_per_sample", "paper_outcomes"),
)


def _metric_delta(before: Any, after: Any) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "status": "unavailable",
            "before": before,
            "after": after,
            "absolute_delta": None,
            "relative_delta": None,
        }
    left = _safe_float(before, math.nan)
    right = _safe_float(after, math.nan)
    if not math.isfinite(left) or not math.isfinite(right):
        return {
            "status": "unavailable",
            "before": before,
            "after": after,
            "absolute_delta": None,
            "relative_delta": None,
        }
    delta = right - left
    return {
        "status": "available",
        "before": round(left, 6),
        "after": round(right, 6),
        "absolute_delta": round(delta, 6),
        "relative_delta": round(delta / abs(left), 6) if abs(left) > 1e-12 else None,
    }


def _profile_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left = {
        str(row.get("profile") or ""): row
        for row in _as_list(_as_dict(before.get("behavior")).get("profile_behavior"))
        if isinstance(row, Mapping) and str(row.get("profile") or "")
    }
    right = {
        str(row.get("profile") or ""): row
        for row in _as_list(_as_dict(after.get("behavior")).get("profile_behavior"))
        if isinstance(row, Mapping) and str(row.get("profile") or "")
    }
    rows: list[dict[str, Any]] = []
    for profile in sorted(set(left) | set(right)):
        before_row = _as_dict(left.get(profile))
        after_row = _as_dict(right.get(profile))
        rows.append(
            {
                "profile": profile,
                "present_before": bool(before_row),
                "present_after": bool(after_row),
                "decision_count": _metric_delta(
                    before_row.get("decision_count") if before_row else None,
                    after_row.get("decision_count") if after_row else None,
                ),
                "directional_action_rate": _metric_delta(
                    before_row.get("directional_action_rate") if before_row else None,
                    after_row.get("directional_action_rate") if after_row else None,
                ),
                "no_edge_hold_rate": _metric_delta(
                    before_row.get("no_edge_hold_rate") if before_row else None,
                    after_row.get("no_edge_hold_rate") if after_row else None,
                ),
            }
        )
    return rows


def _comparison(
    generation_rows: Sequence[Mapping[str, Any]],
    *,
    from_generation: int,
    to_generation: int,
    minimum_decisions: int,
    minimum_observation_hours: float,
) -> dict[str, Any]:
    by_generation = {
        _safe_int(row.get("generation"), 0): dict(row) for row in generation_rows
    }
    before = _as_dict(by_generation.get(from_generation))
    after = _as_dict(by_generation.get(to_generation))
    blockers: list[str] = []
    if not before:
        blockers.append("from_generation_missing")
    if not after:
        blockers.append("to_generation_missing")
    before_behavior = _as_dict(before.get("behavior"))
    after_behavior = _as_dict(after.get("behavior"))
    for label, row in (("from", before_behavior), ("to", after_behavior)):
        if _safe_int(row.get("decision_count"), 0) < minimum_decisions:
            blockers.append(f"{label}_generation_decision_floor_pending")
        if (
            _safe_float(row.get("decision_observation_hours"), 0.0)
            < minimum_observation_hours
        ):
            blockers.append(f"{label}_generation_observation_time_floor_pending")
    common_profiles = sorted(
        set(before_behavior.get("profiles") or []).intersection(
            set(after_behavior.get("profiles") or [])
        )
    )
    if not common_profiles:
        blockers.append("no_common_profile_coverage")
    change_events = [
        {
            "generation": _safe_int(row.get("generation"), 0),
            "candidate_id": str(row.get("candidate_id") or ""),
            "started_utc": str(row.get("started_utc") or ""),
            "change_reason": str(row.get("change_reason") or ""),
            "changed_scopes": list(row.get("changed_scopes") or []),
        }
        for row in generation_rows
        if from_generation < _safe_int(row.get("generation"), 0) <= to_generation
    ]
    scope_counts = Counter(
        scope for row in change_events for scope in _as_list(row.get("changed_scopes"))
    )
    metrics = {}
    for metric, section in COMPARISON_METRICS:
        metrics[metric] = _metric_delta(
            _as_dict(before.get(section)).get(metric) if before else None,
            _as_dict(after.get(section)).get(metric) if after else None,
        )
    paper_ready = bool(
        _safe_int(_as_dict(before.get("paper_outcomes")).get("sample_count"), 0) > 0
        and _safe_int(_as_dict(after.get("paper_outcomes")).get("sample_count"), 0) > 0
    )
    legacy_window_association_involved = bool(
        _safe_int(before_behavior.get("legacy_window_associated_decision_count"), 0)
        or _safe_int(after_behavior.get("legacy_window_associated_decision_count"), 0)
    )
    identity_bound_behavior_comparison_ready = bool(
        not blockers
        and _safe_int(before_behavior.get("identity_bound_decision_count"), 0)
        >= minimum_decisions
        and _safe_int(after_behavior.get("identity_bound_decision_count"), 0)
        >= minimum_decisions
    )
    status = (
        "insufficient_behavior_evidence"
        if blockers
        else (
            "behavior_comparison_ready_with_legacy_window_association"
            if legacy_window_association_involved
            else "identity_bound_behavior_comparison_ready"
        )
    )
    return {
        "from_generation": from_generation,
        "to_generation": to_generation,
        "from_candidate_id": str(before.get("candidate_id") or ""),
        "to_candidate_id": str(after.get("candidate_id") or ""),
        "status": status,
        "behavior_comparison_ready": not blockers,
        "identity_bound_behavior_comparison_ready": identity_bound_behavior_comparison_ready,
        "legacy_window_association_involved": legacy_window_association_involved,
        "economic_comparison_ready": paper_ready,
        "blockers": sorted(set(blockers)),
        "evidence_limitations": (
            [
                "legacy_unstamped_records_are_associated_only_by_immutable_candidate_time_windows",
                "legacy_window_association_is_descriptive_and_never_promotion_grade",
            ]
            if legacy_window_association_involved
            else []
        ),
        "minimum_decisions_per_generation": minimum_decisions,
        "minimum_observation_hours_per_generation": minimum_observation_hours,
        "common_profiles": common_profiles,
        "metric_deltas": metrics,
        "profile_deltas": _profile_delta(before, after) if before and after else [],
        "intervening_accepted_change_count": len(change_events),
        "intervening_changed_scope_counts": dict(sorted(scope_counts.items())),
        "intervening_change_events": change_events,
        "interpretation": {
            "association_only": True,
            "causal_claim_allowed": False,
            "confounded_by_market_regime_and_intervening_changes": True,
            "missing_evidence_is_zero": False,
            "better_trade_frequency_alone_proves_alpha": False,
            "post_cost_outcomes_required_for_economic_claims": True,
        },
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    from_generation: int = 65,
    to_generation: int = 99,
    last_days: int = 21,
    minimum_decisions: int = 20,
    minimum_observation_hours: float = 0.25,
    allow_legacy_window_association: bool = True,
    source_files: Sequence[Path] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    now = _utc(generated_at_utc) or datetime.now(timezone.utc)
    event_path = root / DEFAULT_EVENT_PATH
    events, chain = _read_events(event_path)
    windows, by_candidate = _candidate_windows(events, now=now)
    earliest = min(
        (
            _utc(row.get("started_utc"))
            for row in windows
            if _utc(row.get("started_utc"))
        ),
        default=now,
    )
    scan_start = max(earliest, now - timedelta(days=max(last_days, 1)))
    if source_files is None:
        files = _decision_files(root, start=scan_start, end=now)
    else:
        files = [
            {
                "path": Path(path),
                "priority": 0,
                "source_class": "explicit_test_or_operator_source",
                "day": "",
            }
            for path in source_files
        ]
    decisions, scan = _scan_decisions(
        files,
        windows=by_candidate,
        allow_legacy_window_association=allow_legacy_window_association,
    )
    performance = load_json(root / DEFAULT_PERFORMANCE_PATH)
    flows = _paper_flows(performance)
    generation_rows = _generation_rows(windows, decisions, flows)
    comparison = _comparison(
        generation_rows,
        from_generation=from_generation,
        to_generation=to_generation,
        minimum_decisions=max(minimum_decisions, 1),
        minimum_observation_hours=max(minimum_observation_hours, 0.0),
    )
    soak = load_json(root / DEFAULT_SOAK_PATH)
    candidate = load_json(root / DEFAULT_CANDIDATE_PATH)
    chain_valid = bool(chain.get("ok", False) and events)
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": chain_valid,
        "overall_status": (
            comparison.get("status") if chain_valid else "candidate_event_chain_blocked"
        ),
        "candidate_event_chain": {
            "valid": chain_valid,
            "event_count": _safe_int(chain.get("event_count"), 0),
            "chain_head": str(chain.get("chain_head") or ""),
            "errors": list(chain.get("errors") or []),
            "path": str(event_path),
        },
        "current_candidate": {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "generation": _safe_int(candidate.get("generation"), 0),
            "accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        },
        "cumulative_soak_context": {
            "status": str(soak.get("overall_status") or "missing"),
            "control_grade": str(soak.get("control_grade") or ""),
            "main_soak_counting_mode": str(soak.get("main_soak_counting_mode") or ""),
            "main_soak_elapsed_hours": soak.get("main_soak_elapsed_hours"),
            "main_soak_active_runtime_evidence_hours": soak.get(
                "main_soak_active_runtime_evidence_hours"
            ),
            "main_soak_planned_maintenance_excluded_hours": soak.get(
                "main_soak_planned_maintenance_excluded_hours"
            ),
            "main_soak_includes_pre_reset_time": bool(
                soak.get("main_soak_includes_pre_reset_time", False)
            ),
            "clean_window_started_utc": str(soak.get("clean_window_started_utc") or ""),
            "clean_window_elapsed_hours": soak.get("clean_window_elapsed_hours"),
            "historical_segment_count": _safe_int(
                _as_dict(soak.get("historical_soak_evidence")).get("segment_count"), 0
            ),
            "cumulative_history_preserved": True,
            "historical_segments_grade_current_candidate": False,
            "historical_segments_earn_current_clean_window_credit": False,
        },
        "scan_window": {
            "started_utc": scan_start.isoformat(),
            "ended_utc": now.isoformat(),
            "requested_last_days": max(last_days, 1),
            "source_file_count": len(files),
            "source_class_counts": dict(
                sorted(
                    Counter(
                        str(row.get("source_class") or "unknown") for row in files
                    ).items()
                )
            ),
            "counters": scan,
            "deduplication_key": "message_id_else_candidate_timestamp_profile_symbol_snapshot_action_hash",
            "legacy_window_association_enabled": allow_legacy_window_association,
        },
        "comparison": comparison,
        "generation_rows": generation_rows,
        "policy": {
            "cumulative_soak_is_segmented_exposure_context": True,
            "candidate_forward_identity_and_time_binding_required": True,
            "candidate_event_timeline_is_runtime_heartbeat_proof": False,
            "decision_observation_hours_are_runtime_uptime_proof": False,
            "historical_relabeling_allowed": False,
            "legacy_window_association_is_historical_relabeling": False,
            "legacy_window_association_never_promotion_grade": True,
            "association_is_causation": False,
            "profitability_guaranteed": False,
            "automatic_threshold_change_allowed": False,
            "automatic_strategy_promotion_allowed": False,
            "automatic_allocation_allowed": False,
            "paper_order_authority": False,
            "live_execution_authority": False,
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comparison = _as_dict(payload.get("comparison"))
    soak = _as_dict(payload.get("cumulative_soak_context"))
    lines = [
        "# Generation Behavior Attribution",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Comparison: `G{comparison.get('from_generation', 0)}` to `G{comparison.get('to_generation', 0)}`",
        "",
        "## Cumulative Soak Context",
        "",
        f"- Segmented wall-clock history: `{soak.get('main_soak_elapsed_hours', 'n/a')}` hours",
        f"- Active runtime evidence: `{soak.get('main_soak_active_runtime_evidence_hours', 'n/a')}` hours",
        f"- Current clean candidate window: `{soak.get('clean_window_elapsed_hours', 'n/a')}` hours",
        "- Historical segments remain developmental context and do not earn current-candidate promotion credit.",
        "",
        "## Comparison",
        "",
        f"- Behavior comparison ready: `{comparison.get('behavior_comparison_ready', False)}`",
        f"- Identity-bound comparison ready: `{comparison.get('identity_bound_behavior_comparison_ready', False)}`",
        f"- Legacy window association involved: `{comparison.get('legacy_window_association_involved', False)}`",
        f"- Economic comparison ready: `{comparison.get('economic_comparison_ready', False)}`",
        f"- Common profiles: `{len(comparison.get('common_profiles') or [])}`",
        f"- Intervening accepted changes: `{comparison.get('intervening_accepted_change_count', 0)}`",
    ]
    blockers = list(comparison.get("blockers") or [])
    if blockers:
        lines.append(f"- Evidence blockers: `{', '.join(blockers)}`")
    limitations = list(comparison.get("evidence_limitations") or [])
    if limitations:
        lines.append(f"- Evidence limitations: `{', '.join(limitations)}`")
    lines.extend(
        [
            "",
            "| Metric | Before | After | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, row in _as_dict(comparison.get("metric_deltas")).items():
        delta = _as_dict(row)
        lines.append(
            f"| `{metric}` | {delta.get('before')} | {delta.get('after')} | {delta.get('absolute_delta')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "This report measures candidate-associated behavior. Unstamped legacy rows may be associated only by immutable candidate time windows and never count as promotion-grade evidence. The report does not establish causation, convert missing evidence to zero, grant promotion, alter thresholds, create trades, allocate capital, or authorize live execution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate-bound behavior between accepted production generations."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--from-generation", type=int, default=65)
    parser.add_argument("--to-generation", type=int, default=99)
    parser.add_argument("--last-days", type=int, default=21)
    parser.add_argument("--minimum-decisions", type=int, default=20)
    parser.add_argument("--minimum-observation-hours", type=float, default=0.25)
    parser.add_argument(
        "--no-legacy-window-association",
        action="store_true",
        help="Exclude unstamped legacy decisions instead of associating them descriptively by candidate time window.",
    )
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    out_path = Path(args.out_file).expanduser()
    markdown_path = Path(args.markdown_file).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    payload = build_payload(
        root,
        from_generation=args.from_generation,
        to_generation=args.to_generation,
        last_days=args.last_days,
        minimum_decisions=args.minimum_decisions,
        minimum_observation_hours=args.minimum_observation_hours,
        allow_legacy_window_association=not args.no_legacy_window_association,
    )
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        comparison = _as_dict(payload.get("comparison"))
        print(
            "generation_behavior_attribution "
            f"status={payload.get('overall_status', '')} "
            f"from=G{comparison.get('from_generation', 0)} "
            f"to=G{comparison.get('to_generation', 0)} "
            f"behavior_ready={comparison.get('behavior_comparison_ready', False)} "
            f"economic_ready={comparison.get('economic_comparison_ready', False)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
