"""Bounded native decision reader for the isolated SCHD rehearsal, never orders."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from core.decision_price_evidence import build_evidence, digest, number, timestamp
from core.schd_decision_rehearsal import advance, initial_state
from core.storage_router import inspect_storage_path

SOURCE_MODE = "shadow_dividend_equities"
SOURCE_STRATEGY = "grand_master_bot"
MAX_SCAN_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
SCAN_SECONDS = 8


def checked(path, root):
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_native_decision_route")
    return Path(path)


def read_latest(root, *, now, max_bytes=MAX_SCAN_BYTES, max_seconds=SCAN_SECONDS):
    """Inspect a fixed tail; a partial, unstable or malformed scan is not admission."""
    deadline = time.monotonic() + min(max_seconds, SCAN_SECONDS)
    budget = min(max_bytes, MAX_SCAN_BYTES)
    if budget <= 0 or max_seconds <= 0:
        raise ValueError("positive_native_scan_budget_required")
    selected = None
    selected_receipt = None
    candidates = {}
    scan = {
        "mode": SOURCE_MODE,
        "strategy": SOURCE_STRATEGY,
        "bytes_read": 0,
        "selected": False,
        "issues": [],
        "files": [],
        "selection_scope": "current_and_previous_UTC_day_bounded_tail_not_full_history",
    }
    for day in (now.date(), (now - timedelta(days=1)).date()):
        path = checked(
            root / "decisions" / SOURCE_MODE / f"trade_decisions_{day:%Y%m%d}.jsonl",
            root,
        )
        if not path.exists():
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("native_decision_regular_file_required")
            remaining = budget - scan["bytes_read"]
            start = max(0, before.st_size - remaining)
            handle.seek(start)
            if start:
                skipped = handle.readline(min(MAX_ROW_BYTES + 1, remaining))
                scan["bytes_read"] += len(skipped)
                if not skipped.endswith(b"\n"):
                    scan["issues"].append("native_partial_row_exceeds_bound")
                    break
            boundary = before.st_size
            while handle.tell() < boundary:
                if time.monotonic() >= deadline or scan["bytes_read"] >= budget:
                    scan["issues"].append("native_scan_budget_exhausted")
                    break
                offset = handle.tell()
                line = handle.readline(
                    min(
                        MAX_ROW_BYTES + 1,
                        boundary - offset,
                        budget - scan["bytes_read"],
                    )
                )
                scan["bytes_read"] += len(line)
                if not line.endswith(b"\n"):
                    scan["issues"].append("native_uncommitted_or_oversize_tail")
                    break
                if len(line) > MAX_ROW_BYTES:
                    scan["issues"].append("native_row_exceeds_bound")
                    break
                # Parse every row in the inspected region; malformed rows cannot
                # silently hide a newer veto behind a previous BUY.
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("object_required")
                except (ValueError, UnicodeError):
                    scan["issues"].append("native_malformed_row")
                    continue
                if (
                    row.get("symbol") != "SCHD"
                    or row.get("strategy") != SOURCE_STRATEGY
                ):
                    continue
                try:
                    when = timestamp(row["timestamp_utc"])
                    identity = str(row["decision_id"])
                    if not identity or when > now:
                        raise ValueError("future_or_missing_identity")
                    fingerprint = digest(row)
                except (KeyError, ValueError, TypeError):
                    scan["issues"].append("native_invalid_decision_identity_or_time")
                    continue
                if identity in candidates and candidates[identity] != fingerprint:
                    scan["issues"].append("native_conflicting_decision_id")
                candidates[identity] = fingerprint
                if (
                    selected is not None
                    and when == timestamp(selected["timestamp_utc"])
                    and identity != selected["decision_id"]
                ):
                    scan["issues"].append("native_ambiguous_same_time_decisions")
                if selected is None or when >= timestamp(selected["timestamp_utc"]):
                    selected = row
                    selected_receipt = {
                        "source_path": str(path),
                        "byte_offset": offset,
                        "row_bytes": len(line),
                        "raw_row_sha256": hashlib.sha256(line).hexdigest(),
                        "row_sha256": fingerprint,
                        "device": before.st_dev,
                        "inode": before.st_ino,
                        "file_size_at_scan": boundary,
                        "observed_at_utc": now.isoformat(),
                    }
            after = os.fstat(handle.fileno())
            identity = checked(path, root).lstat()
            if (identity.st_dev, identity.st_ino) != (
                before.st_dev,
                before.st_ino,
            ) or after.st_size < before.st_size:
                scan["issues"].append("native_source_changed_during_scan")
            if (
                after.st_mtime_ns != before.st_mtime_ns
                and after.st_size == before.st_size
            ):
                scan["issues"].append("native_source_rewritten_during_scan")
            scan["files"].append(
                {
                    "path": str(path),
                    "scan_start_offset": start,
                    "scan_end_offset": handle.tell(),
                    "captured_eof": boundary,
                    "full_history_scanned": start == 0,
                }
            )
        # A selected current-day record is sufficient; never search older files
        # for a more convenient action. Timestamp freshness is evaluated later.
        if selected is not None or scan["issues"] or scan["bytes_read"] >= budget:
            break
    if selected is None:
        scan["issues"].append("no_schd_grand_master_decision_in_bounded_scan")
    scan["issues"] = list(dict.fromkeys(scan["issues"]))
    scan["selected"] = selected is not None
    return selected, selected_receipt, scan


def native_packet(row, receipt, scan, market, candidate, *, now):
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    when = timestamp(row["timestamp_utc"])
    blockers = list(scan["issues"])
    candidate_id = str(row.get("production_candidate_id") or "")
    binding = row.get("candidate_binding", {})
    if (
        not candidate_id
        or candidate_id != candidate.get("candidate_id")
        or md.get("production_candidate_id") != candidate_id
        or binding.get("candidate_bound") is not True
        or binding.get("expected_candidate_id") != candidate_id
        or binding.get("observed_candidate_id") != candidate_id
        or not candidate.get("overall_sha256")
        or md.get("production_candidate_receipt_sha256")
        != candidate.get("overall_sha256")
    ):
        blockers.append("native_candidate_binding_mismatch")
    try:
        cutoff = timestamp(binding["candidate_scope_cutoff_utc"])
        if when < cutoff:
            blockers.append("native_decision_before_candidate_scope")
    except (KeyError, TypeError, ValueError):
        blockers.append("native_candidate_cutoff_missing")
    if (
        md.get("source_profile") != "dividend"
        or md.get("layer") != "grand_master"
        or md.get("mode") not in {"shadow", "paper"}
        or row.get("strategy") != SOURCE_STRATEGY
    ):
        blockers.append("native_strategy_scope_mismatch")
    if (
        row.get("schema_valid") is not True
        or row.get("source_broker") != "schwab"
        or row.get("source_provider") != "schwab"
        or row.get("source_quality_label") != "broker_native"
    ):
        blockers.append("native_source_provenance_incomplete")
    try:
        if number(row.get("source_quality_score")) < 0.9:
            blockers.append("native_source_quality_below_floor")
    except ValueError:
        blockers.append("native_source_quality_missing")
    source = market.get("source", {})
    context = md.get("schd_candle_context", {})
    try:
        if (
            context.get("state") != "observed_context_not_claimed_model_input"
            or context.get("capture_sha256") != digest(market)
            or not timestamp(source["fetch_started_at_utc"])
            <= timestamp(context["observed_at_utc"])
            <= when
            or not 0 <= (when - timestamp(source["fetch_started_at_utc"])).total_seconds() <= 300
        ):
            raise ValueError("invalid_context_binding")
    except (KeyError, TypeError, ValueError):
        blockers.append("native_original_candle_context_binding_missing_or_changed")
    if source.get("provider") != "schwab" or source.get("symbol") != "SCHD":
        blockers.append("native_schwab_candle_source_missing")
    reconstructed = True
    try:
        captured = timestamp(source["fetch_started_at_utc"])
        reconstructed = captured > when
        if captured > now:
            blockers.append("native_market_capture_in_future")
        if reconstructed:
            blockers.append("native_candles_retrieved_after_decision_context_only")
    except (KeyError, TypeError, ValueError):
        blockers.append("native_market_capture_time_missing")
    candles, excluded = {}, 0
    for name in ("1m", "5m", "1d"):
        candles[name] = []
        for candle in market.get("candles", {}).get(name, []):
            if timestamp(candle["end_utc"]) <= when:
                candles[name].append(candle)
            else:
                excluded += 1
    quote = {}
    # A local collector clock is useful diagnostic evidence but cannot be
    # relabeled as the exchange/provider quote timestamp.
    provider_time = features.get("provider_quote_ts_utc")
    local_time = features.get("snapshot_ts_utc")
    time_basis = (
        "provider_quote_time"
        if provider_time is not None
        else "local_snapshot_time_not_provider_time"
    )
    if provider_time is None:
        blockers.append("native_provider_quote_timestamp_missing")
    if features.get("provider_quote_realtime_norm") != 1:
        blockers.append("native_original_realtime_quote_unverified")
    try:
        observed = datetime.fromtimestamp(
            number(
                provider_time if provider_time is not None else local_time,
                positive=True,
            ),
            timezone.utc,
        )
        quote = {
            "symbol": "SCHD",
            "provider": "schwab",
            "source_quality_label": row.get("source_quality_label"),
            "snapshot_id": md.get("snapshot_id"),
            "timestamp_utc": observed.isoformat(),
            "timestamp_basis": time_basis,
            **{
                target: number(features[key], positive=True)
                for target, key in (
                    ("last", "provider_quote_last_price"),
                    ("bid", "bid_price"),
                    ("ask", "ask_price"),
                    ("bid_size", "bid_size"),
                    ("ask_size", "ask_size"),
                )
            },
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        blockers.append("native_recorded_quote_fields_incomplete")
    decision = {
        key: deepcopy(row.get(key))
        for key in (
            "timestamp_utc",
            "symbol",
            "decision_id",
            "strategy",
            "action",
            "decision",
            "model_score",
            "threshold",
            "reasons",
            "gates",
            "features",
            "feature_compaction_contract",
        )
    }
    decision["metadata"] = {
        key: deepcopy(md.get(key))
        for key in ("snapshot_id", "invalidation_conditions", "decision_horizon")
        if key in md
    }
    packet = {
        "schema_version": 1,
        "symbol": "SCHD",
        "evidence_kind": "recorded",
        "candidate_id": candidate_id or "unbound_native_decision",
        "price_basis": source.get("price_adjustment_basis", "unknown"),
        "decision": decision,
        "quote": quote,
        "candles": candles,
        "native_validation": {
            "blockers": list(dict.fromkeys(blockers)),
            "receipt": receipt,
            "scan": scan,
            "candidate_binding": binding,
            "quote_timestamp_basis": time_basis,
            "post_decision_candles_excluded": excluded,
            "chart_context_reconstructed_after_decision": reconstructed,
            "original_decision_action": row.get("action"),
            "action_changed": False,
            "gates_changed": False,
            "chart_source": source,
        },
    }
    return packet


def advance_native(state, packet, market, *, now, source_digest):
    if state is None:
        state = initial_state(
            candidate_id=packet["candidate_id"],
            evidence_kind="recorded",
            implementation_sha256=source_digest,
        )
    # Bind every observation, including fills, to the original candidate/source.
    if (
        state["candidate_id"] != packet["candidate_id"]
        or state["implementation_sha256"] != source_digest
    ):
        raise ValueError("native_candidate_or_source_changed_review_required")
    request = packet
    if state.get("pending"):
        pending = state["pending"]
        decision = packet["decision"]
        newer_veto = timestamp(decision["timestamp_utc"]) > timestamp(
            pending["submitted_at_utc"]
        ) and (
            decision["action"] != pending["side"]
            or decision["decision"] != "EXECUTE"
            or not decision["gates"]
            or not all(v is True for v in decision["gates"].values())
        )
        if newer_veto:
            result = deepcopy(state)
            result.update(
                phase="cancelled",
                pending=None,
                wait_reason="new_native_decision_veto_no_retry",
            )
        else:
            # Missing provider proof must not be repaired with a newly fetched
            # quote masquerading as the quote originally seen by the bot.
            request = {
                "candidate_id": packet["candidate_id"],
                "evidence_kind": "recorded",
            }
            fill_blockers = [
                b
                for b in packet["native_validation"]["blockers"]
                if b != "native_candles_retrieved_after_decision_context_only"
            ]
            if not fill_blockers:
                request["fill_quote"] = market.get("quote")
            result = advance(
                state, request, now=now, implementation_sha256=source_digest
            )
        result["last_report"] = build_evidence(packet, now=now)
    else:
        result = advance(state, request, now=now, implementation_sha256=source_digest)
    result["native_connection"] = {
        "mode": "explicit_bounded_artifact_observation_not_background_stream",
        "source_mode": SOURCE_MODE,
        "strategy": SOURCE_STRATEGY,
        "observed_at_utc": now.isoformat(),
        "decision_timestamp_utc": packet["decision"]["timestamp_utc"],
        "decision_id": packet["decision"]["decision_id"],
        "decision_action": packet["decision"]["action"],
        "row_sha256": packet["native_validation"]["receipt"]["row_sha256"],
    }
    return result
