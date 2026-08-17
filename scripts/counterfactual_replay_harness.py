#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "counterfactual_replay_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "counterfactual_replay_state.json"
SOURCE_FINGERPRINT_WINDOW_BYTES = 4096


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _execution_lane_paths(project_root: Path, stem: str) -> list[Path]:
    pattern_groups = [
        [
            str(project_root / "governance" / "execution_lanes" / f"{stem}_*.jsonl"),
            str(project_root / "governance" / "execution_lanes" / f"{stem}_*.jsonl.gz"),
        ],
        [
            str(project_root / "local_fallback_storage" / "governance" / "execution_lanes" / f"{stem}_*.jsonl"),
            str(project_root / "local_fallback_storage" / "governance" / "execution_lanes" / f"{stem}_*.jsonl.gz"),
        ],
        [
            str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes") / f"{stem}_*.jsonl"),
            str(Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/execution_lanes") / f"{stem}_*.jsonl.gz"),
        ],
    ]
    for patterns in pattern_groups:
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
        if paths:
            return _dedupe_paths(paths)
    return []


def _glob_source_paths(project_root: Path) -> list[Path]:
    bridge_paths = [
        Path(path)
        for path in sorted(
            glob.glob(str(project_root / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_*.jsonl"))
        )[-2:]
    ]
    if bridge_paths:
        return bridge_paths

    result_paths = _execution_lane_paths(project_root, "execution_results")[-4:]
    if result_paths:
        return result_paths

    return _execution_lane_paths(project_root, "execution_intents")[-4:]


def _source_kind(path: Path) -> str:
    name = path.name
    if name.startswith("execution_results_"):
        return "execution_result"
    if name.startswith("execution_intents_"):
        return "execution_intent"
    return "paper_bridge"


def _iter_jsonl_rows(path: Path, *, offset_bytes: int = 0) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == ".gz":
            if int(offset_bytes or 0) > 0:
                return [], int(offset_bytes)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            return rows, int(path.stat().st_size)

        with path.open("rb") as handle:
            handle.seek(max(int(offset_bytes), 0))
            for raw in handle:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows, int(handle.tell())
    except Exception:
        return [], max(int(offset_bytes), 0)


def _payload_context(row: dict[str, Any], source_kind: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if source_kind == "execution_result":
        intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
        return (decision or intent), intent, row
    if source_kind == "execution_intent":
        return row, row, row
    return row, {}, row


def _first_number(*values: Any, default: float) -> float:
    for value in values:
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _optional_number(*values: Any) -> float | None:
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            return number
    return None


def _paper_bridge_context(row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    evidence = row.get("order_intent_evidence") if isinstance(row.get("order_intent_evidence"), dict) else {}
    quote = evidence.get("quote_snapshot") if isinstance(evidence.get("quote_snapshot"), dict) else {}
    tradeability = _optional_number(
        row.get("tradeability_score"),
        row.get("tradeability_norm"),
        metadata.get("tradeability_score"),
        metadata.get("tradeability_norm"),
        quote.get("tradeability_norm"),
    )
    conflict = _optional_number(
        row.get("allocation_conflict_norm"),
        metadata.get("allocation_conflict_norm"),
    )
    merged["tradeability_score"] = tradeability
    merged["allocation_conflict_norm"] = conflict
    return merged


def _normalize_execution_result_row(row: dict[str, Any]) -> dict[str, Any] | None:
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    payload = decision or intent
    if not payload:
        return None
    mode = str(row.get("mode") or intent.get("target_mode") or payload.get("mode") or "").strip().lower()
    if mode != "paper":
        return None
    timestamp = (
        str(payload.get("timestamp_utc") or "").strip()
        or str(row.get("timestamp_utc") or "").strip()
        or str(intent.get("timestamp_utc") or "").strip()
        or str(row.get("intent_created_at") or "").strip()
    )
    if not timestamp:
        return None
    return {
        "timestamp_utc": timestamp,
        "symbol": str(payload.get("symbol") or intent.get("symbol") or "").upper(),
        "action": str(payload.get("action") or intent.get("action") or "").upper(),
        "quantity": _first_number(payload.get("quantity"), intent.get("quantity"), default=0.0),
        "model_score": _first_number(payload.get("model_score"), intent.get("model_score"), default=0.0),
        "threshold": _first_number(payload.get("threshold"), intent.get("threshold"), default=0.0),
        "strategy": str(payload.get("strategy") or intent.get("strategy") or ""),
        "realized_pnl": _first_number(payload.get("realized_pnl"), intent.get("realized_pnl"), default=0.0),
        "unrealized_pnl": _first_number(payload.get("unrealized_pnl"), intent.get("unrealized_pnl"), default=0.0),
        "mode": "paper",
    }


def _normalize_execution_intent_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("target_mode") or row.get("mode") or "").strip().lower() != "paper":
        return None
    timestamp = str(row.get("timestamp_utc") or "").strip()
    if not timestamp:
        return None
    return {
        "timestamp_utc": timestamp,
        "symbol": str(row.get("symbol") or "").upper(),
        "action": str(row.get("action") or "").upper(),
        "quantity": _first_number(row.get("quantity"), default=0.0),
        "model_score": _first_number(row.get("model_score"), default=0.0),
        "threshold": _first_number(row.get("threshold"), default=0.0),
        "strategy": str(row.get("strategy") or ""),
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "mode": "paper",
    }


def _merge_source_context(row: dict[str, Any], original: dict[str, Any], source_kind: str) -> dict[str, Any]:
    payload, intent, envelope = _payload_context(original, source_kind)
    merged = dict(row)
    metadata: dict[str, Any] = {}
    for candidate in (envelope.get("metadata"), intent.get("metadata"), payload.get("metadata")):
        if isinstance(candidate, dict):
            metadata.update(candidate)
    if metadata:
        merged["metadata"] = metadata
    profile = str(
        metadata.get("source_profile")
        or payload.get("profile")
        or intent.get("profile")
        or envelope.get("profile")
        or ""
    ).strip()
    if profile:
        merged["profile"] = profile
    merged["tradeability_score"] = _first_number(
        payload.get("tradeability_score"),
        payload.get("tradeability_norm"),
        intent.get("tradeability_score"),
        intent.get("tradeability_norm"),
        metadata.get("tradeability_score"),
        metadata.get("tradeability_norm"),
        envelope.get("tradeability_score"),
        envelope.get("tradeability_norm"),
        default=1.0,
    )
    merged["allocation_conflict_norm"] = _first_number(
        payload.get("allocation_conflict_norm"),
        intent.get("allocation_conflict_norm"),
        metadata.get("allocation_conflict_norm"),
        envelope.get("allocation_conflict_norm"),
        default=0.0,
    )
    return merged


def _normalize_source_rows(rows: list[dict[str, Any]], source_kind: str) -> list[dict[str, Any]]:
    if source_kind == "paper_bridge":
        return [_paper_bridge_context(row) for row in rows if isinstance(row, dict)]

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if source_kind == "execution_result":
            replay_row = _normalize_execution_result_row(row)
        elif source_kind == "execution_intent":
            replay_row = _normalize_execution_intent_row(row)
        else:
            replay_row = None
        if replay_row:
            normalized.append(_merge_source_context(replay_row, row, source_kind))
    return normalized


def _empty_runtime(max_rows: int) -> dict[str, Any]:
    return {
        "max_rows": int(max(max_rows, 1)),
        "latest_ts": "",
        "rolling_rows": [],
    }


def _state_to_runtime(state: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    runtime = _empty_runtime(max_rows)
    runtime["latest_ts"] = str(state.get("latest_ts") or "")
    rows = state.get("rolling_rows") if isinstance(state.get("rolling_rows"), list) else []
    runtime["rolling_rows"] = [dict(row) for row in rows if isinstance(row, dict)][-int(max(max_rows, 1)) :]
    return runtime


def _runtime_to_state(
    runtime: dict[str, Any],
    *,
    source_files: dict[str, dict[str, Any]],
    processing_mode: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in (runtime.get("rolling_rows") or []) if isinstance(row, dict)]
    return {
        "updated_at_utc": _utc_now(),
        "max_rows": int(runtime.get("max_rows", 0) or 0),
        "latest_ts": str(runtime.get("latest_ts") or ""),
        "processing_mode": str(processing_mode or "rebuild"),
        "source_files": source_files,
        "rolling_rows": rows[-int(max(runtime.get("max_rows", 1) or 1, 1)) :],
    }


def _can_incrementally_reuse(state: dict[str, Any], current_paths: list[Path], *, max_rows: int) -> bool:
    if int(state.get("max_rows", 0) or 0) != int(max(max_rows, 1)):
        return False
    tracked = state.get("source_files") if isinstance(state.get("source_files"), dict) else {}
    tracked_paths = {str(path) for path in tracked.keys()}
    current_path_set = {str(path) for path in current_paths}
    if tracked_paths != current_path_set:
        return False
    for path in current_paths:
        meta = tracked.get(str(path))
        if not isinstance(meta, dict):
            return False
        try:
            st = path.stat()
        except Exception:
            return False
        tracked_inode = int(meta.get("inode", 0) or 0)
        tracked_offset = int(meta.get("offset_bytes", 0) or 0)
        if tracked_inode and int(st.st_ino) != tracked_inode:
            return False
        if int(st.st_size) < tracked_offset:
            return False
        expected_fingerprint = str(meta.get("consumed_prefix_fingerprint") or "").strip()
        if not expected_fingerprint:
            return False
        if _consumed_prefix_fingerprint(path, tracked_offset) != expected_fingerprint:
            return False
    return True


def _consumed_prefix_fingerprint(path: Path, consumed_bytes: int) -> str:
    limit = min(max(int(consumed_bytes), 0), int(path.stat().st_size))
    digest = hashlib.sha256()
    digest.update(str(limit).encode("ascii"))
    if limit <= 0:
        return digest.hexdigest()

    window = min(SOURCE_FINGERPRINT_WINDOW_BYTES, limit)
    offsets = sorted(
        {
            0,
            max((limit // 4) - (window // 2), 0),
            max((limit // 2) - (window // 2), 0),
            max(((3 * limit) // 4) - (window // 2), 0),
            max(limit - window, 0),
        }
    )
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(min(window, limit - offset))
            digest.update(str(offset).encode("ascii"))
            digest.update(b"\0")
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(path: Path, offset_bytes: int, source_kind: str) -> dict[str, Any]:
    try:
        st = path.stat()
        consumed = min(max(int(offset_bytes), 0), int(st.st_size))
        return {
            "inode": int(st.st_ino),
            "offset_bytes": consumed,
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "source_kind": source_kind,
            "consumed_prefix_fingerprint": _consumed_prefix_fingerprint(path, consumed),
        }
    except Exception:
        return {
            "inode": 0,
            "offset_bytes": max(int(offset_bytes), 0),
            "size_bytes": 0,
            "mtime_ns": 0,
            "source_kind": source_kind,
            "consumed_prefix_fingerprint": "",
        }


def _update_runtime(runtime: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    buffer = runtime.get("rolling_rows") if isinstance(runtime.get("rolling_rows"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        buffer.append(dict(row))
        ts = str(row.get("timestamp_utc") or "").strip()
        if ts and ts > str(runtime.get("latest_ts") or ""):
            runtime["latest_ts"] = ts
    runtime["rolling_rows"] = buffer[-int(max(runtime.get("max_rows", 1) or 1, 1)) :]


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _row_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key, value in (
        ("decision", row.get("decision_id")),
        ("decision", metadata.get("decision_id")),
        ("message", row.get("message_id")),
    ):
        text = str(value or "").strip()
        if text:
            return f"{key}:{text}"
    return ""


def _dedupe_runtime_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    identity_to_index: dict[str, int] = {}
    duplicate_count = 0
    for row in rows:
        identity = _row_identity(row)
        if not identity:
            deduped.append(row)
            continue
        prior_index = identity_to_index.get(identity)
        if prior_index is None:
            identity_to_index[identity] = len(deduped)
            deduped.append(row)
            continue
        deduped[prior_index] = row
        duplicate_count += 1
    return deduped, duplicate_count


def _profile_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text = str(meta.get("source_profile") or row.get("profile") or "").strip().lower()
    return text or "default"


def _event_pnl(row: dict[str, Any]) -> tuple[float | None, str]:
    for key in (
        "post_cost_pnl_delta",
        "paper_profile_net_pnl_delta",
        "paper_ledger_net_pnl_delta",
        "paper_strategy_net_pnl_delta",
    ):
        value = _optional_number(row.get(key))
        if value is not None:
            return float(value), key

    realized = _optional_number(row.get("realized_pnl"), row.get("realized"))
    unrealized = _optional_number(row.get("unrealized_pnl"), row.get("unrealized"))
    if realized is not None or unrealized is not None:
        return float((realized or 0.0) + (unrealized or 0.0)), "event_realized_plus_unrealized"

    realized_total = _optional_number(row.get("realized_pnl_total"))
    unrealized_total = _optional_number(row.get("unrealized_pnl_total"))
    if realized_total is not None or unrealized_total is not None:
        return float((realized_total or 0.0) + (unrealized_total or 0.0)), "legacy_cumulative_snapshot"
    return None, "unattributed"


def _net_total(row: dict[str, Any]) -> float:
    value, _ = _event_pnl(row)
    return float(value or 0.0)


def _decision_margin(row: dict[str, Any]) -> float | None:
    model_score = _optional_number(row.get("model_score"))
    threshold = _optional_number(row.get("threshold"))
    if model_score is None or threshold is None:
        return None
    action = str(row.get("action") or "").strip().upper()
    if action in {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
        return float((1.0 - threshold) - model_score)
    if action in {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}:
        return float(model_score - threshold)
    return None


def _candidate_rows() -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for threshold_delta in (-0.03, -0.01, 0.0, 0.01, 0.03):
        for tradeability_floor in (0.0, 0.35, 0.5):
            for max_conflict in (1.0, 0.7, 0.45):
                out.append(
                    {
                        "threshold_delta": float(threshold_delta),
                        "tradeability_floor": float(tradeability_floor),
                        "max_conflict_norm": float(max_conflict),
                    }
                )
    return out


def build_counterfactual_report(
    project_root: Path,
    *,
    max_rows: int,
    state_file: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    state_path = (
        Path(state_file).resolve()
        if state_file is not None
        else project_root / "governance" / "health" / DEFAULT_STATE_PATH.name
    )
    current_paths = _glob_source_paths(project_root)
    state = _load_json(state_path)

    processing_mode = "rebuild"
    file_scan_counts = {"full_files": 0, "incremental_files": 0, "reused_files": 0}
    if _can_incrementally_reuse(state, current_paths, max_rows=max_rows):
        runtime = _state_to_runtime(state, max_rows=max_rows)
        tracked = state.get("source_files") if isinstance(state.get("source_files"), dict) else {}
        processing_mode = "incremental"
    else:
        runtime = _empty_runtime(max_rows)
        tracked = {}

    next_source_files: dict[str, dict[str, Any]] = {}
    for path in current_paths:
        meta = tracked.get(str(path)) if isinstance(tracked, dict) else None
        offset = int(meta.get("offset_bytes", 0) or 0) if isinstance(meta, dict) else 0
        rows, final_offset = _iter_jsonl_rows(path, offset_bytes=offset)
        source_kind = _source_kind(path)
        _update_runtime(runtime, _normalize_source_rows(rows, source_kind))
        if isinstance(meta, dict):
            if final_offset > offset:
                file_scan_counts["incremental_files"] += 1
            else:
                file_scan_counts["reused_files"] += 1
        else:
            file_scan_counts["full_files"] += 1
        next_source_files[str(path)] = _source_state(
            path,
            final_offset if final_offset > 0 else int(path.stat().st_size),
            source_kind,
        )

    raw_rows = [dict(row) for row in (runtime.get("rolling_rows") or []) if isinstance(row, dict)]
    rows, duplicate_rows_dropped = _dedupe_runtime_rows(raw_rows)
    runtime["rolling_rows"] = rows[-int(max(max_rows, 1)) :]
    rows = [dict(row) for row in runtime["rolling_rows"]]
    _write_json(
        state_path,
        _runtime_to_state(runtime, source_files=next_source_files, processing_mode=processing_mode),
    )

    candidates = _candidate_rows()
    profile_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for profile in sorted({ _profile_of(row) for row in rows }):
        profile_rows = [row for row in rows if _profile_of(row) == profile]
        if not profile_rows:
            continue
        for candidate in candidates:
            kept: list[dict[str, Any]] = []
            for row in profile_rows:
                decision_margin = _decision_margin(row)
                tradeability = _optional_number(row.get("tradeability_score"))
                conflict = _optional_number(row.get("allocation_conflict_norm"))
                if decision_margin is None or decision_margin < float(candidate["threshold_delta"]):
                    continue
                if (tradeability if tradeability is not None else 0.0) < float(candidate["tradeability_floor"]):
                    continue
                if (conflict if conflict is not None else 1.0) > float(candidate["max_conflict_norm"]):
                    continue
                kept.append(row)
            if not kept:
                continue
            attributed_outcomes = [_event_pnl(row) for row in kept]
            pnl_values = [float(value) for value, _ in attributed_outcomes if value is not None]
            outcome_source_counts: dict[str, int] = defaultdict(int)
            for _, source in attributed_outcomes:
                outcome_source_counts[source] += 1
            non_flat = [value for value in pnl_values if value != 0.0]
            win_rate = (sum(1 for value in non_flat if value > 0.0) / max(len(non_flat), 1)) if non_flat else None
            profile_scores[profile].append(
                {
                    **candidate,
                    "kept_count": int(len(kept)),
                    "attributed_count": int(len(pnl_values)),
                    "unattributed_count": int(len(kept) - len(pnl_values)),
                    "attribution_ratio": round(len(pnl_values) / max(len(kept), 1), 6),
                    "mean_net_pnl_total": round(sum(pnl_values) / max(len(pnl_values), 1), 6),
                    "aggregate_net_pnl_total": round(sum(pnl_values), 6),
                    "win_rate": (round(float(win_rate), 6) if win_rate is not None else None),
                    "outcome_source_counts": dict(sorted(outcome_source_counts.items())),
                }
            )

    top_candidates: list[dict[str, Any]] = []
    for profile, values in profile_scores.items():
        values.sort(
            key=lambda row: (
                float(row.get("aggregate_net_pnl_total", 0.0) or 0.0),
                -float(row.get("threshold_delta", 0.0) or 0.0),
                float(row.get("tradeability_floor", 0.0) or 0.0),
            ),
            reverse=True,
        )
        if values:
            top_candidates.append({"profile": profile, **values[0]})
    top_candidates.sort(key=lambda row: (float(row.get("aggregate_net_pnl_total", 0.0) or 0.0), row.get("profile", "")), reverse=True)

    return {
        "timestamp_utc": _utc_now(),
        "ok": True,
        "profiles_reviewed": sorted(profile_scores.keys()),
        "candidate_count": int(sum(len(rows) for rows in profile_scores.values())),
        "top_candidates": top_candidates[:12],
        "source_files": [str(path) for path in current_paths],
        "processing": {
            "mode": processing_mode,
            "state_file": str(state_path),
            "full_files": int(file_scan_counts["full_files"]),
            "incremental_files": int(file_scan_counts["incremental_files"]),
            "reused_files": int(file_scan_counts["reused_files"]),
            "row_buffer_size": int(len(rows)),
            "raw_row_buffer_size": int(len(raw_rows)),
            "duplicate_rows_dropped": int(duplicate_rows_dropped),
            "latest_event_timestamp_utc": str(runtime.get("latest_ts") or ""),
            "source_snapshots": [
                {"path": path, **dict(metadata)} for path, metadata in sorted(next_source_files.items())
            ],
            "decision_filter_mode": "action_aware_margin",
            "pnl_attribution_policy": "post_cost_event_delta_first",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast counterfactual replay summary over recent paper decisions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--out-file", default="")
    parser.add_argument("--state-file", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_counterfactual_report(
        project_root,
        max_rows=int(args.max_rows),
        state_file=Path(args.state_file).resolve() if args.state_file else None,
    )
    out_path = Path(args.out_file) if args.out_file else project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "counterfactual_replay "
            f"profiles={len(payload.get('profiles_reviewed') or [])} "
            f"candidates={int(payload.get('candidate_count', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
