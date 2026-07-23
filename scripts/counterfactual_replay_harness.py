#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "counterfactual_replay_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "counterfactual_replay_state.json"


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
        return [dict(row) for row in rows if isinstance(row, dict)]

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
    return True


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


def _profile_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text = str(meta.get("source_profile") or row.get("profile") or "").strip().lower()
    return text or "default"


def _net_total(row: dict[str, Any]) -> float:
    realized = _safe_float(row.get("realized_pnl"), _safe_float(row.get("realized"), _safe_float(row.get("realized_pnl_total"))))
    unrealized = _safe_float(
        row.get("unrealized_pnl"),
        _safe_float(row.get("unrealized"), _safe_float(row.get("unrealized_pnl_total"))),
    )
    return float(realized + unrealized)


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
    state_path = Path(state_file).resolve() if state_file is not None else DEFAULT_STATE_PATH
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
        try:
            st = path.stat()
            next_source_files[str(path)] = {
                "inode": int(st.st_ino),
                "offset_bytes": int(final_offset if final_offset > 0 else st.st_size),
                "mtime": float(st.st_mtime),
                "source_kind": source_kind,
            }
        except Exception:
            next_source_files[str(path)] = {
                "inode": 0,
                "offset_bytes": int(final_offset),
                "mtime": 0.0,
                "source_kind": source_kind,
            }

    _write_json(
        state_path,
        _runtime_to_state(runtime, source_files=next_source_files, processing_mode=processing_mode),
    )

    rows = [dict(row) for row in (runtime.get("rolling_rows") or []) if isinstance(row, dict)]
    candidates = _candidate_rows()
    profile_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for profile in sorted({ _profile_of(row) for row in rows }):
        profile_rows = [row for row in rows if _profile_of(row) == profile]
        if not profile_rows:
            continue
        for candidate in candidates:
            kept = []
            for row in profile_rows:
                model_score = _safe_float(row.get("model_score"), 0.0)
                threshold = _safe_float(row.get("threshold"), 0.0)
                tradeability = _safe_float(row.get("tradeability_score"), 1.0)
                conflict = _safe_float(row.get("allocation_conflict_norm"), 0.0)
                if model_score < threshold + float(candidate["threshold_delta"]):
                    continue
                if tradeability < float(candidate["tradeability_floor"]):
                    continue
                if conflict > float(candidate["max_conflict_norm"]):
                    continue
                kept.append(row)
            if not kept:
                continue
            pnl_values = [_net_total(row) for row in kept]
            non_flat = [value for value in pnl_values if value != 0.0]
            win_rate = (sum(1 for value in non_flat if value > 0.0) / max(len(non_flat), 1)) if non_flat else None
            profile_scores[profile].append(
                {
                    **candidate,
                    "kept_count": int(len(kept)),
                    "mean_net_pnl_total": round(sum(pnl_values) / max(len(pnl_values), 1), 6),
                    "aggregate_net_pnl_total": round(sum(pnl_values), 6),
                    "win_rate": (round(float(win_rate), 6) if win_rate is not None else None),
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
            "latest_event_timestamp_utc": str(runtime.get("latest_ts") or ""),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast counterfactual replay summary over recent paper decisions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_counterfactual_report(
        Path(args.project_root).resolve(),
        max_rows=int(args.max_rows),
        state_file=Path(args.state_file).resolve(),
    )
    out_path = Path(args.out_file)
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
