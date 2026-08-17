#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_OUT = Path("governance/health/market_replay_fill_capture_latest.json")
DEFAULT_INBOX_FILE = Path("exports/independent_fill_inbox/market_replay_fills_current.jsonl")
SCHEMA_VERSION = 2
DEFAULT_MAX_BYTES_PER_OBSERVATION_FILE = 8 * 1024 * 1024
DEFAULT_UNMATCHED_RESCAN_GRACE_SECONDS = 3600.0
BUY_ACTIONS = {"BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE"}
SELL_ACTIONS = {"SELL", "SELL_TO_OPEN", "SELL_TO_CLOSE"}
SUPPORTED_ACTIONS = BUY_ACTIONS | SELL_ACTIONS


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _resolve(project_root: Path, path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else project_root / path


def _candidate_binding(project_root: Path) -> tuple[datetime | None, dict[str, Any]]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    windows = state.get("scope_windows_started_utc") if isinstance(state.get("scope_windows_started_utc"), dict) else {}
    starts = [parse_iso_utc(windows.get(scope)) for scope in ("execution", "data", "dependencies")]
    cutoff = max((value for value in starts if value is not None), default=None)
    candidate_id = str(state.get("candidate_id") or "").strip()
    return cutoff, {
        "candidate_id": candidate_id,
        "generation": int(_safe_float(state.get("generation"), 0.0)),
        "cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
        "bound": bool(candidate_id and cutoff is not None),
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_rows(
    paths: Iterable[Path],
    *,
    tail_bytes_per_file: int = 0,
) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        if path.suffix != ".gz" and int(tail_bytes_per_file) > 0:
            try:
                size = path.stat().st_size
                start = max(size - int(tail_bytes_per_file), 0)
                handle = path.open("rb")
            except OSError:
                continue
            with handle:
                handle.seek(start)
                if start > 0:
                    handle.readline()
                for line_number, raw in enumerate(handle, start=1):
                    text = raw.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield path, line_number, row
            continue
        try:
            handle = _open_text(path)
        except OSError:
            continue
        with handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield path, line_number, row


def _path_day(path: Path) -> str:
    match = re.search(r"(?:^|_)(20\d{6})(?:\D|$)", path.name)
    return str(match.group(1)) if match else ""


def _candidate_paths(
    project_root: Path,
    patterns: Iterable[str],
    *,
    max_files: int,
    earliest_day: str = "",
    latest_day: str = "",
) -> list[Path]:
    rows: dict[str, Path] = {}
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if not path.is_file():
                continue
            path_day = _path_day(path)
            if path_day and earliest_day and path_day < earliest_day:
                continue
            if path_day and latest_day and path_day > latest_day:
                continue
            rows[str(path.resolve(strict=False))] = path
    ranked = sorted(
        rows.values(),
        key=lambda path: (path.stat().st_mtime_ns if path.exists() else 0, str(path)),
        reverse=True,
    )
    return sorted(ranked[: max(int(max_files), 1)])


def _row_id(row: dict[str, Any], _path: Path, _line_number: int, *, namespace: str) -> str:
    for key in ("message_id", "decision_id", "external_fill_id", "execution_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    material = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{namespace}:{material}".encode("utf-8")).hexdigest()[:32]


def _order_rows(
    project_root: Path,
    *,
    cutoff: datetime,
    max_files: int,
    max_orders: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = _candidate_paths(
        project_root,
        (
            "exports/paper_broker_bridge/paper/paper_bridge_orders_*.jsonl",
            "exports/paper_broker_bridge/paper/paper_bridge_orders_*.jsonl.gz",
        ),
        max_files=max_files,
    )
    rows: list[dict[str, Any]] = []
    for path, line_number, row in _iter_rows(paths):
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        action = str(row.get("action") or "").strip().upper()
        if timestamp is None or timestamp < cutoff:
            continue
        if str(row.get("status") or "").strip().upper() != "PAPER_EXECUTED":
            continue
        if action not in SUPPORTED_ACTIONS:
            continue
        if _safe_float(row.get("quantity")) <= 0.0:
            continue
        reference_price = _safe_float(row.get("reference_price"), _safe_float(row.get("mark_price")))
        expected_fill_price = _safe_float(row.get("expected_fill_price"))
        if reference_price <= 0.0 or expected_fill_price <= 0.0:
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        rows.append(
            {
                "timestamp": timestamp,
                "timestamp_utc": timestamp.isoformat(),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "action": action,
                "quantity": _safe_float(row.get("quantity")),
                "reference_price": reference_price,
                "expected_fill_price": expected_fill_price,
                "expected_slippage_bps": _safe_float(row.get("expected_slippage_bps")),
                "source_profile": str(metadata.get("source_profile") or row.get("paper_profile") or "paper").strip().lower(),
                "source_broker": str(row.get("source_broker") or "").strip().lower(),
                "source_provider": str(row.get("source_provider") or "").strip().lower(),
                "source_venue": str(row.get("source_venue") or "").strip().lower(),
                "order_id": _row_id(row, path, line_number, namespace="paper_order"),
                "snapshot_id": str(metadata.get("snapshot_id") or "").strip(),
                "source_path": str(path),
                "source_line": line_number,
            }
        )
    rows.sort(key=lambda row: (row["timestamp"], row["order_id"]))
    if len(rows) > max(int(max_orders), 1):
        rows = rows[-max(int(max_orders), 1) :]
    return rows, [str(path) for path in paths]


def _observation_rows(
    project_root: Path,
    *,
    cutoff: datetime,
    through: datetime,
    symbols: set[str],
    max_files: int,
    max_rows: int,
    max_bytes_per_file: int,
    minimum_source_quality: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], int]:
    paths = _candidate_paths(
        project_root,
        (
            "decision_explanations/**/decision_explanations_*.jsonl",
            "decision_explanations/**/decision_explanations_*.jsonl.gz",
        ),
        max_files=max_files,
        earliest_day=cutoff.strftime("%Y%m%d"),
        latest_day=through.strftime("%Y%m%d"),
    )
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted = 0
    for path, line_number, row in _iter_rows(paths, tail_bytes_per_file=max_bytes_per_file):
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        if timestamp is None or timestamp < cutoff or timestamp > through:
            continue
        if row.get("schema_valid") is False:
            continue
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        last_price = _safe_float(features.get("last_price"))
        spread_bps = max(_safe_float(features.get("spread_bps")), 0.0)
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol not in symbols:
            continue
        quality_label = str(row.get("source_quality_label") or "").strip().lower()
        quality_score = _safe_float(row.get("source_quality_score"))
        if not symbol or last_price <= 0.0:
            continue
        if quality_label != "broker_native" or quality_score < max(float(minimum_source_quality), 0.0):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        by_symbol[symbol].append(
            {
                "timestamp": timestamp,
                "timestamp_utc": timestamp.isoformat(),
                "last_price": last_price,
                "spread_bps": spread_bps,
                "source_quality_label": quality_label,
                "source_quality_score": quality_score,
                "source_broker": str(row.get("source_broker") or "").strip().lower(),
                "source_provider": str(row.get("source_provider") or "").strip().lower(),
                "source_venue": str(row.get("source_venue") or "").strip().lower(),
                "observation_id": _row_id(row, path, line_number, namespace="market_observation"),
                "snapshot_id": str(metadata.get("snapshot_id") or "").strip(),
                "source_path": str(path),
                "source_line": line_number,
            }
        )
        accepted += 1
        if accepted >= max(int(max_rows), 1):
            break
    for rows in by_symbol.values():
        rows.sort(key=lambda row: (row["timestamp"], row["observation_id"]))
    return dict(by_symbol), [str(path) for path in paths], accepted


def _match_observation(
    order: dict[str, Any],
    observations: list[dict[str, Any]],
    observation_epochs: list[float],
    *,
    min_latency_seconds: float,
    max_latency_seconds: float,
) -> dict[str, Any] | None:
    start = order["timestamp"].timestamp() + max(float(min_latency_seconds), 0.0)
    index = bisect_left(observation_epochs, start)
    while index < len(observations):
        row = observations[index]
        latency = (row["timestamp"] - order["timestamp"]).total_seconds()
        if latency > max(float(max_latency_seconds), 0.0):
            return None
        if row.get("snapshot_id") and row.get("snapshot_id") == order.get("snapshot_id"):
            index += 1
            continue
        return {**row, "latency_seconds": latency}
    return None


def _replay_row(order: dict[str, Any], observation: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    half_spread = observation["last_price"] * max(observation["spread_bps"], 0.0) / 20_000.0
    fill_price = observation["last_price"] + half_spread if order["action"] in BUY_ACTIONS else observation["last_price"] - half_spread
    source_provider = observation.get("source_provider") or observation.get("source_broker") or "broker_native_market_data"
    source_record_id = f"{order['order_id']}:{observation['observation_id']}"
    replay_material = json.dumps(
        {
            "capture_contract": "market_replay_fill_capture_v2",
            "source_record_id": source_record_id,
            "symbol": order["symbol"],
            "order_timestamp_utc": order["timestamp_utc"],
            "observation_timestamp_utc": observation["timestamp_utc"],
            "source_provider": source_provider,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    replay_dataset_id = hashlib.sha256(replay_material.encode("utf-8")).hexdigest()
    return {
        "timestamp_utc": order["timestamp_utc"],
        "observed_at_utc": observation["timestamp_utc"],
        "symbol": order["symbol"],
        "action": order["action"],
        "quantity": order["quantity"],
        "reference_price": order["reference_price"],
        "intended_price": order["reference_price"],
        "fill_price": round(max(fill_price, 0.00000001), 12),
        "expected_fill_price": order["expected_fill_price"],
        "expected_slippage_bps": order["expected_slippage_bps"],
        "paper_fill_source": "market_replay_fill",
        "source_broker": observation.get("source_broker") or order.get("source_broker") or source_provider,
        "source_provider": source_provider,
        "source_venue": observation.get("source_venue") or order.get("source_venue") or source_provider,
        "external_fill_id": source_record_id,
        "account_mode": "replay",
        "replay_dataset_id": replay_dataset_id,
        "metadata": {
            "order_id": order["order_id"],
            "source_profile": order["source_profile"],
            "account_mode": "replay",
            "capture_kind": "broker_native_delayed_quote_replay",
            "observation_latency_seconds": round(float(observation["latency_seconds"]), 6),
            "source_quality_label": observation["source_quality_label"],
            "source_quality_score": observation["source_quality_score"],
            "candidate_id": candidate.get("candidate_id"),
        },
        "provenance": {
            "source_system": source_provider,
            "source_record_id": source_record_id,
            "account_mode": "replay",
            "captured_at_utc": observation["timestamp_utc"],
            "replay_dataset_id": replay_dataset_id,
            "order_source_path": order["source_path"],
            "order_source_line": order["source_line"],
            "observation_source_path": observation["source_path"],
            "observation_source_line": observation["source_line"],
            "capture_contract": "market_replay_fill_capture_v1",
        },
    }


def _existing_captures(path: Path, *, cutoff: datetime, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _source_path, _line_number, row in _iter_rows([path]):
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        if timestamp is None or timestamp < cutoff:
            continue
        if str(row.get("paper_fill_source") or "").strip().lower() != "market_replay_fill":
            continue
        if not str(row.get("external_fill_id") or "").strip():
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("timestamp_utc") or ""), str(row.get("external_fill_id") or "")))
    return rows[-max(int(max_rows), 1) :]


def _capture_order_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    explicit = str(metadata.get("order_id") or "").strip()
    if explicit:
        return explicit
    return str(row.get("external_fill_id") or "").partition(":")[0].strip()


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    inbox_file: Path = DEFAULT_INBOX_FILE,
    apply: bool = False,
    min_latency_seconds: float = 1.0,
    max_latency_seconds: float = 900.0,
    minimum_source_quality: float = 0.9,
    max_files: int = 256,
    max_orders: int = 10_000,
    max_observations: int = 200_000,
    max_bytes_per_observation_file: int = DEFAULT_MAX_BYTES_PER_OBSERVATION_FILE,
    unmatched_rescan_grace_seconds: float = DEFAULT_UNMATCHED_RESCAN_GRACE_SECONDS,
) -> dict[str, Any]:
    cutoff, candidate = _candidate_binding(project_root)
    target = _resolve(project_root, inbox_file)
    if cutoff is None or not candidate.get("bound", False):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": iso_now(),
            "ok": False,
            "overall_status": "blocked",
            "apply": bool(apply),
            "candidate_binding": candidate,
            "blockers": ["production_candidate_binding_missing"],
            "capture_count": 0,
            "inbox_file": str(target),
        }
        return payload

    orders, order_paths = _order_rows(
        project_root,
        cutoff=cutoff,
        max_files=max_files,
        max_orders=max_orders,
    )
    retained_captures = _existing_captures(target, cutoff=cutoff, max_rows=max_orders)
    captured_order_ids = {_capture_order_id(row) for row in retained_captures}
    uncaptured_orders = [order for order in orders if str(order.get("order_id") or "") not in captured_order_ids]
    latest_order_timestamp = max((order["timestamp"] for order in orders), default=cutoff)
    rescan_window_seconds = max(float(max_latency_seconds), 0.0) + max(float(unmatched_rescan_grace_seconds), 0.0)
    pending_orders = [
        order
        for order in uncaptured_orders
        if (latest_order_timestamp - order["timestamp"]).total_seconds() <= rescan_window_seconds
    ]
    expired_unmatched_order_count = max(len(uncaptured_orders) - len(pending_orders), 0)
    observations: dict[str, list[dict[str, Any]]] = {}
    observation_paths: list[str] = []
    observation_count = 0
    if pending_orders:
        replay_start = min(order["timestamp"] for order in pending_orders)
        replay_end = max(order["timestamp"] for order in pending_orders)
        replay_end = datetime.fromtimestamp(
            replay_end.timestamp() + max(float(max_latency_seconds), 0.0),
            tz=timezone.utc,
        )
        observations, observation_paths, observation_count = _observation_rows(
            project_root,
            cutoff=replay_start,
            through=replay_end,
            symbols={str(order["symbol"]) for order in pending_orders},
            max_files=max_files,
            max_rows=max_observations,
            max_bytes_per_file=max(int(max_bytes_per_observation_file), 1),
            minimum_source_quality=minimum_source_quality,
        )
    observation_epochs = {
        symbol: [row["timestamp"].timestamp() for row in rows]
        for symbol, rows in observations.items()
    }
    new_captures: list[dict[str, Any]] = []
    unmatched = 0
    for order in pending_orders:
        match = _match_observation(
            order,
            observations.get(order["symbol"], []),
            observation_epochs.get(order["symbol"], []),
            min_latency_seconds=min_latency_seconds,
            max_latency_seconds=max_latency_seconds,
        )
        if match is None:
            unmatched += 1
            continue
        new_captures.append(_replay_row(order, match, candidate))
    captures_by_id = {
        str(row.get("external_fill_id") or ""): row
        for row in [*retained_captures, *new_captures]
        if str(row.get("external_fill_id") or "").strip()
    }
    captures = list(captures_by_id.values())
    captures.sort(key=lambda row: (str(row.get("timestamp_utc") or ""), str(row.get("external_fill_id") or "")))
    if apply:
        _atomic_write_jsonl(target, captures)

    status = "ready" if captures else ("waiting_for_observations" if orders else "waiting_for_paper_orders")
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "ok": True,
        "overall_status": status,
        "apply": bool(apply),
        "candidate_binding": candidate,
        "paper_order_count": len(orders),
        "pending_order_count": len(pending_orders),
        "expired_unmatched_order_count": expired_unmatched_order_count,
        "market_observation_count": observation_count,
        "capture_count": len(captures),
        "retained_capture_count": len(retained_captures),
        "new_capture_count": len(new_captures),
        "unmatched_order_count": unmatched,
        "order_source_files": order_paths,
        "observation_source_files": observation_paths,
        "inbox_file": str(target),
        "latency_window_seconds": {
            "minimum": max(float(min_latency_seconds), 0.0),
            "maximum": max(float(max_latency_seconds), 0.0),
        },
        "minimum_source_quality": max(float(minimum_source_quality), 0.0),
        "scan_budget": {
            "max_files": max(int(max_files), 1),
            "max_orders": max(int(max_orders), 1),
            "max_observations": max(int(max_observations), 1),
            "max_bytes_per_active_observation_file": max(int(max_bytes_per_observation_file), 1),
            "unmatched_rescan_grace_seconds": max(float(unmatched_rescan_grace_seconds), 0.0),
            "date_partition_pruning": True,
            "matched_order_rescan_suppressed": True,
        },
        "control_contract": {
            "candidate_cutoff_enforced": True,
            "broker_native_observations_required": True,
            "future_observation_required": True,
            "execution_model_fill_reuse_forbidden": True,
            "archive_path_independent_record_identity": True,
            "content_stable_replay_dataset_identity": True,
            "existing_candidate_window_captures_retained": True,
            "observation_scan_is_date_and_byte_bounded": True,
            "conservative_side_of_spread_replay": True,
            "market_replay_is_not_a_broker_fill_receipt": True,
            "automatic_live_execution_authority": False,
            "automatic_live_promotion_authority": False,
        },
        "blockers": [],
        "recommended_actions": [
            "let candidate-bound paper executions and later broker-native observations accumulate",
            "retain broker-paper or venue receipts as the stronger independent source when available",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture candidate-bound delayed-quote market replay fills.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inbox-file", type=Path, default=DEFAULT_INBOX_FILE)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-latency-seconds", type=float, default=1.0)
    parser.add_argument("--max-latency-seconds", type=float, default=900.0)
    parser.add_argument("--minimum-source-quality", type=float, default=0.9)
    parser.add_argument("--max-files", type=int, default=256)
    parser.add_argument("--max-orders", type=int, default=10_000)
    parser.add_argument("--max-observations", type=int, default=200_000)
    parser.add_argument(
        "--max-bytes-per-observation-file",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_OBSERVATION_FILE,
    )
    parser.add_argument(
        "--unmatched-rescan-grace-seconds",
        type=float,
        default=DEFAULT_UNMATCHED_RESCAN_GRACE_SECONDS,
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(
        project_root,
        inbox_file=args.inbox_file,
        apply=bool(args.apply),
        min_latency_seconds=float(args.min_latency_seconds),
        max_latency_seconds=float(args.max_latency_seconds),
        minimum_source_quality=float(args.minimum_source_quality),
        max_files=max(int(args.max_files), 1),
        max_orders=max(int(args.max_orders), 1),
        max_observations=max(int(args.max_observations), 1),
        max_bytes_per_observation_file=max(int(args.max_bytes_per_observation_file), 1),
        unmatched_rescan_grace_seconds=max(float(args.unmatched_rescan_grace_seconds), 0.0),
    )
    write_payload(_resolve(project_root, args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "market_replay_fill_capture "
            f"status={payload.get('overall_status')} captures={payload.get('capture_count', 0)} "
            f"unmatched={payload.get('unmatched_order_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
