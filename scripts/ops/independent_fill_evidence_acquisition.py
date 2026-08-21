#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload


DEFAULT_INBOX = Path("exports/independent_fill_inbox")
DEFAULT_LEDGER_DIR = Path("governance/evidence/independent_fill_records")
DEFAULT_TRADE_LOG_DIR = Path("exports/trade_logs/independent_fills")
DEFAULT_STATE = Path("governance/runtime/independent_fill_acquisition_state.json")
DEFAULT_OUT = Path("governance/health/independent_fill_evidence_acquisition_latest.json")
SCHEMA_VERSION = 2
INDEPENDENT_SOURCES = {
    "explicit_fill",
    "broker_paper_fill",
    "observed_fill",
    "market_replay_fill",
    "venue_replay_fill",
}
PAPER_ACCOUNT_MODES = {"paper", "sandbox", "replay"}
MODEL_SOURCES = {"expected_fill_model", "mark_price", "model", "execution_simulator", "simulated_fill"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _resolve(project_root: Path, path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else project_root / path


def _candidate_cutoff(project_root: Path) -> tuple[datetime | None, dict[str, Any]]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    windows = state.get("scope_windows_started_utc") if isinstance(state.get("scope_windows_started_utc"), dict) else {}
    values = [parse_iso_utc(windows.get(scope)) for scope in ("execution", "data", "dependencies")]
    cutoff = max((value for value in values if value is not None), default=None)
    return cutoff, {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": int(_safe_float(state.get("generation"), 0.0)),
        "cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
        "bound": bool(str(state.get("candidate_id") or "").strip() and cutoff is not None),
    }


def _record_digest(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_material_digest(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    material = {
        "timestamp_utc": record.get("timestamp_utc"),
        "symbol": record.get("symbol"),
        "action": record.get("action"),
        "quantity": record.get("quantity"),
        "reference_price": record.get("reference_price"),
        "intended_price": record.get("intended_price"),
        "fill_price": record.get("fill_price"),
        "expected_fill_price": record.get("expected_fill_price"),
        "expected_slippage_bps": record.get("expected_slippage_bps"),
        "paper_fill_source": record.get("paper_fill_source"),
        "source_broker": record.get("source_broker"),
        "source_provider": record.get("source_provider"),
        "source_venue": record.get("source_venue"),
        "external_fill_id": record.get("external_fill_id"),
        "source_profile": metadata.get("source_profile"),
        "account_mode": metadata.get("account_mode"),
        "source_system": provenance.get("source_system"),
        "source_record_id": provenance.get("source_record_id"),
        "captured_at_utc": provenance.get("captured_at_utc"),
    }
    return _record_digest(material)


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize(
    row: dict[str, Any],
    *,
    source_file: Path,
    line_number: int,
    cutoff: datetime | None,
    candidate: dict[str, Any],
    now: datetime,
) -> tuple[dict[str, Any] | None, list[str], str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source = _first_text(row.get("paper_fill_source"), metadata.get("paper_fill_source")).lower()
    source_system = _first_text(
        provenance.get("source_system"),
        row.get("source_system"),
        row.get("source_broker"),
        row.get("broker"),
        row.get("source_provider"),
    ).lower()
    source_record_id = _first_text(
        provenance.get("source_record_id"),
        row.get("external_fill_id"),
        row.get("fill_id"),
        row.get("execution_id"),
    )
    account_mode = _first_text(provenance.get("account_mode"), row.get("account_mode"), metadata.get("account_mode")).lower()
    timestamp = parse_iso_utc(row.get("timestamp_utc") or row.get("executed_at_utc"))
    observed_at = parse_iso_utc(provenance.get("captured_at_utc") or row.get("observed_at_utc"))
    symbol = str(row.get("symbol") or "").strip().upper()
    action = str(row.get("action") or row.get("instruction") or "").strip().upper()
    fill_price = _safe_float(row.get("fill_price"))
    reference_price = _safe_float(row.get("reference_price", row.get("intended_price")))
    expected_fill_price = _safe_float(row.get("expected_fill_price"))
    quantity = _safe_float(row.get("quantity", row.get("filled_quantity")))
    replay_dataset_id = _first_text(provenance.get("replay_dataset_id"), row.get("replay_dataset_id"))
    current_candidate_id = str(candidate.get("candidate_id") or "").strip()
    declared_candidate_id = _first_text(
        row.get("candidate_id"),
        row.get("production_candidate_id"),
        metadata.get("candidate_id"),
        metadata.get("production_candidate_id"),
        provenance.get("candidate_id"),
        provenance.get("production_candidate_id"),
    )
    errors: list[str] = []
    if source in MODEL_SOURCES:
        errors.append("model_derived_source_not_independent")
    elif source not in INDEPENDENT_SOURCES:
        errors.append("unsupported_or_missing_independent_source")
    if not source_system:
        errors.append("source_system_missing")
    if not source_record_id:
        errors.append("source_record_id_missing")
    if account_mode not in PAPER_ACCOUNT_MODES:
        errors.append("paper_or_replay_account_mode_required")
    if timestamp is None:
        errors.append("execution_timestamp_missing_or_invalid")
    if observed_at is None:
        errors.append("capture_timestamp_missing_or_invalid")
    if timestamp is not None and observed_at is not None and observed_at < timestamp:
        errors.append("capture_timestamp_precedes_execution")
    if timestamp is not None and timestamp > now + timedelta(minutes=5):
        errors.append("execution_timestamp_in_future")
    if observed_at is not None and observed_at > now + timedelta(minutes=5):
        errors.append("capture_timestamp_in_future")
    if cutoff is not None and timestamp is not None and timestamp < cutoff:
        errors.append("before_candidate_evidence_cutoff")
    if declared_candidate_id and declared_candidate_id != current_candidate_id:
        errors.append("source_candidate_id_mismatch")
    if not symbol:
        errors.append("symbol_missing")
    if action not in {"BUY", "SELL", "BUY_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
        errors.append("supported_trade_action_required")
    if fill_price <= 0.0:
        errors.append("positive_fill_price_required")
    if reference_price <= 0.0:
        errors.append("positive_reference_price_required")
    if expected_fill_price <= 0.0:
        errors.append("positive_expected_fill_price_required")
    if quantity <= 0.0:
        errors.append("positive_fill_quantity_required")
    if source in {"market_replay_fill", "venue_replay_fill"} and not replay_dataset_id:
        errors.append("replay_dataset_id_required")
    identity = f"{source_system}:{source}:{source_record_id}" if source_system and source and source_record_id else ""
    if errors:
        return None, ordered_unique(errors), identity
    profile = _first_text(metadata.get("source_profile"), row.get("profile"), "independent_fill")
    normalized = {
        "timestamp_utc": timestamp.isoformat(),
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "reference_price": reference_price,
        "intended_price": _safe_float(row.get("intended_price"), reference_price),
        "fill_price": fill_price,
        "expected_fill_price": expected_fill_price,
        "expected_slippage_bps": _safe_float(row.get("expected_slippage_bps")),
        "paper_fill_source": source,
        "source_broker": str(row.get("source_broker") or row.get("broker") or source_system).strip().lower(),
        "source_provider": str(row.get("source_provider") or source_system).strip().lower(),
        "source_venue": str(row.get("source_venue") or row.get("venue") or source_system).strip().lower(),
        "external_fill_id": source_record_id,
        "metadata": {
            "source_profile": profile.strip().lower(),
            "account_mode": account_mode,
            "independent_fill_evidence": True,
            "candidate_id": current_candidate_id,
        },
        "provenance": {
            "source_system": source_system,
            "source_record_id": source_record_id,
            "account_mode": account_mode,
            "captured_at_utc": observed_at.isoformat(),
            "replay_dataset_id": replay_dataset_id,
            "candidate_id": current_candidate_id,
            "candidate_generation": int(candidate.get("generation") or 0),
            "candidate_cutoff_utc": str(candidate.get("cutoff_utc") or ""),
            "source_declared_candidate_id": declared_candidate_id,
            "normalizer": "independent_fill_evidence_acquisition_v2",
        },
        "candidate_id": current_candidate_id,
        "candidate_generation": int(candidate.get("generation") or 0),
        "candidate_cutoff_utc": str(candidate.get("cutoff_utc") or ""),
        "evidence_identity": identity,
        "promotion_evidence_eligible": True,
    }
    normalized["evidence_sha256"] = _record_digest(normalized)
    normalized["intake"] = {"file": str(source_file), "line_number": int(line_number)}
    return normalized, [], identity


def _iter_inbox_rows(inbox: Path) -> Iterable[tuple[Path, int, dict[str, Any], str]]:
    for path in sorted(inbox.glob("*.jsonl")) if inbox.exists() else []:
        try:
            handle = path.open("r", encoding="utf-8")
        except Exception:
            continue
        with handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    yield path, line_number, {}, "invalid_json"
                    continue
                if not isinstance(payload, dict):
                    yield path, line_number, {}, "json_object_required"
                    continue
                yield path, line_number, payload, ""


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_ledger_rows(ledger_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ledger_dir.glob("*.json")) if ledger_dir.exists() else []:
        payload = load_json(path)
        if payload and payload.get("promotion_evidence_eligible") is True:
            rows.append(payload)
    rows.sort(key=lambda row: (str(row.get("timestamp_utc") or ""), str(row.get("evidence_identity") or "")))
    return rows


def _record_candidate_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return _first_text(
        row.get("candidate_id"),
        row.get("production_candidate_id"),
        metadata.get("candidate_id"),
        metadata.get("production_candidate_id"),
        provenance.get("candidate_id"),
        provenance.get("production_candidate_id"),
    )


def _candidate_eligible_rows(
    rows: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    cutoff: datetime | None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    current_candidate_id = str(candidate.get("candidate_id") or "").strip()
    eligible: list[dict[str, Any]] = []
    mismatched = 0
    missing_identity = 0
    before_cutoff = 0
    for row in rows:
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        if cutoff is None or timestamp is None or timestamp < cutoff:
            before_cutoff += 1
            continue
        row_candidate_id = _record_candidate_id(row)
        if not row_candidate_id:
            missing_identity += 1
            continue
        if row_candidate_id != current_candidate_id:
            mismatched += 1
            continue
        eligible.append(row)
    return eligible, mismatched, missing_identity, before_cutoff


def _materialize_trade_logs(trade_log_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        if timestamp is None:
            continue
        by_day.setdefault(timestamp.strftime("%Y%m%d"), []).append(row)
    trade_log_dir.mkdir(parents=True, exist_ok=True)
    expected_paths: set[Path] = set()
    for day, day_rows in sorted(by_day.items()):
        path = trade_log_dir / f"paper_trades_{day}.jsonl"
        _atomic_write_jsonl(path, day_rows)
        expected_paths.add(path)
    for stale in trade_log_dir.glob("paper_trades_*.jsonl"):
        if stale not in expected_paths:
            _atomic_write_jsonl(stale, [])
    return [str(path) for path in sorted(expected_paths)]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    inbox: Path = DEFAULT_INBOX,
    ledger_dir: Path = DEFAULT_LEDGER_DIR,
    trade_log_dir: Path = DEFAULT_TRADE_LOG_DIR,
    state_path: Path = DEFAULT_STATE,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    inbox_path = _resolve(project_root, inbox)
    ledger_path = _resolve(project_root, ledger_dir)
    trade_log_path = _resolve(project_root, trade_log_dir)
    effective_state_path = _resolve(project_root, state_path)
    prior_state = load_json(effective_state_path)
    existing_ledger_rows = _load_ledger_rows(ledger_path)
    prior_identities = prior_state.get("identity_hashes") if isinstance(prior_state.get("identity_hashes"), dict) else {}
    identities = {str(key): str(value) for key, value in prior_identities.items() if str(key).strip() and str(value).strip()}
    prior_material_hashes = (
        prior_state.get("identity_material_hashes")
        if isinstance(prior_state.get("identity_material_hashes"), dict)
        else {}
    )
    identity_material_hashes = {
        str(key): str(value)
        for key, value in prior_material_hashes.items()
        if str(key).strip() and str(value).strip()
    }
    for row in existing_ledger_rows:
        identity = str(row.get("evidence_identity") or "").strip()
        digest = str(row.get("evidence_sha256") or "").strip()
        if identity and digest:
            identities.setdefault(identity, digest)
            identity_material_hashes.setdefault(identity, _identity_material_digest(row))
    cutoff, candidate = _candidate_cutoff(project_root)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    scanned = 0
    valid_rows_seen = 0
    duplicate_rows_seen = 0
    for path, line_number, raw, parse_error in _iter_inbox_rows(inbox_path):
        scanned += 1
        if parse_error:
            rejected.append({"file": str(path), "line": line_number, "reasons": [parse_error]})
            continue
        normalized, errors, identity = _normalize(
            raw,
            source_file=path,
            line_number=line_number,
            cutoff=cutoff,
            candidate=candidate,
            now=current,
        )
        if cutoff is None:
            errors = ordered_unique(errors + ["production_candidate_binding_missing"])
            normalized = None
        if normalized is None:
            rejected.append(
                {
                    "file": str(path),
                    "line": line_number,
                    "identity": identity,
                    "source_sha256": _record_digest(raw),
                    "reasons": errors,
                }
            )
            continue
        digest = str(normalized["evidence_sha256"])
        material_digest = _identity_material_digest(normalized)
        previous_digest = identities.get(identity)
        previous_material_digest = identity_material_hashes.get(identity)
        material_conflict = bool(previous_material_digest and previous_material_digest != material_digest)
        legacy_digest_conflict = bool(
            previous_digest
            and not previous_material_digest
            and previous_digest != digest
        )
        if material_conflict or legacy_digest_conflict:
            conflicts.append(
                {
                    "file": str(path),
                    "line": line_number,
                    "identity": identity,
                    "accepted_sha256": previous_digest,
                    "conflicting_sha256": digest,
                    "accepted_identity_material_sha256": previous_material_digest,
                    "conflicting_identity_material_sha256": material_digest,
                    "reason": "immutable_source_record_id_reused_with_different_content",
                }
            )
            continue
        valid_rows_seen += 1
        source_counts[str(normalized.get("paper_fill_source") or "unknown")] += 1
        if previous_digest or previous_material_digest:
            duplicate_rows_seen += 1
            continue
        identities[identity] = digest
        identity_material_hashes[identity] = material_digest
        accepted.append(normalized)

    unique_new: dict[str, dict[str, Any]] = {str(row["evidence_sha256"]): row for row in accepted}
    new_record_count = 0
    materialized_paths: list[str] = []
    if apply:
        ledger_path.mkdir(parents=True, exist_ok=True)
        for digest, row in sorted(unique_new.items()):
            record_path = ledger_path / f"{digest}.json"
            if not record_path.exists():
                write_payload(record_path, row)
                new_record_count += 1
        ledger_rows = _load_ledger_rows(ledger_path)
        (
            eligible_ledger_rows,
            candidate_mismatch_records,
            candidate_identity_missing_records,
            before_cutoff_records,
        ) = _candidate_eligible_rows(
            ledger_rows,
            candidate=candidate,
            cutoff=cutoff,
        )
        materialized_paths = _materialize_trade_logs(trade_log_path, eligible_ledger_rows)
        next_state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": iso_now(),
            "identity_hashes": dict(sorted(identities.items())),
            "identity_material_hashes": dict(sorted(identity_material_hashes.items())),
            "ledger_record_count": len(ledger_rows),
            "candidate_eligible_ledger_record_count": len(eligible_ledger_rows),
            "candidate_binding": candidate,
        }
        write_payload(effective_state_path, next_state)
    else:
        ledger_rows = existing_ledger_rows
        (
            eligible_ledger_rows,
            candidate_mismatch_records,
            candidate_identity_missing_records,
            before_cutoff_records,
        ) = _candidate_eligible_rows(
            ledger_rows,
            candidate=candidate,
            cutoff=cutoff,
        )

    total_accepted = len(ledger_rows)
    candidate_eligible = len(eligible_ledger_rows)
    status = "blocked" if not candidate.get("bound", False) else "conflict" if conflicts else "ready" if candidate_eligible else "waiting_for_source"
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": status,
        "ok": bool(candidate.get("bound", False) and not conflicts),
        "apply": bool(apply),
        "candidate_binding": candidate,
        "inbox": str(inbox_path),
        "ledger_dir": str(ledger_path),
        "trade_log_dir": str(trade_log_path),
        "rows_scanned": scanned,
        "valid_rows_seen": valid_rows_seen,
        "duplicate_rows_seen": duplicate_rows_seen,
        "new_ledger_records": new_record_count,
        "accepted_ledger_records": total_accepted,
        "candidate_eligible_ledger_records": candidate_eligible,
        "candidate_mismatch_ledger_records": candidate_mismatch_records,
        "candidate_identity_missing_ledger_records": candidate_identity_missing_records,
        "before_candidate_cutoff_ledger_records": before_cutoff_records,
        "rejected_count": len(rejected),
        "conflict_count": len(conflicts),
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_tail": rejected[-20:],
        "conflicts": conflicts[-20:],
        "materialized_trade_logs": materialized_paths,
        "control_contract": {
            "model_derived_fills_never_accepted": True,
            "paper_or_replay_account_mode_required": True,
            "candidate_cutoff_enforced": cutoff is not None,
            "exact_candidate_identity_required": True,
            "legacy_unbound_records_remain_lifetime_only": True,
            "source_declared_candidate_mismatch_rejected": True,
            "content_addressed_evidence_ledger": True,
            "source_record_id_conflicts_fail_closed": True,
            "identity_material_excludes_storage_location": True,
            "provenance_relocation_preserves_immutable_identity": True,
            "idempotent_trade_log_materialization": True,
            "live_execution_authority": False,
        },
        "recommended_actions": ordered_unique(
            [
                "route broker-paper execution receipts or licensed market-replay observations into the independent-fill inbox",
                "keep expected-fill-model rows in simulator diagnostics only",
                "investigate immutable source-record conflicts before using affected evidence",
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire independently observed paper/replay fills for calibration evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    parser.add_argument("--trade-log-dir", type=Path, default=DEFAULT_TRADE_LOG_DIR)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(
        project_root,
        inbox=args.inbox,
        ledger_dir=args.ledger_dir,
        trade_log_dir=args.trade_log_dir,
        state_path=args.state_file,
        apply=bool(args.apply),
    )
    write_payload(_resolve(project_root, args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "independent_fill_evidence_acquisition "
            f"status={payload['overall_status']} accepted={payload['accepted_ledger_records']} "
            f"new={payload['new_ledger_records']} rejected={payload['rejected_count']} conflicts={payload['conflict_count']}"
        )
    return 2 if payload["conflict_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
