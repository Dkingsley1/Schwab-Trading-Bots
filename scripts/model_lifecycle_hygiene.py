import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import write_registry_mutation_journal

_LOG_ARTIFACT_SUFFIXES = (".json", ".jsonl", ".log", ".txt")


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _collect_backups(root: Path) -> list[Path]:
    out: list[Path] = []
    out.extend(sorted(root.glob("master_bot_registry.backup*.json")))
    out.extend(sorted(root.glob("registry_backup_before_retrain*.json")))
    out.extend(sorted((root / "governance").glob("master_bot_registry.backup*.json")))
    out.extend(sorted((root / "governance").glob("registry_backup_before_retrain*.json")))
    out.extend(sorted((root / "governance" / "lifecycle").glob("master_bot_registry.repair_backup_*.json")))
    # de-dup while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append(p)
    return uniq


def _as_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


def _mtime_value(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _safe_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: object, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _latest_artifact_for_bot(base: Path, bot_id: str, suffixes: tuple[str, ...]) -> Path | None:
    if not bot_id or (not base.exists()):
        return None
    rows: list[Path] = []
    patterns = []
    for suffix in suffixes:
        patterns.extend(
            [
                f"{bot_id}_*{suffix}",
                f"{bot_id}{suffix}",
            ]
        )
    for pattern in patterns:
        rows.extend([p for p in base.glob(pattern) if p.is_file()])
    if not rows:
        return None
    uniq = {str(p): p for p in rows}
    ordered = sorted(uniq.values(), key=_mtime_value, reverse=True)
    return ordered[0]


def _latest_log_artifact_for_bot(root: Path, bot_id: str) -> Path | None:
    search_bases = [
        root / "logs",
        root / "governance" / "training_diagnostics",
        root / "governance" / "walk_forward",
    ]
    for base in search_bases:
        artifact = _latest_artifact_for_bot(base, bot_id, _LOG_ARTIFACT_SUFFIXES)
        if artifact is not None:
            return artifact
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _training_diagnostic_path(root: Path, bot_id: str) -> Path | None:
    if not bot_id:
        return None
    path = root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"
    return path if path.exists() else None


def _age_hours(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return max((datetime.now(timezone.utc) - modified).total_seconds() / 3600.0, 0.0)
    except Exception:
        return None


def _artifact_state(*, model_ok: bool, log_ok: bool) -> str:
    if model_ok and log_ok:
        return "ok"
    if model_ok and (not log_ok):
        return "missing_log_only"
    if (not model_ok) and log_ok:
        return "missing_model_only"
    return "missing_both"


def _diag_int(diag: dict, key: str) -> int:
    try:
        return int(diag.get(key, 0) or 0)
    except Exception:
        return 0


def _runtime_input_gap_cause(diag: dict) -> str:
    status = str(diag.get("status") or "").strip().lower()
    if status != "deferred_sample_starved":
        return ""
    sample_count = _diag_int(diag, "sample_count")
    eligible_sequences = _diag_int(diag, "eligible_sequences")
    sequence_count = _diag_int(diag, "sequence_count")
    if sample_count == 0 and eligible_sequences == 0 and sequence_count == 0:
        return "shared_runtime_input_gap"
    if sample_count == 0 and eligible_sequences == 0:
        return "sequence_depth_gap"
    return ""


def _bot_family(bot_id: str) -> str:
    lowered = str(bot_id or "").strip().lower()
    for token, family in (
        ("intraday", "intraday"),
        ("swing", "swing"),
        ("crypto", "crypto"),
        ("bond", "bond"),
        ("fx", "fx"),
        ("dividend", "dividend"),
        ("futures", "futures"),
        ("risk_budget", "infrastructure"),
        ("allocator", "infrastructure"),
        ("seasonal", "signal"),
    ):
        if token in lowered:
            return family
    return "general"


def _recovered_diagnostic(*, bot_id: str, log_payload: dict, log_path: Path, model_path: Path | None) -> dict:
    metrics = log_payload.get("metrics") if isinstance(log_payload.get("metrics"), dict) else {}
    acted_count = max(
        _safe_int(metrics.get("acted_count"), 0),
        _safe_int(metrics.get("long_acted_count"), 0) + _safe_int(metrics.get("short_acted_count"), 0),
    )
    sample_count = max(acted_count, 1 if metrics else 0)
    acted_accuracy = _safe_float(metrics.get("acted_accuracy"), -1.0)
    accuracy_lift = _safe_float(metrics.get("accuracy_lift_over_majority"), 0.0)
    failures: list[str] = []
    if acted_accuracy >= 0.0 and acted_accuracy < 0.53:
        failures.append(f"acted_accuracy={acted_accuracy:.4f} < recovered_min_acted_accuracy=0.5300")
    if accuracy_lift < 0.0:
        failures.append(
            "accuracy_lift_over_majority="
            f"{accuracy_lift:.4f} < recovered_min_accuracy_lift_over_majority=0.0000"
        )
    status = "failed" if failures else "passed"
    diag_timestamp = str(log_payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
    positive_rate = _safe_float(metrics.get("positive_rate"), 0.5)
    runtime_meta = {
        "sample_count": int(sample_count),
        "eligible_sequences": int(1 if sample_count > 0 else 0),
        "sequence_count": int(1 if sample_count > 0 else 0),
        "observation_count": int(sample_count),
        "positive_rate": round(float(positive_rate), 6),
        "skipped_filtered": 0,
        "skipped_low_confidence": 0,
        "skipped_labels": 0,
        "recovered_from_training_log": True,
        "recovery_source_log_path": str(log_path),
        "recovery_source_model_path": str(model_path) if model_path else "",
    }
    return {
        "timestamp_utc": diag_timestamp,
        "run_tag": bot_id,
        "status": status,
        "family": _bot_family(bot_id),
        "sample_count": int(sample_count),
        "eligible_sequences": int(runtime_meta["eligible_sequences"]),
        "sequence_count": int(runtime_meta["sequence_count"]),
        "observation_count": int(runtime_meta["observation_count"]),
        "positive_rate": round(float(positive_rate), 6),
        "skipped_filtered": 0,
        "skipped_low_confidence": 0,
        "skipped_labels": 0,
        "metrics": metrics,
        "runtime_meta": runtime_meta,
        "quality_failures": failures,
        "failure_categories": (["quality_guard_failure"] if failures else []),
        "repaired_from_log": True,
    }


def _diagnostic_from_source_payload(*, bot_id: str, source_payload: dict, source_path: Path, model_path: Path | None) -> dict:
    if isinstance(source_payload.get("metrics"), dict):
        return _recovered_diagnostic(
            bot_id=bot_id,
            log_payload=source_payload,
            log_path=source_path,
            model_path=model_path,
        )
    payload = dict(source_payload)
    payload.setdefault("timestamp_utc", str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()))
    payload.setdefault("run_tag", bot_id)
    runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
    payload["runtime_meta"] = {
        **runtime_meta,
        "recovered_from_existing_diagnostic": True,
        "recovery_source_log_path": str(source_path),
        "recovery_source_model_path": str(model_path) if model_path else "",
    }
    payload["repair_source_path"] = str(source_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Model lifecycle hygiene checks, manifest, and backup maintenance.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--keep-backups", type=int, default=25)
    parser.add_argument("--apply-prune", action="store_true")
    parser.add_argument("--update-last-known-good", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--max-missing-active-artifacts", type=int, default=2)
    parser.add_argument("--max-training-diagnostic-age-hours", type=float, default=72.0)
    parser.add_argument("--max-stale-active-diagnostics", type=int, default=0)
    parser.add_argument("--repair-stale-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-repair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply-diagnostic-downgrade", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply-runtime-input-downgrade", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "lifecycle" / "model_lifecycle_latest.json"))
    parser.add_argument("--manifest-file", default=str(PROJECT_ROOT / "governance" / "lifecycle" / "model_manifest_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    reg = _load(registry_path)
    sub_bots = reg.get("sub_bots") if isinstance(reg.get("sub_bots"), list) else []

    active_rows = [r for r in sub_bots if isinstance(r, dict) and bool(r.get("active", False))]
    missing_active_hard = []
    missing_log_only = []
    stale_diagnostics = []
    manifest_rows = []
    repaired_rows = []
    downgraded_rows = []
    runtime_input_downgraded_rows = []
    registry_before = json.loads(json.dumps(reg)) if isinstance(reg, dict) else {}

    for row in active_rows:
        bot_id = str(row.get("bot_id", "")).strip()
        model_path = _as_path(row.get("model_path"))
        log_path = _as_path(row.get("log_file"))
        model_ok = bool(model_path and model_path.exists())
        log_ok = bool(log_path and log_path.exists())

        original_model_path = str(model_path) if model_path is not None else ""
        original_log_path = str(log_path) if log_path is not None else ""
        row_repairs: dict[str, str] = {}

        if args.repair_stale_artifacts:
            if not model_ok:
                repaired_model = _latest_artifact_for_bot(PROJECT_ROOT / "models", bot_id, (".npz",))
                if repaired_model is not None:
                    row["model_path"] = str(repaired_model)
                    model_path = repaired_model
                    model_ok = True
                    row_repairs["model_path"] = str(repaired_model)
            if not log_ok:
                repaired_log = _latest_log_artifact_for_bot(PROJECT_ROOT, bot_id)
                if repaired_log is not None:
                    row["log_file"] = str(repaired_log)
                    log_path = repaired_log
                    log_ok = True
                    row_repairs["log_file"] = str(repaired_log)

        if row_repairs:
            repaired_rows.append(
                {
                    "bot_id": bot_id,
                    "from_model_path": original_model_path,
                    "from_log_path": original_log_path,
                    "to_model_path": row_repairs.get("model_path", original_model_path),
                    "to_log_path": row_repairs.get("log_file", original_log_path),
                }
            )

        artifact_state = _artifact_state(model_ok=model_ok, log_ok=log_ok)
        diagnostic_path = _training_diagnostic_path(PROJECT_ROOT, bot_id)
        diagnostic_payload = _load(diagnostic_path) if diagnostic_path is not None else {}
        diagnostic_age_hours = _age_hours(diagnostic_path)
        latest_log_newer_than_diag = bool(
            log_path
            and diagnostic_path is not None
            and log_path.exists()
            and log_path.stat().st_mtime > diagnostic_path.stat().st_mtime
        )
        diagnostic_rebuilt = False
        if args.repair_stale_artifacts and log_ok and (
            diagnostic_path is None
            or latest_log_newer_than_diag
            or (diagnostic_age_hours is not None and diagnostic_age_hours > float(args.max_training_diagnostic_age_hours))
        ):
            repair_target = PROJECT_ROOT / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"
            recovered = _diagnostic_from_source_payload(
                bot_id=bot_id,
                source_payload=_load(log_path) if log_path is not None else {},
                source_path=log_path if log_path is not None else repair_target,
                model_path=model_path,
            )
            _write_json(repair_target, recovered)
            diagnostic_path = repair_target
            diagnostic_payload = recovered
            diagnostic_age_hours = _age_hours(diagnostic_path)
            diagnostic_rebuilt = True
        diagnostic_fresh = bool(
            diagnostic_age_hours is not None and float(diagnostic_age_hours) <= float(args.max_training_diagnostic_age_hours)
        )
        runtime_input_gap_cause = _runtime_input_gap_cause(diagnostic_payload)
        artifact_row = {
            "bot_id": bot_id,
            "artifact_state": artifact_state,
            "model_exists": bool(model_ok),
            "log_exists": bool(log_ok),
            "model_path": str(model_path) if model_path is not None else "",
            "log_path": str(log_path) if log_path is not None else "",
            "diagnostic_path": str(diagnostic_path) if diagnostic_path is not None else "",
            "diagnostic_age_hours": round(float(diagnostic_age_hours), 3) if diagnostic_age_hours is not None else None,
            "diagnostic_fresh": bool(diagnostic_fresh),
            "runtime_input_gap_cause": runtime_input_gap_cause,
        }
        if diagnostic_rebuilt:
            existing = next((item for item in repaired_rows if item.get("bot_id") == bot_id), None)
            if existing is None:
                repaired_rows.append(
                    {
                        "bot_id": bot_id,
                        "from_model_path": original_model_path,
                        "from_log_path": original_log_path,
                        "to_model_path": str(model_path) if model_path is not None else original_model_path,
                        "to_log_path": str(log_path) if log_path is not None else original_log_path,
                        "diagnostic_path": str(diagnostic_path) if diagnostic_path is not None else "",
                    }
                )
            else:
                existing["diagnostic_path"] = str(diagnostic_path) if diagnostic_path is not None else ""
        if artifact_state == "missing_log_only":
            missing_log_only.append(artifact_row)
        elif artifact_state != "ok":
            missing_active_hard.append(artifact_row)
        if not diagnostic_fresh:
            stale_diagnostics.append(artifact_row)
            if args.apply_diagnostic_downgrade:
                previous_reason = str(row.get("reason", "") or "")
                row["active"] = False
                row["lifecycle_state"] = "probation"
                row["reason"] = "stale_training_diagnostic"
                row["stale_training_diagnostic_age_hours"] = round(float(diagnostic_age_hours), 3) if diagnostic_age_hours is not None else None
                downgraded_rows.append(
                    {
                        "bot_id": bot_id,
                        "from_reason": previous_reason,
                        "to_reason": "stale_training_diagnostic",
                        "diagnostic_age_hours": artifact_row["diagnostic_age_hours"],
                    }
                )
        elif args.apply_runtime_input_downgrade and runtime_input_gap_cause:
            previous_reason = str(row.get("reason", "") or "")
            row["active"] = False
            row["lifecycle_state"] = "probation"
            row["reason"] = "unsupported_runtime_inputs"
            row["runtime_input_gap_cause"] = runtime_input_gap_cause
            runtime_input_downgraded_rows.append(
                {
                    "bot_id": bot_id,
                    "from_reason": previous_reason,
                    "to_reason": "unsupported_runtime_inputs",
                    "runtime_input_gap_cause": runtime_input_gap_cause,
                }
            )

        manifest_rows.append(
            {
                "bot_id": bot_id,
                "model_path": str(model_path) if model_path is not None else "",
                "model_mtime_utc": _mtime_iso(model_path) if model_ok and model_path is not None else None,
                "log_path": str(log_path) if log_path is not None else "",
                "log_mtime_utc": _mtime_iso(log_path) if log_ok and log_path is not None else None,
                "diagnostic_path": str(diagnostic_path) if diagnostic_path is not None else "",
                "diagnostic_mtime_utc": _mtime_iso(diagnostic_path) if diagnostic_path is not None else None,
                "diagnostic_fresh": bool(diagnostic_fresh),
                "artifact_state": artifact_state,
                "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                "test_accuracy": row.get("test_accuracy"),
            }
        )

    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / (1024.0 ** 3)

    backups = _collect_backups(PROJECT_ROOT)
    pruned = []
    if args.apply_prune and len(backups) > int(args.keep_backups):
        stale = backups[: len(backups) - int(args.keep_backups)]
        for p in stale:
            try:
                p.unlink(missing_ok=True)
                pruned.append(str(p))
            except Exception:
                continue

    lifecycle_dir = PROJECT_ROOT / "governance" / "lifecycle"
    lifecycle_dir.mkdir(parents=True, exist_ok=True)

    repair_backup_file = None
    repair_error = None
    registry_updated = False
    if (args.repair_stale_artifacts and args.apply_repair and repaired_rows and isinstance(reg, dict)) or (
        args.apply_diagnostic_downgrade and downgraded_rows and isinstance(reg, dict)
    ) or (
        args.apply_runtime_input_downgrade and runtime_input_downgraded_rows and isinstance(reg, dict)
    ):
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if registry_path.exists():
                backup_path = lifecycle_dir / f"master_bot_registry.repair_backup_{stamp}.json"
                backup_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
                repair_backup_file = str(backup_path)
            if isinstance(reg.get("summary"), dict):
                rows = reg.get("sub_bots") if isinstance(reg.get("sub_bots"), list) else []
                active_rows_now = [r for r in rows if isinstance(r, dict) and bool(r.get("active", False))]
                reg["summary"]["active_bots"] = len(active_rows_now)
                reg["summary"]["inactive_bots"] = max(len(rows) - len(active_rows_now), 0)
            registry_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")
            if isinstance(registry_before, dict) and isinstance(reg, dict):
                write_registry_mutation_journal(
                    project_root=str(PROJECT_ROOT),
                    actor="model_lifecycle_hygiene",
                    reason="repair_stale_or_unsupported_active_bots",
                    before=registry_before,
                    after=reg,
                    extra={
                        "repaired_rows": repaired_rows[:40],
                        "stale_diagnostic_downgrades": downgraded_rows[:40],
                        "runtime_input_downgrades": runtime_input_downgraded_rows[:40],
                    },
                )
            registry_updated = True
        except Exception as exc:
            repair_error = str(exc)

    if args.update_last_known_good and registry_path.exists():
        lk = lifecycle_dir / "registry_last_known_good.json"
        lk.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_bots": len(active_rows),
        "rows": manifest_rows,
    }
    Path(args.manifest_file).write_text(json.dumps(manifest_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    ok = (
        (len(missing_active_hard) <= int(args.max_missing_active_artifacts))
        and (len(stale_diagnostics) <= int(args.max_stale_active_diagnostics))
        and (free_gb >= float(args.min_free_gb))
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "thresholds": {
            "min_free_gb": float(args.min_free_gb),
            "max_missing_active_artifacts": int(args.max_missing_active_artifacts),
            "max_training_diagnostic_age_hours": float(args.max_training_diagnostic_age_hours),
            "max_stale_active_diagnostics": int(args.max_stale_active_diagnostics),
            "keep_backups": int(args.keep_backups),
        },
        "disk": {"free_gb": round(free_gb, 2)},
        "active_bots": len(active_rows),
        "missing_active_artifacts": len(missing_active_hard),
        "missing_active_artifacts_total": len(missing_active_hard) + len(missing_log_only),
        "missing_log_only_artifacts": len(missing_log_only),
        "stale_active_training_diagnostics": len(stale_diagnostics),
        "missing_active_examples": missing_active_hard[:40],
        "missing_log_only_examples": missing_log_only[:40],
        "stale_active_diagnostic_examples": stale_diagnostics[:40],
        "backup_files": len(backups),
        "pruned_backups": pruned,
        "manifest_file": str(Path(args.manifest_file)),
        "repair": {
            "enabled": bool(args.repair_stale_artifacts),
            "apply": bool(args.apply_repair),
            "apply_diagnostic_downgrade": bool(args.apply_diagnostic_downgrade),
            "apply_runtime_input_downgrade": bool(args.apply_runtime_input_downgrade),
            "fixed_count": len(repaired_rows),
            "downgraded_for_stale_diagnostics": len(downgraded_rows),
            "downgraded_for_runtime_input_gaps": len(runtime_input_downgraded_rows),
            "registry_updated": bool(registry_updated),
            "backup_file": repair_backup_file,
            "error": repair_error,
            "examples": repaired_rows[:40],
            "diagnostic_downgrade_examples": downgraded_rows[:40],
            "runtime_input_downgrade_examples": runtime_input_downgraded_rows[:40],
        },
    }

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "model_lifecycle_hygiene "
            f"ok={str(payload['ok']).lower()} free_gb={payload['disk']['free_gb']:.2f} "
            f"missing_active={payload['missing_active_artifacts']} pruned={len(pruned)}"
        )

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
