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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_cleanliness_autopilot_latest.json"
PYTHON_BIN = Path(sys.executable)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


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
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _opsctl(project_root: Path, *args: str) -> list[str]:
    return [str(project_root / "scripts" / "ops" / "opsctl.sh"), *args]


def _script(project_root: Path, rel: str, *args: str) -> list[str]:
    return [str(PYTHON_BIN), str(project_root / rel), *args]


def _health(project_root: Path, name: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "health" / name)


def _champion(project_root: Path, name: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "champion_challenger" / name)


def _walk_forward(project_root: Path, name: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "walk_forward" / name)


def _collector_row(contracts: dict[str, Any], name: str) -> dict[str, Any]:
    rows = contracts.get("rows") if isinstance(contracts.get("rows"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return {}


def _plan_row(layer: str, name: str, reason: str, cmd: list[str], *, gated_by: str = "") -> dict[str, Any]:
    return {
        "layer": layer,
        "name": name,
        "reason": reason,
        "cmd": cmd,
        "gated_by": gated_by,
    }


def _layer_status(blocked: bool, degraded: bool = False) -> str:
    if blocked:
        return "blocked"
    if degraded:
        return "degraded"
    return "ready"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    collectors = _health(project_root, "collector_contracts_latest.json")
    source_verification = _health(project_root, "source_verification_latest.json")
    training = _health(project_root, "training_quality_control_latest.json")
    paper = _health(project_root, "paper_performance_latest.json")
    replay_hash = _health(project_root, "replay_hash_registry_guard_latest.json")
    golden_replay = _health(project_root, "golden_replay_regression_latest.json")
    promotion_quality = _health(project_root, "promotion_quality_gate_latest.json")
    promotion_packet = _champion(project_root, "promotion_autopilot_packet_latest.json")
    coverage_seed = _walk_forward(project_root, "coverage_seed_latest.json")
    coverage_gap = _walk_forward(project_root, "coverage_gap_closer_latest.json")

    repair_plan: list[dict[str, Any]] = []
    operator_followups: list[str] = []

    storage_status = str(storage.get("overall_status") or "missing").strip().lower()
    storage_blocked = storage_status in {"blocked", "critical", "missing"}
    storage_degraded = storage_status in {"degraded", "needs_work"}
    retention_debt_gb = _safe_float(((storage.get("storage") or {}).get("retention_debt_gb")), 0.0)
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    repair_plan.append(
        _plan_row(
            "storage_backpressure",
            "storage_pressure_clearance",
            f"storage_status={storage_status} pressure_index={pressure_index:.3f} retention_debt_gb={retention_debt_gb:.3f}",
            _opsctl(project_root, "storage-pressure-clearance", "--apply", "--force-clear-stale-gate", "--json"),
        )
    ) if storage_blocked or storage_degraded or retention_debt_gb > 0.25 else None
    repair_plan.append(
        _plan_row(
            "storage_backpressure",
            "storage_backpressure_autopilot",
            "clear queue pressure, retention debt, and writer backpressure before training",
            _opsctl(
                project_root,
                "storage-backpressure-autopilot",
                "--apply",
                "--poll-seconds",
                "5",
                "--wait-timeout-seconds",
                "30",
                "--command-timeout-seconds",
                "180",
                "--json",
            ),
        )
    ) if storage_blocked or retention_debt_gb > 1.0 else None
    repair_plan.append(
        _plan_row(
            "storage_backpressure",
            "stateful_storage_regression_guard",
            "verify local/external stateful storage routing before trusting recovery",
            _opsctl(project_root, "stateful-storage-regression-guard", "--apply", "--json"),
        )
    ) if storage_blocked or storage_degraded else None

    required_failures = [str(x) for x in collectors.get("required_failures", []) if str(x).strip()]
    soft_failures = [str(x) for x in collectors.get("soft_failures", []) if str(x).strip()]
    source_overall = source_verification.get("overall") if isinstance(source_verification.get("overall"), dict) else {}
    unverified_sources = [str(x) for x in source_overall.get("unverified_sources", []) if str(x).strip()]
    collector_blocked = bool(required_failures)
    collector_degraded = bool(soft_failures or unverified_sources)

    if "market_micro_context" in required_failures or "market_micro_context" in unverified_sources:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "bounded_market_micro_sync",
                "market_micro_context is the required collector failure; run bounded sync/fallback",
                _opsctl(project_root, "market-micro-sync", "--json"),
            )
        )
    if "tradingeconomics_guest" in soft_failures:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "tradingeconomics_sync",
                "tradingeconomics guest payload is stale",
                _opsctl(project_root, "tradingeconomics-sync", "--json"),
            )
        )
    if "bls_census" in soft_failures or "public_macro_feeds" in unverified_sources or "macro_crossstack" in unverified_sources:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "macro_context_sync",
                "public macro feeds or macro cross-stack verification are stale/partial",
                _opsctl(project_root, "macro-context-sync", "--json"),
            )
        )
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "macro_crosscheck",
                "refresh cross-artifact macro verification after macro sync",
                _opsctl(project_root, "macro-crosscheck", "--json"),
            )
        )
    if "sec_edgar_context" in soft_failures or "sec_edgar_context" in unverified_sources:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "sec_edgar_sync",
                "SEC EDGAR artifact is stale or empty",
                _opsctl(project_root, "sec-edgar-sync", "--json"),
            )
        )
    if "extended_quant_context" in soft_failures or "extended_quant_context" in unverified_sources:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "extended_quant_sync",
                "extended quant context artifact is stale or empty",
                _opsctl(project_root, "extended-quant-sync", "--json"),
            )
        )
    if "options_flow_context" in soft_failures:
        repair_plan.append(
            _plan_row(
                "collectors_sources",
                "options_flow_sync",
                "options flow context is stale or below contract",
                _opsctl(project_root, "options-flow-sync", "--json"),
            )
        )
    repair_plan.append(
        _plan_row(
            "collectors_sources",
            "collector_contract_recheck",
            "recheck collector contracts after refreshes",
            _opsctl(project_root, "collector-contracts", "--json"),
        )
    ) if collector_blocked or collector_degraded else None
    repair_plan.append(
        _plan_row(
            "collectors_sources",
            "source_verification_recheck",
            "recheck source verification after refreshes",
            _opsctl(project_root, "source-verification", "--json"),
        )
    ) if collector_blocked or collector_degraded else None

    training_status = str(training.get("overall_status") or "missing").strip().lower()
    training_blocked = training_status in {"blocked", "missing"}
    training_degraded = training_status in {"degraded", "needs_work"}
    training_metrics = {
        "training_quality_score": _safe_float(training.get("training_quality_score"), 0.0),
        "retention_debt_gb": _safe_float(((training.get("data_ops") or {}).get("retention_debt_gb")), 0.0),
        "top_mode_share": _safe_float(((training.get("dataset_shape") or {}).get("top_mode_share")), 0.0),
        "considered_bots": _safe_int(((training.get("rollout") or {}).get("considered_bots")), 0),
        "min_considered_bots": _safe_int(((training.get("rollout") or {}).get("min_considered_bots")), 4),
    }
    training_gate = "storage_backpressure" if storage_blocked else ("collectors_sources" if collector_blocked else "")
    repair_plan.append(
        _plan_row(
            "training_eligibility",
            "runtime_training_snapshot",
            "refresh runtime snapshot for lane-specific eligibility checks",
            _opsctl(project_root, "runtime-training-snapshot", "--json"),
            gated_by=training_gate,
        )
    ) if training_blocked or training_degraded else None
    repair_plan.append(
        _plan_row(
            "training_eligibility",
            "coverage_seed",
            "seed walk-forward coverage before promotion/retrain",
            _opsctl(project_root, "coverage-seed", "--write-queue", "--json"),
            gated_by=training_gate,
        )
    ) if training_blocked or training_metrics["considered_bots"] < training_metrics["min_considered_bots"] else None
    repair_plan.append(
        _plan_row(
            "training_eligibility",
            "coverage_gap_closer_stage",
            "stage coverage repair candidates without broad retrain",
            _opsctl(project_root, "coverage-gap-closer", "--apply-stage", "--json"),
            gated_by=training_gate,
        )
    ) if training_blocked or training_degraded else None
    repair_plan.append(
        _plan_row(
            "training_eligibility",
            "training_quality_recheck",
            "recheck training eligibility after storage/source/coverage repairs",
            _opsctl(project_root, "training-quality", "--json"),
            gated_by=training_gate,
        )
    ) if training_blocked or training_degraded else None

    weak_sleeves = ((training.get("targeted_actions") or {}).get("weak_sleeves") or [])
    paper_blocked = bool(weak_sleeves)
    if paper_blocked:
        repair_plan.append(
            _plan_row(
                "paper_feedback",
                "bot_quality_autopilot",
                f"weak_sleeve_count={len(weak_sleeves)}; feed losses into probation/threshold work",
                _opsctl(project_root, "bot-quality-autopilot", "--apply", "--json"),
                gated_by=training_gate,
            )
        )
        repair_plan.append(
            _plan_row(
                "paper_feedback",
                "one_numbers_regression_guard",
                "guard the paper/report metrics after weak sleeve feedback",
                _opsctl(project_root, "one-numbers-regression-guard", "--apply", "--json"),
                gated_by=training_gate,
            )
        )

    replay_blocked = not bool(replay_hash.get("ok", False)) or not bool(golden_replay.get("ok", False))
    promotion_ready = bool(promotion_packet.get("promotion_ready", False)) or bool(promotion_quality.get("ok", False))
    promotion_blocked = replay_blocked or not promotion_ready
    promotion_gate = "training_eligibility" if (training_blocked or storage_blocked or collector_blocked) else ""
    repair_plan.append(
        _plan_row(
            "promotion_replay",
            "replay_hash_registry",
            "refresh replay hash registry before promotion",
            _opsctl(project_root, "replay-hash-registry", "--json"),
            gated_by=promotion_gate,
        )
    ) if promotion_blocked else None
    repair_plan.append(
        _plan_row(
            "promotion_replay",
            "golden_replay_regression",
            "refresh golden replay regression proof",
            _opsctl(project_root, "golden-replay-regression", "--json"),
            gated_by=promotion_gate,
        )
    ) if promotion_blocked else None
    repair_plan.append(
        _plan_row(
            "promotion_replay",
            "promotion_autopilot_packet",
            "assemble/refresh promotion packet only after upstream gates are visible",
            _opsctl(project_root, "promotion-autopilot", "--json"),
            gated_by=promotion_gate,
        )
    ) if promotion_blocked else None
    repair_plan.append(
        _plan_row(
            "promotion_replay",
            "promotion_quality_gate",
            "recheck promotion quality gate",
            _script(project_root, "scripts/promotion_quality_gate.py", "--json"),
            gated_by=promotion_gate,
        )
    ) if promotion_blocked else None

    layer_statuses = {
        "storage_backpressure": _layer_status(storage_blocked, storage_degraded or retention_debt_gb > 0.25),
        "collectors_sources": _layer_status(collector_blocked, collector_degraded),
        "training_eligibility": _layer_status(training_blocked, training_degraded),
        "paper_feedback": _layer_status(False, paper_blocked),
        "promotion_replay": _layer_status(promotion_blocked, False),
    }
    blocked_layers = [name for name, status in layer_statuses.items() if status == "blocked"]
    degraded_layers = [name for name, status in layer_statuses.items() if status == "degraded"]

    attempts: list[dict[str, Any]] = []
    if apply:
        for row in repair_plan:
            cmd = list(row.get("cmd") or [])
            if not cmd:
                continue
            gated_by = str(row.get("gated_by") or "").strip()
            if gated_by and layer_statuses.get(gated_by) == "blocked":
                attempts.append(
                    {
                        "name": row.get("name"),
                        "layer": row.get("layer"),
                        "cmd": cmd,
                        "rc": 0,
                        "timed_out": False,
                        "skipped": True,
                        "skip_reason": f"blocked_by_{gated_by}",
                    }
                )
                continue
            attempts.append({**_run_json(cmd, cwd=project_root, timeout_sec=timeout_sec), "name": row.get("name"), "layer": row.get("layer")})

    hard_failed_attempts = [
        row for row in attempts if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2, 124}
    ]
    timeout_attempts = [row for row in attempts if bool(row.get("timed_out", False)) or int(row.get("rc", 0)) == 124]
    if timeout_attempts:
        operator_followups.append("review timed-out bounded repair attempts; they were recorded without blocking the whole autopilot loop")

    overall_status = "ready"
    if blocked_layers or hard_failed_attempts:
        overall_status = "blocked"
    elif degraded_layers or repair_plan or timeout_attempts:
        overall_status = "degraded"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "layer_statuses": layer_statuses,
        "blocked_layers": blocked_layers,
        "degraded_layers": degraded_layers,
        "repair_plan": repair_plan,
        "applyable_repair_count": len(repair_plan),
        "attempts": [
            {
                "name": row.get("name"),
                "layer": row.get("layer"),
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
                "skipped": bool(row.get("skipped", False)),
                "skip_reason": str(row.get("skip_reason") or ""),
            }
            for row in attempts
        ],
        "operator_followups": ordered_unique(operator_followups),
        "metrics": {
            "pressure_index": pressure_index,
            "retention_debt_gb": retention_debt_gb,
            "required_collector_failures": required_failures,
            "soft_collector_failures": soft_failures,
            "unverified_sources": unverified_sources,
            **training_metrics,
            "weak_sleeve_count": len(weak_sleeves),
            "coverage_seed_queue_size": _safe_int(coverage_seed.get("coverage_seed_queue_size"), 0),
            "coverage_gap_status": str((coverage_gap.get("autopilot_contract") or {}).get("overall_status") or coverage_gap.get("overall_status") or ""),
            "replay_hash_ok": bool(replay_hash.get("ok", False)),
            "golden_replay_ok": bool(golden_replay.get("ok", False)),
            "promotion_ready": bool(promotion_ready),
        },
        "assigned_infrabot": "system_cleanliness_infrabot",
        "infra_assistants": [
            "storage_pressure_clearance_bot",
            "storage_backpressure_autopilot",
            "stateful_storage_regression_guard",
            "collector_contracts",
            "source_verification_report",
            "bot_quality_autopilot",
            "one_numbers_regression_guard",
            "replay_hash_registry_guard",
            "golden_replay_regression_guard",
            "promotion_autopilot_packet",
            "writer_lock_handoff_infrabot",
            "health_truth_reconciler_infrabot",
            "provider_cross_verification_infrabot",
            "paper_feedback_repair_infrabot",
            "promotion_replay_gate_infrabot",
            "bot_data_labeling_targeter_infrabot",
            "recovery_drill_infrabot",
            "self_audit_freshness_infrabot",
            "cotenant_headroom_guard_infrabot",
            "protected_volume_boundary_infrabot",
            "training_batch_readiness_infrabot",
            "paper_execution_queue_reconciler_infrabot",
            "duplicate_alpha_compression_infrabot",
            "livefeed_mirror_continuity_infrabot",
            "auth_lease_preflight_infrabot",
            "market_explanation_evidence_infrabot",
        ],
        "recommended_actions": ordered_unique(
            [
                "clear storage/backpressure before broad retrains" if storage_blocked else "",
                "repair market_micro_context before marking collector contracts clean" if collector_blocked else "",
                "keep new sleeves collect-only until source coverage and training eligibility are ready",
                "use lane-specific retraining and dominance caps after ingestion clears" if training_blocked or training_degraded else "",
                "hold promotion until replay lineage and packet gates are clean" if promotion_blocked else "",
            ]
        ),
        "source_files": {
            "storage": str(health_root / "ingestion_storage_control_latest.json"),
            "collectors": str(health_root / "collector_contracts_latest.json"),
            "source_verification": str(health_root / "source_verification_latest.json"),
            "training_quality": str(health_root / "training_quality_control_latest.json"),
            "paper_performance": str(health_root / "paper_performance_latest.json"),
            "replay_hash": str(health_root / "replay_hash_registry_guard_latest.json"),
            "golden_replay": str(health_root / "golden_replay_regression_latest.json"),
            "promotion_autopilot": str(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Five-layer system cleanliness autopilot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_cleanliness_autopilot "
            f"overall_status={payload.get('overall_status')} "
            f"repairs={payload.get('applyable_repair_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
