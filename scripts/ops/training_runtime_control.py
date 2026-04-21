#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_runtime_control_latest.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml_backend_contract import resolve_backend_contract
from core.runtime_python import resolve_runtime_python


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_minutes(raw: Any) -> float | None:
    ts = _parse_ts(raw)
    if ts is None:
        return None
    return max((datetime.now(timezone.utc) - ts).total_seconds() / 60.0, 0.0)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _runtime_backend_probe(project_root: Path) -> dict[str, Any]:
    runtime_python = resolve_runtime_python(project_root)
    current_python = Path(sys.executable).resolve()
    probe = {
        "runtime_python_path": str(runtime_python),
        "current_python_path": str(current_python),
        "runtime_python_exists": runtime_python.exists(),
        "runtime_matches_current": runtime_python.exists() and runtime_python.resolve() == current_python,
        "installed_backends": {},
        "native_contract": {},
        "portable_contract": {},
        "probe_rc": 127,
        "probe_error": "",
        "parity_state": "missing_runtime_python",
    }
    if not runtime_python.exists():
        return probe

    cmd = [
        str(runtime_python),
        "-c",
        (
            "import importlib.util, json, platform, sys; "
            "mods={name:(importlib.util.find_spec(name) is not None) for name in ('mlx','torch','onnxruntime','tensorflow','jax')}; "
            "print(json.dumps({'python': sys.version.split()[0], 'platform': platform.platform(), 'modules': mods}, ensure_ascii=True))"
        ),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        probe["probe_rc"] = int(proc.returncode)
        if proc.returncode == 0:
            parsed = {}
            for raw in reversed([line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]):
                try:
                    candidate = json.loads(raw)
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            modules = parsed.get("modules") if isinstance(parsed.get("modules"), dict) else {}
            installed = {
                "mlx": bool(modules.get("mlx", False)),
                "pytorch": bool(modules.get("torch", False)),
                "onnx": bool(modules.get("onnxruntime", False)),
                "tensorflow": bool(modules.get("tensorflow", False)),
                "jax": bool(modules.get("jax", False)),
            }
            probe["installed_backends"] = installed
            probe["runtime_python_version"] = str(parsed.get("python") or "")
            probe["runtime_platform"] = str(parsed.get("platform") or "")
            probe["native_contract"] = resolve_backend_contract("native_default", mode="native", installed=installed)
            probe["portable_contract"] = resolve_backend_contract("portable_auto", mode="portable", installed=installed)
        else:
            probe["probe_error"] = "\n".join((proc.stderr or "").splitlines()[-8:]) or "\n".join((proc.stdout or "").splitlines()[-8:])
    except Exception as exc:
        probe["probe_error"] = str(exc)

    native_contract = probe.get("native_contract") if isinstance(probe.get("native_contract"), dict) else {}
    if int(probe.get("probe_rc", 127)) != 0:
        probe["parity_state"] = "runtime_probe_failed"
    elif bool(native_contract.get("runtime_training_supported", False)):
        probe["parity_state"] = "ready"
    elif bool((probe.get("portable_contract") or {}).get("shadow_replay_supported", False)):
        probe["parity_state"] = "portable_only"
    else:
        probe["parity_state"] = "native_backend_missing"
    return probe


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
    ):
        if token in lowered:
            return family
    return "general"


def _sequence_timeout_reason(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("reason") or ""),
            str(row.get("stdout_tail") or ""),
            str(row.get("stderr_tail") or ""),
        ]
    ).lower()
    if "loading_sequences" in text:
        return "loading_sequences_timeout"
    if "memory_guard" in text:
        return "memory_guard"
    if "timeout" in text:
        return "runtime_timeout"
    return ""


def _build_precompute_targets(
    *,
    training_quality: dict[str, Any],
    retrain_scorecard: dict[str, Any],
    coverage_seed: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}

    def ensure(bot_id: str) -> dict[str, Any]:
        row = targets.setdefault(
            bot_id,
            {
                "bot_id": bot_id,
                "family": _bot_family(bot_id),
                "priority": 0.0,
                "reasons": [],
                "actions": [],
            },
        )
        return row

    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    for bot_id in targeted_actions.get("targeted_retrain_bot_ids") or []:
        bot = str(bot_id or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 20.0
        row["reasons"].append("targeted_retrain")
        row["actions"].append("precompute_or_refresh_shared_snapshot")

    for bot_id in targeted_actions.get("quality_probation_bot_ids") or []:
        bot = str(bot_id or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 12.0
        row["reasons"].append("quality_probation")
        row["actions"].append("retry_after_runtime_cache_refresh")

    for failure in retrain_scorecard.get("failure_details") or []:
        if not isinstance(failure, dict):
            continue
        bot = str(failure.get("bot_id") or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + 30.0
        timeout_reason = _sequence_timeout_reason(failure)
        row["reasons"].append(timeout_reason or "retrain_failure")
        row["actions"].append("pin_sequence_cache_before_retry")

    for seed_row in coverage_seed.get("seed_queue") or []:
        if not isinstance(seed_row, dict):
            continue
        bot = str(seed_row.get("bot_id") or "").strip()
        if not bot:
            continue
        row = ensure(bot)
        row["priority"] = float(row["priority"]) + min(_safe_float(seed_row.get("priority"), 0.0), 15.0)
        row["reasons"].append("coverage_seed")
        row["actions"].append("reuse_shared_snapshot_for_walk_forward")

    out: list[dict[str, Any]] = []
    for row in targets.values():
        row["reasons"] = _ordered_unique(list(row.get("reasons") or []))
        row["actions"] = _ordered_unique(list(row.get("actions") or []))
        out.append(row)
    out.sort(key=lambda row: (-_safe_float(row.get("priority"), 0.0), str(row.get("bot_id") or "")))
    return out


def build_payload(project_root: Path = PROJECT_ROOT, *, fresh_minutes: int = 360, limit: int = 8) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    snapshot = _load_json(health_root / "runtime_training_snapshot_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    retrain_scorecard = _load_json(health_root / "retrain_scorecard_latest.json")
    training_success = _load_json(health_root / "training_success_latest.json")
    resource_guard = _load_json(health_root / "resource_guard_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")
    coverage_seed = _load_json(walk_root / "coverage_seed_latest.json")
    runtime_probe = _runtime_backend_probe(project_root)

    snapshot_age_minutes = _age_minutes(snapshot.get("timestamp_utc"))
    snapshot_fresh = bool(
        snapshot
        and _safe_int(snapshot.get("sequence_count"), 0) > 0
        and _safe_int(snapshot.get("row_count"), 0) > 0
        and snapshot_age_minutes is not None
        and snapshot_age_minutes <= max(int(fresh_minutes), 1)
    )
    sequence_count = _safe_int(snapshot.get("sequence_count"), 0)
    row_count = _safe_int(snapshot.get("row_count"), 0)
    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    precompute_targets = _build_precompute_targets(
        training_quality=training_quality,
        retrain_scorecard=retrain_scorecard,
        coverage_seed=coverage_seed,
    )
    resource_guard_ok = bool(resource_guard.get("resource_guard_ok", True))
    memory_pressure_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower() or "unknown"
    operating_mode = str(health_gates.get("recommended_operating_mode") or "").strip() or "unknown"
    backpressure_severe = bool(health_gates.get("inputs", {}).get("backpressure_overload_severe", False)) if isinstance(health_gates.get("inputs"), dict) else False

    parity_state = str(runtime_probe.get("parity_state") or "")
    training_failure_details = training_success.get("failure_details") if isinstance(training_success.get("failure_details"), list) else []
    training_quality_blocked = str(training_quality.get("overall_status") or "").strip().lower() == "blocked"
    mlx_failure_detected = any(
        "no module named 'mlx'" in " ".join(
            str((row or {}).get(field) or "")
            for field in ("reason", "stdout_tail", "stderr_tail")
        ).lower()
        for row in training_failure_details
        if isinstance(row, dict)
    )
    mlx_runtime_available = bool(((runtime_probe.get("installed_backends") or {}).get("mlx", False)))
    mlx_failure_active = bool(mlx_failure_detected and not mlx_runtime_available)
    core_runtime_ready = bool(
        snapshot_fresh
        and resource_guard_ok
        and parity_state not in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"}
        and not mlx_failure_active
    )
    coverage_repair_ready = bool(
        core_runtime_ready
        and (
            _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0) > 0
            or len(precompute_targets) > 0
        )
    )

    overall_status = "ready"
    if not snapshot_fresh or not resource_guard_ok:
        overall_status = "constrained"
    if backpressure_severe:
        overall_status = "blocked"
    elif parity_state in {"missing_runtime_python", "runtime_probe_failed", "native_backend_missing"} or mlx_failure_active:
        overall_status = "blocked"
    elif training_quality_blocked:
        overall_status = "degraded" if coverage_repair_ready else "blocked"

    recommended_actions: list[str] = []
    if not snapshot_fresh:
        recommended_actions.append("refresh the shared runtime training snapshot before retrying targeted retrains")
    if any("loading_sequences_timeout" in list(row.get("reasons") or []) for row in precompute_targets[: max(int(limit), 1)]):
        recommended_actions.append("precompute or reuse shared sequence caches for bots that timed out in loading_sequences")
    if not resource_guard_ok or memory_pressure_state not in {"green", "unknown"}:
        recommended_actions.append("wait for green memory pressure before forcing targeted retrains that expand sequence windows")
    if backpressure_severe:
        recommended_actions.append("treat retrain workers as background-only until ingestion backpressure exits the severe state")
    if _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0) > 0:
        recommended_actions.append("reuse the shared snapshot when seeding walk-forward coverage so promotion coverage improves without rebuilding runtime inputs")
    if parity_state in {"missing_runtime_python", "runtime_probe_failed"}:
        recommended_actions.append("repair the runtime python selection before retrying MLX-backed retrains")
    elif parity_state == "native_backend_missing" or mlx_failure_active:
        recommended_actions.append("install or repair MLX inside the runtime interpreter so native retrains stop failing before model code loads")
    elif parity_state == "portable_only":
        recommended_actions.append("keep non-MLX backends in replay and sidecar duty until the native runtime regains MLX support")

    retry_pack = retrain_scorecard.get("retry_pack") if isinstance(retrain_scorecard.get("retry_pack"), dict) else {}
    snapshot_coverage = snapshot.get("coverage") if isinstance(snapshot.get("coverage"), dict) else {}
    repair_contract = {
        "parity_state": parity_state,
        "runtime_python_path": str(runtime_probe.get("runtime_python_path") or ""),
        "runtime_matches_current": bool(runtime_probe.get("runtime_matches_current", False)),
        "probe_rc": int(runtime_probe.get("probe_rc", 0) or 0),
        "verify_runtime_command": [
            str(runtime_probe.get("runtime_python_path") or ""),
            "-c",
            "import mlx, sys; print(sys.executable)",
        ]
        if str(runtime_probe.get("runtime_python_path") or "")
        else [],
        "retry_pack_command": list(retry_pack.get("command") or []),
        "portable_contract_roles": list(((runtime_probe.get("portable_contract") or {}).get("roles_supported") or [])),
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "overall_status": overall_status,
        "snapshot_ready": snapshot_fresh,
        "snapshot_age_minutes": round(float(snapshot_age_minutes), 3) if snapshot_age_minutes is not None else None,
        "fresh_window_minutes": int(max(int(fresh_minutes), 1)),
        "snapshot": {
            "sequence_count": sequence_count,
            "row_count": row_count,
            "rows_path": str(snapshot.get("rows_path") or ""),
            "lookback_days": _safe_int(snapshot.get("lookback_days"), 0),
            "top_modes": list(snapshot_coverage.get("top_modes") or [])[:5],
            "top_sequences": list(snapshot_coverage.get("top_sequences") or [])[:5],
        },
        "resource_guard": {
            "ok": resource_guard_ok,
            "memory_pressure_state": memory_pressure_state,
            "swap_used_gb": round(_safe_float(resource_guard.get("swap_used_gb"), 0.0), 3),
        },
        "runtime_backend_parity": {
            "parity_state": parity_state,
            "runtime_python_path": str(runtime_probe.get("runtime_python_path") or ""),
            "current_python_path": str(runtime_probe.get("current_python_path") or ""),
            "runtime_python_exists": bool(runtime_probe.get("runtime_python_exists", False)),
            "runtime_matches_current": bool(runtime_probe.get("runtime_matches_current", False)),
            "runtime_python_version": str(runtime_probe.get("runtime_python_version") or ""),
            "runtime_platform": str(runtime_probe.get("runtime_platform") or ""),
            "installed_backends": runtime_probe.get("installed_backends") if isinstance(runtime_probe.get("installed_backends"), dict) else {},
            "native_contract": runtime_probe.get("native_contract") if isinstance(runtime_probe.get("native_contract"), dict) else {},
            "portable_contract": runtime_probe.get("portable_contract") if isinstance(runtime_probe.get("portable_contract"), dict) else {},
            "probe_error": str(runtime_probe.get("probe_error") or ""),
            "mlx_failure_detected": mlx_failure_detected,
            "mlx_failure_active": mlx_failure_active,
        },
        "training_quality": {
            "overall_status": str(training_quality.get("overall_status") or ""),
            "training_quality_score": round(_safe_float(training_quality.get("training_quality_score"), 0.0), 3),
            "top_priorities": list(training_quality.get("top_priorities") or [])[:6],
            "targeted_retrain_bot_ids": list(targeted_actions.get("targeted_retrain_bot_ids") or []),
        },
        "coverage_seed": {
            "coverage_shortfall_bots": _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0),
            "seed_queue_size": len(coverage_seed.get("seed_queue") or []),
        },
        "coverage_repair_ready": bool(coverage_repair_ready),
        "retry_pack": {
            "command": list(retry_pack.get("command") or []),
            "include_bot_ids": list(retry_pack.get("include_bot_ids") or []),
            "skip_master_update": bool(retry_pack.get("skip_master_update", False)),
        },
        "operating_mode": operating_mode,
        "precompute_targets": precompute_targets[: max(int(limit), 1)],
        "repair_contract": repair_contract,
        "recommended_actions": _ordered_unique(recommended_actions),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a training-runtime control plane for snapshot reuse, cache posture, and targeted precompute retries.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--fresh-minutes", type=int, default=360)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), fresh_minutes=int(args.fresh_minutes), limit=int(args.limit))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_runtime_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"snapshot_ready={int(bool(payload.get('snapshot_ready', False)))} "
            f"precompute_targets={len(payload.get('precompute_targets') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
