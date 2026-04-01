import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_THRESHOLDS_FILE = PROJECT_ROOT / "governance" / "ops_thresholds.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _promotion_defaults() -> dict[str, Any]:
    payload = _load_json(OPS_THRESHOLDS_FILE)
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), dict) else {}
    return gates.get("promotion_gate") if isinstance(gates.get("promotion_gate"), dict) else {}


def _resolve_daily_verify_failures(
    daily_verify: dict[str, Any],
    *,
    graduation_gate: dict[str, Any],
    replay_hash_registry_gate: dict[str, Any],
    promotion_gate: dict[str, Any],
) -> tuple[list[str], list[str]]:
    failed = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    unresolved: list[str] = []
    resolved: list[str] = []
    for item in failed:
        name = str(item or "").strip()
        if name == "incomplete_run_recovered":
            resolved.append(name)
            continue
        if name == "new_bot_graduation_gate" and bool(graduation_gate.get("ok", False)):
            resolved.append(name)
            continue
        if name == "replay_hash_registry_guard" and bool(replay_hash_registry_gate.get("ok", False)):
            resolved.append(name)
            continue
        if name == "promotion_quality_gate" and bool(promotion_gate.get("promote_ok", False)):
            resolved.append(name)
            continue
        unresolved.append(name)
    return unresolved, resolved


def evaluate_quality(
    promotion_gate: dict[str, Any],
    daily_verify: dict[str, Any],
    graduation_gate: dict[str, Any],
    leak_overfit: dict[str, Any],
    replay_gate: dict[str, Any],
    replay_hash_registry_gate: dict[str, Any],
    reconciliation_slo: dict[str, Any],
    *,
    max_fail_share: float,
    min_considered_bots: int,
    require_replay: bool,
    require_reconciliation_slo: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    failed: list[str] = []

    considered = int(promotion_gate.get("considered_bots", 0) or 0)
    raw_fail_share = promotion_gate.get("fail_share", 1.0)
    fail_share = float(1.0 if raw_fail_share is None else raw_fail_share)
    promote_ok = bool(promotion_gate.get("promote_ok", False))
    unresolved_daily_verify, resolved_daily_verify = _resolve_daily_verify_failures(
        daily_verify,
        graduation_gate=graduation_gate,
        replay_hash_registry_gate=replay_hash_registry_gate,
        promotion_gate=promotion_gate,
    )

    if not promote_ok:
        failed.append("promotion_gate_blocked")
    if considered < int(min_considered_bots):
        failed.append("insufficient_considered_bots")
    if fail_share > float(max_fail_share):
        failed.append("fail_share_above_limit")

    if unresolved_daily_verify:
        failed.append("daily_verify_not_ok")

    if not bool(graduation_gate.get("ok", False)):
        failed.append("new_bot_graduation_not_ok")

    if not bool(leak_overfit.get("ok", False)):
        failed.append("leak_overfit_not_ok")

    if require_replay and not bool(replay_gate.get("ok", False)):
        failed.append("replay_determinism_not_ok")

    if require_reconciliation_slo and not bool(reconciliation_slo.get("ok", False)):
        failed.append("reconciliation_slo_not_ok")

    details = {
        "promotion": {
            "promote_ok": promote_ok,
            "considered_bots": considered,
            "fail_share": round(fail_share, 6),
        },
        "daily_verify_ok": len(unresolved_daily_verify) == 0,
        "daily_verify_unresolved_failed_checks": unresolved_daily_verify,
        "daily_verify_resolved_failed_checks": resolved_daily_verify,
        "graduation_ok": bool(graduation_gate.get("ok", False)),
        "leak_overfit_ok": bool(leak_overfit.get("ok", False)),
        "replay_ok": bool(replay_gate.get("ok", False)),
        "replay_hash_registry_ok": bool(replay_hash_registry_gate.get("ok", False)),
        "reconciliation_slo_ok": bool(reconciliation_slo.get("ok", False)),
    }
    return len(failed) == 0, failed, details


def main() -> int:
    defaults = _promotion_defaults()
    parser = argparse.ArgumentParser(description="Stricter promotion quality gate.")
    parser.add_argument("--promotion-gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--daily-verify-file", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--graduation-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--leak-overfit-file", default=str(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json"))
    parser.add_argument("--replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_end_to_end_latest.json"))
    parser.add_argument("--replay-hash-registry-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"))
    parser.add_argument("--reconciliation-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--max-fail-share", type=float, default=float(defaults.get("max_fail_share", 0.25)))
    parser.add_argument("--min-considered-bots", type=int, default=int(defaults.get("min_considered_bots", 4)))
    parser.add_argument("--require-replay", action="store_true", default=True)
    parser.add_argument("--no-require-replay", dest="require_replay", action="store_false")
    parser.add_argument("--require-reconciliation-slo", action="store_true", default=True)
    parser.add_argument("--no-require-reconciliation-slo", dest="require_reconciliation_slo", action="store_false")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    promotion = _load_json(Path(args.promotion_gate_file))
    daily_verify = _load_json(Path(args.daily_verify_file))
    graduation = _load_json(Path(args.graduation_file))
    leak_overfit = _load_json(Path(args.leak_overfit_file))
    replay = _load_json(Path(args.replay_file))
    replay_hash_registry = _load_json(Path(args.replay_hash_registry_file))
    reconciliation = _load_json(Path(args.reconciliation_file))

    ok, failed_checks, details = evaluate_quality(
        promotion,
        daily_verify,
        graduation,
        leak_overfit,
        replay,
        replay_hash_registry,
        reconciliation,
        max_fail_share=float(args.max_fail_share),
        min_considered_bots=int(args.min_considered_bots),
        require_replay=bool(args.require_replay),
        require_reconciliation_slo=bool(args.require_reconciliation_slo),
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "failed_checks": failed_checks,
        "thresholds": {
            "max_fail_share": float(args.max_fail_share),
            "min_considered_bots": int(args.min_considered_bots),
            "require_replay": bool(args.require_replay),
            "require_reconciliation_slo": bool(args.require_reconciliation_slo),
        },
        "details": details,
        "source_files": {
            "promotion_gate": str(args.promotion_gate_file),
            "daily_verify": str(args.daily_verify_file),
            "graduation": str(args.graduation_file),
            "leak_overfit": str(args.leak_overfit_file),
            "replay": str(args.replay_file),
            "replay_hash_registry": str(args.replay_hash_registry_file),
            "reconciliation": str(args.reconciliation_file),
        },
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        checks = ",".join(failed_checks) if failed_checks else "none"
        print(f"promotion_quality_gate_ok={int(ok)} failed_checks={checks}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
