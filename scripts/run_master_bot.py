import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.master_bot import MasterBot


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_gate_debug(gate: dict) -> dict:
    thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    fail_examples = gate.get("fail_examples") if isinstance(gate.get("fail_examples"), list) else []
    return {
        "promote_ok": bool(gate.get("promote_ok", False)),
        "coverage_ok": bool(gate.get("coverage_ok", False)),
        "considered_bots": int(gate.get("considered_bots", 0) or 0),
        "failed_bots": int(gate.get("failed_bots", 0) or 0),
        "fail_share": float(gate.get("fail_share", 0.0) or 0.0),
        "thresholds": {
            "min_forward_mean": thresholds.get("min_forward_mean"),
            "min_delta": thresholds.get("min_delta"),
            "min_trading_quality_score": thresholds.get("min_trading_quality_score"),
            "max_overfit_gap": thresholds.get("max_overfit_gap"),
            "max_fail_share": thresholds.get("max_fail_share"),
            "max_severe_overfit_share": thresholds.get("max_severe_overfit_share"),
            "min_runs_per_bot": thresholds.get("min_runs_per_bot"),
            "min_considered_bots": thresholds.get("min_considered_bots"),
        },
        "top_fail_examples": fail_examples[:5],
    }


def _bool_gate_ok(path: Path, gate_name: str) -> tuple[bool, str, dict]:
    if not path.exists():
        return False, f"missing_{gate_name}:{path}", {}
    obj = _read_json(path)
    if bool(obj.get("ok", False)):
        return True, "ok", {}
    detail = {
        "path": str(path),
        "ok": bool(obj.get("ok", False)),
        "thresholds": obj.get("thresholds", {}),
    }
    if isinstance(obj.get("immature_active_examples"), list):
        detail["immature_active_examples"] = obj.get("immature_active_examples", [])[:5]
    if isinstance(obj.get("overfit_examples"), list):
        detail["overfit_examples"] = obj.get("overfit_examples", [])[:5]
    if isinstance(obj.get("leak_like_examples"), list):
        detail["leak_like_examples"] = obj.get("leak_like_examples", [])[:5]
    if isinstance(obj.get("failed_checks"), list):
        detail["failed_checks"] = obj.get("failed_checks", [])[:10]
    return False, f"{gate_name}_blocked", detail


def _canary_gate_ok(
    promotion_gate_file: Path,
    stability_file: Path,
    require_graduation_gate: bool,
    graduation_file: Path,
    require_leak_overfit_gate: bool,
    leak_overfit_file: Path,
) -> tuple[bool, str, dict]:
    if not promotion_gate_file.exists():
        return False, f"missing_promotion_gate:{promotion_gate_file}", {}
    gate = _read_json(promotion_gate_file)
    if not bool(gate.get("promote_ok", False)):
        return False, "promotion_gate_blocked", _build_gate_debug(gate)

    if not stability_file.exists():
        return False, f"missing_stability_file:{stability_file}", {}
    stability = _read_json(stability_file)
    if not bool(stability.get("ok", False)):
        failed = stability.get("failed_checks") or []
        detail = {"failed_checks": failed if isinstance(failed, list) else ["unknown"]}
        return False, f"stability_failed:{','.join(failed) if isinstance(failed, list) else 'unknown'}", detail

    if require_graduation_gate:
        ok, reason, detail = _bool_gate_ok(graduation_file, "graduation_gate")
        if not ok:
            return False, reason, detail

    if require_leak_overfit_gate:
        ok, reason, detail = _bool_gate_ok(leak_overfit_file, "leak_overfit_gate")
        if not ok:
            return False, reason, detail

    return True, "ok", {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/update master bot policy from sub-bot outcomes")
    parser.add_argument("--preferred-low", type=float, default=0.55)
    parser.add_argument("--preferred-high", type=float, default=0.65)
    parser.add_argument("--deactivate-below", type=float, default=0.50)
    parser.add_argument("--quality-floor", type=float, default=0.52)
    parser.add_argument("--promotion-margin", type=float, default=0.005)
    parser.add_argument("--no-improvement-retire-streak", type=int, default=3)
    parser.add_argument("--min-active-bots", type=int, default=int(os.getenv("MIN_ACTIVE_BOTS", "20")))
    parser.add_argument("--correlation-prune-threshold", type=float, default=float(os.getenv("CORRELATION_PRUNE_THRESHOLD", "0.92")))

    parser.add_argument("--require-canary-gate", action="store_true", default=os.getenv("REQUIRE_CANARY_PROMOTION_GATE", "1") == "1")
    parser.add_argument("--no-require-canary-gate", dest="require_canary_gate", action="store_false")

    parser.add_argument("--require-graduation-gate", action="store_true", default=os.getenv("REQUIRE_NEW_BOT_GRADUATION_GATE", "1") == "1")
    parser.add_argument("--no-require-graduation-gate", dest="require_graduation_gate", action="store_false")

    parser.add_argument("--require-leak-overfit-gate", action="store_true", default=os.getenv("REQUIRE_LEAK_OVERFIT_GATE", "1") == "1")
    parser.add_argument("--no-require-leak-overfit-gate", dest="require_leak_overfit_gate", action="store_false")

    parser.add_argument("--promotion-gate-file", default=str(Path(PROJECT_ROOT) / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--stability-file", default=str(Path(PROJECT_ROOT) / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--graduation-file", default=str(Path(PROJECT_ROOT) / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--leak-overfit-file", default=str(Path(PROJECT_ROOT) / "governance" / "health" / "leak_overfit_guard_latest.json"))

    parser.add_argument("--print-full", action="store_true")
    args = parser.parse_args()

    if args.require_canary_gate:
        ok, reason, detail = _canary_gate_ok(
            Path(args.promotion_gate_file),
            Path(args.stability_file),
            require_graduation_gate=bool(args.require_graduation_gate),
            graduation_file=Path(args.graduation_file),
            require_leak_overfit_gate=bool(args.require_leak_overfit_gate),
            leak_overfit_file=Path(args.leak_overfit_file),
        )
        if not ok:
            print(f"Master bot update blocked by canary gate: {reason}")
            if detail:
                print("Canary gate detail:")
                print(json.dumps(detail, ensure_ascii=True, indent=2))
            raise SystemExit(2)

    master = MasterBot(
        project_root=PROJECT_ROOT,
        preferred_low=args.preferred_low,
        preferred_high=args.preferred_high,
        deactivate_below=args.deactivate_below,
        quality_floor=args.quality_floor,
        promotion_margin=args.promotion_margin,
        no_improvement_retire_streak=args.no_improvement_retire_streak,
        min_active_bots=args.min_active_bots,
        correlation_prune_threshold=args.correlation_prune_threshold,
    )

    payload = master.train_from_outcomes()

    print("Master bot registry updated")
    print(f"Registry: {os.path.join(PROJECT_ROOT, 'master_bot_registry.json')}")
    print(json.dumps(payload.get("summary", {}), indent=2))

    if args.print_full:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
