import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def evaluate_quality(
    promotion_gate: dict[str, Any],
    daily_verify: dict[str, Any],
    graduation_gate: dict[str, Any],
    leak_overfit: dict[str, Any],
    replay_gate: dict[str, Any],
    reconciliation_slo: dict[str, Any],
    *,
    max_fail_share: float,
    min_considered_bots: int,
    require_replay: bool,
    require_reconciliation_slo: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    failed: list[str] = []

    considered = int(promotion_gate.get("considered_bots", 0) or 0)
    fail_share = float(promotion_gate.get("fail_share", 1.0) or 1.0)
    promote_ok = bool(promotion_gate.get("promote_ok", False))

    if not promote_ok:
        failed.append("promotion_gate_blocked")
    if considered < int(min_considered_bots):
        failed.append("insufficient_considered_bots")
    if fail_share > float(max_fail_share):
        failed.append("fail_share_above_limit")

    if not bool(daily_verify.get("ok", False)):
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
        "daily_verify_ok": bool(daily_verify.get("ok", False)),
        "graduation_ok": bool(graduation_gate.get("ok", False)),
        "leak_overfit_ok": bool(leak_overfit.get("ok", False)),
        "replay_ok": bool(replay_gate.get("ok", False)),
        "reconciliation_slo_ok": bool(reconciliation_slo.get("ok", False)),
    }
    return len(failed) == 0, failed, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Stricter promotion quality gate.")
    parser.add_argument("--promotion-gate-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--daily-verify-file", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--graduation-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--leak-overfit-file", default=str(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json"))
    parser.add_argument("--replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_end_to_end_latest.json"))
    parser.add_argument("--reconciliation-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--max-fail-share", type=float, default=0.25)
    parser.add_argument("--min-considered-bots", type=int, default=20)
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
    reconciliation = _load_json(Path(args.reconciliation_file))

    ok, failed_checks, details = evaluate_quality(
        promotion,
        daily_verify,
        graduation,
        leak_overfit,
        replay,
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
