import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, v])


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Build daily executive dashboard CSV/MD.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "exports" / "executive_dashboard"))
    parser.add_argument("--one-numbers", default=str(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"))
    parser.add_argument("--bot-stack", default=str(PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.json"))
    parser.add_argument("--daily-verify", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--slo", default=str(PROJECT_ROOT / "governance" / "watchdog" / "sleeve_slo_latest.json"))
    parser.add_argument("--allocator", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--risk", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--budget", default=str(PROJECT_ROOT / "governance" / "risk" / "execution_budget_latest.json"))
    parser.add_argument("--distill", default=str(PROJECT_ROOT / "governance" / "distillation" / "teacher_student_plan_latest.json"))
    args = parser.parse_args()

    one = _read_json(Path(args.one_numbers))
    stack = _read_json(Path(args.bot_stack))
    verify = _read_json(Path(args.daily_verify))
    slo = _read_json(Path(args.slo))
    allocator = _read_json(Path(args.allocator))
    risk = _read_json(Path(args.risk))
    budget = _read_json(Path(args.budget))
    distill = _read_json(Path(args.distill))

    dq = _safe_float(one.get("data_quality_score"), 0.0)
    blocked_rate = _safe_float(one.get("combined_blocked_rate"), 0.0)
    stocks_pnl = _safe_float(one.get("stocks_pnl_proxy"), 0.0)
    crypto_pnl = _safe_float(one.get("crypto_pnl_proxy"), 0.0)

    failed_checks = ",".join((verify.get("failed_checks") or [])) if isinstance(verify.get("failed_checks"), list) else "none"
    alerts = slo.get("alerts") if isinstance(slo.get("alerts"), list) else []

    weights = allocator.get("target_weights") or {}
    risk_level = str(risk.get("risk_level", "unknown"))
    risk_score = _safe_float(risk.get("risk_score"), 0.0)
    gross_cap = _safe_float(((risk.get("limits") or {}).get("gross_exposure_cap")), 0.0)
    total_actions = int(((budget.get("global") or {}).get("max_total_actions_per_hour") or 0))
    total_open_orders = int(((budget.get("global") or {}).get("max_total_open_orders") or 0))

    distill_summary = (distill.get("summary") or {}) if isinstance(distill, dict) else {}
    distill_teacher_count = int(distill_summary.get("teacher_count") or 0)
    distill_student_count = int(distill_summary.get("student_count") or 0)
    distill_assignments = int(distill_summary.get("assignment_count") or 0)

    rows: list[tuple[str, str]] = [
        ("day_utc", args.day),
        ("generated_utc", datetime.now(timezone.utc).isoformat()),
        ("pipeline_data_quality_score", f"{dq:.2f}"),
        ("combined_blocked_rate", f"{blocked_rate:.6f}"),
        ("stocks_pnl_proxy", f"{stocks_pnl:.6f}"),
        ("crypto_pnl_proxy", f"{crypto_pnl:.6f}"),
        ("watchdog_restarts", str(one.get("watchdog_restarts", "0"))),
        ("sleeve_slo_overall_ok", str(bool(slo.get("overall_ok", False))).lower()),
        ("sleeve_slo_alert_count", str(len(alerts))),
        ("daily_auto_verify_ok", str(bool(verify.get("ok", False))).lower()),
        ("daily_auto_verify_failed_checks", failed_checks if failed_checks else "none"),
        ("bot_stack_status", str(((stack.get("overall_health") or {}).get("status") or "unknown"))),
        ("bot_stack_active_bots", str((((stack.get("registry") or {}).get("counts") or {}).get("active") or 0))),
        ("allocator_weight_core", str(weights.get("core", "n/a"))),
        ("allocator_weight_aggressive", str(weights.get("aggressive", "n/a"))),
        ("allocator_weight_dividend", str(weights.get("dividend", "n/a"))),
        ("allocator_weight_bond", str(weights.get("bond", "n/a"))),
        ("allocator_weight_crypto", str(weights.get("crypto", "n/a"))),
        ("portfolio_risk_level", risk_level),
        ("portfolio_risk_score", f"{risk_score:.4f}"),
        ("portfolio_gross_exposure_cap", f"{gross_cap:.6f}"),
        ("execution_total_actions_per_hour", str(total_actions)),
        ("execution_total_open_orders", str(total_open_orders)),
        ("distill_teacher_count", str(distill_teacher_count)),
        ("distill_student_count", str(distill_student_count)),
        ("distill_assignment_count", str(distill_assignments)),
        ("top_stock_1", str(one.get("stocks_top_symbol_1", "n/a"))),
        ("top_stock_2", str(one.get("stocks_top_symbol_2", "n/a"))),
        ("top_stock_3", str(one.get("stocks_top_symbol_3", "n/a"))),
        ("top_crypto_1", str(one.get("crypto_top_symbol_1", "n/a"))),
        ("top_crypto_2", str(one.get("crypto_top_symbol_2", "n/a"))),
        ("top_crypto_3", str(one.get("crypto_top_symbol_3", "n/a"))),
    ]

    for i, alert in enumerate(alerts[:5], start=1):
        rows.append((f"slo_alert_{i}", f"{alert.get('name','unknown')}:{'|'.join(alert.get('breaches',[]) or [])}"))
    for i in range(len(alerts[:5]) + 1, 6):
        rows.append((f"slo_alert_{i}", "none"))

    out_dir = Path(args.out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"executive_dashboard_{args.day}_{stamp}.csv"
    md_path = out_dir / f"executive_dashboard_{args.day}_{stamp}.md"

    _write_csv(csv_path, rows)

    md_lines = [
        f"# Executive Dashboard ({args.day})",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Health",
        f"- Data quality score: {dq:.2f}",
        f"- SLO overall ok: {str(bool(slo.get('overall_ok', False))).lower()} (alerts={len(alerts)})",
        f"- Daily verify ok: {str(bool(verify.get('ok', False))).lower()}",
        f"- Failed checks: {failed_checks if failed_checks else 'none'}",
        "",
        "## Portfolio Control",
        f"- Allocator weights (core/aggressive/dividend/bond/crypto): {weights.get('core','n/a')}/{weights.get('aggressive','n/a')}/{weights.get('dividend','n/a')}/{weights.get('bond','n/a')}/{weights.get('crypto','n/a')}",
        f"- Risk level/score: {risk_level}/{risk_score:.4f}",
        f"- Gross exposure cap: {gross_cap:.4f}",
        f"- Execution budget (actions/hr, open orders): {total_actions}, {total_open_orders}",
        "",
        "## Distillation",
        f"- Teachers: {distill_teacher_count}",
        f"- Students: {distill_student_count}",
        f"- Assignments: {distill_assignments}",
        "",
        "## Runtime",
        f"- Blocked rate: {blocked_rate:.4f}",
        f"- Stocks pnl proxy: {stocks_pnl:.6f}",
        f"- Crypto pnl proxy: {crypto_pnl:.6f}",
        f"- Watchdog restarts: {one.get('watchdog_restarts', '0')}",
        "",
        "## Bot Stack",
        f"- Status: {((stack.get('overall_health') or {}).get('status') or 'unknown')}",
        f"- Active bots: {(((stack.get('registry') or {}).get('counts') or {}).get('active') or 0)}",
        "",
        f"CSV: `{csv_path}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latest_csv = out_dir / "latest.csv"
    latest_md = out_dir / "latest.md"
    latest_json = out_dir / "latest.json"

    if latest_csv.exists() or latest_csv.is_symlink():
        latest_csv.unlink()
    if latest_md.exists() or latest_md.is_symlink():
        latest_md.unlink()

    latest_csv.symlink_to(csv_path)
    latest_md.symlink_to(md_path)
    latest_json.write_text(json.dumps({k: v for k, v in rows}, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
