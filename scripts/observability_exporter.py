import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric(name: str, value: float, labels: dict | None = None) -> str:
    if labels:
        lbl = ",".join(f'{k}="{v}"' for k, v in labels.items())
        return f"{name}{{{lbl}}} {value}"
    return f"{name} {value}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export health metrics in Prometheus text format.")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "exports" / "metrics" / "trading_system.prom"))
    args = parser.parse_args()

    health = _load(PROJECT_ROOT / "governance" / "health" / "health_gates_latest.json")
    verify = _load(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json")
    ready = _load(PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json")
    slo = _load(PROJECT_ROOT / "governance" / "health" / "slo_burn_latest.json")
    cc = _load(PROJECT_ROOT / "governance" / "champion_challenger" / "registry.json")
    regime = _load(PROJECT_ROOT / "governance" / "walk_forward" / "regime_segmented_latest.json")

    lines = [
        _metric("trading_health_score", float(health.get("data_quality_score", 0.0) or 0.0)),
        _metric("trading_hard_gate_triggered", 1.0 if health.get("hard_gate_triggered", False) else 0.0),
        _metric("trading_auto_verify_ok", 1.0 if verify.get("ok", False) else 0.0),
        _metric("trading_session_ready", 1.0 if ready.get("ok", False) else 0.0),
        _metric("trading_slo_burn_score", float(slo.get("slo_burn_score", 0.0) or 0.0)),
        _metric("trading_champion_present", 1.0 if bool(cc.get("champion")) else 0.0),
        _metric("trading_metrics_generated_utc", float(datetime.now(timezone.utc).timestamp())),
    ]

    for seg, row in (regime.get("segments", {}) or {}).items():
        try:
            lines.append(_metric("trading_regime_pass_rate", float(row.get("pass_rate", 0.0) or 0.0), {"segment": seg}))
            lines.append(_metric("trading_regime_fail_count", float(row.get("fail_count", 0.0) or 0.0), {"segment": seg}))
        except Exception:
            continue

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
