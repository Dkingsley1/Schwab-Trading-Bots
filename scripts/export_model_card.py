import argparse
import glob
import hashlib
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}




def _feature_hash(names: list[str]) -> str:
    blob = json.dumps(names, ensure_ascii=True, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _latest(pattern: str) -> Path | None:
    rows = sorted(glob.glob(pattern))
    return Path(rows[-1]) if rows else None


def _model_artifact_info(path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "load_ok": False,
        "feature_dim": 0,
        "feature_names": [],
        "feature_names_count": 0,
        "feature_names_sha256": "",
        "error": "",
    }
    if (path is None) or (not path.exists()):
        return out

    try:
        arr = np.load(path, allow_pickle=False)
        w = np.asarray(arr["W"])
        mu = np.asarray(arr["mu"]).reshape(-1)
        sigma = np.asarray(arr["sigma"]).reshape(-1)
        model_dim = int(w.shape[0]) if w.ndim == 2 else 0
        effective_dim = int(min(model_dim, int(mu.shape[0]), int(sigma.shape[0])))

        feature_names: list[str] = []
        if "feature_names" in set(arr.files):
            raw = np.asarray(arr["feature_names"]).reshape(-1)
            for value in raw.tolist():
                if isinstance(value, bytes):
                    name = value.decode("utf-8", errors="ignore").strip()
                else:
                    name = str(value).strip()
                if name:
                    feature_names.append(name)

        out.update(
            {
                "load_ok": True,
                "feature_dim": int(effective_dim),
                "feature_names": feature_names,
                "feature_names_count": int(len(feature_names)),
                "feature_names_sha256": _feature_hash(feature_names) if feature_names else "",
            }
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export model card for latest retrain/trade behavior promotion context.")
    ap.add_argument("--retrain-scorecard", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"))
    ap.add_argument("--training-success", default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"))
    ap.add_argument("--daily-verify", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    ap.add_argument("--promotion-quality", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    ap.add_argument("--trade-log", default="")
    ap.add_argument("--out-file", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    score = _load(Path(args.retrain_scorecard))
    success = _load(Path(args.training_success))
    verify = _load(Path(args.daily_verify))
    pq = _load(Path(args.promotion_quality))

    trade_log_path = Path(args.trade_log) if args.trade_log else _latest(str(PROJECT_ROOT / "logs" / "trade_behavior_policy_*.json"))
    trade = _load(trade_log_path) if trade_log_path else {}
    model_path_from_log = Path(str(trade.get("model_path", "") or "")).expanduser() if str(trade.get("model_path", "") or "").strip() else None
    model_artifact = _model_artifact_info(model_path_from_log)

    failures = score.get("failures") if isinstance(score.get("failures"), list) else []
    failed_checks = verify.get("failed_checks") if isinstance(verify.get("failed_checks"), list) else []
    top_failure_reasons = [str(x) for x in failures[:5]] + [str(x) for x in failed_checks[:5]]

    artifact_feature_names = [str(x) for x in (model_artifact.get("feature_names") or []) if str(x)]
    feature_names = artifact_feature_names
    if not feature_names:
        feature_names = trade.get("feature_names") if isinstance(trade.get("feature_names"), list) else []
    if not feature_names:
        ds = _load(PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json")
        feature_names = ds.get("feature_names") if isinstance(ds.get("feature_names"), list) else []
    feature_names = [str(x) for x in feature_names if str(x)]
    artifact_dim_hint = int(model_artifact.get("feature_dim", 0) or 0)
    if artifact_dim_hint > 0 and len(feature_names) > artifact_dim_hint:
        feature_names = feature_names[:artifact_dim_hint]
    lineage = trade.get("lineage") if isinstance(trade.get("lineage"), dict) else {}

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "model": {
            "path": str(trade.get("model_path", "")),
            "promoted": bool(trade.get("promoted", False)),
            "champion_seed": trade.get("champion_seed"),
            "candidate_score": trade.get("candidate_score"),
            "previous_score": trade.get("previous_score"),
            "feature_dim": int(model_artifact.get("feature_dim", 0) or len(feature_names)),
            "feature_names": feature_names,
            "feature_names_sha256": _feature_hash(feature_names) if feature_names else "",
            "artifact_has_feature_names": bool(model_artifact.get("feature_names_count", 0) or 0),
            "artifact_feature_names_count": int(model_artifact.get("feature_names_count", 0) or 0),
            "artifact_load_ok": bool(model_artifact.get("load_ok", False)),
            "artifact_error": str(model_artifact.get("error", "")),
            "feature_schema_version": str(lineage.get("feature_schema_version", "")),
        },
        "gates": {
            "training_success_confirmed": bool(success.get("confirmed_training_success", False)),
            "training_success_reason": str(success.get("reason", "")),
            "master_update_status": str(score.get("master_update_status", "")),
            "promotion_quality_ok": bool(pq.get("ok", False)),
            "daily_auto_verify_ok": bool(verify.get("ok", False)),
        },
        "top_failure_reasons": top_failure_reasons,
        "promotion_decision": {
            "trade_behavior_promoted": bool(trade.get("promoted", False)),
            "rollback_enabled": bool(((trade.get("promotion_gate") or {}).get("rollback_enabled", False)) if isinstance(trade.get("promotion_gate"), dict) else False),
            "promotion_reasons": ((trade.get("promotion_gate") or {}).get("reasons", []) if isinstance(trade.get("promotion_gate"), dict) else []),
        },
        "sources": {
            "retrain_scorecard": str(args.retrain_scorecard),
            "training_success": str(args.training_success),
            "daily_verify": str(args.daily_verify),
            "promotion_quality": str(args.promotion_quality),
            "trade_behavior_log": str(trade_log_path) if trade_log_path else "",
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_file) if args.out_file else (PROJECT_ROOT / "exports" / "sql_reports" / f"model_card_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    latest = PROJECT_ROOT / "governance" / "health" / "model_card_latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"model_card_exported={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
