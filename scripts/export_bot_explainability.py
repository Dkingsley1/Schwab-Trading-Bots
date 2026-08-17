import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
OUT_DIR = PROJECT_ROOT / "exports" / "sql_reports"
LATEST_PATH = PROJECT_ROOT / "governance" / "health" / "bot_explainability_latest.json"
TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _bot_id_from_path(path: Path) -> str:
    return TIMESTAMP_SUFFIX_RE.sub("", path.stem)


def _latest_logs(limit: int, bot_ids: set[str]) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    for raw in sorted(glob.glob(str(LOGS_DIR / "brain_refinery_v*.json")), reverse=True):
        path = Path(raw)
        bot_id = _bot_id_from_path(path)
        if bot_ids and bot_id not in bot_ids:
            continue
        if bot_id in seen:
            continue
        selected.append(path)
        seen.add(bot_id)
        if len(selected) >= max(limit, 1):
            break
    return selected


def _ranked_feature_importance(model_path: Path, feature_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_load_ok": False,
        "input_dim": 0,
        "feature_dim": len(feature_names),
        "window_steps": 0,
        "top_features": [],
        "top_time_steps": [],
        "error": "",
    }
    if not model_path.exists() or not feature_names:
        out["error"] = "missing_model_or_feature_names"
        return out
    try:
        arr = np.load(model_path, allow_pickle=False)
        if "layer1.weight" not in set(arr.files):
            out["error"] = "missing_layer1_weight"
            return out
        w1 = np.asarray(arr["layer1.weight"], dtype=np.float64)
        input_dim = int(w1.shape[1]) if w1.ndim == 2 else 0
        feature_dim = len(feature_names)
        if input_dim <= 0 or feature_dim <= 0:
            out["error"] = "invalid_dimensions"
            return out
        effective_dim = (input_dim // feature_dim) * feature_dim
        if effective_dim <= 0:
            out["error"] = f"input_dim_not_multiple feature_dim={feature_dim} input_dim={input_dim}"
            return out
        step_count = effective_dim // feature_dim
        base = np.mean(np.abs(w1[:, :effective_dim]), axis=0)
        feature_scores = {name: 0.0 for name in feature_names}
        step_scores: list[dict[str, Any]] = []
        for step in range(step_count):
            start = step * feature_dim
            stop = start + feature_dim
            block = base[start:stop]
            step_scores.append(
                {
                    "step": int(step),
                    "score": float(np.mean(block)),
                    "max_score": float(np.max(block)),
                }
            )
            for idx, name in enumerate(feature_names):
                feature_scores[name] += float(block[idx])
        ranked_features = sorted(
            (
                {
                    "feature": name,
                    "score": float(score / max(step_count, 1)),
                }
                for name, score in feature_scores.items()
            ),
            key=lambda row: (-float(row["score"]), row["feature"]),
        )
        ranked_steps = sorted(step_scores, key=lambda row: (-float(row["score"]), int(row["step"])))
        out.update(
            {
                "model_load_ok": True,
                "input_dim": int(input_dim),
                "feature_dim": int(feature_dim),
                "window_steps": int(step_count),
                "top_features": ranked_features[:12],
                "top_time_steps": ranked_steps[:8],
            }
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export explainability summary for latest bot models.")
    parser.add_argument("--bot-ids", default="", help="Comma-separated bot ids to restrict the export.")
    parser.add_argument("--limit", type=int, default=12, help="Max latest unique bot logs to include.")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    requested = {x.strip() for x in str(args.bot_ids or "").split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    for log_path in _latest_logs(limit=max(int(args.limit), 1), bot_ids=requested):
        payload = _load_json(log_path)
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        feature_names = [str(x) for x in (config.get("features") or []) if str(x)]
        model_path = Path(str(payload.get("model_path", "") or "")).expanduser()
        explain = _ranked_feature_importance(model_path, feature_names)
        rows.append(
            {
                "bot_id": _bot_id_from_path(log_path),
                "log_path": str(log_path),
                "model_path": str(model_path),
                "timestamp": str(payload.get("timestamp", "")),
                "metrics": metrics,
                "feature_names_count": len(feature_names),
                "explainability": explain,
            }
        )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "bot_count": len(rows),
        "bots": rows,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_file) if args.out_file else (OUT_DIR / f"bot_explainability_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"bot_explainability_exported={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
