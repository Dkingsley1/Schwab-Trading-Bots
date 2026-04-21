#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_attribution_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "strategy_attribution_latest.md"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_attribution_state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _glob_day_paths(project_root: Path, day: str) -> list[Path]:
    return sorted((project_root / "governance").glob(f"shadow*/shadow_pnl_attribution_{day}.jsonl"))


def _default_state_path(project_root: Path) -> Path:
    return project_root / "governance" / "health" / "strategy_attribution_state.json"


def _empty_group() -> dict[str, Any]:
    return {
        "samples": 0,
        "pnl_proxy_sum": 0.0,
        "return_1m_sum": 0.0,
        "symbols": set(),
        "bots": set(),
    }


def _empty_runtime(day: str) -> dict[str, Any]:
    return {
        "day": day,
        "latest_ts": "",
        "row_count": 0,
        "total_pnl_proxy": 0.0,
        "action_counts": defaultdict(int),
        "lane_rollup": defaultdict(_empty_group),
        "layer_rollup": defaultdict(_empty_group),
        "symbol_rollup": defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0}),
        "bot_rollup": defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0}),
    }


def _state_to_runtime(state: dict[str, Any], day: str) -> dict[str, Any]:
    runtime = _empty_runtime(day)
    agg = state.get("aggregates") if isinstance(state.get("aggregates"), dict) else {}
    runtime["latest_ts"] = str(agg.get("latest_ts") or "")
    runtime["row_count"] = int(agg.get("row_count", 0) or 0)
    runtime["total_pnl_proxy"] = float(agg.get("total_pnl_proxy", 0.0) or 0.0)

    action_counts = agg.get("action_counts") if isinstance(agg.get("action_counts"), dict) else {}
    for key, value in action_counts.items():
        runtime["action_counts"][str(key)] = int(value or 0)

    for group_name in ("lane_rollup", "layer_rollup"):
        src = agg.get(group_name) if isinstance(agg.get(group_name), dict) else {}
        dst = runtime[group_name]
        for key, row in src.items():
            if not isinstance(row, dict):
                continue
            entry = dst[str(key)]
            entry["samples"] = int(row.get("samples", 0) or 0)
            entry["pnl_proxy_sum"] = float(row.get("pnl_proxy_sum", 0.0) or 0.0)
            entry["return_1m_sum"] = float(row.get("return_1m_sum", 0.0) or 0.0)
            entry["symbols"] = {str(item) for item in (row.get("symbols") or []) if str(item)}
            entry["bots"] = {str(item) for item in (row.get("bots") or []) if str(item)}

    for group_name in ("symbol_rollup", "bot_rollup"):
        src = agg.get(group_name) if isinstance(agg.get(group_name), dict) else {}
        dst = runtime[group_name]
        for key, row in src.items():
            if not isinstance(row, dict):
                continue
            dst[str(key)] = {
                "samples": int(row.get("samples", 0) or 0),
                "pnl_proxy_sum": float(row.get("pnl_proxy_sum", 0.0) or 0.0),
            }
    return runtime


def _runtime_to_state(runtime: dict[str, Any], *, source_files: dict[str, dict[str, Any]], processing_mode: str) -> dict[str, Any]:
    lane_rollup = {
        lane: {
            "samples": int(values["samples"]),
            "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8),
            "return_1m_sum": round(float(values["return_1m_sum"]), 8),
            "symbols": sorted(str(item) for item in values["symbols"]),
            "bots": sorted(str(item) for item in values["bots"]),
        }
        for lane, values in runtime["lane_rollup"].items()
    }
    layer_rollup = {
        layer: {
            "samples": int(values["samples"]),
            "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8),
            "return_1m_sum": round(float(values["return_1m_sum"]), 8),
            "symbols": sorted(str(item) for item in values["symbols"]),
            "bots": sorted(str(item) for item in values["bots"]),
        }
        for layer, values in runtime["layer_rollup"].items()
    }
    symbol_rollup = {
        symbol: {
            "samples": int(values["samples"]),
            "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8),
        }
        for symbol, values in runtime["symbol_rollup"].items()
    }
    bot_rollup = {
        bot_id: {
            "samples": int(values["samples"]),
            "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8),
        }
        for bot_id, values in runtime["bot_rollup"].items()
    }
    return {
        "day": str(runtime.get("day") or ""),
        "processing_mode": processing_mode,
        "updated_at_utc": _utc_now(),
        "source_files": source_files,
        "aggregates": {
            "latest_ts": str(runtime.get("latest_ts") or ""),
            "row_count": int(runtime.get("row_count", 0) or 0),
            "total_pnl_proxy": round(float(runtime.get("total_pnl_proxy", 0.0) or 0.0), 8),
            "action_counts": {str(key): int(value) for key, value in runtime["action_counts"].items()},
            "lane_rollup": lane_rollup,
            "layer_rollup": layer_rollup,
            "symbol_rollup": symbol_rollup,
            "bot_rollup": bot_rollup,
        },
    }


def _iter_jsonl_rows(path: Path, *, offset_bytes: int = 0) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            handle.seek(max(int(offset_bytes), 0))
            for raw in handle:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows, int(handle.tell())
    except Exception:
        return [], max(int(offset_bytes), 0)


def _update_runtime_from_row(runtime: dict[str, Any], lane: str, row: dict[str, Any]) -> None:
    runtime["row_count"] += 1
    ts = str(row.get("timestamp_utc") or "").strip()
    if ts and ts > runtime["latest_ts"]:
        runtime["latest_ts"] = ts

    layer = str(row.get("layer") or "unknown").strip() or "unknown"
    symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
    bot_id = str(row.get("bot_id") or "unknown").strip() or "unknown"
    action = str(row.get("action") or "UNKNOWN").strip().upper() or "UNKNOWN"
    pnl_proxy = _coerce_float(row.get("pnl_proxy"), 0.0)
    return_1m = _coerce_float(row.get("return_1m"), 0.0)

    runtime["total_pnl_proxy"] += pnl_proxy
    runtime["action_counts"][action] += 1

    lane_row = runtime["lane_rollup"][lane]
    lane_row["samples"] += 1
    lane_row["pnl_proxy_sum"] += pnl_proxy
    lane_row["return_1m_sum"] += return_1m
    lane_row["symbols"].add(symbol)
    lane_row["bots"].add(bot_id)

    layer_row = runtime["layer_rollup"][layer]
    layer_row["samples"] += 1
    layer_row["pnl_proxy_sum"] += pnl_proxy
    layer_row["return_1m_sum"] += return_1m
    layer_row["symbols"].add(symbol)
    layer_row["bots"].add(bot_id)

    runtime["symbol_rollup"][symbol]["samples"] += 1
    runtime["symbol_rollup"][symbol]["pnl_proxy_sum"] += pnl_proxy

    runtime["bot_rollup"][bot_id]["samples"] += 1
    runtime["bot_rollup"][bot_id]["pnl_proxy_sum"] += pnl_proxy


def _can_incrementally_reuse(state: dict[str, Any], day: str, current_paths: list[Path]) -> bool:
    if str(state.get("day") or "") != str(day):
        return False
    tracked = state.get("source_files") if isinstance(state.get("source_files"), dict) else {}
    tracked_paths = {str(path) for path in tracked.keys()}
    current_path_set = {str(path) for path in current_paths}
    if not tracked_paths.issubset(current_path_set):
        return False
    for path in current_paths:
        meta = tracked.get(str(path))
        if not isinstance(meta, dict):
            continue
        try:
            st = path.stat()
        except Exception:
            return False
        tracked_inode = int(meta.get("inode", 0) or 0)
        tracked_offset = int(meta.get("offset_bytes", 0) or 0)
        if tracked_inode and int(st.st_ino) != tracked_inode:
            return False
        if int(st.st_size) < tracked_offset:
            return False
    return True


def _summarize_group(rows: dict[str, Any]) -> dict[str, Any]:
    samples = int(rows.get("samples", 0) or 0)
    pnl_proxy_sum = float(rows.get("pnl_proxy_sum", 0.0) or 0.0)
    return {
        "samples": samples,
        "pnl_proxy_sum": round(pnl_proxy_sum, 8),
        "avg_pnl_proxy": round((pnl_proxy_sum / samples) if samples > 0 else 0.0, 8),
        "avg_return_1m": round((float(rows.get("return_1m_sum", 0.0) or 0.0) / samples) if samples > 0 else 0.0, 8),
        "unique_symbols": int(len(rows.get("symbols", set()))),
        "unique_bots": int(len(rows.get("bots", set()))),
    }


def build_strategy_attribution_report(project_root: Path, *, day: str, state_file: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    state_path = Path(state_file).resolve() if state_file is not None else _default_state_path(project_root)
    current_paths = _glob_day_paths(project_root, day)
    state = _load_json(state_path)

    processing_mode = "rebuild"
    file_scan_counts = {"full_files": 0, "incremental_files": 0, "reused_files": 0}

    if _can_incrementally_reuse(state, day, current_paths):
        runtime = _state_to_runtime(state, day)
        tracked = state.get("source_files") if isinstance(state.get("source_files"), dict) else {}
        processing_mode = "incremental"
    else:
        runtime = _empty_runtime(day)
        tracked = {}

    next_source_files: dict[str, dict[str, Any]] = {}
    if processing_mode == "rebuild" and state_path.exists():
        tracked = {}

    for path in current_paths:
        meta = tracked.get(str(path)) if isinstance(tracked, dict) else None
        offset = int(meta.get("offset_bytes", 0) or 0) if isinstance(meta, dict) else 0
        rows, final_offset = _iter_jsonl_rows(path, offset_bytes=offset)
        lane = path.parent.name
        for row in rows:
            _update_runtime_from_row(runtime, lane, row)
        if isinstance(meta, dict):
            if final_offset > offset:
                file_scan_counts["incremental_files"] += 1
            else:
                file_scan_counts["reused_files"] += 1
        else:
            file_scan_counts["full_files"] += 1
        try:
            st = path.stat()
            next_source_files[str(path)] = {
                "inode": int(st.st_ino),
                "offset_bytes": int(final_offset if final_offset > 0 else st.st_size),
                "mtime": float(st.st_mtime),
                "lane": lane,
            }
        except Exception:
            next_source_files[str(path)] = {
                "inode": 0,
                "offset_bytes": int(final_offset),
                "mtime": 0.0,
                "lane": lane,
            }

    persisted_state = _runtime_to_state(runtime, source_files=next_source_files, processing_mode=processing_mode)
    _write_json(state_path, persisted_state)

    by_lane = [
        {"lane": lane, **_summarize_group(values)}
        for lane, values in runtime["lane_rollup"].items()
    ]
    by_lane.sort(key=lambda row: (-abs(float(row["pnl_proxy_sum"])), row["lane"]))

    by_layer = [
        {"layer": layer, **_summarize_group(values)}
        for layer, values in runtime["layer_rollup"].items()
    ]
    by_layer.sort(key=lambda row: (-abs(float(row["pnl_proxy_sum"])), row["layer"]))

    top_positive_symbols = [
        {"symbol": symbol, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for symbol, values in sorted(runtime["symbol_rollup"].items(), key=lambda item: (-float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_negative_symbols = [
        {"symbol": symbol, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for symbol, values in sorted(runtime["symbol_rollup"].items(), key=lambda item: (float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_positive_bots = [
        {"bot_id": bot_id, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for bot_id, values in sorted(runtime["bot_rollup"].items(), key=lambda item: (-float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_negative_bots = [
        {"bot_id": bot_id, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for bot_id, values in sorted(runtime["bot_rollup"].items(), key=lambda item: (float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]

    top_lane = by_lane[0]["lane"] if by_lane else ""
    top_layer = by_layer[0]["layer"] if by_layer else ""

    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 2,
        "ok": runtime["row_count"] > 0,
        "day": day,
        "row_count": int(runtime["row_count"]),
        "file_count": int(len(current_paths)),
        "source_files": [str(path) for path in current_paths],
        "latest_event_timestamp_utc": runtime["latest_ts"],
        "total_pnl_proxy": round(float(runtime["total_pnl_proxy"]), 8),
        "top_lane": top_lane,
        "top_layer": top_layer,
        "action_counts": {key: int(value) for key, value in sorted(runtime["action_counts"].items())},
        "by_lane": by_lane,
        "by_layer": by_layer,
        "top_positive_symbols": top_positive_symbols,
        "top_negative_symbols": top_negative_symbols,
        "top_positive_bots": top_positive_bots,
        "top_negative_bots": top_negative_bots,
        "processing": {
            "mode": processing_mode,
            "state_file": str(state_path),
            "full_files": int(file_scan_counts["full_files"]),
            "incremental_files": int(file_scan_counts["incremental_files"]),
            "reused_files": int(file_scan_counts["reused_files"]),
        },
    }


def render_strategy_attribution_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strategy Attribution",
        "",
        f"- generated_utc: {payload.get('timestamp_utc', '')}",
        f"- day: {payload.get('day', '')}",
        f"- ok: {payload.get('ok', False)}",
        f"- files: {payload.get('file_count', 0)}",
        f"- rows: {payload.get('row_count', 0)}",
        f"- total_pnl_proxy: {payload.get('total_pnl_proxy', 0.0):.8f}",
        f"- latest_event_timestamp_utc: {payload.get('latest_event_timestamp_utc', '')}",
        f"- processing_mode: {((payload.get('processing') or {}).get('mode', 'unknown'))}",
        "",
        "## Lane Attribution",
        "",
    ]
    lines.extend(
        _render_table(
            ["Lane", "Samples", "PnL Proxy", "Avg PnL", "Avg Return 1m", "Symbols", "Bots"],
            [
                [
                    str(row.get("lane", "")),
                    str(row.get("samples", 0)),
                    f"{float(row.get('pnl_proxy_sum', 0.0)):.8f}",
                    f"{float(row.get('avg_pnl_proxy', 0.0)):.8f}",
                    f"{float(row.get('avg_return_1m', 0.0)):.8f}",
                    str(row.get("unique_symbols", 0)),
                    str(row.get("unique_bots", 0)),
                ]
                for row in payload.get("by_lane", [])[:12]
            ],
        )
    )
    lines.extend(["", "## Layer Attribution", ""])
    lines.extend(
        _render_table(
            ["Layer", "Samples", "PnL Proxy", "Avg PnL", "Avg Return 1m", "Symbols", "Bots"],
            [
                [
                    str(row.get("layer", "")),
                    str(row.get("samples", 0)),
                    f"{float(row.get('pnl_proxy_sum', 0.0)):.8f}",
                    f"{float(row.get('avg_pnl_proxy', 0.0)):.8f}",
                    f"{float(row.get('avg_return_1m', 0.0)):.8f}",
                    str(row.get("unique_symbols", 0)),
                    str(row.get("unique_bots", 0)),
                ]
                for row in payload.get("by_layer", [])[:12]
            ],
        )
    )
    lines.extend(["", "## Top Positive Symbols", ""])
    lines.extend(
        _render_table(
            ["Symbol", "Samples", "PnL Proxy"],
            [
                [
                    str(row.get("symbol", "")),
                    str(row.get("samples", 0)),
                    f"{float(row.get('pnl_proxy_sum', 0.0)):.8f}",
                ]
                for row in payload.get("top_positive_symbols", [])[:10]
            ],
        )
    )
    lines.extend(["", "## Top Negative Symbols", ""])
    lines.extend(
        _render_table(
            ["Symbol", "Samples", "PnL Proxy"],
            [
                [
                    str(row.get("symbol", "")),
                    str(row.get("samples", 0)),
                    f"{float(row.get('pnl_proxy_sum', 0.0)):.8f}",
                ]
                for row in payload.get("top_negative_symbols", [])[:10]
            ],
        )
    )
    lines.extend(["", "## Top Positive Bots", ""])
    lines.extend(
        _render_table(
            ["Bot", "Samples", "PnL Proxy"],
            [
                [
                    str(row.get("bot_id", "")),
                    str(row.get("samples", 0)),
                    f"{float(row.get('pnl_proxy_sum', 0.0)):.8f}",
                ]
                for row in payload.get("top_positive_bots", [])[:10]
            ],
        )
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lane/layer strategy attribution summary from shadow PnL attribution logs.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_strategy_attribution_report(
        PROJECT_ROOT,
        day=str(args.day),
        state_file=Path(args.state_file).resolve(),
    )
    out_path = Path(args.out_file)
    md_path = Path(args.md_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(render_strategy_attribution_markdown(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "strategy_attribution "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"files={int(payload.get('file_count', 0))} "
            f"rows={int(payload.get('row_count', 0))} "
            f"top_lane={payload.get('top_lane', '') or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
