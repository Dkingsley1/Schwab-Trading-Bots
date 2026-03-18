#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_attribution_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "strategy_attribution_latest.md"


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


def build_strategy_attribution_report(project_root: Path, *, day: str) -> dict[str, Any]:
    paths = sorted((project_root / "governance").glob(f"shadow*/shadow_pnl_attribution_{day}.jsonl"))
    lane_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0, "return_1m_sum": 0.0, "symbols": set(), "bots": set()})
    layer_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0, "return_1m_sum": 0.0, "symbols": set(), "bots": set()})
    symbol_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0})
    bot_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "pnl_proxy_sum": 0.0})
    action_counts: dict[str, int] = defaultdict(int)

    latest_ts = ""
    row_count = 0
    total_pnl_proxy = 0.0

    for path in paths:
        lane = path.parent.name
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue

                    row_count += 1
                    ts = str(row.get("timestamp_utc") or "").strip()
                    if ts and ts > latest_ts:
                        latest_ts = ts

                    layer = str(row.get("layer") or "unknown").strip() or "unknown"
                    symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
                    bot_id = str(row.get("bot_id") or "unknown").strip() or "unknown"
                    action = str(row.get("action") or "UNKNOWN").strip().upper() or "UNKNOWN"
                    pnl_proxy = _coerce_float(row.get("pnl_proxy"), 0.0)
                    return_1m = _coerce_float(row.get("return_1m"), 0.0)

                    total_pnl_proxy += pnl_proxy
                    action_counts[action] += 1

                    lane_row = lane_rollup[lane]
                    lane_row["samples"] += 1
                    lane_row["pnl_proxy_sum"] += pnl_proxy
                    lane_row["return_1m_sum"] += return_1m
                    lane_row["symbols"].add(symbol)
                    lane_row["bots"].add(bot_id)

                    layer_row = layer_rollup[layer]
                    layer_row["samples"] += 1
                    layer_row["pnl_proxy_sum"] += pnl_proxy
                    layer_row["return_1m_sum"] += return_1m
                    layer_row["symbols"].add(symbol)
                    layer_row["bots"].add(bot_id)

                    symbol_rollup[symbol]["samples"] += 1
                    symbol_rollup[symbol]["pnl_proxy_sum"] += pnl_proxy

                    bot_rollup[bot_id]["samples"] += 1
                    bot_rollup[bot_id]["pnl_proxy_sum"] += pnl_proxy
        except Exception:
            continue

    by_lane = [
        {"lane": lane, **_summarize_group(values)}
        for lane, values in lane_rollup.items()
    ]
    by_lane.sort(key=lambda row: (-abs(float(row["pnl_proxy_sum"])), row["lane"]))

    by_layer = [
        {"layer": layer, **_summarize_group(values)}
        for layer, values in layer_rollup.items()
    ]
    by_layer.sort(key=lambda row: (-abs(float(row["pnl_proxy_sum"])), row["layer"]))

    top_positive_symbols = [
        {"symbol": symbol, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for symbol, values in sorted(symbol_rollup.items(), key=lambda item: (-float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_negative_symbols = [
        {"symbol": symbol, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for symbol, values in sorted(symbol_rollup.items(), key=lambda item: (float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_positive_bots = [
        {"bot_id": bot_id, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for bot_id, values in sorted(bot_rollup.items(), key=lambda item: (-float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]
    top_negative_bots = [
        {"bot_id": bot_id, "samples": int(values["samples"]), "pnl_proxy_sum": round(float(values["pnl_proxy_sum"]), 8)}
        for bot_id, values in sorted(bot_rollup.items(), key=lambda item: (float(item[1]["pnl_proxy_sum"]), item[0]))[:10]
    ]

    top_lane = by_lane[0]["lane"] if by_lane else ""
    top_layer = by_layer[0]["layer"] if by_layer else ""

    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": row_count > 0,
        "day": day,
        "row_count": int(row_count),
        "file_count": int(len(paths)),
        "source_files": [str(path) for path in paths],
        "latest_event_timestamp_utc": latest_ts,
        "total_pnl_proxy": round(float(total_pnl_proxy), 8),
        "top_lane": top_lane,
        "top_layer": top_layer,
        "action_counts": {key: int(value) for key, value in sorted(action_counts.items())},
        "by_lane": by_lane,
        "by_layer": by_layer,
        "top_positive_symbols": top_positive_symbols,
        "top_negative_symbols": top_negative_symbols,
        "top_positive_bots": top_positive_bots,
        "top_negative_bots": top_negative_bots,
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_strategy_attribution_report(PROJECT_ROOT, day=str(args.day))
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
