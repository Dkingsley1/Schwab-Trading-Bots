#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.counterfactual_replay_harness import build_counterfactual_report
from scripts.strategy_attribution_report import (
    build_strategy_attribution_report,
    render_strategy_attribution_markdown,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _tail(text: str, *, limit: int) -> str:
    return (text or "")[-max(int(limit), 1) :]


def _parse_payload_ts(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _artifact_age_minutes(path: Path, payload: dict[str, Any]) -> float | None:
    ts: datetime | None = None
    for key in ("timestamp_utc", "updated_at_utc", "generated_utc", "created_at"):
        if isinstance(payload, dict):
            ts = _parse_payload_ts(str(payload.get(key) or ""))
        if ts is not None:
            break
    if ts is None:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
    return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 60.0


def _load_fresh_artifact(
    path: Path,
    *,
    max_age_minutes: float,
    require_ok: bool = False,
) -> tuple[dict[str, Any], float | None, bool]:
    if max(float(max_age_minutes), 0.0) <= 0.0:
        return {}, None, False
    payload = _read_json(path)
    if not payload:
        return {}, None, False
    age_minutes = _artifact_age_minutes(path, payload)
    if age_minutes is None:
        return payload, None, False
    fresh = age_minutes <= max(float(max_age_minutes), 0.0)
    if fresh and require_ok and (payload.get("ok") is False):
        fresh = False
    return payload, age_minutes, fresh


def _artifact_step(name: str, path: Path, payload: dict[str, Any], *, age_minutes: float | None, skipped: bool = False) -> dict[str, Any]:
    step = {
        "name": name,
        "mode": "artifact",
        "ok": True,
        "rc": 0,
        "out_file": str(path),
        "reused": True,
    }
    if age_minutes is not None:
        step["age_minutes"] = round(float(age_minutes), 3)
    if skipped:
        step["skipped"] = True
    if isinstance(payload, dict):
        for key in ("timestamp_utc", "updated_at_utc", "generated_utc"):
            if payload.get(key):
                step["artifact_timestamp_utc"] = str(payload.get(key))
                break
    return step


def _run_inline_step(
    name: str,
    builder: Callable[[], dict[str, Any]],
    *,
    out_file: Path,
    md_out_file: Path | None = None,
    md_builder: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    try:
        payload = builder()
        _write_json(out_file, payload)
        if md_out_file is not None and md_builder is not None:
            md_out_file.parent.mkdir(parents=True, exist_ok=True)
            md_out_file.write_text(md_builder(payload), encoding="utf-8")
        ok = True
        error = ""
    except Exception as exc:
        payload = {}
        ok = False
        error = f"{type(exc).__name__}:{exc}"
    duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
    step = {
        "name": name,
        "mode": "inline",
        "ok": ok,
        "rc": 0 if ok else 1,
        "duration_ms": duration_ms,
        "out_file": str(out_file),
    }
    if md_out_file is not None:
        step["md_out_file"] = str(md_out_file)
    if error:
        step["error"] = error
    return payload, step


def _run_json_command(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    out_file: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

    payload = _read_json(out_file)
    if not payload:
        stdout = (proc.stdout or "").strip()
        if stdout.startswith("{") and stdout.endswith("}"):
            try:
                payload = json.loads(stdout)
            except Exception:
                payload = {}

    step = {
        "name": name,
        "mode": "subprocess",
        "ok": proc.returncode == 0,
        "rc": int(proc.returncode),
        "duration_ms": duration_ms,
        "cmd": [str(part) for part in cmd],
        "out_file": str(out_file),
        "stdout_tail": _tail(proc.stdout or "", limit=4000),
        "stderr_tail": _tail(proc.stderr or "", limit=2000),
    }
    return payload, step


def _summarize_strategy_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(payload.get("ok", False)),
        "row_count": _safe_int(payload.get("row_count", 0)),
        "file_count": _safe_int(payload.get("file_count", 0)),
        "top_lane": str(payload.get("top_lane") or ""),
        "top_layer": str(payload.get("top_layer") or ""),
        "total_pnl_proxy": float(payload.get("total_pnl_proxy", 0.0) or 0.0),
    }


def _summarize_counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    top_candidate = {}
    rows = payload.get("top_candidates")
    if isinstance(rows, list) and rows:
        top_candidate = rows[0] if isinstance(rows[0], dict) else {}
    return {
        "ok": bool(payload.get("ok", False)),
        "profiles_reviewed_count": len(payload.get("profiles_reviewed") or []),
        "candidate_count": _safe_int(payload.get("candidate_count", 0)),
        "top_candidate": top_candidate,
    }


def _summarize_research_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    failed_steps = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rc = _safe_int(row.get("rc", 1), default=1)
        if rc not in {0, 2}:
            cmd = row.get("cmd") if isinstance(row.get("cmd"), list) else []
            failed_steps.append(str(cmd[-1] if cmd else row.get("name") or "unknown"))
    return {
        "ok": bool(payload.get("ok", False)),
        "step_count": len(rows),
        "failed_steps": failed_steps,
    }


def _summarize_promotion(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "promote_ok": bool(payload.get("promote_ok", False)),
        "coverage_ok": bool(payload.get("coverage_ok", False)),
        "considered_bots": _safe_int(payload.get("considered_bots", 0)),
        "failed_bots": _safe_int(payload.get("failed_bots", 0)),
        "readiness_margin": float(payload.get("readiness_margin", 0.0) or 0.0),
        "recommended_retrain": payload.get("recommended_retrain", {}),
    }


def _summarize_champion(payload: dict[str, Any]) -> dict[str, Any]:
    champion = payload.get("champion") if isinstance(payload.get("champion"), dict) else {}
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    last_event = payload.get("last_event") if isinstance(payload.get("last_event"), dict) else {}
    return {
        "name": str(champion.get("name") or ""),
        "stage": str(champion.get("stage") or ""),
        "since_utc": str(champion.get("since_utc") or ""),
        "history_count": len(history),
        "last_action": str(last_event.get("action") or ""),
    }


def _summarize_derived_state(payload: dict[str, Any]) -> dict[str, Any]:
    source_paths = payload.get("source_paths") if isinstance(payload.get("source_paths"), dict) else {}
    return {
        "ok": bool(payload.get("ok", False)),
        "risk_level": str(payload.get("risk_level") or ""),
        "gross_exposure_cap": float(payload.get("gross_exposure_cap", 0.0) or 0.0),
        "max_total_actions_per_hour": int(payload.get("max_total_actions_per_hour", 0) or 0),
        "sleeve_count": int(len(payload.get("sleeves") or {})),
        "source_paths": {str(key): str(value) for key, value in source_paths.items()},
    }


def build_strategy_research_payload(
    project_root: Path,
    *,
    day: str,
    max_rows: int,
    skip_sandbox: bool = False,
    max_age_minutes: float = 0.0,
    sandbox_max_age_minutes: float | None = None,
    command_runner: Callable[[str, list[str]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    health_dir = project_root / "governance" / "health"
    walk_forward_dir = project_root / "governance" / "walk_forward"
    reports_dir = project_root / "exports" / "reports"
    sandbox_out = project_root / "exports" / "research_sandbox" / "latest.json"
    attribution_out = health_dir / "strategy_attribution_latest.json"
    attribution_md_out = reports_dir / "strategy_attribution_latest.md"
    counterfactual_out = health_dir / "counterfactual_replay_latest.json"
    derived_state_out = health_dir / "derived_state_latest.json"
    promotion_out = walk_forward_dir / "promotion_readiness_latest.json"
    promotion_fail_out = walk_forward_dir / "promotion_fail_bots_latest.json"
    promotion_history_out = walk_forward_dir / "promotion_readiness_history.jsonl"
    promotion_gate_file = walk_forward_dir / "promotion_gate_latest.json"
    walk_forward_file = walk_forward_dir / "walk_forward_latest.json"
    step_max_age_minutes = max(float(max_age_minutes), 0.0)
    effective_sandbox_max_age_minutes = max(
        float(sandbox_max_age_minutes if sandbox_max_age_minutes is not None else step_max_age_minutes),
        0.0,
    )

    if command_runner is None:
        def command_runner(name: str, cmd: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
            if name == "research_sandbox":
                out_file = sandbox_out
            else:
                out_file = promotion_out
            return _run_json_command(name, cmd, cwd=project_root, out_file=out_file)

    attribution_payload, attribution_age_minutes, attribution_fresh = _load_fresh_artifact(
        attribution_out,
        max_age_minutes=step_max_age_minutes,
        require_ok=True,
    )
    if attribution_fresh:
        attribution_step = _artifact_step(
            "strategy_attribution",
            attribution_out,
            attribution_payload,
            age_minutes=attribution_age_minutes,
        )
    else:
        attribution_payload, attribution_step = _run_inline_step(
            "strategy_attribution",
            lambda: build_strategy_attribution_report(project_root, day=day),
            out_file=attribution_out,
            md_out_file=attribution_md_out,
            md_builder=render_strategy_attribution_markdown,
        )

    counterfactual_payload, counterfactual_age_minutes, counterfactual_fresh = _load_fresh_artifact(
        counterfactual_out,
        max_age_minutes=step_max_age_minutes,
        require_ok=True,
    )
    if counterfactual_fresh:
        counterfactual_step = _artifact_step(
            "counterfactual_replay",
            counterfactual_out,
            counterfactual_payload,
            age_minutes=counterfactual_age_minutes,
        )
    else:
        counterfactual_payload, counterfactual_step = _run_inline_step(
            "counterfactual_replay",
            lambda: build_counterfactual_report(project_root, max_rows=max_rows),
            out_file=counterfactual_out,
        )

    if skip_sandbox:
        research_sandbox_payload, research_sandbox_age_minutes, research_sandbox_fresh = _load_fresh_artifact(
            sandbox_out,
            max_age_minutes=effective_sandbox_max_age_minutes,
            require_ok=True,
        )
        if research_sandbox_fresh:
            research_sandbox_step = _artifact_step(
                "research_sandbox",
                sandbox_out,
                research_sandbox_payload,
                age_minutes=research_sandbox_age_minutes,
                skipped=True,
            )
        else:
            research_sandbox_payload = _read_json(sandbox_out)
            if not research_sandbox_payload:
                research_sandbox_payload = {
                    "ok": True,
                    "skipped": True,
                    "reason": "skip_sandbox_without_existing_artifact",
                    "steps": [],
                }
            research_sandbox_step = {
                "name": "research_sandbox",
                "mode": "artifact",
                "ok": True,
                "rc": 0,
                "skipped": True,
                "reason": "skip_sandbox_without_existing_artifact" if not sandbox_out.exists() else "skip_sandbox_reused_artifact",
                "out_file": str(sandbox_out),
            }
    else:
        research_sandbox_payload, research_sandbox_age_minutes, research_sandbox_fresh = _load_fresh_artifact(
            sandbox_out,
            max_age_minutes=effective_sandbox_max_age_minutes,
            require_ok=True,
        )
        if research_sandbox_fresh:
            research_sandbox_step = _artifact_step(
                "research_sandbox",
                sandbox_out,
                research_sandbox_payload,
                age_minutes=research_sandbox_age_minutes,
            )
        else:
            research_sandbox_payload, research_sandbox_step = command_runner(
                "research_sandbox",
                [
                    str(Path(sys.executable).resolve()),
                    str(project_root / "scripts" / "run_research_sandbox.py"),
                    "--out",
                    str(sandbox_out),
                    "--json",
                ],
            )

    promotion_payload, promotion_age_minutes, promotion_fresh = _load_fresh_artifact(
        promotion_out,
        max_age_minutes=step_max_age_minutes,
    )
    if promotion_fresh:
        promotion_step = _artifact_step(
            "promotion_readiness",
            promotion_out,
            promotion_payload,
            age_minutes=promotion_age_minutes,
        )
    else:
        promotion_payload, promotion_step = command_runner(
            "promotion_readiness",
            [
                str(Path(sys.executable).resolve()),
                str(project_root / "scripts" / "promotion_readiness_summary.py"),
                "--gate-file",
                str(promotion_gate_file),
                "--walk-forward-file",
                str(walk_forward_file),
                "--history-jsonl",
                str(promotion_history_out),
                "--latest-out",
                str(promotion_out),
                "--fail-list-out",
                str(promotion_fail_out),
                "--json",
            ],
        )

    champion_payload = _read_json(project_root / "governance" / "champion_challenger" / "registry.json")
    derived_state_payload = _read_json(derived_state_out)

    steps = [
        attribution_step,
        counterfactual_step,
        research_sandbox_step,
        promotion_step,
    ]

    strategy_attribution = _summarize_strategy_attribution(attribution_payload)
    counterfactual_replay = _summarize_counterfactual(counterfactual_payload)
    research_sandbox = _summarize_research_sandbox(research_sandbox_payload)
    promotion_readiness = _summarize_promotion(promotion_payload)
    champion = _summarize_champion(champion_payload)
    derived_state = _summarize_derived_state(derived_state_payload)

    top_counterfactual_candidate = counterfactual_replay.get("top_candidate") if isinstance(counterfactual_replay.get("top_candidate"), dict) else {}
    recommended_retrain = promotion_readiness.get("recommended_retrain") if isinstance(promotion_readiness.get("recommended_retrain"), dict) else {}

    if promotion_readiness.get("promote_ok", False):
        recommended_action = "review_promotion_candidate"
    elif recommended_retrain.get("include_bot_ids"):
        recommended_action = "run_targeted_retrain"
    else:
        recommended_action = "monitor_and_refresh"

    return {
        "timestamp_utc": _utc_now(),
        "ok": all(bool(step.get("ok", False)) for step in steps),
        "day": day,
        "project_root": str(project_root),
        "promotable": bool(promotion_readiness.get("promote_ok", False)),
        "research_sandbox_ok": bool(research_sandbox.get("ok", False)),
        "summary": {
            "recommended_action": recommended_action,
            "top_lane": strategy_attribution.get("top_lane", ""),
            "top_layer": strategy_attribution.get("top_layer", ""),
            "top_counterfactual_profile": str(top_counterfactual_candidate.get("profile") or ""),
            "recommended_regime_focus": str(recommended_retrain.get("regime_focus") or ""),
            "current_champion": champion.get("name", ""),
        },
        "artifacts": {
            "strategy_attribution_json": str(attribution_out),
            "strategy_attribution_md": str(attribution_md_out),
            "counterfactual_replay_json": str(counterfactual_out),
            "research_sandbox_json": str(sandbox_out),
            "promotion_readiness_json": str(promotion_out),
            "promotion_fail_bots_json": str(promotion_fail_out),
            "derived_state_json": str(derived_state_out),
        },
        "strategy_attribution": strategy_attribution,
        "counterfactual_replay": counterfactual_replay,
        "research_sandbox": research_sandbox,
        "promotion_readiness": promotion_readiness,
        "champion": champion,
        "derived_state": derived_state,
        "recommendations": {
            "recommended_retrain": recommended_retrain,
            "top_counterfactual_candidate": top_counterfactual_candidate,
        },
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical strategy research lane summary from existing research and promotion artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--skip-sandbox", action="store_true")
    parser.add_argument("--max-age-minutes", type=float, default=float(os.getenv("STRATEGY_RESEARCH_MAX_AGE_MINUTES", "0")))
    parser.add_argument(
        "--sandbox-max-age-minutes",
        type=float,
        default=float(os.getenv("STRATEGY_RESEARCH_SANDBOX_MAX_AGE_MINUTES", "0")),
    )
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "strategy_research_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_strategy_research_payload(
        Path(args.project_root),
        day=str(args.day),
        max_rows=int(args.max_rows),
        skip_sandbox=bool(args.skip_sandbox),
        max_age_minutes=float(args.max_age_minutes),
        sandbox_max_age_minutes=float(args.sandbox_max_age_minutes),
    )
    out_path = Path(args.out_file)
    _write_json(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "strategy_research "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"promotable={int(bool(payload.get('promotable', False)))} "
            f"recommended_action={payload.get('summary', {}).get('recommended_action', '')} "
            f"top_lane={payload.get('summary', {}).get('top_lane', '') or 'none'}"
        )
    return 0 if payload.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
