#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
    from scripts.ops import source_verification_report as report_src
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload
    from . import source_verification_report as report_src


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "source_verification_autorefresh_latest.json"
HEAVY_REFRESH_MARKERS = {
    "schwab-symbol-news-sync",
    "ticker-news-sync",
    "sec-edgar-sync",
    "extended-quant-sync",
}
COMMAND_TIMEOUT_CAPS = {
    "macro-crosscheck": 60,
    "schwab-symbol-news-sync": 180,
    "ticker-news-sync": 240,
    "sec-edgar-sync": 180,
    "extended-quant-sync": 180,
}
MACRO_CROSSCHECK_STALE_DEPENDENCIES = {"public_macro_feeds", "market_micro_context"}
COMMAND_MARKER_SOURCE_IDS = {
    "fx-market-sync": "fx_market_context",
    "macro-crosscheck": "macro_crossstack",
    "macro-context-sync": "public_macro_feeds",
    "market-micro-sync": "market_micro_context",
    "free-equity-reference-sync": "free_equity_reference_context",
    "schwab-symbol-news-sync": "schwab_symbol_news",
    "ticker-news-sync": "ticker_news_context",
    "sec-edgar-sync": "sec_edgar_context",
    "extended-quant-sync": "extended_quant_context",
    "public-policy-sync": "public_policy_context",
}
DOWNSTREAM_RECHECK_TIMEOUT_SECONDS = 120


def _opsctl(project_root: Path, *args: str) -> list[str]:
    return [str(project_root / "scripts" / "ops" / "opsctl.sh"), *args]


def _tail_text(text: str, *, line_count: int = 8, char_limit: int = 4000) -> str:
    tail = "\n".join((text or "").splitlines()[-max(int(line_count), 1):])
    if len(tail) <= char_limit:
        return tail
    return tail[-char_limit:]


def _command_key(command: list[str]) -> tuple[str, ...]:
    return tuple(str(part) for part in command)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _command_policy(command: list[str], *, default_timeout_seconds: int) -> dict[str, Any]:
    joined = " ".join(str(part) for part in command)
    command_name = next((marker for marker in COMMAND_TIMEOUT_CAPS if marker in joined), "")
    heavy = any(marker in joined for marker in HEAVY_REFRESH_MARKERS)
    timeout_cap = COMMAND_TIMEOUT_CAPS.get(command_name, int(default_timeout_seconds))
    return {
        "command_name": command_name or (Path(str(command[0])).name if command else ""),
        "heavy": heavy,
        "tier": "optional_heavy_source_refresh" if heavy else "core_verification_refresh",
        "timeout_seconds": max(1, min(int(default_timeout_seconds), int(timeout_cap))),
    }


def _source_id_for_command(command: list[str], source_by_command: dict[tuple[str, ...], str]) -> str:
    mapped = source_by_command.get(_command_key(command), "")
    if mapped:
        return mapped
    joined = " ".join(str(part) for part in command)
    for marker, source_id in COMMAND_MARKER_SOURCE_IDS.items():
        if marker in joined:
            return source_id
    return ""


def _append_option_if_missing(command: list[str], option: str, value: str) -> list[str]:
    if option in command:
        return command
    out = list(command)
    try:
        json_index = out.index("--json")
    except ValueError:
        json_index = len(out)
    out[json_index:json_index] = [option, value]
    return out


def _set_option(command: list[str], option: str, value: str) -> list[str]:
    out = list(command)
    try:
        option_index = out.index(option)
    except ValueError:
        return _append_option_if_missing(out, option, value)
    if option_index + 1 < len(out):
        out[option_index + 1] = value
    else:
        out.append(value)
    return out


def _bounded_heavy_command(
    command: list[str],
    *,
    outer_timeout_seconds: int,
) -> list[str]:
    joined = " ".join(str(part) for part in command)
    if "ticker-news-sync" not in joined:
        return command
    configured_runtime_seconds = max(
        _safe_int(
            os.getenv(
                "SOURCE_VERIFICATION_TICKER_MAX_RUNTIME_SECONDS",
                os.getenv("SOURCE_VERIFICATION_GUARDED_TICKER_MAX_RUNTIME_SECONDS", "180"),
            ),
            180,
        ),
        30,
    )
    child_runtime_seconds = min(
        configured_runtime_seconds,
        max(int(outer_timeout_seconds) - 30, 30),
    )
    out = list(command)
    out = _append_option_if_missing(
        out,
        "--max-symbols",
        str(
            max(
                _safe_int(
                    os.getenv(
                        "SOURCE_VERIFICATION_TICKER_MAX_SYMBOLS",
                        os.getenv("SOURCE_VERIFICATION_GUARDED_TICKER_MAX_SYMBOLS", "300"),
                    ),
                    300,
                ),
                1,
            )
        ),
    )
    out = _append_option_if_missing(
        out,
        "--limit-per-symbol",
        str(
            max(
                _safe_int(
                    os.getenv(
                        "SOURCE_VERIFICATION_TICKER_LIMIT_PER_SYMBOL",
                        os.getenv("SOURCE_VERIFICATION_GUARDED_TICKER_LIMIT_PER_SYMBOL", "6"),
                    ),
                    6,
                ),
                1,
            )
        ),
    )
    out = _append_option_if_missing(
        out,
        "--timeout-seconds",
        str(
            max(
                _safe_float(
                    os.getenv(
                        "SOURCE_VERIFICATION_TICKER_TIMEOUT_SECONDS",
                        os.getenv("SOURCE_VERIFICATION_GUARDED_TICKER_TIMEOUT_SECONDS", "2.5"),
                    ),
                    2.5,
                ),
                0.5,
            )
        ),
    )
    out = _set_option(out, "--max-runtime-seconds", str(child_runtime_seconds))
    return out


def _runtime_refresh_contract(project_root: Path) -> dict[str, Any]:
    runtime = _load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    mac = runtime.get("mac_fluidity_contract") if isinstance(runtime.get("mac_fluidity_contract"), dict) else {}
    health_fast = _load_json(project_root / "governance" / "health" / "health_fast_latest.json")
    host_score = _safe_float(runtime.get("host_saturation_score"), 100.0)
    compute = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory = str(runtime.get("memory_pressure_level") or "").strip().lower()
    runtime_status = str(runtime.get("overall_status") or "").strip().lower()
    mac_status = str(mac.get("overall_status") or "").strip().lower()
    mac_score = _safe_float(mac.get("fluidity_score"), 0.0)
    support_pause = bool(mac.get("support_pause_recommended", False))
    health_fast_strict_all_clear = bool(health_fast.get("strict_all_clear", False))
    paper_policy = runtime.get("paper_execution_policy") if isinstance(runtime.get("paper_execution_policy"), dict) else {}
    paper_downshifted_safe = bool(
        not paper_policy
        or (
            bool(paper_policy.get("paper_execution_allowed", False))
            and not bool(paper_policy.get("pause_paper_execution", False))
        )
    )
    strict_heavy_allowed = bool(
        runtime_status == "ready"
        and compute == "normal"
        and memory == "normal"
        and host_score < 45.0
        and mac_status == "ready"
        and mac_score >= 90.0
        and not support_pause
    )
    guarded_heavy_allowed = bool(
        health_fast_strict_all_clear
        and memory == "normal"
        and compute in {"normal", "elevated"}
        and runtime_status in {"ready", "advisory", "degraded"}
        and host_score < 50.0
        and mac_score >= 70.0
        and not support_pause
        and paper_downshifted_safe
    )
    heavy_allowed = bool(strict_heavy_allowed or guarded_heavy_allowed)
    return {
        "active": True,
        "runtime_status": runtime_status,
        "host_saturation_score": round(host_score, 3),
        "compute_pressure_level": compute,
        "memory_pressure_level": memory,
        "mac_fluidity_status": mac_status,
        "mac_fluidity_score": round(mac_score, 3),
        "health_fast_strict_all_clear": health_fast_strict_all_clear,
        "heavy_refresh_allowed": heavy_allowed,
        "heavy_refresh_mode": "strict_ready" if strict_heavy_allowed else ("guarded_single_heavy" if guarded_heavy_allowed else "deferred"),
        "max_command_batch": 2 if strict_heavy_allowed else 1,
        "paper_downshifted_safe": paper_downshifted_safe,
        "policy": "refresh_source_mesh_in_strict_batches_or_single_guarded_heavy_batch_when_fast_health_is_clear",
    }


def _run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_seconds), 1),
        )
        return {
            "command": [str(part) for part in command],
            "rc": int(proc.returncode),
            "ok": proc.returncode == 0,
            "stdout_tail": _tail_text(proc.stdout or ""),
            "stderr_tail": _tail_text(proc.stderr or ""),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "command": [str(part) for part in command],
            "rc": 124,
            "ok": False,
            "stdout_tail": _tail_text(stdout),
            "stderr_tail": _tail_text(stderr),
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "command": [str(part) for part in command],
            "rc": 1,
            "ok": False,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "timed_out": False,
        }


def _write_latest_source_report(project_root: Path, payload: dict[str, Any]) -> None:
    health = project_root / "governance" / "health"
    reports = project_root / "exports" / "reports"
    health.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    (health / "source_verification_latest.json").write_text(text, encoding="utf-8")
    (reports / "source_verification_latest.md").write_text(report_src._render_markdown(payload), encoding="utf-8")


def _downstream_recheck_commands(project_root: Path) -> list[list[str]]:
    return [
        _opsctl(project_root, "collector-contracts", "--json"),
        _opsctl(project_root, "health-gates", "--json"),
    ]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    max_commands: int = 2,
    timeout_seconds: int = 240,
    max_heavy_commands: int = 1,
) -> dict[str, Any]:
    before = report_src.build_source_verification_payload(project_root)
    runtime_contract = _runtime_refresh_contract(project_root)
    commands = before.get("recommended_refresh_commands") if isinstance(before.get("recommended_refresh_commands"), list) else []
    unique_commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in commands:
        if not isinstance(raw, list):
            continue
        command = [str(part) for part in raw if str(part).strip()]
        if not command:
            continue
        key = _command_key(command)
        if key in seen:
            continue
        seen.add(key)
        unique_commands.append(command)

    refresh_candidates = [
        command for command in unique_commands if not any(str(part) == "source-verification" for part in command)
    ]
    degraded_sources = [str(item) for item in (before.get("degraded_artifacts") or []) if str(item)]
    source_rows = {
        str(row.get("source_id") or ""): row
        for row in (before.get("sources") if isinstance(before.get("sources"), list) else [])
        if isinstance(row, dict) and str(row.get("source_id") or "").strip()
    }
    source_by_command: dict[tuple[str, ...], str] = {}
    for source_id in degraded_sources:
        try:
            command = report_src._refresh_command_for_source(project_root, source_id)
        except Exception:
            continue
        source_by_command.setdefault(_command_key([str(part) for part in command]), source_id)
    stale_sources = {str(item) for item in (before.get("stale_artifacts") or []) if str(item)}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    heavy_count = 0
    batch_cap = min(max(int(max_commands), 0), _safe_int(runtime_contract.get("max_command_batch"), 1))
    heavy_cap = max(_safe_int(max_heavy_commands, 0), 0)
    for command in refresh_candidates:
        policy = _command_policy(command, default_timeout_seconds=int(timeout_seconds))
        command = _bounded_heavy_command(
            command,
            outer_timeout_seconds=_safe_int(policy.get("timeout_seconds"), int(timeout_seconds)),
        )
        policy = _command_policy(command, default_timeout_seconds=int(timeout_seconds))
        source_id = _source_id_for_command(command, source_by_command)
        if source_id == "macro_crossstack" and stale_sources.intersection(MACRO_CROSSCHECK_STALE_DEPENDENCIES):
            skipped.append(
                {
                    "command": command,
                    "reason": "dependent_stale_sources_waiting",
                    "source_id": source_id,
                    "stale_dependencies": sorted(stale_sources.intersection(MACRO_CROSSCHECK_STALE_DEPENDENCIES)),
                    "policy": policy,
                }
            )
            continue
        source_row = source_rows.get(source_id, {}) if source_id else {}
        if (
            source_id
            and stale_sources
            and source_id not in stale_sources
            and bool(source_row.get("fresh", False))
            and bool(source_row.get("ok", False))
        ):
            skipped.append(
                {
                    "command": command,
                    "reason": "fresh_ok_source_confidence_debt_waiting",
                    "source_id": source_id,
                    "waiting_stale_sources": sorted(stale_sources),
                    "policy": policy,
                }
            )
            continue
        if len(selected) >= batch_cap:
            skipped.append({"command": command, "reason": "bounded_batch_cap", "policy": policy})
            continue
        if bool(policy.get("heavy", False)):
            if not bool(runtime_contract.get("heavy_refresh_allowed", False)):
                skipped.append({"command": command, "reason": "runtime_or_mac_fluidity_not_ready_for_heavy_refresh", "policy": policy})
                continue
            if heavy_count >= heavy_cap:
                skipped.append({"command": command, "reason": "heavy_refresh_batch_cap", "policy": policy})
                continue
            heavy_count += 1
        selected.append({"command": command, "policy": policy})

    results: list[dict[str, Any]] = []
    downstream_recheck_commands = _downstream_recheck_commands(project_root)
    downstream_recheck_results: list[dict[str, Any]] = []
    after = before
    if apply and selected:
        for row in selected:
            command = row["command"]
            policy = row["policy"] if isinstance(row.get("policy"), dict) else {}
            results.append(_run_command(command, cwd=project_root, timeout_seconds=_safe_int(policy.get("timeout_seconds"), int(timeout_seconds))))
        after = report_src.build_source_verification_payload(project_root)
        _write_latest_source_report(project_root, after)
        for command in downstream_recheck_commands:
            downstream_recheck_results.append(
                _run_command(command, cwd=project_root, timeout_seconds=DOWNSTREAM_RECHECK_TIMEOUT_SECONDS)
            )
    elif apply:
        _write_latest_source_report(project_root, after)
        for command in downstream_recheck_commands:
            downstream_recheck_results.append(
                _run_command(command, cwd=project_root, timeout_seconds=DOWNSTREAM_RECHECK_TIMEOUT_SECONDS)
            )

    failed = [row for row in results if not bool(row.get("ok", False))]
    downstream_failed = [row for row in downstream_recheck_results if not bool(row.get("ok", False))]
    status = "ready" if bool(after.get("ok", False)) else "needs_refresh"
    if apply and failed:
        status = "applied_with_failures"
    elif apply and downstream_failed:
        status = "applied_with_recheck_failures"
    elif apply and results:
        status = "applied" if bool(after.get("ok", False)) else "applied_still_degraded"
    elif not selected and skipped:
        status = "deferred_by_runtime_governor"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": status in {"ready", "applied", "needs_refresh", "deferred_by_runtime_governor"},
        "overall_status": status,
        "apply": bool(apply),
        "runtime_refresh_contract": runtime_contract,
        "before": {
            "overall_status": str(before.get("overall_status") or ""),
            "unverified_sources": list(before.get("unverified_sources") or []),
            "stale_artifacts": list(before.get("stale_artifacts") or []),
            "degraded_artifacts": list(before.get("degraded_artifacts") or []),
        },
        "after": {
            "overall_status": str(after.get("overall_status") or ""),
            "unverified_sources": list(after.get("unverified_sources") or []),
            "stale_artifacts": list(after.get("stale_artifacts") or []),
            "degraded_artifacts": list(after.get("degraded_artifacts") or []),
        },
        "planned_commands": unique_commands,
        "selected_commands": [row["command"] for row in selected],
        "skipped_commands": skipped,
        "applied_commands": [row["command"] for row in selected] if apply else [],
        "results": results,
        "downstream_recheck_commands": downstream_recheck_commands,
        "downstream_recheck_results": downstream_recheck_results,
        "recommended_actions": [
            "source refresh is deferred until runtime and Mac fluidity are ready"
            if status == "deferred_by_runtime_governor"
            else "post-refresh downstream rechecks failed; inspect collector-contracts and health-gates artifacts"
            if downstream_failed
            else "apply source-verification-refresh to refresh degraded artifacts in bounded batches"
            if not apply and selected
            else "rerun source-verification after failed refresh commands"
            if failed
            else "rerun source-verification-refresh for the next bounded batch"
            if skipped
            else "source verification autorefresh completed",
        ],
        "source_verification": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh stale/degraded source-verification artifacts and rerun the verification report.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-commands", type=int, default=int(os.getenv("SOURCE_VERIFICATION_REFRESH_MAX_COMMANDS", "2")))
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("SOURCE_VERIFICATION_REFRESH_TIMEOUT_SECONDS", "240")))
    parser.add_argument("--max-heavy-commands", type=int, default=int(os.getenv("SOURCE_VERIFICATION_REFRESH_MAX_HEAVY_COMMANDS", "1")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        max_commands=int(args.max_commands),
        timeout_seconds=int(args.timeout_seconds),
        max_heavy_commands=int(args.max_heavy_commands),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "source_verification_autorefresh "
            f"status={payload.get('overall_status', '')} "
            f"planned={len(payload.get('planned_commands') or [])} "
            f"selected={len(payload.get('selected_commands') or [])} "
            f"applied={len(payload.get('applied_commands') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_refresh", "deferred_by_runtime_governor", "applied", "applied_still_degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
