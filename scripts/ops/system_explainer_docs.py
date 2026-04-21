#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "exports" / "reports" / "system_explainers"
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
FEATURE_STORE_PATH = PROJECT_ROOT / "governance" / "feature_store" / "latest.json"
MANIFEST_PATH = HEALTH_DIR / "system_explainer_docs_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _system_state(health_dir: Path, feature_store_path: Path) -> dict[str, Any]:
    collectors = _load_json(health_dir / "collector_contracts_latest.json")
    verification = _load_json(health_dir / "source_verification_latest.json")
    ingestion = _load_json(health_dir / "ingestion_storage_control_latest.json")
    gates = _load_json(health_dir / "health_gates_latest.json")
    sql_link = _load_json(health_dir / "sql_link_service_latest.json")
    paper_performance = _load_json(health_dir / "paper_performance_latest.json")
    feature_store = _load_json(feature_store_path)
    return {
        "collectors": collectors,
        "verification": verification,
        "ingestion": ingestion,
        "gates": gates,
        "sql_link": sql_link,
        "paper_performance": paper_performance,
        "feature_store": feature_store,
    }


def _framework_html(generated_utc: str, state: dict[str, Any]) -> str:
    collectors = state["collectors"]
    verification = state["verification"]
    ingestion = state["ingestion"]
    gates = state["gates"]
    sql_link = state["sql_link"]
    paper_performance = state["paper_performance"]
    feature_store = state["feature_store"]
    overall = verification.get("overall") if isinstance(verification.get("overall"), dict) else {}
    point_in_time = feature_store.get("point_in_time_contract") if isinstance(feature_store.get("point_in_time_contract"), dict) else {}
    steady_state = ingestion.get("steady_state") if isinstance(ingestion.get("steady_state"), dict) else {}
    steady_targets = steady_state.get("targets") if isinstance(steady_state.get("targets"), dict) else {}
    steady_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    active_paper = paper_performance.get("active_paper_profiles_today") if isinstance(paper_performance.get("active_paper_profiles_today"), list) else []
    active_paper_labels = ", ".join(
        str(item.get("profile") or "")
        for item in active_paper
        if isinstance(item, dict) and str(item.get("profile") or "").strip()
    ) or "n/a"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Framework Map v2</title>
  <style>
    :root {{
      --bg: #f4f7f8;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #5b6471;
      --line: #d7e0e6;
      --teal: #2ca6a4;
      --blue: #7fa8d1;
      --gold: #d8a93a;
      --red: #c65a5a;
      --green: #1e8e5a;
      --purple: #6b5bd2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ padding: 28px 30px 36px; }}
    .hero {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: 28px; }}
    .sub {{ margin-top: 8px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-top: 20px; }}
    .box {{ background: var(--card); border-radius: 16px; border: 2px solid var(--line); padding: 14px; min-height: 154px; }}
    .box h3 {{ font-size: 18px; margin-bottom: 10px; }}
    .box ul {{ margin: 0; padding-left: 18px; }}
    .box.teal {{ border-color: var(--teal); }}
    .box.blue {{ border-color: var(--blue); }}
    .box.gold {{ border-color: var(--gold); }}
    .box.purple {{ border-color: var(--purple); }}
    .box.green {{ border-color: var(--green); }}
    .box.red {{ border-color: var(--red); }}
    .flow-wrap {{ margin-top: 20px; background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px 18px 8px; }}
    .flow-title {{ font-size: 18px; font-weight: 700; margin-bottom: 14px; }}
    .flow {{ display: flex; align-items: stretch; gap: 8px; }}
    .flow .box {{ flex: 1 1 0; min-height: 150px; }}
    .arrow-col {{ width: 34px; display: flex; align-items: center; justify-content: center; position: relative; }}
    .arrow-line {{ width: 100%; height: 2px; background: #8a98a8; position: relative; }}
    .arrow-line::after {{
      content: "";
      position: absolute;
      right: -1px;
      top: -4px;
      border-left: 10px solid #8a98a8;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
    }}
    .arrow-label {{ position: absolute; top: -18px; left: 0; width: 100%; text-align: center; font-size: 10px; color: var(--muted); }}
    .row {{ display: grid; grid-template-columns: 1.4fr 1.2fr 1.4fr; gap: 14px; margin-top: 18px; }}
    .note {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; }}
    .note ul {{ margin: 8px 0 0; padding-left: 18px; }}
    .section-card {{ margin-top: 18px; background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px; }}
    .section-card h2 {{ margin-bottom: 12px; font-size: 20px; }}
    .mini-map {{ margin-top: 10px; }}
    .mini-row {{ display: grid; grid-template-columns: 1fr 48px 1fr 48px 1fr; gap: 8px; align-items: stretch; }}
    .mini-row.four {{ grid-template-columns: 1fr 48px 1fr 48px 1fr 48px 1fr; }}
    .mini-box {{ background: #fbfcfd; border-radius: 14px; border: 2px solid var(--line); padding: 12px; min-height: 114px; }}
    .mini-box h3 {{ font-size: 16px; margin-bottom: 8px; }}
    .mini-box p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .mini-box.teal {{ border-color: var(--teal); }}
    .mini-box.blue {{ border-color: var(--blue); }}
    .mini-box.gold {{ border-color: var(--gold); }}
    .mini-box.purple {{ border-color: var(--purple); }}
    .mini-box.green {{ border-color: var(--green); }}
    .mini-box.red {{ border-color: var(--red); }}
    .mini-arrow {{ display: flex; align-items: center; justify-content: center; position: relative; }}
    .mini-arrow .arrow-line {{ height: 2px; }}
    .mini-note {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .meta {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Schwab Trading Bot Framework Map v2</h1>
      <div class="sub">Generated {generated_utc}. This version shows the real planes that matter in practice: runtime sleeves, cross-sleeve allocation, ingestion/storage, and health-gate feedback.</div>
    </section>

    <section class="flow-wrap">
      <div class="flow-title">Top-Level System Flow</div>
      <div class="flow">
        <div class="box teal">
          <h3>Sources</h3>
          <ul>
            <li>Schwab and Coinbase market / account surfaces</li>
            <li>Macro, FX, crypto, issuer, filing, and education feeds</li>
            <li>Current collector count: {int(collectors.get("collector_count", 0) or 0)}</li>
          </ul>
        </div>
        <div class="arrow-col"><div class="arrow-label">collect</div><div class="arrow-line"></div></div>
        <div class="box blue">
          <h3>Collectors</h3>
          <ul>
            <li>Cached collectors and sync artifacts</li>
            <li>Required failures: {int(collectors.get("required_failure_count", 0) or 0)}</li>
            <li>Average quality: {_fmt_num(collectors.get("average_quality_score", 0.0))}</li>
          </ul>
        </div>
        <div class="arrow-col"><div class="arrow-label">write</div><div class="arrow-line"></div></div>
        <div class="box gold">
          <h3>Artifacts</h3>
          <ul>
            <li>JSONL runtime logs and decision records</li>
            <li>Health latest.json snapshots</li>
            <li>External context blobs and report sources</li>
          </ul>
        </div>
        <div class="arrow-col"><div class="arrow-label">ingest</div><div class="arrow-line"></div></div>
        <div class="box purple">
          <h3>Ingestion + Shards</h3>
          <ul>
            <li>link_jsonl_to_sql + shard manager</li>
            <li>Mode: {sql_link.get("mode", "unknown")}</li>
            <li>Link mode: {sql_link.get("link_mode", "unknown")}</li>
          </ul>
        </div>
        <div class="arrow-col"><div class="arrow-label">serve</div><div class="arrow-line"></div></div>
        <div class="box green">
          <h3>Runtime</h3>
          <ul>
            <li>Sleeves contain specialists, then master / grand-master logic</li>
            <li>Cross-sleeve allocator is downstream</li>
            <li>Execution bridge is separate from sleeve voting</li>
          </ul>
        </div>
        <div class="arrow-col"><div class="arrow-label">learn</div><div class="arrow-line"></div></div>
        <div class="box red">
          <h3>Learning + Ops</h3>
          <ul>
            <li>Feature store strict ok: {_fmt_bool(feature_store.get("strict_ok", False))}</li>
            <li>Event categories: {int(point_in_time.get("event_category_count", 0) or 0)}</li>
            <li>All verified sources: {_fmt_bool(overall.get("all_verified", False))}</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="row">
      <div class="note">
        <h2>Accuracy Note</h2>
        <ul>
          <li>A sleeve is a runtime lane or container, not a passive label.</li>
          <li>Specialists feed sleeve-level master outputs, and grand-master routing sits inside that sleeve path.</li>
          <li>Cross-sleeve allocation and portfolio risk net intents after sleeve-level decisioning.</li>
        </ul>
      </div>
      <div class="note">
        <h2>Control Feedback</h2>
        <ul>
          <li>Ingestion status: {ingestion.get("overall_status", "unknown")}</li>
          <li>Recommended mode: {ingestion.get("recommended_operating_mode", "unknown")}</li>
          <li>Backpressure score: {_fmt_num(steady_state.get("quality_score", 0.0), 2)}/100 with pressure_index {_fmt_num(ingestion.get("pressure_index", 0.0), 3)}</li>
          <li>Health gates can actively push runtime into shadow-only or backlog-protection mode.</li>
        </ul>
      </div>
      <div class="note">
        <h2>Missing From Simpler Maps</h2>
        <ul>
          <li>Storage routing between BOT_LOGS and local fallback</li>
          <li>Shard-pressure and backlog control</li>
          <li>Point-in-time event-store freshness contracts</li>
          <li>Paper lane roster today: {active_paper_labels}</li>
        </ul>
      </div>
    </section>

    <section class="section-card">
      <h2>Folded-In Deep-Dive Maps</h2>
      <div class="mini-note">These are the missing maps folded into the same file so the framework PDF reads like one packet instead of a pile of separate summaries.</div>

      <div class="mini-map">
        <h3>Runtime Hierarchy</h3>
        <div class="mini-row four">
          <div class="mini-box teal">
            <h3>Launcher</h3>
            <p>run_all_sleeves coordinates process groups and checks storage / halt readiness before launch.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box blue">
            <h3>Sleeves</h3>
            <p>Each sleeve is a runtime lane such as aggressive, dividend, dividend_capture, bond, FX, futures, or crypto.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box gold">
            <h3>Specialists</h3>
            <p>Signal, options, futures, and infra observers feed sleeve-level voting.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box green">
            <h3>Master / Grand-Master</h3>
            <p>Weighted master outputs and infra veto logic determine sleeve-level action and intent.</p>
          </div>
        </div>
        <div class="mini-note">Then approved intents flow into cross-sleeve allocation, portfolio risk, and the execution bridge.</div>
      </div>

      <div class="mini-map">
        <h3>Data Intake And Shards</h3>
        <div class="mini-row four">
          <div class="mini-box teal">
            <h3>Collectors</h3>
            <p>Payloads, sync artifacts, watermarks, and provenance metadata get written at collection time.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box gold">
            <h3>Artifacts</h3>
            <p>Hot JSONL logs plus governance/health and external context JSON snapshots.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box purple">
            <h3>Stream IDs</h3>
            <p>link_jsonl_to_sql classifies streams like runtime, governance, feature_store, event_store, and external_context.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box red">
            <h3>Shard Targets</h3>
            <p>Shard families absorb writes before merge and protect the hot path from colder/supporting data.</p>
          </div>
        </div>
        <div class="mini-note">Backpressure scorecard target set: pressure_index&lt;={_fmt_num(steady_targets.get("pressure_index", 0.25), 2)}, core_pending_lines&lt;={int(steady_targets.get("core_pending_lines", 5000) or 0)}, total_drain_minutes&lt;={_fmt_num(steady_targets.get("estimated_total_drain_minutes", 15.0), 1)}. Ready now: {_fmt_bool(steady_status.get("steady_state_ready", False))}.</div>
      </div>

      <div class="mini-map">
        <h3>Health, Halt, And Failover Feedback</h3>
        <div class="mini-row">
          <div class="mini-box red">
            <h3>Health Gates</h3>
            <p>Current recommended mode: {gates.get("recommended_operating_mode", "unknown")}.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box purple">
            <h3>Runtime Controls</h3>
            <p>Shadow-only, backlog protection, halts, kill switches, and broker truth checks can actively change runtime behavior.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box blue">
            <h3>Storage + Recovery</h3>
            <p>BOT_LOGS routing, local fallback, safe eject, failback sync, and split-brain reconcile protect the data plane.</p>
          </div>
        </div>
      </div>

      <div class="mini-map">
        <h3>Storage Routing And Failover</h3>
        <div class="mini-row four">
          <div class="mini-box teal">
            <h3>BOT_LOGS Primary</h3>
            <p>Normal writes land on the routed external storage root so log and SQL growth stay off the internal drive.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box blue">
            <h3>Router</h3>
            <p>storage_router and the switch orchestrator decide whether paths resolve to external storage or local fallback.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box gold">
            <h3>Fallback</h3>
            <p>If BOT_LOGS drops, writers move to local fallback so collection and ingestion can keep running.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box purple">
            <h3>Failback Sync</h3>
            <p>Split-brain reconciliation, failback sync, and route journals help move the plane back to the external root cleanly.</p>
          </div>
        </div>
      </div>

      <div class="mini-map">
        <h3>Broker Truth And Reconciliation</h3>
        <div class="mini-row four">
          <div class="mini-box teal">
            <h3>Broker Snapshot</h3>
            <p>Shared Schwab account truth is fetched and cached for the live lane set.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box green">
            <h3>Lane Truth Files</h3>
            <p>Each lane writes broker-truth health showing ok, mismatch, or transient broker/auth errors.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box red">
            <h3>Overrides + Mismatches</h3>
            <p>Manual override expectations are compared to broker positions so intentional vs accidental drift is visible.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box blue">
            <h3>Execution Reconcile</h3>
            <p>Execution-layer position reconciliation is separate from broker-truth sanity and matters when orders are allowed.</p>
          </div>
        </div>
      </div>

      <div class="mini-map">
        <h3>Training And Promotion</h3>
        <div class="mini-row four">
          <div class="mini-box teal">
            <h3>Runtime Evidence</h3>
            <p>Decisions, runtime snapshots, health artifacts, and external context feed the training surface.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box gold">
            <h3>Point-In-Time Store</h3>
            <p>Event-store freshness and feature-store contracts preserve what the system knew at decision time.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box purple">
            <h3>Retraining</h3>
            <p>Behavior datasets, walk-forward validation, teacher quality, and model card generation prepare candidates.</p>
          </div>
          <div class="mini-arrow"><div class="arrow-line"></div></div>
          <div class="mini-box red">
            <h3>Promotion Gates</h3>
            <p>Promotion quality, canaries, and rollout controls decide whether retrained outputs can feed runtime again.</p>
          </div>
        </div>
      </div>
    </section>

    <div class="meta">Source roots: /governance/health, /governance/feature_store, /exports/reports, /scripts</div>
  </div>
</body>
</html>
"""


def _markdown_doc(title: str, generated_utc: str, intro: list[str], sections: list[tuple[str, list[str]]]) -> str:
    lines: list[str] = [f"# {title}", "", f"- Generated: {generated_utc}"]
    lines.extend(f"- {line}" for line in intro)
    lines.append("")
    for heading, bullets in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.extend(f"- {bullet}" for bullet in bullets)
        lines.append("")
    return "\n".join(lines)


def build_docs(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    generated_utc = datetime.now(timezone.utc).isoformat()
    reports_dir = project_root / "exports" / "reports" / "system_explainers"
    health_dir = project_root / "governance" / "health"
    feature_store_path = project_root / "governance" / "feature_store" / "latest.json"
    manifest_path = health_dir / "system_explainer_docs_latest.json"
    state = _system_state(health_dir, feature_store_path)
    collectors = state["collectors"]
    verification = state["verification"]
    ingestion = state["ingestion"]
    gates = state["gates"]
    paper_performance = state["paper_performance"]
    sql_link = state["sql_link"]
    feature_store = state["feature_store"]
    overall = verification.get("overall") if isinstance(verification.get("overall"), dict) else {}
    point_in_time = feature_store.get("point_in_time_contract") if isinstance(feature_store.get("point_in_time_contract"), dict) else {}
    active_paper = paper_performance.get("active_paper_profiles_today") if isinstance(paper_performance.get("active_paper_profiles_today"), list) else []
    active_paper_labels = ", ".join(
        str(item.get("profile") or "") for item in active_paper if isinstance(item, dict) and str(item.get("profile") or "").strip()
    ) or "n/a"

    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    framework_html_path = reports_dir / "framework_map_v2_latest.html"
    _write_text(framework_html_path, _framework_html(generated_utc, state))
    outputs["framework_map_v2"] = str(framework_html_path)

    runtime_hierarchy = _markdown_doc(
        "Runtime Hierarchy",
        generated_utc,
        intro=[
            "Purpose: explain how sleeves, specialists, master logic, and cross-sleeve allocation fit together.",
            f"Current bundle status: ingestion={ingestion.get('overall_status', 'unknown')} health_gate_mode={gates.get('recommended_operating_mode', 'unknown')}",
            f"Active paper lane roster today: {active_paper_labels}",
        ],
        sections=[
            (
                "Launcher Layer",
                [
                    "run_all_sleeves.py is the coordinator, not the only runtime process.",
                    "It launches mixed lane scripts including parallel shadows, aggressive modes, dedicated dividend/bond/FX lanes, and the execution lane.",
                    "Storage routing is checked before the runtime is allowed to start.",
                ],
            ),
            (
                "Inside A Sleeve",
                [
                    "A sleeve contains specialist sub-bots like signal, options, futures, and infrastructure observers.",
                    "Dividend paper coverage is split between the main dividend lane and a dedicated dividend_capture ex-div lane.",
                    "Those specialist rows feed sleeve-level master outputs and grand-master routing.",
                    "Infrastructure votes and veto signals can bias the sleeve result before intent publication.",
                ],
            ),
            (
                "After Sleeve Voting",
                [
                    "Approved sleeve intents flow into the cross-sleeve allocator and portfolio-risk layer.",
                    "That allocator handles cross-sleeve netting, gross budget, and portfolio caps.",
                    "Execution then happens through the paper or execution bridge rather than directly from each specialist.",
                ],
            ),
        ],
    )
    runtime_hierarchy_path = reports_dir / "runtime_hierarchy_latest.md"
    _write_text(runtime_hierarchy_path, runtime_hierarchy)
    outputs["runtime_hierarchy"] = str(runtime_hierarchy_path)

    data_intake = _markdown_doc(
        "Data Intake And SQL Shards",
        generated_utc,
        intro=[
            "Purpose: show how raw feeds become queryable records and where storage routing fits in.",
            f"Current SQL mode: {sql_link.get('link_mode', 'unknown')} on {sql_link.get('mode', 'unknown')}",
        ],
        sections=[
            (
                "Collection Layer",
                [
                    f"Collectors healthy: {int(collectors.get('collector_count', 0) or 0)} tracked, required_failures={int(collectors.get('required_failure_count', 0) or 0)}, soft_failures={int(collectors.get('soft_failure_count', 0) or 0)}.",
                    "Collectors write payload artifacts plus sync/health files instead of only one latest health blob.",
                    "Watermarks, provenance, and source-verification signals live alongside the payloads.",
                ],
            ),
            (
                "Artifact Layer",
                [
                    "Hot runtime logs live as JSONL event streams.",
                    "Control-plane and context artifacts live as JSON under governance/health, exports, and external_context roots.",
                    "Dead-letter and schema-drift paths catch bad rows before they disappear into aggregate counts.",
                ],
            ),
            (
                "Shard Layer",
                [
                    "link_jsonl_to_sql classifies stream identity, batches rows, and pushes them into shard-specific SQLite targets.",
                    "Shard families include fast health/trading lanes plus broader governance, runtime, attribution, feature_store, event_store, and data/external_context lanes.",
                    "Storage routing decides whether those writes land on BOT_LOGS or the local fallback root.",
                ],
            ),
            (
                "Current Risk",
                [
                    f"Ingestion status is {ingestion.get('overall_status', 'unknown')} with recommended mode {ingestion.get('recommended_operating_mode', 'unknown')}.",
                    f"Core pending lines currently sit at {int((((ingestion.get('backpressure') or {}).get('core_pending_lines', 0)) or 0))}.",
                    (
                        "Backpressure scorecard: "
                        f"quality_score={_fmt_num((((ingestion.get('steady_state') or {}).get('quality_score'))), 2)}/100 "
                        f"pressure_index={_fmt_num(ingestion.get('pressure_index', 0.0), 3)} "
                        f"target_ready={_fmt_bool((((ingestion.get('steady_state') or {}).get('target_status') or {}).get('steady_state_ready', False)))}."
                    ),
                    "The main pain point right now is backlog pressure, not collector quality or SQL sink correctness.",
                ],
            ),
        ],
    )
    data_intake_path = reports_dir / "data_intake_and_shards_latest.md"
    _write_text(data_intake_path, data_intake)
    outputs["data_intake_and_shards"] = str(data_intake_path)

    halt_logic = _markdown_doc(
        "Health Gates And Halt Logic",
        generated_utc,
        intro=[
            "Purpose: explain which controls can slow, halt, or quarantine runtime activity.",
            f"Current recommended mode from health_gates: {gates.get('recommended_operating_mode', 'unknown')}",
        ],
        sections=[
            (
                "Health Gate Inputs",
                [
                    "Health gates consume One Numbers, runtime summaries, SQL ingestion health, backpressure, shard pressure, collector contracts, and storage policy.",
                    f"Current data_quality_score: {_fmt_num(gates.get('data_quality_score', 0.0), 2)}.",
                    f"Hard gate triggered now: {_fmt_bool(gates.get('hard_gate_triggered', False))}.",
                ],
            ),
            (
                "What Can Trip Runtime Protection",
                [
                    "Stale windows, severe backpressure overload, priority shard latency/storage failures, collector contract failures, and SQL progress stalls can all trigger mode changes.",
                    "Those gates do not just report health. They can push runtime toward shadow-only or backlog-protection modes.",
                    "Separate halt and kill-switch paths also exist for global trading halts, lane kills, and incident-driven auto-halts.",
                ],
            ),
            (
                "Why The System Keeps Evidence",
                [
                    "The crash report digest, halt recovery artifacts, broker truth, and storage route status files preserve why a control fired.",
                    "This makes it possible to explain whether the system paused because of data integrity, broker truth, storage, or execution risk.",
                    "That evidence layer is part of the control plane, not a post hoc reporting add-on.",
                ],
            ),
        ],
    )
    halt_logic_path = reports_dir / "health_gates_and_halt_logic_latest.md"
    _write_text(halt_logic_path, halt_logic)
    outputs["health_gates_and_halt_logic"] = str(halt_logic_path)

    storage_routing = _markdown_doc(
        "Storage Routing And Failover",
        generated_utc,
        intro=[
            "Purpose: explain how BOT_LOGS, local fallback, and failback sync protect the log / SQL plane.",
            f"Primary DB realpath: {sql_link.get('primary_db_realpath', sql_link.get('primary_db', 'unknown'))}",
        ],
        sections=[
            (
                "Normal Mode",
                [
                    "The active primary DB and shard files normally live on the routed BOT_LOGS volume.",
                    "The repo path still points at the active root through storage routing instead of hardcoding every consumer to a mount path.",
                    "This keeps most of the heavy log and SQL growth off the internal drive.",
                ],
            ),
            (
                "Fallback Mode",
                [
                    "If the external volume is unavailable, routing can switch to local_fallback_storage.",
                    "The storage eject guard and switch orchestrator are supposed to quiesce writers and flip the routing cleanly.",
                    "Failback sync and split-brain reconciliation then help move the system back toward the external root.",
                ],
            ),
            (
                "Operator Meaning",
                [
                    "Storage routing is not just mount plumbing. It directly affects where the SQL primary, shard DBs, stage areas, and reports are written.",
                    "A safe-eject path matters because Finder ejects can otherwise race with live writers.",
                    "This plane is one of the key differences between a toy bot stack and an operational one.",
                ],
            ),
        ],
    )
    storage_routing_path = reports_dir / "storage_routing_and_failover_latest.md"
    _write_text(storage_routing_path, storage_routing)
    outputs["storage_routing_and_failover"] = str(storage_routing_path)

    broker_truth = _markdown_doc(
        "Broker Truth And Reconciliation",
        generated_utc,
        intro=[
            "Purpose: separate broker truth, manual override reconciliation, and execution-layer reconciliation.",
            "This is one of the easiest parts of the system to misunderstand because several related controls sound similar.",
        ],
        sections=[
            (
                "Broker Truth",
                [
                    "Broker truth checks pull a broker/account snapshot and compare it to expected manual override state per lane.",
                    "A broker-truth error is often an auth/API issue, while a broker-truth mismatch is usually a holdings mismatch or override mismatch.",
                    "That is a different problem from execution slippage or post-trade attribution.",
                ],
            ),
            (
                "Manual Override Reconcile",
                [
                    "Manual trade reconcile state tracks whether the live system sees an active manual difference that needs to be respected or cleared.",
                    "This gives the runtime a way to tolerate operator intervention without pretending the book is unchanged.",
                    "It is especially important when sleeves can trade the same symbols through different decision paths.",
                ],
            ),
            (
                "Execution Reconciliation",
                [
                    "Execution reconciliation looks at fill / execution truth and the live guard rails, not just a shadow account snapshot.",
                    "So broker truth explains account alignment, while execution reconciliation explains order and fill alignment.",
                    "Both belong in the system explainer pack because they answer different operator questions.",
                ],
            ),
        ],
    )
    broker_truth_path = reports_dir / "broker_truth_and_reconciliation_latest.md"
    _write_text(broker_truth_path, broker_truth)
    outputs["broker_truth_and_reconciliation"] = str(broker_truth_path)

    training = _markdown_doc(
        "Training And Promotion",
        generated_utc,
        intro=[
            "Purpose: explain how runtime data becomes learning datasets and how promotion is gated.",
            f"Feature store ok={_fmt_bool(feature_store.get('ok', False))} strict_ok={_fmt_bool(feature_store.get('strict_ok', False))}",
        ],
        sections=[
            (
                "Dataset Layer",
                [
                    "Runtime training snapshots provide append-only rows keyed by timestamp_utc plus snapshot_id, symbol, and mode.",
                    f"Current runtime row count: {int((((feature_store.get('dataset_contract') or {}).get('row_count', 0)) or 0))}.",
                    "Trade-learning labels and behavior datasets then join onto those rows for model and policy learning.",
                ],
            ),
            (
                "Point-In-Time Control",
                [
                    f"Current point-in-time event count: {int((point_in_time.get('event_count', 0) or 0))}.",
                    f"Event store fresh: {_fmt_bool(point_in_time.get('event_store_fresh', False))}.",
                    "The point-in-time event store is what lets training know what the system knew at the time, not just what the operator knows later.",
                ],
            ),
            (
                "Promotion Gates",
                [
                    "Promotion quality gates consume feature-store readiness, contract hashes, data quality, and training-side validation artifacts.",
                    "Model cards, explainability exports, retrain scorecards, and lane scorecards are the operator-facing evidence for that process.",
                    "That is why these PDF reports should live together as one coherent explainer pack instead of scattered single reports.",
                ],
            ),
        ],
    )
    training_path = reports_dir / "training_and_promotion_latest.md"
    _write_text(training_path, training)
    outputs["training_and_promotion"] = str(training_path)

    manifest = {
        "generated_utc": generated_utc,
        "schema_version": 1,
        "output_dir": str(reports_dir),
        "files": outputs,
    }
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=True, indent=2))
    return outputs


def main() -> int:
    outputs = build_docs(PROJECT_ROOT)
    print(json.dumps({"ok": True, "file_count": len(outputs), "files": outputs}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
