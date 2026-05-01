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
    champion_dir = health_dir.parent / "champion_challenger"
    collectors = _load_json(health_dir / "collector_contracts_latest.json")
    verification = _load_json(health_dir / "source_verification_latest.json")
    ingestion = _load_json(health_dir / "ingestion_storage_control_latest.json")
    gates = _load_json(health_dir / "health_gates_latest.json")
    sql_link = _load_json(health_dir / "sql_link_service_latest.json")
    paper_performance = _load_json(health_dir / "paper_performance_latest.json")
    feature_store = _load_json(feature_store_path)
    live_readiness = _load_json(health_dir / "live_readiness_smoke_latest.json")
    autonomy_control = _load_json(health_dir / "autonomy_control_plane_latest.json")
    data_plane_recovery = _load_json(health_dir / "data_plane_recovery_controller_latest.json")
    process_watchdog = _load_json(health_dir / "process_watchdog_latest.json")
    roster_resilience = _load_json(health_dir / "roster_resilience_planner_latest.json")
    portable_brain = _load_json(health_dir / "portable_brain_contract_latest.json")
    macro_event = _load_json(health_dir / "macro_event_intelligence_latest.json")
    promotion_autopilot = _load_json(champion_dir / "promotion_autopilot_packet_latest.json")
    return {
        "collectors": collectors,
        "verification": verification,
        "ingestion": ingestion,
        "gates": gates,
        "sql_link": sql_link,
        "paper_performance": paper_performance,
        "feature_store": feature_store,
        "live_readiness": live_readiness,
        "autonomy_control": autonomy_control,
        "data_plane_recovery": data_plane_recovery,
        "process_watchdog": process_watchdog,
        "roster_resilience": roster_resilience,
        "portable_brain": portable_brain,
        "macro_event": macro_event,
        "promotion_autopilot": promotion_autopilot,
    }


def _framework_html(generated_utc: str, state: dict[str, Any]) -> str:
    collectors = state["collectors"]
    verification = state["verification"]
    ingestion = state["ingestion"]
    gates = state["gates"]
    sql_link = state["sql_link"]
    paper_performance = state["paper_performance"]
    feature_store = state["feature_store"]
    live_readiness = state["live_readiness"]
    autonomy_control = state["autonomy_control"]
    data_plane_recovery = state["data_plane_recovery"]
    process_watchdog = state["process_watchdog"]
    roster_resilience = state["roster_resilience"]
    portable_brain = state["portable_brain"]
    macro_event = state["macro_event"]
    promotion_autopilot = state["promotion_autopilot"]
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
    host_contract = portable_brain.get("host_contract") if isinstance(portable_brain.get("host_contract"), dict) else {}
    cross_platform = portable_brain.get("cross_platform_proof_node") if isinstance(portable_brain.get("cross_platform_proof_node"), dict) else {}
    restart_storms = process_watchdog.get("restart_storms") if isinstance(process_watchdog.get("restart_storms"), list) else []
    supportable = int(roster_resilience.get("active_supportable_bots", 0) or 0)
    queue_depth = int(data_plane_recovery.get("queue_depth", 0) or 0)
    write_failures = int(data_plane_recovery.get("write_failure_count", 0) or 0)
    snapshot_failures = int(data_plane_recovery.get("account_snapshot_failure_count", 0) or 0)
    blocker_count = int(promotion_autopilot.get("blocker_count", 0) or 0)
    live_score = _fmt_num(live_readiness.get("readiness_score", 0.0), 2)
    autonomy_score = _fmt_num(autonomy_control.get("autonomy_score", 0.0), 2)
    collector_quality = _fmt_num(collectors.get("average_quality_score", 0.0), 3)
    host_label = str(host_contract.get("chip") or host_contract.get("host_profile") or "unknown host")
    macro_source = str(macro_event.get("source") or "unknown source")
    macro_status = str(macro_event.get("overall_status") or "unknown")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Framework Map v2</title>
  <style>
    :root {{
      --bg: #eef2f3;
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
      --shadow: 0 18px 40px rgba(21, 33, 52, 0.08);
      --hero: linear-gradient(145deg, rgba(27, 146, 146, 0.12), rgba(107, 91, 210, 0.08) 55%, rgba(216, 169, 58, 0.12));
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top left, #f8fbfc 0, var(--bg) 40%, #e8eeef 100%); color: var(--ink); font: 15px/1.6 "Avenir Next", "Segoe UI", sans-serif; }}
    .page {{ padding: 28px 30px 36px; }}
    .hero {{ background: var(--hero), var(--card); border: 1px solid rgba(128, 151, 166, 0.22); border-radius: 24px; padding: 26px 28px; box-shadow: var(--shadow); }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font: 700 31px/1.12 "Iowan Old Style", "Georgia", serif; letter-spacing: -0.02em; }}
    h2 {{ font: 700 22px/1.18 "Iowan Old Style", "Georgia", serif; }}
    h3 {{ font-size: 18px; margin-bottom: 10px; }}
    .eyebrow {{ display: inline-block; margin-bottom: 10px; color: var(--purple); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}
    .sub {{ margin-top: 10px; color: var(--muted); max-width: 940px; }}
    .hero-grid {{ display: grid; grid-template-columns: 1.55fr 1fr; gap: 20px; align-items: start; }}
    .hero-notes {{ display: grid; gap: 12px; }}
    .hero-callout {{ background: rgba(255, 255, 255, 0.75); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 16px; padding: 14px 16px; }}
    .hero-callout strong {{ display: block; font-size: 13px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric-card {{ background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(128, 151, 166, 0.18); border-radius: 18px; padding: 14px 16px; box-shadow: var(--shadow); }}
    .metric-card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-card .value {{ margin-top: 6px; font-size: 23px; font-weight: 700; line-height: 1.1; }}
    .metric-card .detail {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .report-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .toc-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .brief-card {{ background: rgba(255, 255, 255, 0.94); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
    .brief-card p {{ margin: 0; color: var(--muted); }}
    .brief-card ul {{ margin: 10px 0 0; padding-left: 18px; }}
    .toc-card {{ background: linear-gradient(180deg, #ffffff 0%, #fafcfd 100%); border-radius: 18px; border: 1px solid rgba(128, 151, 166, 0.18); padding: 18px; box-shadow: var(--shadow); }}
    .toc-card h3 {{ margin-bottom: 8px; }}
    .toc-card p {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-top: 20px; }}
    .box {{ background: linear-gradient(180deg, #ffffff 0%, #fbfcfd 100%); border-radius: 18px; border: 2px solid var(--line); padding: 14px; min-height: 154px; }}
    .box h3 {{ font-size: 18px; margin-bottom: 10px; }}
    .box ul {{ margin: 0; padding-left: 18px; }}
    .box.teal {{ border-color: var(--teal); }}
    .box.blue {{ border-color: var(--blue); }}
    .box.gold {{ border-color: var(--gold); }}
    .box.purple {{ border-color: var(--purple); }}
    .box.green {{ border-color: var(--green); }}
    .box.red {{ border-color: var(--red); }}
    .flow-wrap {{ margin-top: 20px; background: var(--card); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 22px; padding: 18px 18px 8px; box-shadow: var(--shadow); }}
    .flow-title {{ font-size: 18px; font-weight: 700; margin-bottom: 6px; }}
    .flow-sub {{ color: var(--muted); margin-bottom: 14px; }}
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
    .note {{ background: linear-gradient(180deg, #ffffff 0%, #fafcfd 100%); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 18px; padding: 16px 18px; box-shadow: var(--shadow); }}
    .note ul {{ margin: 8px 0 0; padding-left: 18px; }}
    .section-card {{ margin-top: 18px; background: var(--card); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 22px; padding: 18px; box-shadow: var(--shadow); }}
    .section-card h2 {{ margin-bottom: 12px; font-size: 20px; }}
    .section-lead {{ color: var(--muted); margin-bottom: 12px; }}
    .mini-map {{ margin-top: 10px; }}
    .mini-row {{ display: grid; grid-template-columns: 1fr 48px 1fr 48px 1fr; gap: 8px; align-items: stretch; }}
    .mini-row.four {{ grid-template-columns: 1fr 48px 1fr 48px 1fr 48px 1fr; }}
    .mini-box {{ background: #fbfcfd; border-radius: 16px; border: 2px solid var(--line); padding: 12px; min-height: 114px; }}
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
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(44, 166, 164, 0.08); color: var(--teal); font-size: 12px; font-weight: 700; }}
    .closing-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .closing-card {{ background: linear-gradient(180deg, #ffffff 0%, #fafcfd 100%); border-radius: 18px; border: 1px solid rgba(128, 151, 166, 0.18); padding: 18px; }}
    .closing-card p {{ margin: 0; color: var(--muted); }}
    .closing-card ul {{ margin: 10px 0 0; padding-left: 18px; }}
    .meta {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Operational Architecture Report</div>
          <h1>Schwab Trading Bot Framework Map v2</h1>
          <div class="sub">Generated {generated_utc}. This report is the operating blueprint for how the platform turns multi-source inputs into sleeve decisions, portfolio intents, and controlled execution while still preserving point-in-time evidence, routing durability, and rollback discipline.</div>
          <div class="sub">The important point is not just that the stack has many modules. It is that collection, storage, runtime, and learning are separated on purpose so the system can keep broker truth, fail small under pressure, and explain what it knew when a decision was made.</div>
        </div>
        <div class="hero-notes">
          <div class="hero-callout">
            <strong>Why This Map Matters</strong>
            It shows where the runtime can move independently, where it must wait for hard evidence, and which layers are allowed to halt or constrain the system before risk leaks into live execution.
          </div>
          <div class="hero-callout">
            <strong>Current Operational Story</strong>
            Live readiness is `{live_readiness.get("overall_status", "unknown")}` at `{live_score}/100`, autonomy is `{autonomy_control.get("overall_status", "unknown")}` at `{autonomy_score}/100`, and the data plane is carrying queue depth `{queue_depth}` with `{write_failures}` write failures / `{snapshot_failures}` snapshot failures.
          </div>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-card"><div class="label">Live Readiness</div><div class="value">{live_score}/100</div><div class="detail">{live_readiness.get("overall_status", "unknown")}</div></div>
        <div class="metric-card"><div class="label">Autonomy</div><div class="value">{autonomy_score}/100</div><div class="detail">{autonomy_control.get("overall_status", "unknown")}</div></div>
        <div class="metric-card"><div class="label">Collector Quality</div><div class="value">{collector_quality}</div><div class="detail">{int(collectors.get("collector_count", 0) or 0)} collectors tracked</div></div>
        <div class="metric-card"><div class="label">Supportable Active Bots</div><div class="value">{supportable}</div><div class="detail">paper lane roster today: {active_paper_labels}</div></div>
        <div class="metric-card"><div class="label">Queue Depth</div><div class="value">{queue_depth}</div><div class="detail">write {write_failures} / snapshot {snapshot_failures} failures</div></div>
        <div class="metric-card"><div class="label">Portable Host</div><div class="value">{host_label}</div><div class="detail">proof node `{cross_platform.get("status", "unknown")}`</div></div>
      </div>
    </section>

    <section class="section-card">
      <h2>How To Read This Report</h2>
      <div class="section-lead">The document is laid out like an operating brief: first the thesis, then the control surfaces, then the subsystem maps, and finally the upgrades that matter most from here.</div>
      <div class="toc-grid">
        <div class="toc-card">
          <h3>1. Thesis And Proof</h3>
          <p>Read the cover and proof strip first. That tells you whether the architecture is merely well designed or actually carrying its operational weight right now.</p>
        </div>
        <div class="toc-card">
          <h3>2. Control Posture</h3>
          <p>The control sections explain where the system can self-govern, where it needs stronger data-plane health, and which gates still block promotion or live freedom.</p>
        </div>
        <div class="toc-card">
          <h3>3. Deep-Dive Maps</h3>
          <p>The folded maps show the mechanics behind the claims: sleeve hierarchy, shards, storage routing, broker truth, and training/promotion discipline.</p>
        </div>
        <div class="toc-card">
          <h3>4. Recommendations</h3>
          <p>The closing recommendations interpret the maps into action so the report can be used for operating decisions, not just architecture admiration.</p>
        </div>
      </div>
    </section>

    <section class="report-grid">
      <div class="brief-card">
        <h2>Executive Summary</h2>
        <p>This architecture is strongest where it treats evidence, risk, and execution as separate responsibilities. The framework is designed to let sleeves think independently, let cross-sleeve controls net or veto intent, and let operational controls override all of that if broker truth or storage health is questionable.</p>
      </div>
      <div class="brief-card">
        <h2>What Makes It Worthy</h2>
        <ul>
          <li>Point-in-time event categories tracked: `{int(point_in_time.get("event_category_count", 0) or 0)}`.</li>
          <li>Portable runtime contract currently recommends `{portable_brain.get("recommended_runtime_mode", "unknown")}` on `{portable_brain.get("recommended_backend", "unknown")}`.</li>
          <li>Latest macro intelligence surface is `{macro_status}` from `{macro_source}`.</li>
        </ul>
      </div>
      <div class="brief-card">
        <h2>Current Watch Items</h2>
        <ul>
          <li>Promotion state is `{promotion_autopilot.get("autopilot_state", "unknown")}` with `{blocker_count}` blockers.</li>
          <li>Restart storms currently tracked: `{len(restart_storms)}`.</li>
          <li>All verified sources: `{_fmt_bool(overall.get("all_verified", False))}`.</li>
        </ul>
      </div>
    </section>

    <section class="flow-wrap">
      <div class="flow-title">Top-Level System Flow</div>
      <div class="flow-sub">This is the actual control sequence: acquire truth, stamp provenance, route durable writes, build sleeve decisions, then let downstream governance decide whether those intents are allowed to survive.</div>
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
            <li>Average quality: {collector_quality}</li>
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
        <h2>Why The Layering Matters</h2>
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
        <h2>What Simpler Maps Miss</h2>
        <ul>
          <li>Storage routing between BOT_LOGS and local fallback</li>
          <li>Shard-pressure and backlog control</li>
          <li>Point-in-time event-store freshness contracts</li>
          <li>Paper lane roster today: {active_paper_labels}</li>
        </ul>
      </div>
    </section>

    <section class="section-card">
      <h2>Operational Proof And Control Posture</h2>
      <div class="section-lead">This is the report layer that tells an operator why the architecture matters right now, not just what modules exist.</div>
      <div class="report-grid">
        <div class="brief-card">
          <span class="pill">Runtime</span>
          <h3>Execution Integrity</h3>
          <p>The runtime is only trustworthy if live readiness, broker/session readiness, and restart discipline stay aligned. Current posture is live `{live_readiness.get("overall_status", "unknown")}` with readiness `{live_score}/100` and restart storms `{len(restart_storms)}`.</p>
        </div>
        <div class="brief-card">
          <span class="pill">Data Plane</span>
          <h3>Storage And Backpressure</h3>
          <p>Ingestion is `{ingestion.get("overall_status", "unknown")}` with recommended mode `{ingestion.get("recommended_operating_mode", "unknown")}`. Queue depth `{queue_depth}` tells you how much operational drag still sits between collection and clean analytical access.</p>
        </div>
        <div class="brief-card">
          <span class="pill">Governance</span>
          <h3>Promotion And Learning</h3>
          <p>Feature store strictness is `{_fmt_bool(feature_store.get("strict_ok", False))}` and promotion is `{promotion_autopilot.get("autopilot_state", "unknown")}`. That is the layer that decides whether new intelligence is allowed to become production behavior.</p>
        </div>
      </div>
    </section>

    <section class="section-card">
      <h2>Interpretation Notes</h2>
      <div class="section-lead">These notes connect the raw metrics to what they mean operationally.</div>
      <div class="report-grid">
        <div class="brief-card">
          <h3>Why Live Readiness Is Not Enough</h3>
          <p>A `100/100` live surface is important, but it is not the whole story. Queue pressure `{queue_depth}` and autonomy `{autonomy_score}/100` show whether the control plane can sustain that readiness without leaning on operator intervention.</p>
        </div>
        <div class="brief-card">
          <h3>Why Point-In-Time Evidence Matters</h3>
          <p>`{int(point_in_time.get("event_category_count", 0) or 0)}` tracked event categories means the learning stack can reconstruct what the system knew when it acted. That is one of the main differences between a research toy and a defensible trading platform.</p>
        </div>
        <div class="brief-card">
          <h3>Why Portability Changes The Story</h3>
          <p>Recognizing `{host_label}` cleanly is good; proving the same architecture can travel to non-Mac replay and research nodes is better. That is how the platform avoids becoming an impressive but isolated local build.</p>
        </div>
      </div>
    </section>

    <section class="section-card">
      <h2>Folded-In Deep-Dive Maps</h2>
      <div class="mini-note">These deeper maps are folded into the same packet so the report moves from high-level architecture into the concrete subsystems that make durability, broker truth, and promotion discipline possible.</div>

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

    <section class="section-card">
      <h2>Architecture Recommendations</h2>
      <div class="section-lead">This closing section translates the current architecture and proofs into the next moves that would improve trust the most.</div>
      <div class="closing-grid">
        <div class="closing-card">
          <h3>Clear The Data Plane First</h3>
          <p>The biggest operational drag is still the write path and queue pressure.</p>
          <ul>
            <li>Queue depth is currently `{queue_depth}`.</li>
            <li>Write failures / snapshot failures are `{write_failures}` / `{snapshot_failures}`.</li>
            <li>Until those numbers calm down, every other proof surface has to carry more strain.</li>
          </ul>
        </div>
        <div class="closing-card">
          <h3>Turn Promotion Into Proof</h3>
          <p>The architecture already separates training from runtime; the remaining gap is promotion confidence.</p>
          <ul>
            <li>Promotion state is `{promotion_autopilot.get("autopilot_state", "unknown")}`.</li>
            <li>Current blocker count is `{blocker_count}`.</li>
            <li>Finishing those gates is what turns the learning layer from promising into production-grade.</li>
          </ul>
        </div>
        <div class="closing-card">
          <h3>Keep Building Portability And Bench Depth</h3>
          <p>The biggest strategic upside is making the platform stronger without making it more fragile.</p>
          <ul>
            <li>Supportable active bots currently sit at `{supportable}`.</li>
            <li>Portable proof node status is `{cross_platform.get("status", "unknown")}`.</li>
            <li>More supportable depth plus stronger non-Mac proof would make the architecture much harder to dismiss.</li>
          </ul>
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
