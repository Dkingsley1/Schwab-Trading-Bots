import json
import signal
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_report as report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_incident_report_build_payload_is_decision_oriented_and_separate_from_timeline(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"

    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "recent_incident_count": 4,
            "open_incident_count": 2,
            "auto_close_contract": {"closure_ready": False, "candidate_count": 0, "review_required": True},
            "open_surfaces": [
                {
                    "surface": "runtime_separation",
                    "status": "blocked",
                    "category": "operations",
                    "severity": "critical",
                    "summary": "coverage debt still blocks promotion",
                    "age_minutes": 14.2,
                },
                {
                    "surface": "auth_lease",
                    "status": "degraded",
                    "category": "auth_lease",
                    "severity": "warning",
                    "summary": "lease window is thin",
                    "age_minutes": 7.4,
                },
            ],
            "recent_incidents": [
                {
                    "timestamp_utc": "2026-04-22T15:22:12Z",
                    "category": "risk_halt",
                    "severity": "critical",
                    "summary": "global halt stayed engaged",
                    "source_rel": "governance/events/live_softguard_20260422.jsonl",
                },
                {
                    "timestamp_utc": "2026-04-22T15:18:10Z",
                    "category": "auth_lease",
                    "severity": "warning",
                    "summary": "lease margin dropped under threshold",
                    "source_rel": "governance/events/auth_events_20260422.jsonl",
                },
            ],
            "recommended_actions": ["clear the auth lease blocker before resuming live writes"],
        },
    )
    _write_json(
        health / "incident_review_packet_latest.json",
        {
            "overall_status": "blocked",
            "review_required": True,
            "review_state": "awaiting_remediation",
            "packet_sha256": "abc123",
            "closure_contract": {"closure_ready": False, "candidate_count": 0, "review_required": True, "closure_reason": "open_surfaces_present"},
            "recommended_actions": ["treat packet hash as the incident-review anchor"],
            "source_snapshot": {"timeline": {"open_incident_count": 2}, "auth": {"lease_state": "warning"}},
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "blocked", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "degraded", "lease_state": "warning"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "degraded", "critical_backlog": {"unacked_count": 3, "unsent_count": 1}})
    _write_json(health / "lane_thaw_controller_latest.json", {"overall_status": "degraded", "paused_lane_count": 2, "candidate_count": 1})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "degraded", "write_failure_count": 5, "account_snapshot_failure_count": 2})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "degraded", "restart_storms": [{"name": "shadow_watchdog"}], "alerts": [{"summary": "restart storm"}]})

    payload = report.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["review_required"] is True
    assert payload["review_state"] == "awaiting_remediation"
    assert payload["incident_scope"]["report_kind"] == "decision_oriented_incident_report"
    assert "separate from the timeline" in payload["incident_scope"]["separation_from_timeline"]
    assert payload["incident_counts"]["open_incident_count"] == 2
    assert payload["incident_counts"]["critical_open_surface_count"] == 1
    assert payload["control_plane_snapshot"]["runtime_clearance_state"] == "awaiting_coverage_cycles"
    assert payload["control_plane_snapshot"]["critical_alert_backlog"]["unacked_count"] == 3
    assert payload["packet_sha256"] == "abc123"
    assert "runtime_separation" in payload["open_surface_names"]
    assert "risk_halt" in payload["recent_categories"]
    assert payload["remediation_packs"][0]["surface"] == "runtime_separation"
    assert payload["closeout_contract"]["closeout_ready"] is False
    assert "use the timeline for chronology and this report for remediation approvals, escalation, and closeout decisions" in payload["recommended_actions"]


def test_incident_report_main_writes_markdown_html_and_pdf(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root_path = project_root
    health = project_root / "governance" / "health"

    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "degraded",
            "recent_incident_count": 2,
            "open_incident_count": 1,
            "auto_close_contract": {"closure_ready": False, "candidate_count": 0, "review_required": True},
            "open_surfaces": [
                {
                    "surface": "process_watchdog",
                    "status": "degraded",
                    "category": "operations",
                    "severity": "warning",
                    "summary": "restart storm cooled but not cleared",
                    "age_minutes": 3.5,
                }
            ],
            "recent_incidents": [
                {
                    "timestamp_utc": "2026-04-22T15:23:53Z",
                    "category": "operations",
                    "severity": "warning",
                    "summary": "watchdog throttled maintenance work",
                    "source_rel": "governance/watchdog/ops_events_20260422.jsonl",
                }
            ],
            "recommended_actions": ["continue monitoring the watchdog surface until it stabilizes"],
        },
    )
    _write_json(
        health / "incident_review_packet_latest.json",
        {
            "overall_status": "degraded",
            "review_required": True,
            "review_state": "awaiting_remediation",
            "packet_sha256": "packet-456",
            "closure_contract": {"closure_ready": False, "candidate_count": 0, "review_required": True, "closure_reason": "watchdog_alerts_present"},
            "recommended_actions": ["keep the packet hash with the remediation notes"],
            "source_snapshot": {"timeline": {"open_incident_count": 1}},
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "degraded", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "degraded", "restart_storms": [], "alerts": [{"summary": "maintenance throttle"}]})

    out_file = health / "incident_report_latest.json"
    md_file = project_root / "exports" / "reports" / "incident_report_latest.md"
    html_file = project_root / "exports" / "reports" / "incident_report_latest.html"
    pdf_file = project_root / "exports" / "reports" / "incident_report_latest.pdf"

    def _fake_render_pdf_from_html(_html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool, project_root: Path) -> tuple[bool, str]:
        assert allow_gui_renderer is True
        assert project_root == project_root_path
        pdf_path.write_bytes(b"%PDF-1.4\n%dummy incident report\n")
        return True, "ok"

    monkeypatch.setattr(report, "_render_pdf_from_html", _fake_render_pdf_from_html)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "incident_report.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out_file),
            "--md-out-file",
            str(md_file),
            "--html-out-file",
            str(html_file),
            "--pdf-out-file",
            str(pdf_file),
            "--no-refresh-supporting-artifacts",
            "--allow-gui-pdf-renderer",
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    markdown = md_file.read_text(encoding="utf-8")
    html = html_file.read_text(encoding="utf-8")

    assert rc == 0
    assert payload["overall_status"] == "degraded"
    assert payload["artifacts"]["mode"] == "full_bundle"
    assert payload["artifacts"]["pdf_available"] is True
    assert payload["artifacts"]["pdf"] == str(pdf_file)
    assert md_file.exists()
    assert html_file.exists()
    assert pdf_file.exists()
    assert "Incident Report" in markdown
    assert "Why This Is Separate From The Timeline" in markdown
    assert "Open Surfaces" in markdown
    assert "Remediation Packs" in markdown
    assert "Closeout Contract" in markdown
    assert "Decision-Oriented Incident Report" in html
    assert "Current Counts" in html
    assert "watchdog throttled maintenance work" in html


def test_incident_report_app_renderer_reaps_headless_chrome_after_pdf(tmp_path: Path, monkeypatch) -> None:
    chrome_bin = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    chrome_bin.parent.mkdir(parents=True)
    chrome_bin.write_text("# fake chrome\n", encoding="utf-8")
    profile_dir = tmp_path / "incident-report-open-test"
    pdf_path = tmp_path / "incident_report_latest.pdf"

    class FakeProc:
        pid = 4321

        def __init__(self) -> None:
            self.waited = False
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.waited = True
            self.returncode = -signal.SIGTERM
            return self.returncode

        def communicate(self):
            return "", ""

    fake_proc = FakeProc()
    popen_calls: list[list[str]] = []
    killpg_calls: list[tuple[int, int]] = []

    def fake_mkdtemp(prefix: str) -> str:
        assert prefix == "incident-report-open-"
        profile_dir.mkdir()
        return str(profile_dir)

    def fake_popen(cmd, **kwargs):
        popen_calls.append(list(cmd))
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == str(tmp_path)
        pdf_path.write_bytes(b"%PDF-1.4\nincident report\n")
        return fake_proc

    def fake_killpg(pid: int, sig: int) -> None:
        killpg_calls.append((pid, sig))

    monkeypatch.setattr(report.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(report.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(report.os, "killpg", fake_killpg)

    ok, detail = report._render_pdf_via_open_app(str(chrome_bin), "file:///tmp/report.html", pdf_path, project_root=tmp_path)

    assert ok is True
    assert detail == "ok"
    assert popen_calls
    assert popen_calls[0][0] == str(chrome_bin)
    assert "--headless=new" in popen_calls[0]
    assert killpg_calls == [(4321, signal.SIGTERM)]
    assert fake_proc.waited is True
    assert not profile_dir.exists()
