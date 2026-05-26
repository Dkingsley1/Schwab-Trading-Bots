from datetime import datetime, timezone
from types import SimpleNamespace

from scripts.ops import process_watchdog as pw
from scripts.ops import watchdog_intelligence as wi


def test_watchdog_contract_treats_fanout_hold_as_intentional() -> None:
    contract = pw._watchdog_intelligence_contract(
        status_rows=[
            {
                "name": "all_sleeves",
                "heartbeat_ok": False,
                "process_live": False,
                "restart_skipped": "startup_not_ready",
                "reason": "process_fanout_guard_active",
            }
        ],
        restarts=[],
        restart_storms=[],
        recent_restart_storms=[],
        alerts=[],
        safety_pause={"active": False},
        creative_pause={"active": False},
        network_payload={"outage_active": False},
    )

    assert contract["overall_status"] == "ready"
    assert contract["active_issue_count"] == 0
    assert contract["intentional_hold_count"] == 1
    assert contract["exact_needs"][0]["status"] == "intentional_hold"
    assert contract["exact_needs"][0]["blocker"] == "process_fanout_guard_active"


def test_watchdog_contract_surfaces_exact_stale_heartbeat_need() -> None:
    contract = pw._watchdog_intelligence_contract(
        status_rows=[
            {
                "name": "coinbase_loop",
                "heartbeat_ok": False,
                "heartbeat_fresh": False,
                "process_live": True,
            }
        ],
        restarts=[],
        restart_storms=[],
        recent_restart_storms=[],
        alerts=[],
        safety_pause={"active": False},
        creative_pause={"active": False},
        network_payload={"outage_active": False},
    )

    assert contract["overall_status"] == "degraded"
    assert contract["active_issue_count"] == 1
    need = contract["exact_needs"][0]
    assert need["status"] == "needs_repair"
    assert need["blocker"] == "heartbeat_stale"
    assert need["exact_command"] == ["./scripts/ops/opsctl.sh", "coinbase-api-health", "--snapshot", "--json"]


def test_watchdog_intelligence_flags_duplicate_singleton_supervisors() -> None:
    now = datetime.now(timezone.utc)
    process_contract = {
        "overall_status": "ready",
        "grade": "A",
        "score": 100.0,
        "target_count": 4,
        "healthy_target_count": 4,
        "active_issue_count": 0,
        "intentional_hold_count": 0,
        "restart_storm_count": 0,
        "alert_count": 0,
        "exact_needs": [],
        "recommended_commands": [["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"]],
    }

    report = wi.build_report(
        process_watchdog={"timestamp_utc": now.isoformat(), "watchdog_intelligence": process_contract},
        fanout_guard={"timestamp_utc": now.isoformat(), "override": {"active": False}},
        mac_notification_state={"timestamp_utc": now.isoformat()},
        all_sleeves_launcher={"timestamp_utc": now.isoformat(), "status": "running"},
        supervisors={
            "ok": True,
            "counts": {"mac_notification_watch": 2},
            "matches": {"mac_notification_watch": [111, 222]},
            "duplicates": [{"name": "mac_notification_watch", "count": 2, "pids": [111, 222]}],
        },
        now=now,
    )

    assert report["overall_status"] == "degraded"
    assert report["section_grades"]["notification_noise"]["grade"] in {"B", "C"}
    assert any(need["status"] == "duplicate_supervisor" for need in report["exact_needs"])


def test_watchdog_intelligence_integrates_launcher_readiness_contract() -> None:
    now = datetime.now(timezone.utc)
    report = wi.build_report(
        process_watchdog={
            "timestamp_utc": now.isoformat(),
            "watchdog_intelligence": {
                "overall_status": "ready",
                "score": 100.0,
                "restart_storm_count": 0,
                "exact_needs": [],
                "recommended_commands": [["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"]],
            },
        },
        fanout_guard={"timestamp_utc": now.isoformat(), "override": {"active": False}},
        mac_notification_state={"timestamp_utc": now.isoformat()},
        all_sleeves_launcher={
            "timestamp_utc": now.isoformat(),
            "status": "running",
            "launcher_readiness_contract": {
                "mode": "sleeve_launcher_readiness_expansion_v2",
                "readiness_score": 82.0,
                "exact_needs": [
                    {
                        "target": "volatility",
                        "status": "needs_repair",
                        "blocker": "exited",
                        "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                    }
                ],
            },
        },
        supervisors={"ok": True, "counts": {}, "matches": {}, "duplicates": []},
        now=now,
    )

    assert report["overall_status"] == "degraded"
    assert report["section_grades"]["sleeve_launcher_readiness"]["score"] == 82.0
    assert report["all_sleeves_launcher"]["readiness_contract"]["mode"] == "sleeve_launcher_readiness_expansion_v2"
    assert any(need.get("source") == "all_sleeves_launcher_readiness" for need in report["exact_needs"])


def test_watchdog_supervisor_scan_ignores_shadow_watchdog_embedded_launch_command(monkeypatch) -> None:
    ps_out = "\n".join(
        [
            "PID COMMAND",
            " 101 python scripts/shadow_watchdog.py --schwab-start-cmd scripts/run_all_sleeves.py",
            " 202 python scripts/run_all_sleeves.py --broker schwab",
        ]
    )
    monkeypatch.setattr(
        wi.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=ps_out, returncode=0),
    )

    supervisors = wi._collect_supervisors(project_marker="")

    assert supervisors["counts"]["shadow_watchdog"] == 1
    assert supervisors["counts"]["all_sleeves_launcher"] == 1
    assert supervisors["duplicates"] == []
