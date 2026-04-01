import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_walk_forward_gate_filters_inactive_deleted_and_infrastructure() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wf_file = td_path / "walk_forward.json"
        registry_file = td_path / "registry.json"
        out_file = td_path / "promotion_gate.json"

        _write_json(
            wf_file,
            {
                "bots": {
                    "brain_refinery_pass_signal": {
                        "runs": 30,
                        "forward_mean": 0.61,
                        "delta": 0.0,
                        "trading_quality_score": 0.63,
                        "overfit_gap": 0.01,
                        "status": "pass",
                    },
                    "brain_refinery_fail_signal": {
                        "runs": 30,
                        "forward_mean": 0.49,
                        "delta": -0.03,
                        "trading_quality_score": 0.40,
                        "overfit_gap": 0.02,
                        "status": "fail",
                    },
                    "brain_refinery_deleted_signal": {
                        "runs": 30,
                        "forward_mean": 0.45,
                        "delta": -0.04,
                        "trading_quality_score": 0.41,
                        "overfit_gap": 0.03,
                        "status": "fail",
                    },
                    "brain_refinery_infra": {
                        "runs": 30,
                        "forward_mean": 0.46,
                        "delta": -0.02,
                        "trading_quality_score": 0.45,
                        "overfit_gap": 0.01,
                        "status": "fail",
                    },
                }
            },
        )
        _write_json(
            registry_file,
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_pass_signal",
                        "active": True,
                        "deleted_from_rotation": False,
                        "bot_role": "signal_sub_bot",
                    },
                    {
                        "bot_id": "brain_refinery_fail_signal",
                        "active": False,
                        "deleted_from_rotation": False,
                        "bot_role": "signal_sub_bot",
                    },
                    {
                        "bot_id": "brain_refinery_deleted_signal",
                        "active": False,
                        "deleted_from_rotation": True,
                        "bot_role": "signal_sub_bot",
                    },
                    {
                        "bot_id": "brain_refinery_infra",
                        "active": True,
                        "deleted_from_rotation": False,
                        "bot_role": "infrastructure_sub_bot",
                    },
                ]
            },
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "walk_forward_promotion_gate.py"),
                "--in-file",
                str(wf_file),
                "--registry-file",
                str(registry_file),
                "--out-file",
                str(out_file),
                "--min-considered-bots",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["considered_bots"] == 1
        assert payload["failed_bots"] == 0
        assert payload["promote_ok"] is True
        assert payload["excluded_counts"] == {
            "inactive": 1,
            "deleted_from_rotation": 1,
            "infrastructure_sub_bot": 1,
        }


def test_lane_gate_filters_same_registry_noise() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wf_file = td_path / "walk_forward_lane.json"
        registry_file = td_path / "registry_lane.json"
        out_file = td_path / "lane_gate.json"

        _write_json(
            wf_file,
            {
                "bots": {
                    "brain_refinery_intraday_pass": {
                        "runs": 30,
                        "forward_mean": 0.62,
                        "delta": 0.0,
                        "trading_quality_score": 0.66,
                        "overfit_gap": 0.01,
                        "status": "pass",
                    },
                    "brain_refinery_intraday_deleted": {
                        "runs": 30,
                        "forward_mean": 0.41,
                        "delta": -0.03,
                        "trading_quality_score": 0.42,
                        "overfit_gap": 0.02,
                        "status": "fail",
                    },
                }
            },
        )
        _write_json(
            registry_file,
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_intraday_pass",
                        "active": True,
                        "deleted_from_rotation": False,
                        "bot_role": "signal_sub_bot",
                    },
                    {
                        "bot_id": "brain_refinery_intraday_deleted",
                        "active": False,
                        "deleted_from_rotation": True,
                        "bot_role": "signal_sub_bot",
                    },
                ]
            },
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "lane_promotion_gate.py"),
                "--in-file",
                str(wf_file),
                "--registry-file",
                str(registry_file),
                "--out-file",
                str(out_file),
                "--min-considered-per-lane",
                "1",
                "--min-covered-lanes",
                "1",
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["considered_bots"] == 1
        assert payload["promote_ok"] is True
        assert payload["excluded_counts"] == {"deleted_from_rotation": 1}


def test_walk_forward_gate_reclassifies_narrow_single_metric_misses_as_near_pass() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wf_file = td_path / "walk_forward.json"
        registry_file = td_path / "registry.json"
        out_file = td_path / "promotion_gate.json"

        _write_json(
            wf_file,
            {
                "bots": {
                    "brain_refinery_v10_seasonal": {
                        "runs": 19,
                        "forward_mean": 0.92,
                        "delta": 0.0,
                        "trading_quality_score": 1.0,
                        "overfit_gap": 0.0,
                        "status": "pass",
                    },
                    "brain_refinery_v35_dmi_state_machine": {
                        "runs": 16,
                        "forward_mean": 0.82,
                        "delta": 0.01,
                        "trading_quality_score": 1.0,
                        "overfit_gap": 0.0,
                        "status": "pass",
                    },
                    "brain_refinery_v12_news_shocks": {
                        "runs": 29,
                        "forward_mean": 0.692457,
                        "delta": -0.024205,
                        "trading_quality_score": 0.730543,
                        "overfit_gap": 0.024205,
                        "status": "fail",
                    },
                    "brain_refinery_v26_relative_strength_cross_section": {
                        "runs": 21,
                        "forward_mean": 0.504624,
                        "delta": 0.004544,
                        "trading_quality_score": 0.613388,
                        "overfit_gap": -0.004544,
                        "status": "fail",
                    },
                    "brain_refinery_v48_position_1m_3m": {
                        "runs": 19,
                        "forward_mean": 0.520615,
                        "delta": 0.005329,
                        "trading_quality_score": 0.617234,
                        "overfit_gap": -0.005329,
                        "status": "fail",
                    },
                }
            },
        )
        _write_json(
            registry_file,
            {
                "sub_bots": [
                    {"bot_id": "brain_refinery_v10_seasonal", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v35_dmi_state_machine", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v12_news_shocks", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v26_relative_strength_cross_section", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v48_position_1m_3m", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                ]
            },
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "walk_forward_promotion_gate.py"),
                "--in-file",
                str(wf_file),
                "--registry-file",
                str(registry_file),
                "--out-file",
                str(out_file),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["promote_ok"] is True
        assert payload["failed_bots"] == 1
        assert payload["raw_failed_bots"] == 3
        assert payload["near_pass_bots"] == 2
        assert {row["bot_id"] for row in payload["near_pass_examples"]} == {
            "brain_refinery_v12_news_shocks",
            "brain_refinery_v48_position_1m_3m",
        }


def test_lane_gate_scales_coverage_to_observed_lanes_and_excludes_near_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wf_file = td_path / "walk_forward_lane.json"
        registry_file = td_path / "registry_lane.json"
        out_file = td_path / "lane_gate.json"

        _write_json(
            wf_file,
            {
                "bots": {
                    "brain_refinery_v10_seasonal": {
                        "runs": 19,
                        "forward_mean": 0.92,
                        "delta": 0.0,
                        "trading_quality_score": 1.0,
                        "overfit_gap": 0.0,
                        "status": "pass",
                    },
                    "brain_refinery_v35_dmi_state_machine": {
                        "runs": 16,
                        "forward_mean": 0.82,
                        "delta": 0.01,
                        "trading_quality_score": 1.0,
                        "overfit_gap": 0.0,
                        "status": "pass",
                    },
                    "brain_refinery_v12_news_shocks": {
                        "runs": 29,
                        "forward_mean": 0.692457,
                        "delta": -0.024205,
                        "trading_quality_score": 0.730543,
                        "overfit_gap": 0.024205,
                        "status": "fail",
                    },
                    "brain_refinery_v26_relative_strength_cross_section": {
                        "runs": 21,
                        "forward_mean": 0.504624,
                        "delta": 0.004544,
                        "trading_quality_score": 0.613388,
                        "overfit_gap": -0.004544,
                        "status": "fail",
                    },
                    "brain_refinery_v48_position_1m_3m": {
                        "runs": 19,
                        "forward_mean": 0.520615,
                        "delta": 0.005329,
                        "trading_quality_score": 0.617234,
                        "overfit_gap": -0.005329,
                        "status": "fail",
                    },
                }
            },
        )
        _write_json(
            registry_file,
            {
                "sub_bots": [
                    {"bot_id": "brain_refinery_v10_seasonal", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v35_dmi_state_machine", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v12_news_shocks", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v26_relative_strength_cross_section", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                    {"bot_id": "brain_refinery_v48_position_1m_3m", "active": True, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                ]
            },
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "lane_promotion_gate.py"),
                "--in-file",
                str(wf_file),
                "--registry-file",
                str(registry_file),
                "--out-file",
                str(out_file),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["promote_ok"] is True
        assert payload["coverage_ok"] is True
        assert payload["effective_thresholds"]["min_covered_lanes"] == 2
        assert payload["lanes"]["equities"]["failed_bots"] == 1
        assert payload["lanes"]["swing"]["failed_bots"] == 0
        assert payload["lanes"]["swing"]["near_pass_bots"] == 1
        assert payload["ranked_lanes"][0]["lane"] == "equities"
        assert payload["ranked_lanes"][0]["fail_share"] == 0.25
