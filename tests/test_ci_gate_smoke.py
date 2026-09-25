import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate_inputs(tmp_path):
    # Synthetic CI evidence stays in pytest's temporary directory, never governance.
    bots = {}
    registry_rows = []
    for bot_id, runs, train, forward, quality in (
        ("brain_refinery_v10_seasonal", 30, 0.58, 0.57, 0.61),
        ("brain_refinery_v21_flash_crash", 28, 0.56, 0.55, 0.59),
    ):
        bots[bot_id] = {
            "runs": runs,
            "train_mean": train,
            "forward_mean": forward,
            "delta": -0.01,
            "trading_quality_score": quality,
            "overfit_gap": 0.01,
            "status": "pass",
        }
        model, log = tmp_path / f"{bot_id}.npz", tmp_path / f"{bot_id}.json"
        model.touch()
        log.write_text("{}", encoding="utf-8")
        registry_rows.append(
            {
                "bot_id": bot_id,
                "active": True,
                "model_path": str(model),
                "log_file": str(log),
                "quality_score": 0.6,
                "test_accuracy": forward,
            }
        )
    source, registry = tmp_path / "walk_forward.json", tmp_path / "registry.json"
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_evidence": {"complete": True},
                "bots": bots,
            }
        ),
        encoding="utf-8",
    )
    registry.write_text(json.dumps({"sub_bots": registry_rows}), encoding="utf-8")
    return source, registry


def run_gate(tmp_path, script, *args):
    output = tmp_path / f"{script}_result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / f"{script}.py"),
            *map(str, args),
            "--out-file",
            str(output),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "MARKET_DATA_ONLY": "1",
            "ALLOW_ORDER_EXECUTION": "0",
            "SCHWAB_READ_ONLY": "1",
            "ALLOW_LIVE": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert output.is_file(), result.stdout + result.stderr
    return result.returncode, json.loads(output.read_text(encoding="utf-8"))


def run_promotion(tmp_path, source, registry):
    return run_gate(
        tmp_path,
        "walk_forward_promotion_gate",
        "--in-file",
        source,
        "--registry-file",
        registry,
        "--min-runs-per-bot",
        "6",
        "--min-considered-bots",
        "2",
        "--min-forward-mean",
        "0.50",
        "--min-delta",
        "-0.05",
        "--min-trading-quality-score",
        "0.40",
        "--max-fail-share",
        "0.5",
    )


def test_ci_gate_clis_accept_complete_fresh_fixture(tmp_path, gate_inputs):
    source, registry = gate_inputs
    code, promotion = run_promotion(tmp_path, source, registry)
    assert code == 0, promotion
    assert promotion["promote_ok"] is True
    assert promotion["source_evidence_reason"] == "complete_fresh_scan"
    assert promotion["considered_bots"] == 2

    code, graduation = run_gate(
        tmp_path,
        "new_bot_graduation_gate",
        "--registry",
        registry,
        "--walk-forward-file",
        source,
        "--min-runs",
        "6",
        "--min-forward-mean",
        "0.50",
        "--min-delta",
        "-0.05",
        "--min-mature-bots",
        "1",
        "--max-immature-active",
        "0",
    )
    assert code == 0, graduation
    assert graduation["ok"] is True
    assert graduation["immature_active_count"] == 0

    code, leakage = run_gate(
        tmp_path,
        "leak_overfit_guard",
        "--walk-forward-file",
        source,
        "--max-overfit-offenders",
        "10",
        "--max-leak-offenders",
        "10",
    )
    assert code == 0, leakage
    assert leakage["ok"] is True
    assert leakage["counts"] == {"overfit": 0, "severe_overfit": 0, "leak_like": 0}


@pytest.mark.parametrize("case", ["missing", "incomplete", "stale", "future"])
def test_ci_gate_rejects_invalid_source_evidence(tmp_path, gate_inputs, case):
    source, registry = gate_inputs
    payload = json.loads(source.read_text(encoding="utf-8"))
    if case == "missing":
        payload.pop("source_evidence")
    elif case == "incomplete":
        payload["source_evidence"]["complete"] = False
    else:
        delta = timedelta(hours=-1 if case == "stale" else 1)
        payload["timestamp_utc"] = (datetime.now(timezone.utc) + delta).isoformat()
    source.write_text(json.dumps(payload), encoding="utf-8")

    code, promotion = run_promotion(tmp_path, source, registry)
    assert code == 2, promotion
    assert promotion["promote_ok"] is False
    assert promotion["source_evidence_ready"] is False
    expected = (
        "walk_forward_source_scan_incomplete"
        if case in {"missing", "incomplete"}
        else "walk_forward_source_stale_or_future"
    )
    assert promotion["source_evidence_reason"] == expected
