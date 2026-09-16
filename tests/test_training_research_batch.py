import copy
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.ops import training_research_batch as batch


@pytest.fixture
def queue(monkeypatch):
    current = {
        "snapshot": {"schema_version": 2, "rows_sha256": "verified",
                     "timestamp_utc": batch.now_iso(), "latest_row_timestamp_utc": batch.now_iso()},
        "version": 1,
    }
    calls = []
    monkeypatch.setattr(batch, "cohort_contract", lambda root, bots: copy.deepcopy(current))
    monkeypatch.setattr(batch, "resource_admission", lambda *args: True)

    def worker(command, **kwargs):
        bot = command[command.index("--worker") + 1]
        calls.append((bot, kwargs))
        result = {
            "bot_id": bot, "status": "evaluated",
            "contract_sha256": command[command.index("--contract-sha256") + 1],
            "preflight": {"results": [{"blockers": []}]},
            "evaluation": {"results": [{"diagnostic_quality_passed": False, "metrics": {"test": {"accuracy": 0.4}}}]},
        }
        return {"rc": 0, "stdout": json.dumps(result), "timed_out": False}

    monkeypatch.setattr(batch, "run_bounded_process_group", worker)
    return current, calls, worker


def test_larger_cohort_runs_serially_and_keeps_shared_latest_untouched(tmp_path, queue):
    _, calls, _ = queue
    result = batch.execute_queue(tmp_path, bot_ids=[], seconds=600)
    assert len(calls) == len(batch.SUPPORTED) == 13
    assert result["overall_status"] == "complete"
    assert result["evaluated_bot_count"] == 13
    assert result["pending_bot_count"] == 0
    assert result["diagnostic_quality_passed_count"] == 0
    assert result["limits"]["workers"] == 1
    for bot, options in calls:
        assert options["timeout_seconds"] == 120
        assert all(options["env"][key] == value for key, value in batch.THREAD_ENV.items())
        assert (tmp_path / batch.RUNS_PATH / result["run_id"] / f"{bot}.json").is_file()
    for key in ("runtime_model_write", "registry_write", "promotion_authority", "live_execution_authority"):
        assert result["authority_contract"][key] is False
    assert not (tmp_path / "governance/health/training_dataset_preflight_latest.json").exists()
    assert not (tmp_path / "governance/health/training_dataset_evaluation_latest.json").exists()


def test_resource_denial_defers_then_resumes_without_repeating_completed_fits(tmp_path, queue, monkeypatch):
    _, calls, _ = queue
    admissions = iter([True, False])
    monkeypatch.setattr(batch, "resource_admission", lambda *args: next(admissions))
    first = batch.execute_queue(tmp_path, bot_ids=list(batch.SUPPORTED[:3]), seconds=600)
    assert first["overall_status"] == "deferred"
    assert first["evaluated_bot_count"] == 1
    assert first["pending_bot_count"] == 2
    monkeypatch.setattr(batch, "resource_admission", lambda *args: True)
    second = batch.execute_queue(tmp_path, bot_ids=list(batch.SUPPORTED[:3]), seconds=600)
    assert second["run_id"] == first["run_id"]
    assert second["overall_status"] == "complete"
    assert [bot for bot, _ in calls] == list(batch.SUPPORTED[:3])
    batch.execute_queue(tmp_path, bot_ids=list(batch.SUPPORTED[:3]), seconds=600)
    assert len(calls) == 3


def test_input_change_requires_new_run_and_preserves_old_receipts(tmp_path, queue):
    contract, calls, _ = queue
    first = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    contract["version"] += 1
    blocked = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    assert blocked["blockers"] == ["cohort_changed_start_new_run"]
    assert len(calls) == 1
    second = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600, new_run=True)
    assert second["run_id"] != first["run_id"]
    assert (tmp_path / batch.RUNS_PATH / first["run_id"] / "progress.json").is_file()


def test_corrupted_saved_receipt_blocks_resume(tmp_path, queue):
    _, calls, _ = queue
    first = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    path = tmp_path / batch.RUNS_PATH / first["run_id"] / f"{batch.SUPPORTED[0]}.json"
    path.write_text("{}")
    result = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    assert result["blockers"] == ["batch_receipt_invalid_start_new_run"]
    assert len(calls) == 1


def test_malformed_state_cannot_silently_restart_the_same_holdouts(tmp_path, queue):
    path = tmp_path / batch.STATE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    result = batch.execute_queue(tmp_path, bot_ids=[], seconds=600)
    assert result["blockers"] == ["batch_state_invalid_start_new_run"]
    assert queue[1] == []


def test_expired_observations_cannot_be_refreshed_by_new_batch_timestamp(tmp_path, queue):
    contract, calls, _ = queue
    contract["snapshot"]["latest_row_timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    result = batch.execute_queue(tmp_path, bot_ids=[], seconds=600)
    assert result["blockers"] == ["snapshot_missing_stale_or_unverified"]
    assert calls == []


def test_changed_inputs_between_bots_stop_queue(tmp_path, queue, monkeypatch):
    contract, calls, worker = queue

    def change(command, **kwargs):
        result = worker(command, **kwargs)
        contract["version"] += 1
        return result

    monkeypatch.setattr(batch, "run_bounded_process_group", change)
    result = batch.execute_queue(tmp_path, bot_ids=[], seconds=600)
    assert len(calls) == 1
    assert result["blockers"] == ["cohort_input_changed_or_expired"]
    assert result["evaluated_bot_count"] == 0


def test_deadline_defers_without_starting_worker(tmp_path, queue, monkeypatch):
    _, calls, _ = queue
    ticks = iter([0, 10])
    monkeypatch.setattr(batch.time, "monotonic", lambda: next(ticks))
    result = batch.execute_queue(tmp_path, bot_ids=[], seconds=150)
    assert result["blockers"] == ["invocation_time_budget"]
    assert calls == []


def test_worker_timeouts_have_bounded_resume_attempts(tmp_path, queue, monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs)
        return {"rc": 124, "timed_out": True, "stdout": ""}

    monkeypatch.setattr(batch, "run_bounded_process_group", timeout)
    for _ in range(2):
        result = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
        assert result["overall_status"] == "deferred"
        assert result["results"][0]["status"] == "pending"
    result = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    assert len(calls) == 2
    assert result["results"][0]["status"] == "failed"
    assert result["evaluated_bot_count"] == 0


def test_failed_worker_does_not_count_as_a_training_success(tmp_path, queue, monkeypatch):
    monkeypatch.setattr(batch, "run_bounded_process_group", lambda *a, **k: {"rc": 2, "timed_out": False})
    result = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    assert result["overall_status"] == "complete"
    assert result["evaluated_bot_count"] == 0
    assert result["results"][0]["status"] == "failed"


def test_data_blocked_bot_does_not_stop_other_bots(tmp_path, queue, monkeypatch):
    _, calls, worker = queue

    def outcome(command, **kwargs):
        raw = worker(command, **kwargs)
        if len(calls) == 1:
            result = json.loads(raw["stdout"])
            result.update(status="data_blocked", evaluation=None,
                          preflight={"results": [{"blockers": ["sample_floor"]}]})
            raw["stdout"] = json.dumps(result)
        return raw

    monkeypatch.setattr(batch, "run_bounded_process_group", outcome)
    result = batch.execute_queue(tmp_path, bot_ids=list(batch.SUPPORTED[:2]), seconds=600)
    assert result["completed_bot_count"] == 2
    assert result["evaluated_bot_count"] == 1
    assert result["results"][0]["blockers"] == ["sample_floor"]


def test_wrong_bot_receipt_is_rejected(tmp_path, queue, monkeypatch):
    _, _, worker = queue

    def mismatch(command, **kwargs):
        raw = worker(command, **kwargs)
        result = json.loads(raw["stdout"])
        result["bot_id"] = "wrong_bot"
        raw["stdout"] = json.dumps(result)
        return raw

    monkeypatch.setattr(batch, "run_bounded_process_group", mismatch)
    result = batch.execute_queue(tmp_path, bot_ids=[batch.SUPPORTED[0]], seconds=600)
    assert result["evaluated_bot_count"] == 0
    assert result["results"][0]["status"] == "failed"


def test_worker_consumes_its_own_preflight_only(tmp_path, queue, monkeypatch):
    contract, _, _ = queue
    prepared = {"results": [{"data_checks_passed": True}]}
    monkeypatch.setattr(batch.preflight, "build_payload", lambda *a, **kw: prepared)

    def evaluate(root, *, preflight):
        assert preflight is prepared
        return {"ok": True}

    monkeypatch.setattr(batch.evaluation, "build_payload", evaluate)
    result = batch.run_worker(tmp_path, batch.SUPPORTED[0], list(batch.SUPPORTED), batch.contract_digest(contract))
    assert result["status"] == "evaluated"


def test_worker_cannot_fit_after_second_resource_denial(tmp_path, queue, monkeypatch):
    contract, _, _ = queue
    monkeypatch.setattr(batch.preflight, "build_payload", lambda *a, **kw: {"results": [{"data_checks_passed": True}]})
    monkeypatch.setattr(batch.evaluation, "build_payload", lambda *a, **kw: pytest.fail("unadmitted fit"))
    monkeypatch.setattr(batch, "resource_admission", lambda *a: False)
    result = batch.run_worker(tmp_path, batch.SUPPORTED[0], list(batch.SUPPORTED), batch.contract_digest(contract))
    assert result["status"] == "deferred"


def test_worker_rechecks_inputs_before_fit(tmp_path, queue, monkeypatch):
    contract, _, _ = queue

    def prepare(*args, **kwargs):
        contract["version"] += 1
        return {"results": [{"data_checks_passed": True}]}

    monkeypatch.setattr(batch.preflight, "build_payload", prepare)
    monkeypatch.setattr(batch.evaluation, "build_payload", lambda *a, **kw: pytest.fail("changed input fit"))
    with pytest.raises(ValueError, match="cohort_input_changed"):
        batch.run_worker(tmp_path, batch.SUPPORTED[0], list(batch.SUPPORTED), batch.contract_digest(contract))


@pytest.mark.parametrize("seconds", [0, 149, 1801])
def test_invalid_duration_rejected_before_io(tmp_path, seconds):
    with pytest.raises(ValueError, match="batch_seconds"):
        batch.execute_queue(tmp_path, bot_ids=[], seconds=seconds)


def test_unknown_bot_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="unsupported_research_cohort"):
        batch.execute_queue(tmp_path, bot_ids=["untrusted_module"], seconds=600)


def test_status_does_not_create_a_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch.sys, "argv", ["batch", "--status", "--json"])
    assert batch.main() == 0
    assert json.loads(capsys.readouterr().out)["overall_status"] == "not_started"
    assert not (tmp_path / batch.RUNS_PATH).exists()


def test_concurrent_invocation_does_not_launch_or_overwrite_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch.sys, "argv", ["batch", "--json"])
    monkeypatch.setattr(batch, "execute_queue", lambda *a, **kw: pytest.fail("concurrent queue"))
    directory = tmp_path / batch.RUNS_PATH
    directory.mkdir(parents=True)
    with (directory / ".lock").open("a") as lock:
        batch.fcntl.flock(lock, batch.fcntl.LOCK_EX | batch.fcntl.LOCK_NB)
        assert batch.main() == 2
    assert json.loads(capsys.readouterr().out)["blockers"] == ["batch_already_running"]
    assert not (tmp_path / batch.STATE_PATH).exists()


@pytest.mark.parametrize("rc,timed_out", [(2, False), (124, True), (0, True), (0, False)])
def test_resource_admission_requires_successful_bounded_guard(tmp_path, monkeypatch, rc, timed_out):
    def run(command, **kwargs):
        assert command[-2:] == ["--profile", "refresh"]
        assert kwargs["timeout_seconds"] == 20
        return {"rc": rc, "timed_out": timed_out}

    monkeypatch.setattr(batch, "run_bounded_process_group", run)
    assert batch.resource_admission(tmp_path, {}) == (rc == 0 and not timed_out)


def test_cohort_identity_binds_snapshot_sources_label_contract_and_split_policy(tmp_path):
    bots = [batch.SUPPORTED[0]]
    owners = ["scripts/ops/training_research_batch.py", "scripts/ops/training_dataset_preflight.py",
              "scripts/ops/training_dataset_evaluation.py", "core/runtime_training_common.py",
              "core/crypto_runtime_bot_common.py", "core/runtime_requested_bot_common.py",
              "core/training_diagnostic_contract.py", "core/indicator_bot_common.py", f"core/{bots[0]}.py"]
    for owner in owners:
        path = tmp_path / owner
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    batch.write_payload(tmp_path / "master_bot_registry.json", {"sub_bots": [{"bot_id": bots[0], "active": True}]})
    first = batch.cohort_contract(tmp_path, bots)
    assert first["bots"][bots[0]]["active"] is True
    assert first["bots"][bots[0]]["split_policy"] == {}
    (tmp_path / owners[-1]).write_text("modified")
    assert batch.contract_digest(first) != batch.contract_digest(batch.cohort_contract(tmp_path, bots))
    batch.write_payload(tmp_path / "governance/training_labeling_intelligence/label_depth_training_dataset_latest.json",
                        {"work_items": [{"bot_id": bots[0], "label_quality_contract": {"split_policy": {"embargo_minutes": 900}}}]})
    assert batch.cohort_contract(tmp_path, bots)["bots"][bots[0]]["split_policy"] == {"embargo_minutes": 900}
