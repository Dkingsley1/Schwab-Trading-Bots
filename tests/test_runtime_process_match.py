from scripts.ops.runtime_process_match import command_matches_pattern, parse_process_rows


def test_worker_script_matches_by_path_suffix_and_ordered_arguments() -> None:
    command = (
        "/opt/homebrew/bin/python3 "
        "/repo/scripts/run_execution_lane.py --mode paper --interval 5"
    )

    assert command_matches_pattern(command, "scripts/run_execution_lane.py --mode paper")


def test_launcher_json_does_not_impersonate_worker_process() -> None:
    command = (
        "/opt/homebrew/bin/python3 /repo/scripts/shadow_watchdog.py "
        "--schwab-start-cmd "
        "'[\"/repo/.venv/bin/python\", \"/repo/scripts/run_all_sleeves.py\", "
        "\"--with-aggressive-modes\"]'"
    )

    assert not command_matches_pattern(command, "scripts/run_all_sleeves.py")
    assert command_matches_pattern(command, "scripts/shadow_watchdog.py")


def test_runtime_cpu_class_between_script_and_broker_still_matches() -> None:
    command = (
        "/repo/.venv/bin/python /repo/scripts/run_shadow_training_loop.py "
        "--runtime-cpu-class market_decision --broker schwab "
        "--profile schwab_futures --max-iterations 0"
    )

    assert command_matches_pattern(
        command,
        "scripts/run_shadow_training_loop.py --broker schwab --profile schwab_futures",
    )


def test_process_rows_preserve_command_and_status() -> None:
    rows = parse_process_rows(
        "  101 S    /repo/.venv/bin/python /repo/scripts/run_all_sleeves.py\n"
        "  102 T    /repo/.venv/bin/python /repo/scripts/run_execution_lane.py --mode paper\n"
    )

    assert [row.pid for row in rows] == [101, 102]
    assert rows[0].command.endswith("scripts/run_all_sleeves.py")
    assert rows[1].stat == "T"
