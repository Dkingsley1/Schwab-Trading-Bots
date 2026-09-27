"""Unprivileged single-worker pacing when OS scheduling controls are unavailable."""

import os
import time


def background_policy():
    policy = {
        "workers": 1,
        "cpu_fraction_of_one_core": 0.25,
        "mode": "nice_and_cooperative",
        "requested_nice_increment": 15,
    }
    try:
        os.nice(15)
    except OSError as exc:
        # Do not retry the denied scheduling operation through another API.
        policy.update(mode="cooperative", priority_api_error=type(exc).__name__)
    return policy


class WorkBudget:
    def __init__(
        self,
        fraction=0.25,
        *,
        wall=time.monotonic,
        cpu=time.process_time,
        sleep=time.sleep
    ):
        if not 0 < fraction <= 0.25:
            raise ValueError("invalid_background_cpu_fraction")
        self.fraction, self.wall, self.cpu, self.sleep = fraction, wall, cpu, sleep
        self.start_wall, self.start_cpu = wall(), cpu()
        self.sleep_seconds = 0.0

    def tick(self):
        elapsed = self.wall() - self.start_wall
        used = self.cpu() - self.start_cpu
        debt = used / self.fraction - elapsed
        while debt >= 0.025:
            interval = min(debt, 0.25)
            self.sleep(interval)
            self.sleep_seconds += interval
            debt = (self.cpu() - self.start_cpu) / self.fraction - (
                self.wall() - self.start_wall
            )

    def snapshot(self):
        return {
            "mode": "cooperative_single_worker",
            "cpu_fraction_limit": self.fraction,
            "elapsed_seconds": self.wall() - self.start_wall,
            "process_cpu_seconds": self.cpu() - self.start_cpu,
            "pacing_sleep_seconds": self.sleep_seconds,
        }
