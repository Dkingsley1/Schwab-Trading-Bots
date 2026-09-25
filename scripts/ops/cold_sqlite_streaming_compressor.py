"""Pinned, CPU-paced AFSC compression of isolated archive copies."""

import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import psutil

from core.background_work_budget import WorkBudget

GIB = 1024**3
# The pinned CLI rejects logical sizes >= u32::MAX before compression.
MAX_FILE_BYTES = 4 * GIB - 2
# AFSC block offsets are 32-bit, including the resource-fork header.
MAX_COMPRESSED_BYTES = 7 * GIB // 2
OVERHEAD_BYTES = 64 * 1024**2
MAX_RSS_BYTES = 256 * 1024**2
EXECUTABLE = Path(__file__).resolve().parents[2] / ".venv314/bin/applesauce"
EXECUTABLE_SHA256 = "38be5419c9b8068781880a22fda1ec4bec509e2e68042874d4faa6f65e7eb918"


def installed():
    return EXECUTABLE.is_file()


def require_executable():
    if not installed() or not os.access(EXECUTABLE, os.X_OK):
        raise RuntimeError("applesauce_not_installed")
    if EXECUTABLE.is_symlink() or EXECUTABLE.stat().st_size > 16 * 1024**2:
        raise RuntimeError("applesauce_executable_untrusted")
    if hashlib.sha256(EXECUTABLE.read_bytes()).hexdigest() != EXECUTABLE_SHA256:
        raise RuntimeError("applesauce_executable_hash_mismatch")
    return str(EXECUTABLE)


def extra_scratch_bytes(size):
    return min(int(size * 0.95), MAX_COMPRESSED_BYTES) + OVERHEAD_BYTES


def run_paced(command, *, parent, deadline, reserve_bytes):
    """One child process shares a quarter-core budget across all its threads."""
    started = time.monotonic()
    peak_rss = 0
    sampled_cpu = 0.0
    child = None
    last_probe = 0.0

    def check():
        nonlocal peak_rss, last_probe
        if time.monotonic() >= deadline:
            raise TimeoutError("streaming_compression_deadline")
        if child is not None:
            try:
                peak_rss = max(peak_rss, child.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak_rss > MAX_RSS_BYTES:
                raise RuntimeError("streaming_compressor_memory_limit")
        if time.monotonic() - last_probe >= 1:
            if shutil.disk_usage(parent).free < reserve_bytes:
                raise RuntimeError("streaming_compressor_reserve_consumed")
            last_probe = time.monotonic()

    def cpu():
        nonlocal sampled_cpu
        try:
            value = child.cpu_times()
            sampled_cpu = max(sampled_cpu, value.user + value.system)
        except psutil.NoSuchProcess:
            pass
        return sampled_cpu

    def sleep(seconds):
        check()
        time.sleep(min(seconds, max(deadline - time.monotonic(), 0)))

    check()
    env = {**os.environ, "TMPDIR": str(parent)}

    def send(proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    with tempfile.TemporaryFile(dir=parent) as stdout, tempfile.TemporaryFile(
        dir=parent
    ) as stderr:
        proc = subprocess.Popen(
            command,
            cwd=parent,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        budget = None
        try:
            child = psutil.Process(proc.pid)
            budget = WorkBudget(cpu=cpu, sleep=sleep)
            while proc.poll() is None:
                check()
                # Suspend every thread while paying back the sampled CPU time.
                send(proc, signal.SIGSTOP)
                budget.tick()
                send(proc, signal.SIGCONT)
                sleep(0.05)
            check()
        finally:
            if proc.poll() is None:
                send(proc, signal.SIGCONT)
                send(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    send(proc, signal.SIGKILL)
                    proc.wait(timeout=2)
        stderr.seek(max(stderr.tell() - 4000, 0))
        return {
            "rc": proc.returncode,
            "stderr": stderr.read(4000).decode("utf-8", errors="replace"),
            "peak_sampled_rss_bytes": peak_rss,
            "sampled_child_cpu_seconds": sampled_cpu,
            "elapsed_seconds": time.monotonic() - started,
            "pacing": budget.snapshot(),
            "poll_seconds": 0.05,
            "process_reaped": True,
        }


def compress(executable, target, *, deadline, reserve_bytes):
    size = target.stat().st_size
    if not 0 < size <= MAX_FILE_BYTES:
        raise ValueError("archive_exceeds_streaming_compressor_size")
    ratio = min(0.95, MAX_COMPRESSED_BYTES / max(size, 1))
    return run_paced(
        [
            executable,
            "compress",
            "--quiet",
            "--compression",
            "lzvn",
            "--minimum-compression-ratio",
            str(ratio),
            str(target),
        ],
        parent=target.parent,
        deadline=deadline,
        reserve_bytes=reserve_bytes,
    )
