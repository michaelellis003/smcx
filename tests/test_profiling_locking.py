# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-wide profiling campaign lock."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.profiling.locking import (
    DEFAULT_CAMPAIGN_LOCK_PATH,
    ConcurrentCampaignError,
    HostCampaignLock,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_default_campaign_lock_path_is_host_wide() -> None:
    """The default is stable across users and working directories."""
    assert (
        Path("/tmp/smcx-profiling-campaign.lock") == DEFAULT_CAMPAIGN_LOCK_PATH
    )
    assert HostCampaignLock().path == DEFAULT_CAMPAIGN_LOCK_PATH


def test_campaign_lock_records_holder_pid(tmp_path: Path) -> None:
    """An acquired lock publishes the current holder PID."""
    lock_path = tmp_path / "campaign.lock"

    with HostCampaignLock(lock_path) as lock:
        assert lock.holder_pid == os.getpid()
        assert lock_path.read_text(encoding="ascii") == f"{os.getpid()}\n"

    assert lock.holder_pid is None


def test_campaign_lock_reports_cross_process_contention(
    tmp_path: Path,
) -> None:
    """A second process fails immediately with holder diagnostics."""
    lock_path = tmp_path / "campaign.lock"
    script = """
import sys
from pathlib import Path

from benchmarks.profiling.locking import (
    ConcurrentCampaignError,
    HostCampaignLock,
)

try:
    with HostCampaignLock(Path(sys.argv[1])):
        raise AssertionError('contended lock was unexpectedly acquired')
except ConcurrentCampaignError as error:
    print(error)
"""

    with HostCampaignLock(lock_path):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(lock_path)],
            capture_output=True,
            check=False,
            cwd=_ROOT,
            text=True,
            timeout=5.0,
        )

    assert completed.returncode == 0, completed.stderr
    assert "another profiling campaign is already running" in completed.stdout
    assert f"holder PID {os.getpid()}" in completed.stdout
    assert str(lock_path) in completed.stdout


def test_campaign_lock_releases_when_body_raises(tmp_path: Path) -> None:
    """Context-manager cleanup never strands the advisory lock."""
    lock_path = tmp_path / "campaign.lock"

    with (
        pytest.raises(LookupError, match="synthetic campaign failure"),
        HostCampaignLock(lock_path),
    ):
        raise LookupError("synthetic campaign failure")

    with HostCampaignLock(lock_path) as replacement:
        assert replacement.holder_pid == os.getpid()


def test_campaign_lock_release_is_idempotent(tmp_path: Path) -> None:
    """Cleanup paths may safely call release more than once."""
    lock = HostCampaignLock(tmp_path / "campaign.lock")

    lock.acquire()
    lock.release()
    lock.release()

    assert lock.holder_pid is None


def test_campaign_lock_rejects_duplicate_acquire(tmp_path: Path) -> None:
    """One lock object cannot silently replace its live descriptor."""
    lock = HostCampaignLock(tmp_path / "campaign.lock")

    with lock, pytest.raises(RuntimeError, match="already acquired"):
        lock.acquire()


def test_concurrent_campaign_error_is_a_runtime_error() -> None:
    """Callers can handle contention as an operational runtime failure."""
    assert issubclass(ConcurrentCampaignError, RuntimeError)
