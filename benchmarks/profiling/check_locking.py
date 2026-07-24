# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Check the security and contention contract of the profiling lock."""

import os
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.profiling.locking import (
    DEFAULT_CAMPAIGN_LOCK_PATH,
    HostCampaignLock,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONTENDER = """
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


def _require(condition: bool, message: str) -> None:
    """Raise a readable self-check failure."""
    if not condition:
        raise AssertionError(message)


def _check_symlink_rejection(directory: Path) -> None:
    """Require rejection without mutating the symlink target."""
    victim = directory / "victim"
    lock_path = directory / "symlink.lock"
    payload = b"valuable-data"
    victim.write_bytes(payload)
    lock_path.symlink_to(victim)

    rejection: RuntimeError | None = None
    try:
        with HostCampaignLock(lock_path):
            pass
    except RuntimeError as error:
        rejection = error

    _require(
        victim.read_bytes() == payload,
        "a symlink lock path changed its target bytes",
    )
    _require(rejection is not None, "a symlink lock path was accepted")
    _require(
        "symbolic link" in str(rejection) and str(lock_path) in str(rejection),
        f"symlink rejection was not actionable: {rejection}",
    )


def _check_metadata_rejection(directory: Path) -> None:
    """Require restrictive mode and single-link validation."""
    permissive = directory / "permissive.lock"
    permissive.write_text("do-not-change", encoding="ascii")
    permissive.chmod(0o644)
    try:
        with HostCampaignLock(permissive):
            pass
    except RuntimeError as error:
        _require(
            "mode 0600" in str(error),
            f"mode rejection was not actionable: {error}",
        )
    else:
        raise AssertionError("a mode-0644 lock file was accepted")

    linked_source = directory / "linked-source"
    linked_lock = directory / "linked.lock"
    linked_source.write_text("do-not-change", encoding="ascii")
    linked_source.chmod(0o600)
    os.link(linked_source, linked_lock)
    try:
        with HostCampaignLock(linked_lock):
            pass
    except RuntimeError as error:
        _require(
            "exactly one hard link" in str(error),
            f"link-count rejection was not actionable: {error}",
        )
    else:
        raise AssertionError("a multiply linked lock file was accepted")


def _check_holder_behavior(directory: Path) -> None:
    """Preserve PID diagnostics and a stable, private lock inode."""
    lock_path = directory / "campaign.lock"
    lock = HostCampaignLock(lock_path)

    with lock:
        metadata = lock_path.stat()
        inode = (metadata.st_dev, metadata.st_ino)
        _require(lock.holder_pid == os.getpid(), "holder_pid was not published")
        _require(
            lock_path.read_text(encoding="ascii") == f"{os.getpid()}\n",
            "the lock file did not record its holder PID",
        )
        _require(
            stat.S_IMODE(metadata.st_mode) == 0o600,
            "a new lock file did not have mode 0600",
        )
        completed = subprocess.run(
            [sys.executable, "-c", _CONTENDER, str(lock_path)],
            capture_output=True,
            check=False,
            cwd=_ROOT,
            text=True,
            timeout=5.0,
        )

    _require(completed.returncode == 0, completed.stderr)
    _require(
        f"holder PID {os.getpid()}" in completed.stdout,
        f"contention did not report the holder PID: {completed.stdout}",
    )
    _require(lock.holder_pid is None, "release retained the holder PID")
    _require(lock_path.exists(), "release unlinked the stable lock file")
    with HostCampaignLock(lock_path):
        replacement = lock_path.stat()
        _require(
            (replacement.st_dev, replacement.st_ino) == inode,
            "reacquisition replaced the stable lock inode",
        )


def main() -> None:
    """Run the standalone profiling-lock contract check."""
    with TemporaryDirectory(prefix="smcx-lock-check-") as temporary:
        directory = Path(temporary)
        _check_symlink_rejection(directory)
        _check_metadata_rejection(directory)
        _check_holder_behavior(directory)
    expected = Path(f"/tmp/smcx-{os.getuid()}-profiling-campaign.lock")
    _require(
        expected == DEFAULT_CAMPAIGN_LOCK_PATH,
        "the default lock path is not stable and scoped to the current UID",
    )
    print("profiling lock contract: ok")


if __name__ == "__main__":
    main()
