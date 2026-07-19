# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host-wide mutual exclusion for profiling campaigns."""

import errno
import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import Final, Self, TextIO

DEFAULT_CAMPAIGN_LOCK_PATH: Final = Path("/tmp/smcx-profiling-campaign.lock")


class ConcurrentCampaignError(RuntimeError):
    """Raised when another profiling campaign holds the host lock."""


class HostCampaignLock:
    """Nonblocking advisory lock shared by profiling processes on one host.

    The lock file is intentionally never unlinked: deleting a live advisory
    lock file could let another process lock a new inode at the same path.
    A successful acquisition replaces any stale file content with its PID.

    Args:
        path: Stable lock-file path shared by every campaign on the host.
    """

    def __init__(
        self,
        path: Path = DEFAULT_CAMPAIGN_LOCK_PATH,
    ) -> None:
        """Initialize an unacquired lock for ``path``."""
        self.path = path
        self._file: TextIO | None = None
        self._holder_pid: int | None = None

    @property
    def holder_pid(self) -> int | None:
        """PID recorded by this instance while it owns the lock."""
        return self._holder_pid

    def acquire(self) -> Self:
        """Acquire the host lock without waiting.

        Returns:
            This acquired lock instance.

        Raises:
            ConcurrentCampaignError: Another process holds the lock.
            RuntimeError: This instance already owns the lock.
        """
        if self._file is not None:
            raise RuntimeError("profiling campaign lock is already acquired")

        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_RDWR,
            0o666,
        )
        try:
            lock_file = os.fdopen(
                descriptor,
                "r+",
                encoding="ascii",
                errors="replace",
            )
        except BaseException:
            os.close(descriptor)
            raise

        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                lock_file.close()
                raise
            holder_pid = _read_holder_pid(lock_file)
            lock_file.close()
            holder = (
                f"holder PID {holder_pid}"
                if holder_pid is not None
                else "holder PID unavailable"
            )
            raise ConcurrentCampaignError(
                "Cannot acquire host-wide profiling campaign lock at "
                f"{self.path}: another profiling campaign is already "
                f"running ({holder})."
            ) from None

        holder_pid = os.getpid()
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"{holder_pid}\n")
            lock_file.flush()
            os.fsync(lock_file.fileno())
        except BaseException:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()
            raise

        self._file = lock_file
        self._holder_pid = holder_pid
        return self

    def release(self) -> None:
        """Release the lock; repeated calls are harmless."""
        lock_file = self._file
        if lock_file is None:
            return

        self._file = None
        self._holder_pid = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def __enter__(self) -> Self:
        """Acquire the lock for a campaign context."""
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock even when the campaign raises."""
        del exception_type, exception, traceback
        self.release()


def _read_holder_pid(lock_file: TextIO) -> int | None:
    """Read a positive holder PID without trusting lock-file contents."""
    lock_file.seek(0)
    value = lock_file.read().strip()
    try:
        holder_pid = int(value)
    except ValueError:
        return None
    return holder_pid if holder_pid > 0 else None
