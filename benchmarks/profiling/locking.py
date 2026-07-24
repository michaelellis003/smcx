# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Same-user, host-local mutual exclusion for profiling campaigns."""

import errno
import fcntl
import os
import stat
from pathlib import Path
from types import TracebackType
from typing import Final, Self, TextIO

_LOCK_MODE: Final = 0o600
DEFAULT_CAMPAIGN_LOCK_PATH: Final = Path(
    f"/tmp/smcx-{os.getuid()}-profiling-campaign.lock"
)


class ConcurrentCampaignError(RuntimeError):
    """Raised when another same-user campaign holds the host-local lock."""


class UnsafeCampaignLockError(RuntimeError):
    """Raised when a profiling lock path fails its safety contract."""


class HostCampaignLock:
    """Nonblocking advisory lock shared by one user's processes on a host.

    The lock file is intentionally never unlinked: deleting a live advisory
    lock file could let another process lock a new inode at the same path.
    A safe lock is a regular file owned by the current UID, has mode ``0600``
    and exactly one hard link, and is opened without following a final
    symlink. A successful acquisition replaces any stale file content with
    its PID.

    Args:
        path: Stable lock-file path shared by this user's campaigns.
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
        """Acquire this user's host-local lock without waiting.

        Returns:
            This acquired lock instance.

        Raises:
            ConcurrentCampaignError: Another process holds the lock.
            RuntimeError: This instance already owns the lock.
            UnsafeCampaignLockError: The lock path cannot be used safely.
        """
        if self._file is not None:
            raise RuntimeError("profiling campaign lock is already acquired")

        descriptor = _open_lock_descriptor(self.path)
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
                "Cannot acquire same-user profiling campaign lock at "
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


def _open_lock_descriptor(path: Path) -> int:
    """Open and validate a lock descriptor without following a symlink."""
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            _LOCK_MODE,
        )
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK}:
            reason = "the path is a symbolic link"
        else:
            reason = error.strerror or f"operating-system error {error.errno}"
        raise UnsafeCampaignLockError(
            f"Cannot safely open profiling campaign lock at {path}: {reason}."
        ) from None

    try:
        _validate_lock_descriptor(descriptor, path)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_lock_descriptor(descriptor: int, path: Path) -> None:
    """Reject a descriptor that could alias or expose another file."""
    metadata = os.fstat(descriptor)
    owner_uid = os.getuid()
    mode = stat.S_IMODE(metadata.st_mode)
    problems: list[str] = []
    if not stat.S_ISREG(metadata.st_mode):
        problems.append(
            f"it is not a regular file ({stat.filemode(metadata.st_mode)})"
        )
    if metadata.st_uid != owner_uid:
        problems.append(
            f"owner UID is {metadata.st_uid}, expected current UID {owner_uid}"
        )
    if mode != _LOCK_MODE:
        problems.append(f"mode is {mode:04o}, expected mode 0600")
    if metadata.st_nlink != 1:
        problems.append(
            f"it has {metadata.st_nlink} hard links, expected exactly one "
            "hard link"
        )
    if problems:
        raise UnsafeCampaignLockError(
            f"Refusing unsafe profiling campaign lock at {path}: "
            f"{'; '.join(problems)}."
        )


def _read_holder_pid(lock_file: TextIO) -> int | None:
    """Read a positive holder PID without trusting lock-file contents."""
    lock_file.seek(0)
    value = lock_file.read().strip()
    try:
        holder_pid = int(value)
    except ValueError:
        return None
    return holder_pid if holder_pid > 0 else None
