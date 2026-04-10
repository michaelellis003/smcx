# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Allow running the package with ``python -m smcjax``."""

from smcjax import __version__


def main() -> None:
    """Print package version."""
    print(f"smcjax {__version__}")


if __name__ == "__main__":
    main()
