# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for smcx."""

import smcx


def test_version_is_set():
    """The package exposes a version string."""
    assert isinstance(smcx.__version__, str)
    assert len(smcx.__version__) > 0
