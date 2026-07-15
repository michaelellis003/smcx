# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Shared test configuration.

Installs the jaxtyping runtime type-checking import hook (backed by
beartype) BEFORE smcx is imported anywhere in the test session, so
every annotated function in the library enforces its shape and dtype
annotations during tests at zero production cost (ADR-0007; mirrors
smcjax's conftest).
"""

from jaxtyping import install_import_hook

install_import_hook("smcx", "beartype.beartype")

import smcx  # noqa: F401  (forces hook-instrumented import first)
