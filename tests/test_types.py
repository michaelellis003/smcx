# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the shared type aliases.

Consumed fully in later cycles; the ADR-0008 callback Protocols join
them with the bootstrap module.
"""

import mlx.core as mx
from beartype.door import is_bearable

from smcx.types import KeyT, Scalar


def test_key_alias_matches_mx_random_key():
    assert is_bearable(mx.random.key(0), KeyT)
    assert not is_bearable(mx.zeros((3,)), KeyT)


def test_scalar_alias_accepts_float_and_0d_array():
    assert is_bearable(1.5, Scalar)
    assert is_bearable(mx.array(1.5), Scalar)
    assert not is_bearable(mx.zeros((2,)), Scalar)
