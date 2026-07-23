# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""The library must run cleanly under JAX's default (float32) config.

The test suite enables x64 on CPU (conftest), so an explicit
``dtype=jnp.float64`` inside the library never warns there — but every
default-config user sees ``UserWarning: Explicitly requested dtype
float64 ... truncated`` on their first filter call. This test runs
the public entry points in a subprocess with the stock configuration
and ``-W error::UserWarning``, so any such warning fails the run.
A subprocess (rather than toggling ``jax_enable_x64`` in-process) is
deliberate: the warning registry deduplicates by call site, which
makes in-process checks order-dependent.
"""

import os
import subprocess
import sys

_SCRIPT = """
import jax.numpy as jnp
import jax.random as jr

import smcx

key = jr.key(0)
emissions = jr.normal(key, (5, 1))


def initial_sampler(key, n):
    return jr.normal(key, (n, 1))


def transition_sampler(key, state):
    return 0.9 * state + 0.5 * jr.normal(key, state.shape)


def log_observation_fn(y, state):
    return -0.5 * (y[0] - state[0]) ** 2


post = smcx.bootstrap_filter(
    key, initial_sampler, transition_sampler, log_observation_fn,
    emissions, 100,
)
assert jnp.isfinite(post.marginal_loglik)
report = smcx.diagnose(post)
assert 'warnings' in report


def trans_p(key, state, params):
    return transition_sampler(key, state)


def log_obs_p(y, state, params):
    return log_observation_fn(y, state)


def param_init(key, n):
    return jr.normal(key, (n, 1))


lw = smcx.liu_west_filter(
    key, initial_sampler, trans_p, log_obs_p, log_obs_p, param_init,
    emissions, 100,
)
assert jnp.isfinite(lw.marginal_loglik)

kalman = smcx.kalman_filter(
    jnp.zeros(1), jnp.eye(1), jnp.eye(1), 0.2 * jnp.eye(1),
    jnp.eye(1), 0.3 * jnp.eye(1), emissions,
)
smoothed = smcx.rts_smoother(kalman, jnp.eye(1))
assert jnp.isfinite(kalman.marginal_loglik)
assert jnp.all(jnp.isfinite(smoothed.smoothed_covariances))
print('OK')
"""


def test_filters_are_silent_under_default_config():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("JAX_ENABLE_X64", "SMCX_TEST_PLATFORM")
    }
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
