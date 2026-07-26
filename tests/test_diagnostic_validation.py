# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public argument contracts for particle-filter diagnostics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterPosterior,
    SMC2Posterior,
)


def _posterior() -> ParticleFilterPosterior:
    num_particles = 4
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(-1.0),
        filtered_particles=jnp.arange(12.0).reshape(3, 4, 1),
        filtered_log_weights=jnp.full(
            (3, num_particles), -jnp.log(num_particles)
        ),
        ancestors=jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32),
            (3, num_particles),
        ),
        ess=jnp.full((3,), float(num_particles)),
        log_evidence_increments=jnp.array([-0.2, -0.3, -0.5]),
    )


def _parameter_posterior() -> LiuWestPosterior:
    posterior = _posterior()
    return LiuWestPosterior(
        *posterior,
        filtered_params=posterior.filtered_particles / 10.0,
    )


def _smc2_posterior() -> SMC2Posterior:
    """Return an SMC² result with the same parameter cloud."""
    posterior = _parameter_posterior()
    return SMC2Posterior(
        marginal_loglik=jnp.asarray(posterior.marginal_loglik),
        filtered_params=posterior.filtered_params,
        filtered_log_weights=posterior.filtered_log_weights,
        ess=posterior.ess,
        log_evidence_increments=posterior.log_evidence_increments,
        acceptance_rates=jnp.zeros_like(posterior.ess),
    )


class _ResearchPosterior(NamedTuple):
    """Caller-owned container satisfying the structural result protocol."""

    marginal_loglik: object
    filtered_particles: object
    filtered_log_weights: object
    ancestors: object
    ess: object
    log_evidence_increments: object


@pytest.mark.parametrize(
    ("summary", "posterior"),
    [
        (smcx.weighted_quantile, _posterior()),
        (smcx.param_weighted_quantile, _parameter_posterior()),
    ],
)
@pytest.mark.parametrize(
    ("q", "message"),
    [
        ([0.5], "must be a JAX array"),
        (jnp.asarray(0.5), "shape \\(num_quantiles,\\)"),
        (jnp.empty((0,)), "num_quantiles >= 1"),
        (jnp.asarray([[0.5]]), "shape \\(num_quantiles,\\)"),
        (jnp.asarray([0], dtype=jnp.int32), "floating dtype"),
    ],
)
def test_quantile_summaries_reject_malformed_q(summary, posterior, q, message):
    with pytest.raises(ValueError, match=message):
        summary(posterior, q)


@pytest.mark.parametrize(
    "q",
    [
        jnp.asarray([-0.1]),
        jnp.asarray([1.1]),
        jnp.asarray([jnp.nan]),
    ],
)
@pytest.mark.parametrize(
    ("summary", "posterior"),
    [
        (smcx.weighted_quantile, _posterior()),
        (smcx.param_weighted_quantile, _parameter_posterior()),
    ],
)
def test_quantile_summaries_reject_eager_values_outside_unit_interval(
    summary, posterior, q
):
    with pytest.raises(ValueError, match="q values must be in \\[0, 1\\]"):
        summary(posterior, q)


@pytest.mark.parametrize(
    ("summary", "posterior"),
    [
        (smcx.weighted_quantile, _posterior()),
        (smcx.param_weighted_quantile, _parameter_posterior()),
    ],
)
def test_quantile_validation_retains_valid_jit(summary, posterior):
    compiled = jax.jit(lambda q: summary(posterior, q))
    result = compiled(jnp.asarray([0.25, 0.75]))
    assert result.shape == (3, 2, 1)
    with pytest.raises(ValueError, match=r"shape \(num_quantiles,\)"):
        compiled(jnp.asarray(0.5))


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([1.0], "must be a JAX array"),
        (jnp.asarray(1.0), "shape \\(num_samples,\\)"),
        (jnp.empty((0,)), "num_samples >= 1"),
        (jnp.asarray([[1.0]]), "shape \\(num_samples,\\)"),
        (jnp.asarray([1], dtype=jnp.int32), "floating dtype"),
    ],
)
def test_crps_rejects_malformed_predictions(predictions, message):
    with pytest.raises(ValueError, match=message):
        smcx.crps(predictions, jnp.asarray(0.0))


@pytest.mark.parametrize(
    "observation",
    [
        jnp.asarray([0.0]),
        jnp.asarray(0, dtype=jnp.int32),
    ],
)
def test_crps_rejects_nonfloating_scalar_observation(observation):
    with pytest.raises(
        ValueError,
        match="observation must be a floating scalar",
    ):
        smcx.crps(jnp.asarray([0.0, 1.0]), observation)


@pytest.mark.parametrize("num_replicates", [0, -1, 1.5, True])
def test_replicated_log_ml_requires_positive_integer_count(num_replicates):
    with pytest.raises(
        ValueError, match="num_replicates must be a positive integer"
    ):
        smcx.replicated_log_ml(
            jr.key(0),
            lambda _key: jnp.asarray(0.0),
            num_replicates,
        )


@pytest.mark.parametrize(
    "result",
    [
        jnp.asarray([0.0]),
        jnp.asarray(0, dtype=jnp.int32),
    ],
)
def test_replicated_log_ml_requires_scalar_float_callback_output(result):
    with pytest.raises(
        ValueError,
        match="filter_fn output must be a floating scalar",
    ):
        smcx.replicated_log_ml(jr.key(0), lambda _key: result, 2)


@pytest.mark.parametrize(
    ("log_ml_1", "log_ml_2"),
    [
        (jnp.asarray([1.0]), jnp.asarray(0.0)),
        (1.0, jnp.asarray([0.0])),
        ("1.0", jnp.asarray(0.0)),
        (jnp.asarray(1.0), jnp.asarray(0, dtype=jnp.int32)),
    ],
)
def test_log_bayes_factor_requires_two_floating_scalars(log_ml_1, log_ml_2):
    with pytest.raises(ValueError, match="must be a floating scalar"):
        smcx.log_bayes_factor(log_ml_1, log_ml_2)


@pytest.mark.parametrize(
    "q",
    [
        0.0,
        -0.1,
        0.5001,
        1.0,
        float("nan"),
        float("inf"),
        None,
        "0.1",
        [0.1],
        jnp.asarray([0.1]),
    ],
)
def test_tail_ess_rejects_invalid_tail_fraction(q):
    with pytest.raises(
        ValueError,
        match=r"q must be finite and in \(0, 0.5\]",
    ):
        smcx.tail_ess(_posterior(), q=q)


@pytest.mark.parametrize("lag", [-1, 1.5, True])
def test_log_ml_variance_requires_nonnegative_integer_lag(lag):
    with pytest.raises(ValueError, match="lag must be nonnegative integer"):
        smcx.log_ml_variance(_posterior(), lag=lag)


@pytest.mark.parametrize("num_samples", [0, -1, 1.5, True])
def test_posterior_predictive_requires_positive_integer_count(num_samples):
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        smcx.posterior_predictive_sample(
            jr.key(0),
            _posterior(),
            lambda _key, state: state,
            lambda _key, state: state,
            num_samples=num_samples,
        )


def test_state_summary_rejects_misaligned_particle_axes():
    """State values and weights must describe the same particle cloud."""
    posterior = _posterior()._replace(
        filtered_particles=jnp.arange(6.0).reshape(1, 6, 1),
        filtered_log_weights=jnp.zeros((1, 1)),
    )
    with pytest.raises(
        ValueError,
        match=r"filtered_particles.*axes.*\(1, 1\).*\(1, 6\)",
    ):
        smcx.weighted_mean(posterior)


def test_state_summaries_require_a_nonempty_event_axis():
    """Euclidean summaries cannot reduce a zero-dimensional event."""
    posterior = _posterior()._replace(filtered_particles=jnp.empty((3, 4, 0)))
    with pytest.raises(ValueError, match="state_dim >= 1"):
        smcx.weighted_mean(posterior)


@pytest.mark.parametrize(
    "posterior",
    [_parameter_posterior(), _smc2_posterior()],
)
def test_parameter_summaries_reject_misaligned_clouds(posterior):
    """Liu-West and SMC² parameter clouds align with outer weights."""
    malformed = posterior._replace(filtered_params=jnp.zeros((3, 3, 1)))
    with pytest.raises(
        ValueError,
        match=r"filtered_params.*axes.*\(3, 4\).*\(3, 3\)",
    ):
        smcx.param_weighted_mean(malformed)


def test_parameter_summaries_require_a_nonempty_event_axis():
    posterior = _parameter_posterior()._replace(
        filtered_params=jnp.empty((3, 4, 0))
    )
    with pytest.raises(ValueError, match="param_dim >= 1"):
        smcx.param_weighted_mean(posterior)


def test_genealogy_rejects_misaligned_ancestor_axes():
    """Genealogy indexing requires one ancestor for every weighted particle."""
    posterior = _posterior()._replace(
        ancestors=jnp.zeros((3, 3), dtype=jnp.int32)
    )
    with pytest.raises(
        ValueError,
        match=r"ancestors.*axes.*\(3, 4\).*\(3, 3\)",
    ):
        smcx.reconstruct_trajectories(posterior)


def test_predictive_rejects_misaligned_structured_particle_history():
    """Every state leaf must share the weight history's leading axes."""
    posterior = _posterior()._replace(
        filtered_particles={
            "position": jnp.zeros((3, 4, 1)),
            "scale": jnp.zeros((2, 4)),
        }
    )
    with pytest.raises(
        ValueError,
        match=r"filtered_particles.*\['scale'\].*\(3, 4\).*\(2, 4\)",
    ):
        smcx.posterior_predictive_sample(
            jr.key(0),
            posterior,
            lambda _key, state: state,
            lambda _key, state: state["position"],
            num_samples=2,
        )


def test_diagnose_rejects_misaligned_ess_trace():
    posterior = _posterior()._replace(ess=jnp.ones(2))
    with pytest.raises(
        ValueError,
        match=r"ess.*time axis.*log_evidence_increments.*3.*2",
    ):
        smcx.diagnose(posterior)


def test_valid_final_only_histories_retain_supported_summaries():
    """Final-only state and predictive summaries intentionally remain valid."""
    posterior = _posterior()
    final = posterior._replace(
        filtered_particles=posterior.filtered_particles[-1:],
        filtered_log_weights=posterior.filtered_log_weights[-1:],
        ancestors=posterior.ancestors[-1:],
    )
    assert smcx.weighted_mean(final).shape == (1, 1)
    predictive = smcx.posterior_predictive_sample(
        jr.key(0),
        final,
        lambda _key, state: state,
        lambda _key, state: state,
        num_samples=2,
    )
    assert predictive.shape == (1, 2, 1)

    with pytest.raises(ValueError, match="store_history=True"):
        smcx.reconstruct_trajectories(final)


def test_valid_caller_owned_posterior_remains_jittable():
    """Diagnostics keep accepting the typed structural extension boundary."""
    posterior = _ResearchPosterior(*_posterior())
    expected = jnp.asarray([[1.5], [5.5], [9.5]])

    assert isinstance(posterior, smcx.ParticleFilterResult)
    assert jnp.array_equal(jax.jit(smcx.weighted_mean)(posterior), expected)

    malformed = posterior._replace(filtered_particles=jnp.zeros((3, 3, 1)))
    with pytest.raises(ValueError, match=r"filtered_particles.*axes"):
        jax.jit(smcx.weighted_mean)(malformed)
