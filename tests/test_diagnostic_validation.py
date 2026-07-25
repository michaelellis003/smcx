# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public argument contracts for particle-filter diagnostics."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx
from smcx.containers import LiuWestPosterior, ParticleFilterPosterior


def _posterior() -> ParticleFilterPosterior:
    """Return a small dense posterior with full genealogy history."""
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
    """Return a parameter posterior for the shared quantile contract."""
    posterior = _posterior()
    return LiuWestPosterior(
        *posterior,
        filtered_params=posterior.filtered_particles / 10.0,
    )


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
    """Quantile levels have one fixed public array contract."""
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
    """Eager quantile values must lie in the documented closed interval."""
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
    """Structural validation remains available while q is traced."""
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
    """CRPS requires a nonempty rank-one floating sample."""
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
    """The CRPS observation is one floating scalar."""
    with pytest.raises(
        ValueError,
        match="observation must be a floating scalar",
    ):
        smcx.crps(jnp.asarray([0.0, 1.0]), observation)


@pytest.mark.parametrize("num_replicates", [0, -1, 1.5, True])
def test_replicated_log_ml_requires_positive_integer_count(num_replicates):
    """Replicate counts fail before key splitting or callback execution."""
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
    """Every replicated filter returns one floating log-evidence value."""
    with pytest.raises(
        ValueError,
        match="filter_fn output must be a floating scalar",
    ):
        smcx.replicated_log_ml(jr.key(0), lambda _key: result, 2)


@pytest.mark.parametrize(
    ("log_ml_1", "log_ml_2"),
    [
        (jnp.asarray([1.0]), jnp.asarray(0.0)),
        (jnp.asarray(1.0), jnp.asarray([0.0])),
        (jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0.0)),
        (jnp.asarray(1.0), jnp.asarray(0, dtype=jnp.int32)),
    ],
)
def test_log_bayes_factor_requires_two_floating_scalars(log_ml_1, log_ml_2):
    """Bayes-factor broadcasting is outside the public contract."""
    with pytest.raises(ValueError, match="must be a floating scalar"):
        smcx.log_bayes_factor(log_ml_1, log_ml_2)


@pytest.mark.parametrize(
    "q",
    [0.0, -0.1, 0.5001, 1.0, float("nan"), float("inf")],
)
def test_tail_ess_rejects_invalid_tail_fraction(q):
    """Tail fractions are finite and cannot overlap."""
    with pytest.raises(
        ValueError,
        match=r"q must be finite and in \(0, 0.5\]",
    ):
        smcx.tail_ess(_posterior(), q=q)


@pytest.mark.parametrize("lag", [-1, 1.5, True])
def test_log_ml_variance_requires_nonnegative_integer_lag(lag):
    """The fixed genealogy lag is an optional nonnegative integer."""
    with pytest.raises(ValueError, match="lag must be nonnegative integer"):
        smcx.log_ml_variance(_posterior(), lag=lag)


@pytest.mark.parametrize("num_samples", [0, -1, 1.5, True])
def test_posterior_predictive_requires_positive_integer_count(num_samples):
    """Predictive sample counts fail before any callback executes."""
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        smcx.posterior_predictive_sample(
            jr.key(0),
            _posterior(),
            lambda _key, state: state,
            lambda _key, state: state,
            num_samples=num_samples,
        )
