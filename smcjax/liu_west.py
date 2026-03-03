# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
r"""Liu-West particle filter for joint state-parameter estimation.

The Liu-West filter (Liu & West, 2001) extends the auxiliary particle
filter to estimate static model parameters alongside latent states.
Parameters are propagated using kernel density smoothing:

.. math::

    \phi_t^i = a \phi_{t-1}^{a_i}
             + (1 - a) \bar{\phi}_{t-1}
             + h \, \varepsilon^i, \quad
    \varepsilon^i \sim \mathcal{N}(0, V_{t-1})

where :math:`a` is the shrinkage parameter, :math:`\bar{\phi}` is the
weighted parameter mean, :math:`V` is the weighted parameter covariance,
and :math:`h^2 = 1 - a^2`.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from blackjax.smc.ess import ess as compute_ess
from blackjax.smc.resampling import systematic
from jax import lax, vmap
from jaxtyping import Array, Float

from smcjax._utils import _conditional_resample, _prepend
from smcjax.containers import LiuWestPosterior
from smcjax.types import PRNGKeyT
from smcjax.weights import log_normalize, normalize

# Carry type: (particles, params, log_weights, log_marginal_likelihood)
_Carry = tuple[Array, Array, Array, Array]


def _init_liu_west(
    init_key: PRNGKeyT,
    initial_sampler: Callable,
    param_initial_sampler: Callable,
    log_observation_fn: Callable,
    first_emission: Array,
    num_particles: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Initialise Liu-West filter at t=0.

    Args:
        init_key: PRNG key for initialisation.
        initial_sampler: State prior sampler.
        param_initial_sampler: Parameter prior sampler.
        log_observation_fn: Observation log-density.
        first_emission: First observation y_0.
        num_particles: Number of particles N.

    Returns:
        Tuple of (particles_0, params_0, log_w_0, log_ev_0, ess_0,
        identity_ancestors).
    """
    log_n = jnp.log(jnp.asarray(num_particles, dtype=jnp.float64))
    k_z, k_p = jr.split(init_key)
    particles_0 = initial_sampler(k_z, num_particles)
    params_0 = param_initial_sampler(k_p, num_particles)

    log_obs_0 = vmap(lambda z, p: log_observation_fn(first_emission, z, p))(
        particles_0, params_0
    )

    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0 = jnp.asarray(compute_ess(log_w_0))
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    return particles_0, params_0, log_w_0, log_ev_0, ess_0, identity


def liu_west_filter(
    key: PRNGKeyT,
    initial_sampler: Callable,
    transition_sampler: Callable,
    log_observation_fn: Callable,
    log_auxiliary_fn: Callable,
    param_initial_sampler: Callable,
    emissions: Float[Array, 'ntime emission_dim'],
    num_particles: int,
    shrinkage: float = 0.95,
    resampling_fn: Callable = systematic,
    resampling_threshold: float = 0.5,
) -> LiuWestPosterior:
    r"""Run a Liu-West particle filter (Liu & West, 2001).

    Jointly estimates latent states and static parameters using
    auxiliary particle filtering with kernel density smoothing for
    parameter propagation.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles) -> particles``
            that draws from the initial state distribution.
        transition_sampler: Function ``(key, state, params) -> state``
            that draws from the transition distribution.  Unlike the
            bootstrap/auxiliary filters, this receives parameters.
        log_observation_fn: Function
            ``(emission, state, params) -> log_prob`` that evaluates
            the observation log-density.  Receives parameters.
        log_auxiliary_fn: Function
            ``(emission, state, params) -> log_prob`` that evaluates
            the look-ahead log-density.  Receives parameters.
        param_initial_sampler: Function
            ``(key, num_particles) -> params`` that draws from the
            prior parameter distribution.  Returns array of shape
            ``(num_particles, param_dim)``.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        shrinkage: Shrinkage parameter :math:`a \in (0, 1)`.
            Controls the balance between the kernel smoothing
            exploration and prior concentration.  Higher values
            give tighter parameter posteriors.

            .. warning::

                The shrinkage parameter has no generative
                interpretation: it introduces artificial dynamics
                into the parameter evolution that do not correspond
                to any probabilistic model.  Results can be
                sensitive to this choice.  We recommend running the
                filter under several values (e.g. 0.95, 0.975,
                0.99) and reporting the range of posterior and
                evidence estimates.
        resampling_fn: Resampling algorithm.  Defaults to systematic.
        resampling_threshold: ESS fraction triggering resampling.

    Returns:
        :class:`~smcjax.containers.LiuWestPosterior` containing
        filtered particles, parameters, log weights, ancestor indices,
        the marginal log-likelihood estimate, and ESS trace.
    """
    key, init_key = jr.split(key)
    log_n = jnp.log(jnp.asarray(num_particles, dtype=jnp.float64))
    a = jnp.asarray(shrinkage, dtype=jnp.float64)
    h_sq = 1.0 - a**2

    (
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity_ancestors,
    ) = _init_liu_west(
        init_key,
        initial_sampler,
        param_initial_sampler,
        log_observation_fn,
        emissions[0],
        num_particles,
    )

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(
        carry: _Carry,
        args: tuple[PRNGKeyT, Array],
    ) -> tuple[_Carry, tuple[Array, Array, Array, Array, Array, Array]]:
        particles, params, log_weights, log_ml = carry
        step_key, y_t = args
        k1, k2, k3 = jr.split(step_key, 3)

        # Weighted parameter moments for kernel smoothing
        w = normalize(log_weights)
        param_mean = jnp.sum(w[:, None] * params, axis=0)
        param_dev = params - param_mean[None, :]
        param_cov = jnp.einsum('n,nd,ne->de', w, param_dev, param_dev)

        # Shrunk means: m_i = a * phi_i + (1-a) * phi_bar
        shrunk = a * params + (1.0 - a) * param_mean[None, :]

        # 1. First-stage weights using shrunk params
        log_aux = vmap(lambda z, p: log_auxiliary_fn(y_t, z, p))(
            particles, shrunk
        )
        log_first_norm, log_first_sum = log_normalize(log_weights + log_aux)

        # 2. Conditionally resample
        threshold = resampling_threshold * num_particles
        do_resample, ancestors = _conditional_resample(
            k1,
            log_first_norm,
            resampling_fn,
            threshold,
            num_particles,
            identity_ancestors,
        )

        # 3. Propagate params via kernel smoothing + propagate states
        param_dim = params.shape[1]
        # Jitter prevents NaN from cholesky on singular covariance
        # (e.g. when all particles share the same parameter value).
        jitter = 1e-8 * jnp.eye(param_dim)
        chol = jnp.linalg.cholesky(h_sq * param_cov + jitter)
        eps = jr.normal(k2, (num_particles, param_dim))
        new_params = shrunk[ancestors] + eps @ chol.T

        keys = jr.split(k3, num_particles)
        propagated = vmap(transition_sampler)(
            keys,
            particles[ancestors],
            new_params,
        )

        # 4. Second-stage weights
        log_obs = vmap(lambda z, p: log_observation_fn(y_t, z, p))(
            propagated, new_params
        )
        log_w_unnorm = jnp.where(
            do_resample,
            log_obs - log_aux[ancestors],
            log_weights + log_obs,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)

        log_ev_inc = jnp.where(
            do_resample,
            log_first_sum + log_sum - log_n,
            log_sum,
        )

        new_carry = (propagated, new_params, log_w_norm, log_ml + log_ev_inc)
        ess_t = jnp.asarray(compute_ess(log_w_norm))
        return new_carry, (
            propagated,
            new_params,
            log_w_norm,
            ancestors,
            ess_t,
            log_ev_inc,
        )

    init_carry: _Carry = (particles_0, params_0, log_w_0, log_ev_0)
    step_keys = jr.split(key, emissions.shape[0] - 1)

    (
        final_carry,
        (
            particles_rest,
            params_rest,
            log_w_rest,
            ancestors_rest,
            ess_rest,
            log_ev_inc_rest,
        ),
    ) = lax.scan(_step, init_carry, (step_keys, emissions[1:]))

    # --- Combine t=0 with t=1..T-1 -----------------------------------------
    _, _, _, final_log_ml = final_carry

    return LiuWestPosterior(
        marginal_loglik=final_log_ml,
        filtered_particles=_prepend(particles_0, particles_rest),
        filtered_log_weights=_prepend(log_w_0, log_w_rest),
        ancestors=_prepend(identity_ancestors, ancestors_rest),
        ess=_prepend(jnp.asarray(ess_0), ess_rest),
        log_evidence_increments=_prepend(
            jnp.asarray(log_ev_0), log_ev_inc_rest
        ),
        filtered_params=_prepend(params_0, params_rest),
    )
