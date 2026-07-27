# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Adaptive tempered SMC for static targets.

Anneals from the prior to the posterior
$\pi_\phi \propto p(x)\, L(x)^\phi$ along an adaptive schedule
[Del Moral, Doucet & Jasra, 2006]: the next temperature solves
``ESS(phi) / ESS(uniform weights) = target_ess`` by bisection on the
*resident* log-likelihood vector when that target lies before
``phi = 1``. The terminal stage may instead stop above the target ESS.
The solve is deterministic and uses no fresh sampling.
Each stage reweights by
$\ell \cdot \Delta\phi$ (evidence increment at the reweight,
pre-move — the Del Moral et al. collapse), resamples, and applies a
$\pi_{\phi'}$-invariant mutation. By default this is random-walk
Metropolis with proposal covariance
$2.38^2/d \cdot \hat\Sigma$ from the *weighted* pre-resample cloud
(Roberts & Rosenthal, 2001) — two-pass in float64 on the host
(single-pass cancels catastrophically at ordinary posterior offsets).

The target ESS ratio is capped one float32 machine epsilon below one.
At exactly one, no positive temperature increment can satisfy the target
for a heterogeneous likelihood; the finite gap also gives the ESS search
one relative unit of float32 resolution below its uniform-cloud maximum.
The search scales that ratio by the ESS of the represented uniform
log-weight vector (mathematically $N$), so reduction rounding cannot put
the accepted target above the backend's computed maximum.

The adaptive schedule is host-driven (bisection reads ESS values), so
``temper`` itself is not jittable; each per-stage mutation sweep is jitted.
"""

import math
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, lax, vmap
from jaxtyping import Array, Bool, Float

from smcx._covariance import _weighted_covariance_factor
from smcx._numerics import _neumaier_add
from smcx._utils import (
    _raise_if_degenerate,
    _raise_invalid_ancestors,
    _validate_ancestors,
    _validate_log_density_batch,
)
from smcx.containers import TemperedPosterior
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import systematic
from smcx.types import (
    DenseInitialSampler,
    PRNGKeyT,
    ResamplingFn,
    StaticLogDensity,
    TemperingMutationInfo,
    TemperingMutationInitFn,
    TemperingMutationState,
    TemperingMutationStepFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

_BISECT_ITERS = 60
_MAX_TARGET_ESS = 1.0 - float(np.finfo(np.float32).eps)
_RWM_SCALE = 2.38


def _mutation_position(
    state: object,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
    *,
    source: str,
) -> Array:
    """Validate and return a structural mutation state's position."""
    if not hasattr(state, "position"):
        raise ValueError(f"{source} must return state with a position field")
    position = cast(TemperingMutationState, state).position
    if not hasattr(position, "shape") or not hasattr(position, "dtype"):
        raise ValueError(f"{source} state.position must be a JAX array")
    if tuple(position.shape) != expected_shape:
        raise ValueError(
            f"{source} state.position must have shape {expected_shape}; "
            f"got {position.shape}"
        )
    if position.dtype != expected_dtype:
        raise ValueError(
            f"{source} state.position must have dtype {expected_dtype}; "
            f"got {position.dtype}"
        )
    return position


def _mutation_acceptance_rate(info: object) -> Array:
    """Validate and return one structural mutation diagnostic."""
    if not hasattr(info, "acceptance_rate"):
        raise ValueError(
            "mutation_step_fn info must have an acceptance_rate field"
        )
    value = cast(TemperingMutationInfo, info).acceptance_rate
    try:
        rate: Array = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "mutation_step_fn acceptance_rate must be a scalar float"
        ) from error
    if rate.ndim != 0 or not jnp.issubdtype(rate.dtype, jnp.floating):
        raise ValueError(
            "mutation_step_fn acceptance_rate must be a scalar float; "
            f"got shape {rate.shape} and dtype {rate.dtype}"
        )
    return rate


def _is_valid_acceptance_rate(
    rate: Float[Array, ""],
) -> Bool[Array, ""]:
    """Return whether a traced mutation acceptance rate is a probability."""
    bit_width = jnp.finfo(rate.dtype).bits
    if bit_width < 16:
        widened_rate = rate.astype(jnp.float32)
        return (
            jnp.isfinite(widened_rate)
            & (widened_rate >= 0.0)
            & (widened_rate <= 1.0)
        )
    unsigned_dtype = {
        16: jnp.uint16,
        32: jnp.uint32,
        64: jnp.uint64,
    }[bit_width]
    bits = lax.bitcast_convert_type(rate, unsigned_dtype)
    sign_mask = jnp.asarray(
        1 << (bit_width - 1),
        dtype=unsigned_dtype,
    )
    # Arithmetic comparisons can flush a negative subnormal to zero. Inspect
    # its sign bit while admitting the sign-only negative-zero encoding.
    nonnegative = ((bits & sign_mask) == 0) | (bits == sign_mask)
    return jnp.isfinite(rate) & nonnegative & (rate <= 1.0)


def temper(
    key: PRNGKeyT,
    initial_sampler: DenseInitialSampler,
    log_prior_fn: StaticLogDensity,
    log_likelihood_fn: StaticLogDensity,
    num_particles: int,
    num_mcmc_steps: int = 5,
    target_ess: float = 0.5,
    resampling_fn: ResamplingFn = systematic,
    *,
    mutation_init_fn: TemperingMutationInitFn | None = None,
    mutation_step_fn: TemperingMutationStepFn | None = None,
    max_stages: int = 1000,
) -> TemperedPosterior:
    r"""Sample a static target by adaptive tempered SMC.

    Args:
        key: JAX PRNG key.
        initial_sampler: ``(key, num_particles) -> (N, d)`` drawing
            from the prior.
        log_prior_fn: Per-particle ``(state) -> scalar`` log-prior;
            vmapped internally. Must return at least float32 precision.
        log_likelihood_fn: Per-particle ``(state) -> scalar``
            log-likelihood; vmapped internally. Must return at least
            float32 precision.
        num_particles: Number of particles N.
        num_mcmc_steps: RWM sweeps per temperature stage. Five may under-mix
            in moderate or high dimensions.
        target_ess: The bisection solves
            ``ESS / ESS(uniform weights) = target_ess`` for nonterminal
            stages, where the denominator is mathematically N. The terminal
            jump to ``phi = 1`` may finish above the target. Must lie in
            ``(0, 1 - numpy.finfo(numpy.float32).eps]``; the common float32
            bound keeps the accepted domain consistent across backends.
            Values near the upper bound can require raising ``max_stages``
            for heterogeneous likelihoods.
        resampling_fn: Resampler applied at every
            stage.
        mutation_init_fn: Optional
            ``(position, tempered_logdensity_fn) -> state`` callback.
            State must be a JAX PyTree with a dense ``position`` field.
        mutation_step_fn: Optional
            ``(key, state, tempered_logdensity_fn) -> (state, info)``
            callback. Info must expose a scalar floating
            ``acceptance_rate`` that is finite and in ``[0, 1]``. Stage
            means retain that scalar's dtype. Supply both mutation callbacks
            or neither.
        max_stages: Safety cap on the number of stages. Near-unit
            ``target_ess`` values may need a larger budget.

    Returns:
        `smcx.containers.TemperedPosterior` with an equal-weight particle
        approximation to the posterior, the log-evidence estimate, and
        per-stage temperature/ESS/acceptance traces.

    Raises:
        ValueError: Particle or mutation counts are invalid, ``target_ess``
            is outside its numerically viable interval, callback pairing is
            incomplete, or a mutation state or diagnostic is malformed or
            outside its documented domain.
        DegenerateWeightsError: A tempering reweight stage cannot be
            normalized.
        RuntimeError: ``max_stages`` exceeded before reaching
            ``phi = 1``.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    if num_mcmc_steps < 1:
        raise ValueError(f"num_mcmc_steps must be >= 1; got {num_mcmc_steps}")
    if not 0.0 < target_ess <= _MAX_TARGET_ESS:
        raise ValueError(
            "target_ess must be in the interval "
            f"(0, 1 - eps32] (upper bound {_MAX_TARGET_ESS}); "
            f"got {target_ess}"
        )
    if max_stages < 1:
        raise ValueError(f"max_stages must be >= 1; got {max_stages}")
    if (mutation_init_fn is None) != (mutation_step_fn is None):
        raise ValueError(
            "mutation_init_fn and mutation_step_fn must be supplied together"
        )
    n = num_particles
    log_n = math.log(n)
    key, k_init = jr.split(key)
    particles = initial_sampler(k_init, n)
    if not isinstance(particles, jax.Array):
        raise ValueError(
            "initial_sampler output must be a JAX array with shape (N, d)"
        )
    if (
        particles.ndim != 2
        or particles.shape[0] != n
        or particles.shape[1] == 0
    ):
        raise ValueError(
            "initial_sampler output must have shape (N, d) with "
            f"N={n} and d >= 1; got {particles.shape}"
        )
    if not jnp.issubdtype(particles.dtype, jnp.floating):
        raise ValueError(
            "initial_sampler output must have a floating dtype; "
            f"got {particles.dtype}"
        )
    dim = particles.shape[1]
    scale2 = _RWM_SCALE**2 / dim

    batch_lik = vmap(log_likelihood_fn)
    batch_prior = vmap(log_prior_fn)

    loglik: Float[Array, " num_particles"] = jnp.asarray(batch_lik(particles))
    logprior: Float[Array, " num_particles"] = jnp.asarray(
        batch_prior(particles)
    )
    _validate_log_density_batch(
        loglik,
        n,
        name="log_likelihood_fn",
    )
    _validate_log_density_batch(
        logprior,
        n,
        name="log_prior_fn",
    )
    log_w = jnp.full((n,), -log_n)  # normalized (LSE == 0)

    @jit
    def rwm_sweep(
        key: PRNGKeyT,
        particles: Float[Array, "num_particles state_dim"],
        loglik: Float[Array, " num_particles"],
        logprior: Float[Array, " num_particles"],
        phi_arr: Float[Array, ""],
        l_prop: Float[Array, "state_dim state_dim"],
    ) -> tuple[
        Float[Array, "num_particles state_dim"],
        Float[Array, " num_particles"],
        Float[Array, " num_particles"],
        Float[Array, ""],
    ]:
        """Run fixed-count RWM sweeps with branchless acceptance."""
        acc = jnp.zeros(())
        for _ in range(num_mcmc_steps):
            kz, ku, key = jr.split(key, 3)
            z = jr.normal(kz, (n, dim), dtype=particles.dtype)
            prop = particles + z @ l_prop.T
            lp = jnp.asarray(batch_prior(prop))
            ll = jnp.asarray(batch_lik(prop))
            _validate_log_density_batch(lp, n, name="log_prior_fn")
            _validate_log_density_batch(ll, n, name="log_likelihood_fn")
            log_alpha = (lp + phi_arr * ll) - (logprior + phi_arr * loglik)
            u = jr.uniform(ku, (n,))
            log_u = jnp.log(jnp.maximum(u, jnp.finfo(u.dtype).tiny))
            accept = log_u < log_alpha
            particles = jnp.where(accept[:, None], prop, particles)
            loglik = jnp.where(accept, ll, loglik)
            logprior = jnp.where(accept, lp, logprior)
            acc = acc + jnp.mean(accept)
        return particles, loglik, logprior, acc / num_mcmc_steps

    mutation_sweep = None
    if mutation_init_fn is not None:
        initialize = mutation_init_fn
        mutate = cast(TemperingMutationStepFn, mutation_step_fn)

        @jit
        def mutation_sweep(
            key: PRNGKeyT,
            particles: Float[Array, "num_particles state_dim"],
            phi_arr: Float[Array, ""],
        ) -> tuple[
            Float[Array, "num_particles state_dim"],
            Float[Array, ""],
            Bool[Array, ""],
        ]:
            """Run a caller-owned fixed-count mutation sweep."""

            def tempered_logdensity(position):
                return log_prior_fn(position) + phi_arr * log_likelihood_fn(
                    position
                )

            def initialize_one(position):
                state = initialize(position, tempered_logdensity)
                _mutation_position(
                    state,
                    (dim,),
                    particles.dtype,
                    source="mutation_init_fn",
                )
                return state

            states = vmap(initialize_one)(particles)
            sweep_keys = jr.split(key, num_mcmc_steps)

            def apply_sweep(states, sweep_key):
                particle_keys = jr.split(sweep_key, n)

                def apply_one(particle_key, state):
                    next_state, info = mutate(
                        particle_key, state, tempered_logdensity
                    )
                    _mutation_position(
                        next_state,
                        (dim,),
                        particles.dtype,
                        source="mutation_step_fn",
                    )
                    rate = _mutation_acceptance_rate(info)
                    return next_state, (
                        rate,
                        _is_valid_acceptance_rate(rate),
                    )

                return vmap(apply_one)(particle_keys, states)

            states, (acceptance_rates, valid_rates) = lax.scan(
                apply_sweep,
                states,
                sweep_keys,
            )
            positions = cast(TemperingMutationState, states).position
            mean_dtype = (
                jnp.float32
                if jnp.finfo(acceptance_rates.dtype).bits < 32
                else acceptance_rates.dtype
            )
            mean_acceptance_rate = jnp.mean(
                acceptance_rates.astype(mean_dtype)
            ).astype(acceptance_rates.dtype)
            return (
                positions,
                mean_acceptance_rate,
                jnp.all(valid_rates),
            )

    def ess_at(phi_new: float, phi: float) -> float:
        return float(compute_ess(log_w + (phi_new - phi) * loglik))

    phi = 0.0
    temps: list[float] = []
    ess_trace: list[float] = []
    acc_trace: list[Array] = []
    total = jnp.zeros(())
    comp = jnp.zeros(())

    for _ in range(max_stages):
        # --- adaptive schedule: bisect ESS(phi') = target ----------
        # Match ess_at's promotion by constructing the zero increment with
        # loglik's dtype before adding it to log_w.
        represented_uniform = log_w + jnp.zeros_like(loglik)
        target = target_ess * float(compute_ess(represented_uniform))
        probe_delta = min(1e-6, 1.0 - phi)
        _, probe_log_sum = log_normalize(log_w + probe_delta * loglik)
        _raise_if_degenerate(probe_log_sum)
        e_full = ess_at(1.0, phi)
        if math.isnan(e_full) and math.isnan(ess_at(phi + 1e-6, phi)):
            raise DegenerateWeightsError(
                "particle weights cannot be normalized at the next "
                "tempering stage"
            )
        if e_full >= target:
            phi_new = 1.0
        else:
            lo, hi = phi, 1.0
            for _ in range(_BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                e_mid = ess_at(mid, phi)
                if math.isnan(e_mid) or e_mid < target:
                    hi = mid
                else:
                    lo = mid
            phi_new = lo if lo > phi else 0.5 * (phi + hi)
        delta = phi_new - phi

        # --- reweight; increment at the reweight stage --------------
        lw_norm, log_sum = log_normalize(log_w + delta * loglik)
        _raise_if_degenerate(log_sum)
        stage_ess = float(compute_ess(lw_norm))
        total, comp = _neumaier_add(total, comp, log_sum)

        # --- adapt the default proposal from the weighted cloud ------
        if mutation_sweep is None:
            l_prop = _weighted_covariance_factor(
                particles,
                jnp.exp(lw_norm),
                scale=scale2,
            )

        # --- resample (always) + pi_{phi'}-invariant moves ----------
        key, kr, km = jr.split(key, 3)
        idx, invalid_resampling = _validate_ancestors(
            resampling_fn(kr, jnp.exp(lw_norm), n),
            n,
            n,
        )
        _raise_invalid_ancestors(invalid_resampling, n)
        particles = particles[idx]
        loglik = loglik[idx]
        logprior = logprior[idx]
        if mutation_sweep is None:
            particles, loglik, logprior, acc = rwm_sweep(
                km,
                particles,
                loglik,
                logprior,
                jnp.asarray(phi_new),
                l_prop,
            )
        else:
            particles, acc, valid_rates = mutation_sweep(
                km,
                particles,
                jnp.asarray(phi_new),
            )
            if not bool(valid_rates):
                raise ValueError(
                    "mutation_step_fn acceptance_rate must be finite "
                    "and in [0, 1]"
                )
            loglik = jnp.asarray(batch_lik(particles))
            logprior = jnp.asarray(batch_prior(particles))
            _validate_log_density_batch(
                loglik,
                n,
                name="log_likelihood_fn",
            )
            _validate_log_density_batch(
                logprior,
                n,
                name="log_prior_fn",
            )
        log_w = jnp.full((n,), -log_n)

        temps.append(phi_new)
        ess_trace.append(stage_ess)
        acc_trace.append(acc)
        phi = phi_new
        if phi >= 1.0:
            break
    else:
        raise RuntimeError(
            f"tempering did not reach phi=1 within {max_stages} stages"
        )

    marginal: Float[Array, ""] = total + comp
    return TemperedPosterior(
        particles=particles,
        log_weights=log_w,
        marginal_loglik=marginal,
        temperatures=jnp.asarray(temps),
        ess=jnp.asarray(ess_trace),
        acceptance_rates=jnp.stack(acc_trace),
    )
