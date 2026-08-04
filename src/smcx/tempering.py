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
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, vmap
from jaxtyping import Array, Bool, Float

from smcx._covariance import _weighted_covariance_factor
from smcx._static_mutation import _run_custom_mutation_sweep
from smcx._static_smc import (
    _static_smc_stage,
    _StaticMoveResult,
    _StaticSMCState,
)
from smcx._utils import (
    _raise_if_degenerate,
    _validate_log_density_batch,
)
from smcx.containers import TemperedPosterior
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import systematic
from smcx.types import (
    DenseInitialSampler,
    ParticleCloud,
    PRNGKeyT,
    ResamplingFn,
    StaticLogDensity,
    TemperingMutationInitFn,
    TemperingMutationStepFn,
    TemperingScheduleFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

_BISECT_ITERS = 60
_MAX_TARGET_ESS = 1.0 - float(np.finfo(np.float32).eps)
_RWM_SCALE = 2.38


class _TemperingPopulation(NamedTuple):
    """Particles and aligned target caches owned by the tempering shell."""

    particles: Float[Array, "num_particles state_dim"]
    log_likelihoods: Float[Array, " num_particles"]
    log_priors: Float[Array, " num_particles"]


def temper(
    key: PRNGKeyT,
    initial_sampler: DenseInitialSampler,
    log_prior_fn: StaticLogDensity,
    log_likelihood_fn: StaticLogDensity,
    num_particles: int,
    *,
    num_mcmc_steps: int = 5,
    target_ess: float = 0.5,
    resampling_fn: ResamplingFn = systematic,
    mutation_init_fn: TemperingMutationInitFn | None = None,
    mutation_step_fn: TemperingMutationStepFn | None = None,
    schedule_fn: TemperingScheduleFn | None = None,
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
        schedule_fn: Optional host callback
            ``(phi, normalized_log_weights, log_likelihoods) -> float``
            choosing the next temperature; it must return a finite
            value in ``(phi, 1]``. When omitted, the adaptive ESS
            bisection above applies. The callback runs host-side once
            per stage; ``target_ess`` is ignored when it is supplied.
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

            return _run_custom_mutation_sweep(
                key,
                particles,
                tempered_logdensity,
                num_steps=num_mcmc_steps,
                initialize=initialize,
                mutate=mutate,
            )

    def ess_at(phi_new: float, phi: float) -> float:
        return float(compute_ess(log_w + (phi_new - phi) * loglik))

    phi = 0.0
    temps: list[float] = []
    ess_trace: list[float] = []
    acc_trace: list[Array] = []
    increment_trace: list[Array] = []
    total = jnp.zeros(())
    comp = jnp.zeros(())

    for _ in range(max_stages):
        # --- adaptive schedule: bisect ESS(phi') = target ----------
        # Match ess_at's promotion by constructing the zero increment with
        # loglik's dtype before adding it to log_w.
        probe_delta = min(1e-6, 1.0 - phi)
        _, probe_log_sum = log_normalize(log_w + probe_delta * loglik)
        _raise_if_degenerate(probe_log_sum)
        e_full = ess_at(1.0, phi)
        if math.isnan(e_full) and math.isnan(ess_at(phi + 1e-6, phi)):
            raise DegenerateWeightsError(
                "particle weights cannot be normalized at the next "
                "tempering stage"
            )
        if schedule_fn is not None:
            phi_new = float(schedule_fn(phi, log_w, loglik))
            if not math.isfinite(phi_new) or not phi < phi_new <= 1.0:
                raise ValueError(
                    "schedule_fn must return a finite temperature in "
                    f"(phi, 1]; got {phi_new} at phi={phi}"
                )
        else:
            represented_uniform = log_w + jnp.zeros_like(loglik)
            target = target_ess * float(compute_ess(represented_uniform))
            if e_full >= target:
                phi_new = 1.0
            else:
                lo, hi = phi, 1.0
                for _ in range(_BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    # Once the midpoint rounds onto an endpoint, every
                    # later iteration recomputes the same midpoint and,
                    # ess_at being deterministic, takes the same branch
                    # — so after applying this update the pair is at
                    # its fixed point and breaking is bitwise-identical
                    # to finishing the loop. Each avoided probe is a
                    # host sync.
                    converged = mid in (lo, hi)
                    e_mid = ess_at(mid, phi)
                    if math.isnan(e_mid) or e_mid < target:
                        hi = mid
                    else:
                        lo = mid
                    if converged:
                        break
                phi_new = lo if lo > phi else 0.5 * (phi + hi)
        delta = phi_new - phi

        # --- shared correction, selection, and invariant move -------
        key, kr, km = jr.split(key, 3)
        stage_phi = jnp.asarray(phi_new)

        def move_population(
            move_key: PRNGKeyT,
            corrected_population: ParticleCloud,
            corrected_log_weights: Float[Array, " num_particles"],
            selected_population: ParticleCloud,
            stage_target: Float[Array, ""] = stage_phi,
        ) -> _StaticMoveResult:
            source = cast(_TemperingPopulation, corrected_population)
            selected = cast(_TemperingPopulation, selected_population)
            if mutation_sweep is None:
                l_prop = _weighted_covariance_factor(
                    source.particles,
                    jnp.exp(corrected_log_weights),
                    scale=scale2,
                )
                moved_particles, moved_loglik, moved_logprior, acc = rwm_sweep(
                    move_key,
                    selected.particles,
                    selected.log_likelihoods,
                    selected.log_priors,
                    stage_target,
                    l_prop,
                )
                valid_rates = jnp.asarray(True)
            else:
                moved_particles, acc, valid_rates = mutation_sweep(
                    move_key,
                    selected.particles,
                    stage_target,
                )
                moved_loglik = jnp.asarray(batch_lik(moved_particles))
                moved_logprior = jnp.asarray(batch_prior(moved_particles))
                _validate_log_density_batch(
                    moved_loglik,
                    n,
                    name="log_likelihood_fn",
                )
                _validate_log_density_batch(
                    moved_logprior,
                    n,
                    name="log_prior_fn",
                )
            return _StaticMoveResult(
                population=_TemperingPopulation(
                    moved_particles,
                    moved_loglik,
                    moved_logprior,
                ),
                acceptance_rate=acc,
                acceptance_valid=valid_rates,
            )

        stage_state, stage_info = _static_smc_stage(
            _StaticSMCState(
                population=_TemperingPopulation(
                    particles,
                    loglik,
                    logprior,
                ),
                log_weights=log_w,
                log_evidence=total,
                log_evidence_correction=comp,
            ),
            delta * loglik,
            kr,
            km,
            resampling_fn=resampling_fn,
            resampling_threshold=None,
            time_index=None,
            move_fn=move_population,
            # temper resamples at every stage, so log_w entering a stage
            # is always the released uniform vector; reusing it as the
            # reset template preserves its exact dtype and weak-type
            # representation, which a full_like rebuild would not.
            uniform_log_weights=log_w,
        )
        population = cast(_TemperingPopulation, stage_state.population)
        particles = population.particles
        loglik = population.log_likelihoods
        logprior = population.log_priors
        log_w = stage_state.log_weights
        total = stage_state.log_evidence
        comp = stage_state.log_evidence_correction
        stage_ess = float(stage_info.selection_ess)

        temps.append(phi_new)
        ess_trace.append(stage_ess)
        acc_trace.append(stage_info.acceptance_rate)
        increment_trace.append(stage_info.log_evidence_increment)
        phi = phi_new
        if phi >= 1.0:
            break
    else:
        raise RuntimeError(
            f"tempering did not reach phi=1 within {max_stages} stages; "
            f"reached phi={phi:.9g} with last increment {delta:.3e} and "
            f"last stage ESS {stage_ess:.6g} against target ESS "
            f"{target_ess}. Raise max_stages or lower target_ess; "
            "near-unit targets can need arbitrarily many stages at "
            "some likelihood scales."
        )

    marginal: Float[Array, ""] = total + comp
    return TemperedPosterior(
        particles=particles,
        log_weights=log_w,
        marginal_loglik=marginal,
        temperatures=jnp.asarray(temps),
        ess=jnp.asarray(ess_trace),
        acceptance_rates=jnp.stack(acc_trace),
        log_evidence_increments=jnp.stack(increment_trace),
    )
