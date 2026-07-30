# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling and validation, PSD covariance
# perturbations, shrinkage guidance, typed callbacks, exogenous inputs, and
# optional history storage.

r"""Liu-West particle filter for joint state-parameter estimation.

The Liu-West filter (Liu & West, 2001) extends the auxiliary particle
filter to estimate static model parameters alongside latent states.
The initial particle cloud represents the joint prior
$p(\phi)\,p(x_0 \mid \phi)$; the state prior may therefore be sampled
from either a parameter-independent callback or an explicitly
parameter-conditioned callback.
Parameters are propagated using kernel density smoothing:

$$
\phi_t^i = a \phi_{t-1}^{a_i}
         + (1 - a) \bar{\phi}_{t-1}
         + h \, \varepsilon^i, \quad
\varepsilon^i \sim \mathcal{N}(0, V_{t-1})
$$

where $a$ is the shrinkage parameter, $\bar{\phi}$ is the weighted
parameter mean, $V$ is the weighted parameter covariance, and
$h^2 = 1 - a^2$.

MPS uses a sequence of one-step scans; other platforms use one full scan.

References:
    Liu, J. and West, M. (2001). Combined Parameter and State Estimation
    in Simulation-Based Filtering. *Sequential Monte Carlo Methods in
    Practice*, 197--223.
    https://doi.org/10.1007/978-1-4757-3437-9_10
    Pébay, P., Terriberry, T. B., Kolla, H., and Bennett, J. (2016).
    Numerically stable, scalable formulas for parallel and online
    computation of higher-order multivariate central moments with
    arbitrary weights. https://doi.org/10.1007/s00180-015-0637-z
    Higham, N. J. (2008). Functions of Matrices: Theory and Computation,
    Chapter 6. https://doi.org/10.1137/1.9780898717778.ch6
"""

import math
import warnings
from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import core, lax, vmap
from jaxtyping import Array, Bool, Float, Int

from smcx._numerics import _neumaier_add
from smcx._utils import (
    _canonicalize_inputs,
    _conditional_resample,
    _filter_scan,
    _prepend,
    _raise_if_degenerate,
    _raise_invalid_ancestors,
    _TreeSignature,
    _validate_filter_inputs,
    _validate_log_density_batch,
    _validate_particle_cloud,
    _validate_resampling_threshold,
    _validate_state_tree,
)
from smcx.containers import LiuWestPosterior
from smcx.resampling import systematic
from smcx.types import (
    DenseInitialSampler,
    DenseInitialSamplerWithInput,
    Emission,
    EmissionSequence,
    InputSequence,
    ModelInput,
    ParamCloudInitialStateSampler,
    ParamCloudInitialStateSamplerWithInput,
    ParamInitialSampler,
    ParamLogObservationFn,
    ParamLogObservationFnWithInput,
    ParamTransitionSampler,
    ParamTransitionSamplerWithInput,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize


class _LiuWestStepCarry(NamedTuple):
    particles: Float[Array, "num_particles state_dim"]
    params: Float[Array, "num_particles param_dim"]
    log_weights: Float[Array, " num_particles"]
    log_marginal_likelihood: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]
    ancestors: Int[Array, " num_particles"]
    invalid_resampling: Bool[Array, ""]


class _LiuWestStepInput(NamedTuple):
    emission: Emission
    input_t: ModelInput | None
    time_index: Int[Array, ""]


class _LiuWestStepOutput(NamedTuple):
    particles: Float[Array, "num_particles state_dim"]
    params: Float[Array, "num_particles param_dim"]
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]
    normalizers_finite: Bool[Array, ""]


def _centered_parameter_moments(
    params: Float[Array, "num_particles param_dim"],
    weights: Float[Array, " num_particles"],
) -> tuple[Array, Array, Array, Array, Array]:
    """Anchor-relative weighted moments shared by shrink and factor.

    Moments are evaluated relative to a maximum-weight parameter,
    avoiding amplification of weight-sum error by a large common
    offset. Represented-zero parameters are masked before centered
    arithmetic.
    """
    working_dtype = jnp.promote_types(params.dtype, jnp.float32)
    working_params = params.astype(working_dtype)
    weights = weights.astype(working_dtype)
    weights = weights / jnp.sum(weights)
    anchor = working_params[jnp.argmax(weights)]
    material = weights[:, None] > 0.0
    offsets = jnp.where(
        material,
        working_params - anchor,
        jnp.zeros_like(working_params),
    )
    offset_mean = jnp.sum(weights[:, None] * offsets, axis=0)
    return anchor, offsets, offset_mean, weights, material


def _shrunk_parameters(
    params: Float[Array, "num_particles param_dim"],
    weights: Float[Array, " num_particles"],
    shrinkage: Float[Array, ""],
) -> Float[Array, "num_particles param_dim"]:
    """Shrink the cloud toward its weighted mean; no factorization.

    This feeds the auxiliary weights and the selection decision, so it
    must not touch the eigendecomposition — its derivative is
    undefined at repeated eigenvalues, and the decision path is
    differentiated on every step (follow-up response review R7).
    """
    working_dtype = jnp.promote_types(params.dtype, jnp.float32)
    anchor, offsets, offset_mean, _, _ = _centered_parameter_moments(
        params, weights
    )
    shrinkage = jnp.asarray(shrinkage, dtype=working_dtype)
    one = jnp.asarray(1.0, dtype=working_dtype)
    shrunk = (
        anchor[None, :]
        + shrinkage * offsets
        + (one - shrinkage) * offset_mean[None, :]
    )
    return shrunk.astype(params.dtype)


def _kernel_covariance_factor(
    params: Float[Array, "num_particles param_dim"],
    weights: Float[Array, " num_particles"],
    kernel_variance: Float[Array, ""],
) -> Float[Array, "param_dim param_dim"]:
    """PSD spectral root of the weighted kernel covariance.

    Spectral modes no larger than ``D * eps * ||covariance||_2`` are
    indistinguishable from eigensolver roundoff and are left
    unperturbed.
    """
    working_dtype = jnp.promote_types(params.dtype, jnp.float32)
    _, _, offset_mean, weights, material = _centered_parameter_moments(
        params, weights
    )
    working_params = params.astype(working_dtype)
    anchor = working_params[jnp.argmax(weights)]
    offsets = jnp.where(
        material,
        working_params - anchor,
        jnp.zeros_like(working_params),
    )
    deviations = jnp.where(
        material,
        offsets - offset_mean[None, :],
        jnp.zeros_like(offsets),
    )
    kernel_variance = jnp.asarray(kernel_variance, dtype=working_dtype)
    covariance = jnp.einsum(
        "n,nd,ne->de",
        weights,
        deviations,
        deviations,
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(kernel_variance * covariance)
    spectral_scale = jnp.max(jnp.abs(eigenvalues))
    tolerance = (
        jnp.asarray(params.shape[1], dtype=working_dtype)
        * jnp.asarray(jnp.finfo(working_dtype).eps, dtype=working_dtype)
        * spectral_scale
    )
    eigenvalues = jnp.where(eigenvalues > tolerance, eigenvalues, 0.0)
    return (eigenvectors * jnp.sqrt(eigenvalues)[None, :]) @ eigenvectors.T


def _parameter_kernel(
    params: Float[Array, "num_particles param_dim"],
    weights: Float[Array, " num_particles"],
    shrinkage: Float[Array, ""],
    kernel_variance: Float[Array, ""],
) -> tuple[
    Float[Array, "num_particles param_dim"],
    Float[Array, "param_dim param_dim"],
]:
    """Return shrunk parameters and a PSD kernel covariance factor."""
    return (
        _shrunk_parameters(params, weights, shrinkage),
        _kernel_covariance_factor(params, weights, kernel_variance),
    )


def _validate_dense_initial_cloud(
    values: object,
    num_particles: int,
    *,
    name: str,
) -> _TreeSignature:
    """Require one nonempty floating matrix with a particle axis."""
    signature = _validate_particle_cloud(
        values,
        num_particles,
        name=name,
    )
    if signature.paths != ("<root>",):
        raise ValueError(
            f"{name} must be a JAX array with shape (num_particles, dimension)"
        )
    array = cast(Array, values)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (num_particles, dimension) with "
            f"dimension >= 1; got {array.shape}"
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(
            f"{name} must have a floating dtype; got {array.dtype}"
        )
    return signature


def _init_liu_west(
    init_key: PRNGKeyT,
    initial_sampler: DenseInitialSampler | DenseInitialSamplerWithInput | None,
    param_initial_sampler: ParamInitialSampler,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    first_emission: Emission,
    num_particles: int,
    input_t: ModelInput | None = None,
    param_initial_state_sampler: ParamCloudInitialStateSampler
    | ParamCloudInitialStateSamplerWithInput
    | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, _TreeSignature]:
    """Initialise Liu-West filter at t=0.

    Args:
        init_key: PRNG key for initialisation.
        initial_sampler: Parameter-independent state prior sampler.
        param_initial_sampler: Parameter prior sampler.
        log_observation_fn: Observation log-density.
        first_emission: First observation y_0.
        num_particles: Number of particles N.
        input_t: Optional t=0 input passed to the state initializer
            and after parameters to the observation callback.
        param_initial_state_sampler: Optional state prior sampler that
            receives the aligned parameter cloud before the optional input.

    Returns:
        Tuple of (particles_0, params_0, log_w_0, log_ev_0, ess_0,
        identity_ancestors, state_signature).
    """
    log_n = jnp.asarray(math.log(num_particles))
    k_z, k_p = jr.split(init_key)
    if param_initial_state_sampler is None:
        if input_t is None:
            state_init = cast(DenseInitialSampler, initial_sampler)
            particles_0 = state_init(k_z, num_particles)
        else:
            state_init_u = cast(DenseInitialSamplerWithInput, initial_sampler)
            particles_0 = state_init_u(k_z, num_particles, input_t)
        params_0 = param_initial_sampler(k_p, num_particles)
        state_name = "initial_sampler output"
    else:
        params_0 = param_initial_sampler(k_p, num_particles)
        _validate_dense_initial_cloud(
            params_0,
            num_particles,
            name="param_initial_sampler output",
        )
        if input_t is None:
            param_state_init = cast(
                ParamCloudInitialStateSampler,
                param_initial_state_sampler,
            )
            particles_0 = param_state_init(k_z, num_particles, params_0)
        else:
            param_state_init_u = cast(
                ParamCloudInitialStateSamplerWithInput,
                param_initial_state_sampler,
            )
            particles_0 = param_state_init_u(
                k_z,
                num_particles,
                params_0,
                input_t,
            )
        state_name = "param_initial_state_sampler output"
    state_signature = _validate_dense_initial_cloud(
        particles_0,
        num_particles,
        name=state_name,
    )
    if param_initial_state_sampler is None:
        _validate_dense_initial_cloud(
            params_0,
            num_particles,
            name="param_initial_sampler output",
        )

    if input_t is None:
        observation_fn = cast(ParamLogObservationFn, log_observation_fn)
        log_obs_0 = cast(
            Array,
            vmap(lambda z, p: observation_fn(first_emission, z, p))(
                particles_0, params_0
            ),
        )
    else:
        observation_fn_u = cast(
            ParamLogObservationFnWithInput, log_observation_fn
        )
        log_obs_0 = cast(
            Array,
            vmap(lambda z, p: observation_fn_u(first_emission, z, p, input_t))(
                particles_0, params_0
            ),
        )

    _validate_log_density_batch(
        log_obs_0,
        num_particles,
        name="log_observation_fn",
    )
    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0 = jnp.asarray(compute_ess(log_w_0))
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    return (
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity,
        state_signature,
    )


def _liu_west_step(
    carry: _LiuWestStepCarry,
    inputs_t: _LiuWestStepInput,
    key_t: PRNGKeyT,
    *,
    transition_sampler: ParamTransitionSampler
    | ParamTransitionSamplerWithInput,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    log_auxiliary_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    log_num_particles: Float[Array, ""],
    shrinkage: Float[Array, ""],
    kernel_variance: Float[Array, ""],
    state_signature: _TreeSignature,
    move_only_on_selection: bool,
    static_never_select: bool,
) -> tuple[_LiuWestStepCarry, _LiuWestStepOutput]:
    particles, params, log_weights, log_ml, correction, _, invalid_seen = carry
    emission_t, input_t, time_index = inputs_t
    num_particles = log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, parameter_key, transition_key = jr.split(key_t, 3)

    def _evaluate(
        callback: ParamLogObservationFn | ParamLogObservationFnWithInput,
        values: Float[Array, "num_particles state_dim"],
        parameter_values: Float[Array, "num_particles param_dim"],
    ) -> Array:
        if input_t is None:
            callback_fn = cast(ParamLogObservationFn, callback)
            result = vmap(lambda z, p: callback_fn(emission_t, z, p))(
                values, parameter_values
            )
        else:
            callback_fn_u = cast(ParamLogObservationFnWithInput, callback)
            result = vmap(
                lambda z, p: callback_fn_u(emission_t, z, p, input_t)
            )(values, parameter_values)
        return cast(Array, result)

    if static_never_select:
        # A concrete zero threshold with moves confined to selection is
        # a pure sequential-importance-sampling pass: the shrink-and-
        # jitter kernel is inert, so its eigendecomposition must not
        # enter the computation graph — its reverse-mode derivative is
        # undefined for repeated eigenvalues and would poison gradients
        # of a run in which the move never happens (follow-up review
        # R7). Zeros stand in for the unused auxiliary quantities.
        do_resample = jnp.asarray(False)
        ancestors = identity
        invalid_resampling = jnp.asarray(False)
        new_params = params
        log_aux = jnp.zeros_like(log_weights)
        log_first_sum = jnp.zeros(())
    else:
        weights = normalize(log_weights)
        # Only the mean shrink feeds the auxiliary weights and the
        # selection decision; the covariance factorization exists for
        # the stochastic move alone (follow-up response review R7).
        shrunk = _shrunk_parameters(params, weights, shrinkage)
        log_aux = _evaluate(log_auxiliary_fn, particles, shrunk)
        _validate_log_density_batch(
            log_aux, num_particles, name="log_auxiliary_fn"
        )
        log_first_norm, log_first_sum = log_normalize(log_weights + log_aux)
        first_ess = jnp.asarray(compute_ess(log_first_norm))
        do_resample, ancestors, invalid_resampling = _conditional_resample(
            resample_key,
            log_first_norm,
            first_ess,
            resampling_fn,
            resampling_threshold,
            num_particles,
            identity,
            time_index,
        )
        param_dim = params.shape[1]
        factor_dtype = jnp.promote_types(params.dtype, jnp.float32)

        def _moved(operand):
            params_op, weights_op, ancestors_op, key_op = operand
            factor = _kernel_covariance_factor(
                params_op, weights_op, kernel_variance
            )
            eps = jr.normal(
                key_op, (num_particles, param_dim), dtype=factor_dtype
            )
            return (
                shrunk[ancestors_op].astype(factor_dtype) + eps @ factor.T
            ).astype(params.dtype)

        # Under the 'on_selection' policy the move — including its
        # eigendecomposition, whose derivative is undefined at
        # repeated eigenvalues — lives inside the selected branch, so
        # an ordinary data-dependent false decision keeps it out of
        # the differentiated graph entirely. The key is split outside
        # the branch, preserving the key schedule; selected-step
        # draws are bitwise-identical to the unconditional form.
        if move_only_on_selection:

            def _kept(operand):
                params_op, _, ancestors_op, _ = operand
                return params_op[ancestors_op]

            new_params = lax.cond(
                do_resample,
                _moved,
                _kept,
                (params, weights, ancestors, parameter_key),
            )
        else:
            new_params = _moved((params, weights, ancestors, parameter_key))
    particle_keys = jr.split(transition_key, num_particles)
    if input_t is None:
        transition_fn = cast(ParamTransitionSampler, transition_sampler)
        propagated = vmap(transition_fn)(
            particle_keys, particles[ancestors], new_params
        )
    else:
        transition_fn_u = cast(
            ParamTransitionSamplerWithInput, transition_sampler
        )
        propagated = vmap(transition_fn_u, in_axes=(0, 0, 0, None))(
            particle_keys, particles[ancestors], new_params, input_t
        )
    batched_shapes = ((num_particles, *state_signature.shapes[0]),)
    batched_signature = state_signature._replace(shapes=batched_shapes)
    _validate_state_tree(
        propagated, batched_signature, name="transition_sampler output"
    )
    log_obs = _evaluate(log_observation_fn, propagated, new_params)
    _validate_log_density_batch(
        log_obs, num_particles, name="log_observation_fn"
    )
    log_w_unnorm = jnp.where(
        do_resample,
        log_obs - log_aux[ancestors],
        log_weights + log_obs,
    )
    log_w_norm, log_sum = log_normalize(log_w_unnorm)
    log_ev_inc = jnp.where(
        do_resample,
        log_first_sum + log_sum - log_num_particles,
        log_sum,
    )
    log_ml, correction = _neumaier_add(log_ml, correction, log_ev_inc)
    new_carry = _LiuWestStepCarry(
        propagated,
        new_params,
        log_w_norm,
        log_ml,
        correction,
        ancestors,
        invalid_seen | invalid_resampling,
    )
    ess = jnp.asarray(compute_ess(log_w_norm))
    output = _LiuWestStepOutput(
        propagated,
        new_params,
        log_w_norm,
        ancestors,
        ess,
        log_ev_inc,
        jnp.isfinite(log_first_sum) & jnp.isfinite(log_sum),
    )
    return new_carry, output


def liu_west_filter(
    key: PRNGKeyT,
    initial_sampler: DenseInitialSampler | DenseInitialSamplerWithInput | None,
    transition_sampler: ParamTransitionSampler
    | ParamTransitionSamplerWithInput,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    log_auxiliary_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    param_initial_sampler: ParamInitialSampler,
    emissions: EmissionSequence,
    num_particles: int,
    *,
    shrinkage: float = 0.95,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion | None = None,
    parameter_moves: str | None = None,
    inputs: InputSequence | None = None,
    param_initial_state_sampler: ParamCloudInitialStateSampler
    | ParamCloudInitialStateSamplerWithInput
    | None = None,
    store_history: bool = True,
) -> LiuWestPosterior:
    r"""Run a Liu-West particle filter (Liu & West, 2001).

    Jointly estimates latent states and static parameters using
    auxiliary particle filtering with kernel density smoothing for
    parameter propagation.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws a parameter-independent state prior as a
            nonempty floating array of shape ``(num_particles, state_dim)``.
            Set to ``None`` when using ``param_initial_state_sampler``.
        transition_sampler: Function
            ``(key, state, params[, input_t]) -> state`` that draws from
            the transition distribution while preserving the initial state
            shape and dtype.
        log_observation_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the observation log-density and returns a scalar
            with at least float32 precision.
        log_auxiliary_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the look-ahead log-density and returns a scalar
            with at least float32 precision.
        param_initial_sampler: Function
            ``(key, num_particles) -> params`` that draws from the
            prior parameter distribution. Returns a nonempty floating array
            of shape ``(num_particles, param_dim)``. Float16 and bfloat16
            kernel arithmetic is promoted to float32, then cast back.
        emissions: Scalar ``(T,)`` or vector ``(T, D)`` observations.
            Rank-one data become ``(T, 1)``; dtype is preserved.
        num_particles: Number of particles $N$.
        shrinkage: Shrinkage parameter $a \in (0, 1)$. Larger values apply
            less shrinkage toward the ensemble mean and inject less kernel
            noise; smaller values apply more of both.

            !!! warning

                The shrinkage parameter has no generative interpretation:
                it introduces artificial dynamics into the parameter
                evolution that do not correspond to any probabilistic
                model. Each move preserves the weighted cloud's mean and
                covariance (the kernel is variance-matched). Results can
                be sensitive to this choice. Run the filter under several
                values (e.g. 0.95, 0.975, 0.99) and report the range of
                posterior and evidence estimates.
        resampling_fn: Resampling algorithm.  Defaults to systematic.
        resampling_threshold: ESS fraction, or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            Numeric values must be finite and nonnegative; zero disables
            resampling and values above one force it at every update.
            Omitting the argument keeps the 2.x default (0.5) and warns:
            in smcx 3.0 the default becomes 2.0 — selection at every
            update, the algorithm of Liu and West (2001). See
            ``parameter_moves`` for what happens to the parameter cloud
            on unselected steps. The callback receives the first-stage
            weights and ESS at the zero-based emission indices 1
            through T - 1.
        parameter_moves: When ``"on_selection"``, the shrink-and-jitter
            parameter move applies only on steps where selection occurs,
            so every move is accounted for by the auxiliary weights and
            the run stays inside the derivation of Liu and West (2001)
            at any threshold; with a concrete zero threshold the inert
            kernel also stays out of the computation graph, keeping
            gradients defined for degenerate parameter clouds.
            ``"always"`` moves the parameters at every observation; on
            unselected steps the importance weights carry no correction
            for that move, so the run is an explicitly approximate
            extension. Omitting the argument keeps the 2.x default
            (``"always"``) and warns; smcx 3.0 defaults to
            ``"on_selection"``. When the move is active, differentiating
            through the kernel's eigendecomposition is undefined at
            repeated eigenvalues of the weighted parameter covariance
            (a JAX ``eigh`` limitation) and can return NaN gradients.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)`` and a nonempty event. Inputs follow ``params`` in
            every callback;
            the parameter initializer remains input-independent.
        param_initial_state_sampler: Optional
            ``(key, num_particles, params[, input_0]) -> particles`` callback.
            The filter samples and validates the parameter cloud first, then
            passes the whole aligned cloud to this callback. Supply exactly
            one of this callback and ``initial_sampler``.
        store_history: When False, only the final step's
            particle/param/weight/ancestor arrays are returned (time
            axis length 1); ``ess``/``log_evidence_increments`` stay
            full.

    Returns:
        `smcx.containers.LiuWestPosterior` containing
        filtered particles, parameters, log weights, ancestor indices,
        the marginal log-likelihood estimate, and ESS trace.

    Raises:
        DegenerateWeightsError: A particle-weight stage cannot be normalized
            (eager execution only; under ``jax.jit`` its nonfinite signal
            propagates).
        ValueError: Initializer selection, observations, inputs, particle
            count, shrinkage, threshold, callback output, or criterion result
            is structurally invalid.
    """
    if resampling_threshold is None or parameter_moves is None:
        warnings.warn(
            "liu_west_filter's defaults change in smcx 3.0 to the "
            "Liu-West (2001) algorithm: selection at every update "
            "(resampling_threshold=2.0) with "
            "parameter_moves='on_selection'. The current defaults "
            "(resampling_threshold=0.5, parameter_moves='always') "
            "apply an uncorrected shrink-and-jitter move on "
            "unselected steps. Pass both arguments explicitly to "
            "keep either behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        if resampling_threshold is None:
            resampling_threshold = 0.5
        if parameter_moves is None:
            parameter_moves = "always"
    _validate_resampling_threshold(resampling_threshold)
    if parameter_moves not in ("on_selection", "always"):
        raise ValueError(
            "parameter_moves must be 'on_selection' or 'always'; "
            f"got {parameter_moves!r}"
        )
    move_only_on_selection = parameter_moves == "on_selection"
    static_never_select = (
        move_only_on_selection
        and not callable(resampling_threshold)
        and not isinstance(resampling_threshold, core.Tracer)
        # Exact zero is the documented disables-resampling sentinel,
        # so exact comparison is intended here.
        and float(resampling_threshold) == 0.0  # noqa: RUF069
    )
    emissions, num_timesteps = _validate_filter_inputs(
        emissions,
        num_particles,
    )
    if not 0.0 < shrinkage < 1.0:
        raise ValueError(
            f"shrinkage must be in the open interval (0, 1); got {shrinkage}"
        )
    if (initial_sampler is None) == (param_initial_state_sampler is None):
        raise ValueError(
            "exactly one of initial_sampler and "
            "param_initial_state_sampler must be supplied"
        )
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    key, init_key = jr.split(key)
    log_n = jnp.asarray(math.log(num_particles))
    (
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity_ancestors,
        state_signature,
    ) = (
        _init_liu_west(
            init_key,
            initial_sampler,
            param_initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
            param_initial_state_sampler=param_initial_state_sampler,
        )
        if inputs_arr is None
        else _init_liu_west(
            init_key,
            initial_sampler,
            param_initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
            inputs_arr[0],
            param_initial_state_sampler,
        )
    )
    kernel_dtype = jnp.promote_types(params_0.dtype, jnp.float32)
    a = jnp.asarray(shrinkage, dtype=kernel_dtype)
    h_sq = jnp.asarray(1.0, dtype=kernel_dtype) - a**2

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(carry: _LiuWestStepCarry, args: tuple[Array, ...]):
        if inputs_arr is None:
            step_key, y_t, time_index = args
            input_t = None
        else:
            step_key, y_t, input_t, time_index = args
        next_carry, output = _liu_west_step(
            carry,
            _LiuWestStepInput(y_t, input_t, time_index),
            step_key,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation_fn,
            log_auxiliary_fn=log_auxiliary_fn,
            resampling_fn=resampling_fn,
            resampling_threshold=resampling_threshold,
            log_num_particles=log_n,
            shrinkage=a,
            kernel_variance=h_sq,
            state_signature=state_signature,
            move_only_on_selection=move_only_on_selection,
            static_never_select=static_never_select,
        )
        if store_history:
            return next_carry, output
        # In final-only mode, the scan stacks only the scalar traces;
        # final arrays come from the carry.
        return next_carry, (
            output.ess,
            output.log_evidence_increment,
            output.normalizers_finite,
        )

    init_carry = _LiuWestStepCarry(
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        jnp.zeros_like(log_ev_0),
        identity_ancestors,
        jnp.asarray(False),
    )
    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    scan_inputs = (
        (step_keys, emissions[1:], time_indices)
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:], time_indices)
    )

    if store_history:
        final_carry, outputs = _filter_scan(_step, init_carry, scan_inputs)
        all_particles = _prepend(particles_0, outputs.particles)
        all_params = _prepend(params_0, outputs.params)
        all_log_w = _prepend(log_w_0, outputs.log_weights)
        all_ancestors = _prepend(identity_ancestors, outputs.ancestors)
        ess_rest = outputs.ess
        log_ev_inc_rest = outputs.log_evidence_increment
        normalizers_finite = outputs.normalizers_finite
    else:
        (
            final_carry,
            (
                ess_rest,
                log_ev_inc_rest,
                normalizers_finite,
            ),
        ) = _filter_scan(_step, init_carry, scan_inputs)
        all_particles = final_carry.particles[None]
        all_params = final_carry.params[None]
        all_log_w = final_carry.log_weights[None]
        all_ancestors = final_carry.ancestors[None]

    final_log_ml = (
        final_carry.log_marginal_likelihood
        + final_carry.log_evidence_compensation
    )

    all_normalizers_finite = jnp.isfinite(log_ev_0) & jnp.all(
        normalizers_finite
    )
    checked_log_ml = jnp.where(
        all_normalizers_finite,
        final_log_ml,
        jnp.nan,
    )
    _raise_invalid_ancestors(final_carry.invalid_resampling, num_particles)
    _raise_if_degenerate(checked_log_ml)

    return LiuWestPosterior(
        marginal_loglik=checked_log_ml,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=_prepend(jnp.asarray(ess_0), ess_rest),
        log_evidence_increments=_prepend(
            jnp.asarray(log_ev_0), log_ev_inc_rest
        ),
        filtered_params=all_params,
    )
