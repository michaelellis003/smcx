# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Optional Dynamax adapter for the preregistered L1 benchmark.

This original glue code calls Dynamax's public state-space-model distribution
methods; no Dynamax implementation code is copied or translated.  The
supported release is Dynamax 1.0.2, tag commit
``a216d7feec0d025560a0a194ed5abab538648375``:

* Public SSM interface (MIT):
  https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/ssm.py#L90-L146
* LGSSM distribution methods (MIT):
  https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/models.py#L173-L202
* Immutable upstream license, Copyright 2022 Probabilistic machine learning:
  https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/LICENSE

Dynamax is imported only when the factory is called.  Importing this module,
the profiling registry, smcx, or the core test graph therefore does not make
the optional notebook dependency a runtime requirement.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, NamedTuple

from jaxtyping import Array, Float, PRNGKeyArray

from benchmarks.profiling.models import LGSSM
from smcx.types import (
    InitialSamplerWithInput,
    LogObservationFnWithInput,
    TransitionSamplerWithInput,
)

DYNAMAX_VERSION = "1.0.2"
DYNAMAX_COMMIT = "a216d7feec0d025560a0a194ed5abab538648375"


class DynamaxLGSSMAdapter(NamedTuple):
    """Dynamax objects and callbacks for one mapped L1 model."""

    model: Any
    params: Any
    initial_sampler: InitialSamplerWithInput
    transition_sampler: TransitionSamplerWithInput
    log_observation_fn: LogObservationFnWithInput


def _linear_gaussian_ssm_class() -> Any:
    """Load the exactly preregistered optional Dynamax implementation."""
    try:
        installed = version("dynamax")
    except PackageNotFoundError as error:
        raise ImportError(
            "Dynamax profiling requires the optional 'notebooks' dependency "
            "group; its resolved Dynamax version must be "
            f"{DYNAMAX_VERSION}"
        ) from error
    if installed != DYNAMAX_VERSION:
        raise RuntimeError(
            "Dynamax profiling is preregistered for version "
            f"{DYNAMAX_VERSION}; found {installed}"
        )

    try:
        from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    except ImportError as error:
        raise ImportError(
            f"Dynamax {DYNAMAX_VERSION} is installed but could not be imported"
        ) from error
    return LinearGaussianSSM


def make_dynamax_lgssm_adapter(model: LGSSM) -> DynamaxLGSSMAdapter:
    """Map the dependency-free scalar L1 definition to Dynamax callbacks.

    The adapter preserves L1's ``(1,)`` state, emission, and input shapes.
    Each callback delegates sampling or density evaluation to the public
    Dynamax distribution returned for the fixed parameter PyTree.  Parameters
    are closed over exactly as they are in the dependency-free callback arm,
    keeping the benchmark comparison at the model-callable boundary.

    Args:
        model: Dependency-free controlled scalar LGSSM specification.

    Returns:
        Dynamax model and parameter objects plus three input-aware callbacks
        accepted by :func:`smcx.bootstrap_filter`.

    Raises:
        ImportError: Dynamax or a required transitive dependency is absent.
        RuntimeError: The installed Dynamax version is not the preregistered
            1.0.2 release.
    """
    linear_gaussian_ssm_class = _linear_gaussian_ssm_class()

    # Imported only on explicit use so importing the adapter stays optional.
    import jax.numpy as jnp

    dynamax_model = linear_gaussian_ssm_class(
        state_dim=1,
        emission_dim=1,
        input_dim=1,
        has_dynamics_bias=False,
        has_emissions_bias=False,
    )
    params, _ = dynamax_model.initialize(
        initial_mean=jnp.asarray([model.m0], dtype=jnp.float32),
        initial_covariance=jnp.asarray([[model.p0]], dtype=jnp.float32),
        dynamics_weights=jnp.asarray([[model.a]], dtype=jnp.float32),
        dynamics_input_weights=jnp.asarray([[model.b]], dtype=jnp.float32),
        dynamics_covariance=jnp.asarray([[model.q]], dtype=jnp.float32),
        emission_weights=jnp.asarray([[1.0]], dtype=jnp.float32),
        emission_input_weights=jnp.asarray([[0.0]], dtype=jnp.float32),
        emission_covariance=jnp.asarray([[model.r]], dtype=jnp.float32),
    )

    def initial_sampler(
        key: PRNGKeyArray,
        num_particles: int,
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, "num_particles 1"]:
        input_value = jnp.asarray(input_t, dtype=jnp.float32)
        distribution = dynamax_model.initial_distribution(params, input_value)
        return distribution.sample(
            sample_shape=(num_particles,),
            seed=key,
        )

    def transition_sampler(
        key: PRNGKeyArray,
        state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " 1"]:
        state_value = jnp.asarray(state, dtype=jnp.float32)
        input_value = jnp.asarray(input_t, dtype=jnp.float32)
        distribution = dynamax_model.transition_distribution(
            params,
            state_value,
            input_value,
        )
        return distribution.sample(seed=key)

    def log_observation_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, ""]:
        emission_value = jnp.asarray(emission, dtype=jnp.float32)
        state_value = jnp.asarray(state, dtype=jnp.float32)
        input_value = jnp.asarray(input_t, dtype=jnp.float32)
        distribution = dynamax_model.emission_distribution(
            params,
            state_value,
            input_value,
        )
        return distribution.log_prob(emission_value)

    return DynamaxLGSSMAdapter(
        model=dynamax_model,
        params=params,
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_observation_fn,
    )
