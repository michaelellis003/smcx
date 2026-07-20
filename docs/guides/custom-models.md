# Author custom models

smcx represents a model by the callbacks needed for one inference
algorithm. You do not need to subclass an smcx model or wrap distributions
in an smcx object. The boundary consists of arrays, PyTrees, and explicit
PRNG keys.

Most callbacks act on one particle. smcx maps them over the particle cloud;
only an initial sampler creates the whole cloud at once. This keeps model
representation with your application while smcx owns resampling, schedules,
and evidence accounting.

## Choose callbacks for the algorithm

Each algorithm asks only for behavior it can use:

| Algorithm | Sampling | Densities or weights |
| --- | --- | --- |
| Bootstrap | Initial cloud and transition | Observation |
| Auxiliary | Initial cloud and transition | Observation and look-ahead |
| Guided | Initial cloud and proposal | Proposal, transition, and observation |
| Static | Initial cloud if needed | Base or target; new-data increment |
| RBPF *(forthcoming)* | Sampled component | Matrices, offsets, covariances |

For static inference or updating, the increment is the new-data likelihood;
the current `temper` entry point also takes an initial sampler. The forthcoming
native RBPF will define callbacks for its conditional-Gaussian matrices,
offsets, and covariances in its own ADR.

A generic model object cannot infer a useful look-ahead function, guided
proposal, parameter transform, or conditional-Gaussian decomposition. Select
the algorithm and construct its callbacks explicitly. Do not use `hasattr`,
signature inspection, optional methods, or `NotImplementedError` to discover
capabilities at runtime.

## Bind a plain JAX model

The following user-owned classes are one possible model representation. The
method names are a convention for this recipe; smcx never imports the classes
or inspects the methods.

```python
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import smcx


class AR1Params(NamedTuple):
    rho: jax.Array
    process_scale: jax.Array
    observation_scale: jax.Array


class GaussianAR1:
    def sample_initial(self, key, num_particles, params):
        scale = params.process_scale / jnp.sqrt(1.0 - params.rho**2)
        return scale * jr.normal(key, (num_particles, 1))

    def sample_transition(self, key, state, params):
        noise = params.process_scale * jr.normal(key, state.shape)
        return params.rho * state + noise

    def log_observation(self, emission, state, params):
        scale = params.observation_scale
        residual = (emission[0] - state[0]) / scale
        return (
            -0.5 * residual**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)
        )
```

A small factory binds the model representation and its parameters into the
three callback signatures expected by `smcx.bootstrap_filter`:

```python
def make_bootstrap_callbacks(model, params):
    def initial(key, num_particles):
        return model.sample_initial(key, num_particles, params)

    def transition(key, state):
        return model.sample_transition(key, state, params)

    def log_observation(emission, state):
        return model.log_observation(emission, state, params)

    return initial, transition, log_observation


model = GaussianAR1()
params = AR1Params(
    rho=jnp.asarray(0.95),
    process_scale=jnp.asarray(0.3),
    observation_scale=jnp.asarray(0.7),
)
initial, transition, log_observation = make_bootstrap_callbacks(model, params)

emissions = jnp.asarray([[0.2], [-0.1], [0.4]])
posterior = smcx.bootstrap_filter(
    jr.key(0),
    initial,
    transition,
    log_observation,
    emissions,
    num_particles=4_096,
)
```

The factory is ordinary application code, not a public callback-bundle type.
An auxiliary or guided factory should return that algorithm's extra
callbacks rather than attach optional methods to one universal object.

## Write input-aware callbacks explicitly

Time-varying inputs use distinct callback signatures. At time zero,
`input_0` reaches the initial sampler and observation callback. At later
times, `input_t` reaches the transition into that time and its observation.
The smcx-facing input is always the final callback argument.

Use a separate factory for an input-aware model:

```python
def make_input_aware_bootstrap_callbacks(model, params):
    def initial(key, num_particles, input_0):
        return model.sample_initial(key, num_particles, input_0, params)

    def transition(key, state, input_t):
        return model.sample_transition(key, state, input_t, params)

    def log_observation(emission, state, input_t):
        return model.log_observation(emission, state, input_t, params)

    return initial, transition, log_observation
```

Choose `make_bootstrap_callbacks` or
`make_input_aware_bootstrap_callbacks` when configuring the run. Keeping both
forms visible prevents an ambiguous runtime dispatch and makes input alignment
part of the model code.

## Keep the two PyTree roles separate

There are two useful, different PyTrees at this boundary.

The **latent-state PyTree** is one particle's evolving state. Bootstrap,
auxiliary, and guided filters accept a nonempty PyTree of arrays. The initial
sampler adds a leading particle axis to every leaf, and each transition
preserves the tree structure, leaf shapes, and dtypes. smcx carries and
resamples this tree as inference state.

The **model or parameter PyTree** belongs to the application. Fixed model
values can be closed over by the callbacks, as `params` is above. Algorithms
whose mutation kernels require Euclidean geometry still take dense parameter
vectors; a user-owned codec can decode those vectors inside conditioned
callbacks.

Do not copy a fixed neural network into every latent particle. Closing over a
fixed model is appropriate for one inference run. Frequently replacing
closed-over parameter arrays may cause JAX to retrace or recompile the run;
profile that usage before changing filter APIs to pass a dynamic model.

## Optional Equinox representation

This static recipe targets `equinox==0.13.8`. Equinox remains outside every
smcx dependency group. Its [Module documentation][equinox-module] defines
modules as PyTrees whose methods have no special transformation semantics, so
a fixed callable module can cross the same closure boundary.

```python
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import smcx


class LinearGaussianTransition(eqx.Module):
    rho: jax.Array
    process_scale: jax.Array

    def __call__(self, key, state):
        noise = self.process_scale * jr.normal(key, state.shape)
        return self.rho * state + noise


def make_equinox_bootstrap_callbacks(
    transition_model,
    initial_scale,
    observation_scale,
):
    def initial(key, num_particles):
        return initial_scale * jr.normal(key, (num_particles, 1))

    def transition(key, state):
        return transition_model(key, state)

    def log_observation(emission, state):
        residual = (emission[0] - state[0]) / observation_scale
        return (
            -0.5 * residual**2
            - jnp.log(observation_scale)
            - 0.5 * math.log(2.0 * math.pi)
        )

    return initial, transition, log_observation


transition_model = LinearGaussianTransition(
    rho=jnp.asarray(0.95),
    process_scale=jnp.asarray(0.3),
)
initial, transition, log_observation = make_equinox_bootstrap_callbacks(
    transition_model,
    initial_scale=jnp.asarray(1.0),
    observation_scale=jnp.asarray(0.7),
)
emissions = jnp.asarray([[0.2], [-0.1], [0.4]])
posterior = smcx.bootstrap_filter(
    jr.key(0),
    initial,
    transition,
    log_observation,
    emissions,
    num_particles=4_096,
)
```

The factory has the same three signatures and returns the same posterior
contract as the plain JAX version. smcx sees only those pure callables. It
does not need an Equinox-aware model class, codec, or adapter.

[equinox-module]: https://docs.kidger.site/equinox/api/module/module/
