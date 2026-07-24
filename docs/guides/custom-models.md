# Author custom models

Nonlinear Gaussian, particle, and tempered methods represent a model by the
callbacks needed for one inference algorithm. You do not need to subclass an
smcx model or wrap distributions in an smcx object. The boundary consists of
arrays, PyTrees, and, for stochastic algorithms, explicit PRNG keys. Exact
linear-Gaussian models instead use the dense-array interface shown in the
[quickstart](quickstart.md#establish-the-exact-baseline).

## Compose a nonlinear Gaussian filter

The extended and unscented Kalman filters share two ordinary mean callbacks.
The extended filter additionally takes explicit Jacobians:

```text
# EKF and UKF
transition_mean(state) -> state_mean
observation_mean(state) -> observation_mean

# EKF only
transition_jacobian(state) -> (state_dim, state_dim)
observation_jacobian(state) -> (observation_dim, state_dim)
```

Jacobians use output-by-input orientation. Each Jacobian can be analytic or
created explicitly by the caller with `jax.jacfwd`; smcx does not select an
automatic-differentiation policy. This example mixes both forms:

```python
import jax
import jax.numpy as jnp

import smcx


def transition_mean(state):
    return jnp.array([
        0.9 * state[0] + 0.1 * jnp.sin(state[1]),
        0.8 * state[1],
    ])


def transition_jacobian(state):
    return jnp.array([
        [0.9, 0.1 * jnp.cos(state[1])],
        [0.0, 0.8],
    ])


def observation_mean(state):
    return jnp.array([state[0] + 0.05 * state[1] ** 2])


observation_jacobian = jax.jacfwd(observation_mean)

emissions = jnp.array([[0.2], [-0.1], [0.4]])
posterior = smcx.extended_kalman_filter(
    jnp.zeros(2),
    jnp.eye(2),
    transition_mean,
    transition_jacobian,
    0.1 * jnp.eye(2),
    observation_mean,
    observation_jacobian,
    jnp.array([[0.3]]),
    emissions,
)
```

The UKF reuses the two mean functions without Jacobians:

```python
unscented = smcx.unscented_kalman_filter(
    jnp.zeros(2),
    jnp.eye(2),
    transition_mean,
    0.1 * jnp.eye(2),
    observation_mean,
    jnp.array([[0.3]]),
    emissions,
)
```

Rule defaults are `alpha=1.0`, `beta=2.0`, and `kappa=0.0`.

The transition covariance may have shape `(state_dim, state_dim)` or
`(ntime - 1, state_dim, state_dim)`. The observation covariance may have
shape `(observation_dim, observation_dim)` or
`(ntime, observation_dim, observation_dim)`. All arrays and callback outputs
share one float32 or float64 dtype.

With `inputs=...`, every supplied callback accepts `(state, input_t)`.
`inputs[t]` reaches the observation at `t` and the transition into `t`;
`inputs[0]` does not transform the supplied prior. A rank-one input sequence
is presented to callbacks as a length-one vector. When compiling a complete
filter, close the callbacks over in a `jax.jit` wrapper rather
than passing them as dynamic array arguments.

Use the EKF to supply a local linearization; use the UKF to apply the fixed
scaled sigma-point rule. Shared means let research code compare them without
a model hierarchy or general sigma-point plug-in layer.

## Choose particle callbacks for the algorithm

Most particle callbacks act on one particle. smcx maps them over the particle
cloud; only an initial sampler creates the whole cloud at once. This keeps
model representation with your application while smcx owns resampling,
schedules, and evidence accounting.

Each callback-driven algorithm asks only for behavior it can use:

| Algorithm | Sampling | Densities or weights |
| --- | --- | --- |
| Bootstrap | Initial cloud and transition | Observation |
| Auxiliary | Initial cloud and transition | Observation and look-ahead |
| Guided | Initial cloud and proposal | Proposal, transition, and observation |
| Tempered SMC | Initial cloud | Prior and likelihood |

Choose the algorithm first, then supply the callbacks shown in the table.
smcx does not inspect a model object to discover optional capabilities.

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

The factory belongs to the application. An auxiliary or guided factory can
return the extra callbacks required by that algorithm.

## Choose when to resample

The four state-space particle filters accept either an ESS fraction or a
caller-owned resampling criterion in `resampling_threshold`. The callback
receives normalized log weights, the corresponding absolute ESS, and the
zero-based emission index:

```python
def every_fifth_step(log_weights, current_ess, time_index):
    del log_weights, current_ess
    return time_index % 5 == 0


posterior = smcx.bootstrap_filter(
    jr.key(0),
    initial,
    transition,
    log_observation,
    emissions,
    num_particles=4_096,
    resampling_threshold=every_fifth_step,
)
```

The result must be a Python Boolean or scalar JAX Boolean. The callback runs
for time indices 1 through T - 1 and can be traced as part of the filter.
Bootstrap and guided filters supply their carried weights and ESS. Auxiliary
and Liu–West filters instead supply the normalized first-stage weights and
their ESS, because those are the quantities governing ancestor selection.
The numeric default `0.5` retains the strict rule `ESS < 0.5 * N`.

## Compose a particle-filter kernel

Use `smcx.run_particle_filter` when a built-in filter does not provide the
algorithmic pieces you want to combine. The runner accepts these callback
contracts when there are no time-varying inputs:

```text
initialize(time_index, emission_t, key_t) -> (carry, record)
step(carry, time_index, emission_t, key_t) -> (carry, record)
```

An input-aware kernel inserts `input_t` before `key_t` in both callbacks:

```text
initialize(time_index, emission_t, input_t, key_t) -> (carry, record)
step(carry, time_index, emission_t, input_t, key_t) -> (carry, record)
```

The carry may be any JAX PyTree of arrays. Its structure, leaf shapes, and
dtypes must remain fixed across steps because the runner uses `jax.lax.scan`.
It is private execution state and is not included in the returned posterior.
Each callback also returns the public standard record:

```python
smcx.ParticleFilterRecord(
    particles,  # PyTree leaves: (num_particles, ...)
    normalized_log_weights,  # (num_particles,)
    ancestor_indices,  # (num_particles,), integer
    log_evidence_increment,  # scalar
)
```

The record describes the current time. Ancestor indices refer to the previous
cloud; an identity map is conventional at time zero. The runner aligns
emissions, optional inputs, and fresh keys; computes ESS; accumulates the
evidence increments; and assembles `smcx.ParticleFilterPosterior`. The
callbacks retain control of resampling, propagation, weighting, and the
increment calculation. Weight normalization and ancestor-index bounds are
callback preconditions.

This always-resampling bootstrap kernel composes only public smcx operations
with the `initial`, `transition`, and `log_observation` callbacks defined
above:

```python
num_particles = 4_096


def weighted_record(particles, emission_t, ancestors):
    log_scores = jax.vmap(log_observation, in_axes=(None, 0))(
        emission_t, particles
    )
    log_weights, log_total = smcx.log_normalize(log_scores)
    increment = log_total - jnp.log(jnp.asarray(num_particles))
    record = smcx.ParticleFilterRecord(
        particles,
        log_weights,
        ancestors,
        increment,
    )
    return log_weights, record


def initialize_kernel(time_index, emission_t, key_t):
    del time_index
    particles = initial(key_t, num_particles)
    ancestors = jnp.arange(num_particles, dtype=jnp.int32)
    log_weights, record = weighted_record(
        particles,
        emission_t,
        ancestors,
    )
    return (particles, log_weights), record


def step_kernel(carry, time_index, emission_t, key_t):
    del time_index
    previous_particles, previous_log_weights = carry
    resample_key, transition_key = jr.split(key_t)
    ancestors = smcx.systematic(
        resample_key,
        smcx.normalize(previous_log_weights),
        num_particles,
    )
    selected = jax.tree.map(
        lambda leaf: leaf[ancestors],
        previous_particles,
    )
    particle_keys = jr.split(transition_key, num_particles)
    particles = jax.vmap(transition)(particle_keys, selected)
    log_weights, record = weighted_record(
        particles,
        emission_t,
        ancestors,
    )
    return (particles, log_weights), record


custom_posterior = smcx.run_particle_filter(
    jr.key(0),
    initialize_kernel,
    step_kernel,
    emissions,
)
```

Initialization receives time zero and the first emission. The step callback
then receives times one through `ntime - 1`. With `store_history=True`, the
posterior stores every particle record. With `store_history=False`, its
particle, weight, and ancestor histories contain only the final record; ESS
and evidence increments remain available for every time step.

## Combine auxiliary selection with a guided proposal

The same runner can combine an auxiliary look-ahead `log_m` with a proposal
`q` that sees the current emission. Keep normalized carried log weights `W`
and use this core inside the step callback:

```python
log_first, first_total = smcx.log_normalize(W + log_m)
do_resample = smcx.ess(log_first) < threshold * num_particles
ancestors = jax.lax.cond(
    do_resample,
    lambda: resampling_fn(
        resample_key, smcx.normalize(log_first), num_particles
    ),
    lambda: jnp.arange(num_particles, dtype=jnp.int32),
)
parents = jax.tree.map(lambda leaf: leaf[ancestors], previous_particles)
particles = jax.vmap(
    lambda key_i, parent: proposal(key_i, parent, emission_t)
)(particle_keys, parents)
log_g = jax.vmap(lambda state: log_observation(emission_t, state))(particles)
log_f = jax.vmap(log_transition)(particles, parents)
log_q = jax.vmap(
    lambda state, parent: log_proposal(emission_t, state, parent)
)(particles, parents)
log_step = log_g + log_f - log_q
log_scores = jnp.where(
    do_resample, log_step - log_m[ancestors], W + log_step
)
log_weights, second_total = smcx.log_normalize(log_scores)
increment = jnp.where(
    do_resample,
    first_total + second_total - jnp.log(num_particles),
    second_total,
)
```

Return `particles`, `log_weights`, `ancestors`, and `increment` in a
`ParticleFilterRecord`. The look-ahead correction appears only after
first-stage resampling; without resampling, the ordinary guided score is
`W + log(g) + log(f) - log(q)`. In an input-aware step, pass the aligned
`input_t` to the look-ahead, proposal, and all three density callbacks.

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

Close over a fixed model rather than copying it into every latent particle.
Replacing closed-over array values can make JAX retrace or recompile the
filter, so pass frequently changing values through an explicit argument.

## Replace the tempering mutation

`smcx.temper` accepts one caller-owned invariant mutation through a paired
structural callback boundary:

```text
mutation_init(position, tempered_logdensity_fn) -> state
mutation_step(key, state, tempered_logdensity_fn) -> (state, info)
```

State is a JAX PyTree with a dense vector `position`; info is a JAX PyTree
with a scalar floating `acceptance_rate`. NamedTuples are a convenient
representation, and either object may carry extra fields. The target passed
to both callbacks is the current stage density
`log_prior + phi * log_likelihood`.

```python
posterior = smcx.temper(
    jr.key(0),
    initial,
    log_prior,
    log_likelihood,
    num_particles=4_096,
    mutation_init_fn=mutation_init,
    mutation_step_fn=mutation_step,
)
```

smcx batches independent states across particles and compiles the fixed-count
sweep; `temper` itself remains host-driven. Mutation state is reinitialized
after each resampling stage. The caller is responsible for making each step
invariant for the supplied target. Omitting both callbacks selects the
existing cloud-adaptive random-walk Metropolis mutation. Pass ordinary
callbacks rather than pre-jitting a function that accepts the target callable.

## Optional Equinox representation

If an application already uses Equinox, a callable module can be captured by
the same closures. This example targets `equinox==0.13.8`; Equinox is not an
smcx dependency. See the Equinox [Module documentation][equinox-module] for
its PyTree behavior.

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

The factory exposes the same three callback signatures as the plain JAX
version. No Equinox-specific adapter is needed.

[equinox-module]: https://docs.kidger.site/equinox/api/module/module/
