# Author custom models

Nonlinear Gaussian, particle, and tempered methods represent a model by the
callbacks needed for one inference algorithm. You do not need to subclass an
smcx model or wrap distributions in an smcx object. The boundary consists of
arrays, PyTrees, and, for stochastic algorithms, explicit PRNG keys. Exact
linear-Gaussian models instead use the dense-array interface shown in the
[quickstart](quickstart.md#establish-the-exact-baseline).

## Compose a nonlinear Gaussian filter and smoother

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

All covariance arrays are finite and symmetric. The EKF permits positive
semidefinite prior and transition covariances, including deterministic state
components, and requires a positive-definite observation covariance. The UKF
also requires a positive-definite prior because it takes Cholesky square roots
of state covariances; its transition covariance remains positive
semidefinite. These value checks run at eager Python entry and are skipped
when the arrays are tracers inside a JAX transformation.
Concrete entries must be zero or normal finite values. Positive-definite
covariances must also be non-indefinite within dtype-scaled roundoff and yield
a finite, positive-diagonal Cholesky factor on the active backend, as described
in the [filtering quickstart](quickstart.md). These eager checks do not
guarantee endpoint behavior under an outer JAX transformation.

With `inputs=...`, every supplied callback accepts `(state, input_t)`.
`inputs[t]` reaches the observation at `t` and the transition into `t`;
`inputs[0]` does not transform the supplied prior. A rank-one input sequence
is presented to callbacks as a length-one vector. When compiling a complete
filter, close the callbacks over in a `jax.jit` wrapper rather
than passing them as dynamic array arguments.

Use the EKF to supply a local linearization; use the UKF to apply the fixed
scaled sigma-point rule. `smcx.gaussian_filter` makes the rule an
exchangeable strategy on one model: pass
`method=smcx.taylor_order1(transition_jacobian, observation_jacobian)`
for first-order linearization or `method=smcx.unscented(alpha, beta,
kappa)` for sigma points, with results identical to the named filters.

The strategy configures the matching nonlinear RTS smoother:

```python
extended_smoothed = smcx.gaussian_smoother(
    posterior,
    transition_mean,
    method=smcx.taylor_order1(
        transition_jacobian,
        observation_jacobian,
    ),
)
unscented_smoothed = smcx.gaussian_smoother(
    unscented,
    transition_mean,
    method=smcx.unscented(),
)
```

The smoother strategy must match the one used for its filter record. Taylor
smoothing evaluates transition Jacobians at the filtered means and does not call
`transition_mean`. Unscented smoothing re-evaluates `transition_mean` at sigma
points. Neither path uses the observation side of the strategy. Unscented
smoothing requires a record produced with the same transition model and
sigma-point rule.

### Match a smoother to its forward result

Filter records store moments, not the identity of the model that produced
them. The caller supplies the matching model pieces to the backward pass:

| Forward result | Retrospective operation | Caller must reuse | Claim |
| --- | --- | --- | --- |
| `kalman_filter` | `rts_smoother` | Static or time-varying transition matrices | Exact for a consistent linear-Gaussian model |
| `kalman_filter` | `posterior_sample` | Static or time-varying transition matrices | Exact joint linear-Gaussian trajectories |
| `extended_kalman_filter` or `gaussian_filter(..., method=taylor_order1(...))` | `gaussian_smoother(..., method=taylor_order1(...))` | Transition Jacobian and `inputs[1:]` if used; supply a full length-`T` input array | Approximate extended RTS moments |
| `unscented_kalman_filter` or `gaussian_filter(..., method=unscented(...))` | `gaussian_smoother(..., method=unscented(...))` | Transition mean, sigma-point rule, and `inputs[1:]` if used; supply a full length-`T` input array | Approximate unscented RTS moments |
| `dlm_filter` | `dlm_smoother` | Static `G`, the same scale-free `W_tilde` or state discount, and `variance_discount=1` | Exact constant-common-variance Student-t marginals |
| `dglm_filter` | `dglm_smoother` | Static `G` and the same static or length-`T - 1` `W`, or the same state discount | Approximate retrospective state moments; exact RTS for a normal-family record with `dispersion_discount=1` |
| Fixed-parameter `ParticleFilterPosterior` | `reconstruct_trajectories` | Full particle and ancestor history; use final filtering weights for path summaries | Approximate genealogy paths; cheap but subject to early coalescence |
| Fixed-parameter `ParticleFilterPosterior` | `backward_simulation` | Full particle and corrected weight history; the same transition log density or mass, parameters, and full input history | Equal-weight draws from the discrete particle FFBS approximation |

`smoothed_cross_covariances` is the exact linear-RTS companion; nonlinear
smoother records do not retain the direct cross-covariances for that claim.

smcx cannot compare these model identities at runtime. DLM and DGLM
retrospective operations cannot verify the resupplied evolution identity.
For particle backward simulation, every stored log-weight row must be the
normalized filtering weights of that same stored cloud. The named bootstrap,
guided, and auxiliary filters and their shared-loop derivations satisfy this
contract, including the auxiliary filter's corrective weight.

`reconstruct_trajectories` can follow the state genealogy of a
`LiuWestPosterior`, but that result is not a fixed-parameter smoother.
`backward_simulation` rejects it because one external parameter PyTree cannot
represent its time- and particle-specific `filtered_params`. A tempered result
has no latent state-time filtering history. The current `SMC2Posterior`
exposes its outer parameter history, not the inner state histories needed by
these consumers. These record shapes therefore have no row in the table.

All filtered covariances and the first predicted covariance must be finite,
symmetric, and positive semidefinite. Positive-time predicted covariances must
be positive definite. Unscented smoothing also factors each nonterminal
filtered covariance, so those matrices must be positive definite; the terminal
filtered covariance may remain positive semidefinite. For the backward
transition from time `t` to `t + 1`, an input-aware callback receives
`inputs[t + 1]`. `inputs[0]` is unused by smoothing.

As reference cases, Särkkä and Svensson (2023) apply ERTSS and URTSS to the
same noisy-pendulum model in
[Examples 13.2 and 14.10](https://doi.org/10.1017/9781108917407); the latter
appears in section 14.4. These are reference cases rather than smcx numerical
oracles; smcx does not use the reported RMSEs as frozen expected values.

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
| Liu-West | Initial state and parameter clouds; state transition | Observation and look-ahead |
| Tempered SMC | Initial cloud | Prior and likelihood |
| SMC² | Initial parameter cloud; conditioned inner state cloud and transition | Parameter prior and observation |
| Caller-owned runner | Initialization and step kernels | Normalized weights and evidence increments |

Choose the algorithm first, then supply the callbacks shown in the table.
smcx does not inspect a model object to discover optional capabilities.
Density callback outputs, probability- and log-weight arrays, and runner
record evidence increments must have at least float32 precision. smcx rejects
float16 and bfloat16 at these boundaries instead of silently changing the
arithmetic or fixed-key draws.

### Observation arrays

Callback-driven particle filters, SMC², and the caller-owned runner accept
JAX observation arrays with shape `(T,)` for scalar events or
`(T, emission_dim)` for vector events. The event dimension must be nonempty.
Scalar sequences become `(T, 1)`, so callbacks always receive `emission_t` as
a vector. Observation dtype is model-owned and preserved; integer and Boolean
data are supported for discrete likelihoods. Incremental bootstrap calls
accept either a scalar or vector observation and apply the same
canonicalization. Gaussian filters also accept scalar observation sequences,
but retain their documented float32/float64 requirements.

Callback inputs preserve dtype: `(T,)` becomes `(T, 1)`; rank-two inputs are
unchanged and empty widths rejected. Callers relying on incremental scalar
callback shapes must use length-one vectors; sampled emissions do likewise.

Log-weight normalization and ESS are invariant to a finite constant offset
whenever the relative differences remain representable in the input dtype.
The absolute log normalizer remains in that dtype, so a correction smaller
than one unit in its last place cannot be represented in the returned scalar.
Earlier releases could restore the offset before deriving normalized results
and erase those differences. Correcting that wrong-result path can change
fixed-key weights, ESS values, and SMC² or tempering paths. Following
[NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html), smcx
treats this as a bug fix; public signatures, shapes, and dtypes are unchanged.

## Learn the observational variance exactly

One special structure admits exact sequential learning of a static
parameter: the linear-Gaussian model whose single unknown
observational variance scales every covariance (West and Harrison
1997, ch. 4). `smcx.dlm_filter` carries its Normal-Inverse-Gamma
posterior in closed form and returns the exact Student-t marginal
likelihood. Covariances are supplied scale-free (divided by the
unknown variance), and the evolution covariance may instead be stated
by a discount factor — a modeling device, not an estimator:

```python
posterior = smcx.dlm_filter(
    jnp.zeros(1),
    jnp.eye(1),  # prior covariance / V
    jnp.eye(1),
    jnp.ones(1),
    emissions,
    discount=0.95,
    prior_shape=4.0,  # Inverse-Gamma degrees of freedom
    prior_scale=1.0,  # prior point estimate of V
)
smoothed = smcx.dlm_smoother(posterior, jnp.eye(1), discount=0.95)
scale_matrices = (
    posterior.scale_estimates[:, None, None]
    * posterior.filtered_scale_free_covariances
)
# These are Student-t scale matrices, not covariances; the filtered
# covariance carries the tail factor n / (n - 2) and exists for n > 2.
dof = posterior.scale_shapes[:, None, None]
filtered_covariances = dof / (dof - 2.0) * scale_matrices
smoothed_scale_matrices = (
    posterior.scale_estimates[-1] * smoothed.smoothed_scale_free_covariances
)
```

The smoothed scale matrices use the final variance estimate and degrees of
freedom at every time. Here `discount=0.95` is the state-evolution discount;
the exact retrospective claim still requires `variance_discount=1`.

A `variance_discount` below one instead tracks a slowly changing
variance (exact under the implied beta-gamma random walk on the
precision). Learning several free covariances breaks the conjugacy;
that is where the particle methods below take over.

## Filter count or binary observations and smooth state moments

Between the exact conjugate case and the particle methods sits the
dynamic generalized linear model (West, Harrison, and Migon 1985).
`smcx.dglm_filter` runs exponential-family emissions — Poisson
counts, Bernoulli or binomial outcomes — over a linear state
evolution carried by moments only. Each step matches a conjugate
prior to the linear predictor's two moments, updates it exactly on
the observation, and feeds the posterior moments back to the state by
linear Bayes estimation. The recursion is deterministic and
closed-form, and it is approximate: the docstring states the three
approximation points, and the particle filters below are the natural
accuracy check.

```python
counts = jnp.array([1, 0, 3, 2])
transition = jnp.eye(1)
posterior = smcx.dglm_filter(
    jnp.zeros(1),
    jnp.eye(1),
    transition,
    jnp.ones(1),
    counts,
    family=smcx.poisson(),
    discount=0.95,
)
smoothed = smcx.dglm_smoother(posterior, transition, discount=0.95)
posterior.marginal_loglik  # sum of exact negative-binomial forecasts
smoothed.smoothed_means.shape  # (4, 1)
```

The smoother takes no observation family, observation vector, emissions, or
dispersion discount. It does require the same `G`, `W`, or state discount as
the filter, which the record cannot verify. Its retained conjugate parameters
remain filtering-time quantities. General smoothed output is a moment-only
summary, not a distribution or credible interval; a normal-family record with
`dispersion_discount=1` reduces exactly to RTS.

The observation family is a `smcx.DGLMFamily` record of four pure
callables (moment matching, forecast log density, conjugate update,
posterior moments), so a new family is user-definable without
touching the filter. The library's own tests build a normal family
through this record to prove the recursion reduces exactly to the
Kalman filter. The built-in factories also check emission support
eagerly at the filter boundary. A user-defined family's emissions
pass through unchecked. `smcx.bernoulli()` and
`smcx.binomial(trials=n)` cover binary and bounded counts, and
`dispersion_discount` adds Berry and West's random-effects
extra-dispersion.

## Bind a model record

A `smcx.StateSpaceModel` groups the per-particle callables that define
one model. Parameters stay an explicit PyTree argument that smcx
threads to every callable, so there is no binding factory to write,
gradients with respect to parameters flow through filters, and
changing parameters cannot retrace. Every callable takes a trailing
`input_t`, which is `None` when the run has no exogenous inputs; a
model that ignores inputs simply ignores the argument.

```python
import jax.numpy as jnp
import jax.random as jr

import smcx


def sample_initial(key, params, input_0):
    scale = params["process_scale"] / jnp.sqrt(1.0 - params["rho"] ** 2)
    return scale * jr.normal(key, (1,))


def sample_transition(key, state, params, input_t):
    noise = params["process_scale"] * jr.normal(key, state.shape)
    return params["rho"] * state + noise


def log_observation(emission, state, params, input_t):
    residual = (emission[0] - state[0]) / params["observation_scale"]
    return -0.5 * residual**2 - jnp.log(params["observation_scale"])


model = smcx.StateSpaceModel(
    sample_initial=sample_initial,
    sample_transition=sample_transition,
    log_observation=log_observation,
)
params = {
    "rho": jnp.asarray(0.95),
    "process_scale": jnp.asarray(0.3),
    "observation_scale": jnp.asarray(0.7),
}

emissions = jnp.asarray([[0.2], [-0.1], [0.4]])
posterior = smcx.run_smc(
    jr.key(0),
    smcx.bootstrap_fk(model, params, emissions),
    num_particles=4_096,
)
```

`smcx.bootstrap_fk`, `smcx.guided_fk`, and `smcx.auxiliary_fk` derive
the algorithm object — a `smcx.FeynmanKac` — from the same record; an
algorithm that needs an optional capability (`sample_proposal`,
`log_proposal`, `log_transition`, or `log_lookahead`) raises a named
error when the field is `None`. The record is data, not a base class:
smcx never inspects it beyond reading its fields (a Dynamax or other
model-library adapter is just a function returning one).

The positional-callback filters (`smcx.bootstrap_filter` and
friends) remain supported with their original signatures. A model
record can be reused across every derivation that consumes its
fields.

## Write a custom Feynman-Kac model

An algorithm of resample-mutate-reweight form fits
`smcx.FeynmanKac`: an initial law ``m0``, a per-particle mutation
kernel ``m``, and a per-particle log-potential ``log_g``, over a
context PyTree whose leading axis is time. `smcx.run_smc` then
supplies the conditional-resampling loop, the branch weight rule,
the evidence accounting, and the posterior container. An algorithm
that must own its full step belongs on `smcx.run_particle_filter`
instead. As one record-level example, scaling the potential by a
fixed constant replaces ``log_g`` on the derived record. This
defines one likelihood-powered target, not an annealing path:

```python
fk = smcx.bootstrap_fk(model, params, emissions)


def scaled_log_g(parent, state, context_t):
    return 0.5 * fk.log_g(parent, state, context_t)


scaled = fk._replace(log_g=scaled_log_g)
posterior = smcx.run_smc(jr.key(0), scaled, num_particles=4_096)
```

An optional ``log_eta`` field adds an auxiliary-filter look-ahead
twist (first-stage selection, ancestor correction, and the two-factor
evidence increment are loop-owned), and ``log_g_batch`` lets a
composite potential own its per-callback validation, as the guided
derivation does. For algorithms whose step does not fit the
resample-mutate-reweight shape at all, `smcx.run_particle_filter`
below hands the whole step to your code.

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
Numeric thresholds must be finite and nonnegative. Zero disables resampling;
because the comparison is strict and ESS cannot exceed N, values above one
force resampling at every eligible step. The default `0.5` retains the rule
`ESS < 0.5 * N`. For `liu_west_filter` the selection threshold and the
`parameter_moves` policy are announced to change in smcx 3.0 (to the
always-select Liu-West algorithm with moves on selection); omitting
either argument keeps the 2.x behavior and emits a `FutureWarning`.

The separate `resampling_fn` callback must return a JAX array of exactly
`num_samples` ancestor indices with dtype `int32`. Every index must be in
`[0, num_particles)`. Shape and dtype errors are reported while the filter
is traced. Index values are checked after the filter loop returns, including
with `store_history=False`; under an outer `jax.jit`, Python exceptions cannot
be raised and the value check is skipped. Until issue #38 closes,
multi-observation MPS filters use a sequence of one-step scans.

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
dtypes must remain fixed across steps. It is private execution state and is
not included in the returned posterior. Each callback also returns the public
standard record:

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

CPU and other backends execute the later steps in one `jax.lax.scan` and
support wrapping the runner in JAX transformations. Until
[smcx #38](https://github.com/michaelellis003/smcx/issues/38) closes, MPS
instead uses a sequence of one-step scans to contain an upstream Metal
history-corruption defect. Traced calls stage both paths and select the
execution-platform branch during outer `jax.jit` lowering, so CPU-placed
compiled inputs retain the full scan even when MPS is the default backend.
The contained branch remains compatible with `jax.vmap` and gradients on the
selected backend. Eager MPS calls add per-observation dispatch overhead;
other backends retain the compiled scan.

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
and evidence increments remain available for every time step. Trajectory
reconstruction, genealogy-based variance, the combined diagnostic summary,
and ArviZ export require the default full history.

The final-only option retains the evidence estimate needed by an external
particle marginal Metropolis--Hastings (PMMH) kernel, but smcx has no
standalone PMMH API. Such a kernel can use a proposed parameter's fresh
`posterior.marginal_loglik` as $\log \widehat Z$ when
$\widehat Z=\exp(\mathtt{marginal\_loglik})$ comes from an unbiased particle
filter and resampling scheme; the log estimate itself is Jensen-biased. On a
rejection, the kernel must retain the accepted estimate rather than rerun it.
`backward_simulation` instead conditions on one fixed-parameter, full-history
record. `smc2` uses PMMH moves internally; it is not a user-facing PMMH
chain. See
[Andrieu, Doucet, and Holenstein (2010)](https://doi.org/10.1111/j.1467-9868.2009.00736.x).

`log_ml_variance` is calibrated only when the filter used multinomial
resampling. Posterior containers do not retain resampler provenance, so values
from the default systematic resampler or another scheme are heuristic. Its
optional lag is also an exploratory ancestry-window diagnostic for log
evidence, not an independently calibrated fixed-lag estimator.

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
particles = jax.vmap(lambda key_i, parent: proposal(key_i, parent, emission_t))(
    particle_keys, parents
)
log_g = jax.vmap(lambda state: log_observation(emission_t, state))(particles)
log_f = jax.vmap(log_transition)(particles, parents)
log_q = jax.vmap(lambda state, parent: log_proposal(emission_t, state, parent))(
    particles, parents
)
log_step = log_g + log_f - log_q
log_scores = jnp.where(do_resample, log_step - log_m[ancestors], W + log_step)
log_weights, second_total = smcx.log_normalize(log_scores)
increment = jnp.where(
    jnp.isfinite(first_total),
    jnp.where(
        do_resample,
        first_total + second_total - jnp.log(num_particles),
        second_total,
    ),
    first_total,
)
```

Return `particles`, `log_weights`, `ancestors`, and `increment` in a
`ParticleFilterRecord`. The look-ahead correction appears only after
first-stage resampling; without resampling, the ordinary guided score is
`W + log(g) + log(f) - log(q)`. The outer `where` carries a nonfinite
first-stage normalizer into the runner's eager evidence check even when
resampling is skipped. In an input-aware step, pass the aligned `input_t` to
the look-ahead, proposal, and all three density callbacks.

## Thread time-varying inputs

At time zero, `inputs[0]` reaches the initial sampler and observation
callback. At later times, `inputs[t]` reaches the transition into
that time and its observation. With the model record there is nothing
else to write: pass `inputs=...` to the derivation and every callable
receives the aligned `input_t` as its final argument (a rank-one
input sequence arrives as a length-one vector):

```python
posterior = smcx.run_smc(
    jr.key(0),
    smcx.bootstrap_fk(model, params, emissions, inputs=inputs),
    num_particles=4_096,
)
```

For the positional-callback filters, input-aware runs instead use the
`WithInput` callback arities documented on each filter.

## Align posterior predictions

`posterior_predictive_sample` draws one step beyond every retained filtering
row. For an input-aware model, pass `future_inputs` with the same number of
rows as the retained particle history. `future_inputs[t]` reaches both the
transition out of filtered row `t` and the resulting emission, so it denotes
$u_{t+1}$ rather than the input that produced row `t`. A final-only posterior
therefore needs one future input even though its evidence trace still covers
the complete observed series.

For `LiuWestPosterior`, use `param_posterior_predictive_sample`. Its callbacks
are `(key, state, params[, input_t])`; smcx resamples each aligned state and
parameter pair with one index, retains that static parameter for the forecast,
and passes it to both callbacks. Calling the state-only helper with a Liu-West
result raises `ValueError` since smcx 2.0: it would silently ignore
`filtered_params`, so the ambiguous path is an error rather than a wrong
forecast.

## Keep the two PyTree roles separate

The latent-state PyTree flows through resampling and mutation as
inference state. The parameter PyTree flows into every callback as
explicit data.

The three base fields (`sample_initial`, `sample_transition`,
`log_observation`) are required by the record's typed contract.
The optional derivations validate only their additional capabilities
and raise a named error at construction when one is `None`. The
table lists the fields each operation consumes:

| Operation | Fields consumed |
| --- | --- |
| `bootstrap_fk` | `sample_initial`, `sample_transition`, `log_observation` |
| `guided_fk` | `sample_initial`, `log_observation`, plus `sample_proposal`, `log_proposal`, `log_transition` |
| `auxiliary_fk` | bootstrap fields plus `log_lookahead` |
| `backward_simulation` | `log_transition`, passed explicitly with the same `params` and full input history |

The callback-first named filters are the short on-ramp. The record
and Feynman–Kac path is the reusable workbench layer, where one
model definition serves every derivation that consumes its fields.

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
with a scalar floating `acceptance_rate` that is finite and in `[0, 1]`.
NamedTuples are a convenient representation, and either object may carry
extra fields. The target passed to both callbacks is the current stage
density `log_prior + phi * log_likelihood`.

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
sweep; `temper` itself remains host-driven. The temperature ladder is
also caller-replaceable: a keyword-only ``schedule_fn(phi,
normalized_log_weights, log_likelihoods)`` host callback returns the
next temperature in ``(phi, 1]``, and omitting it keeps the adaptive
ESS bisection. Mutation state is reinitialized
after each resampling stage. Every acceptance rate is checked when the sweep
returns to that host-driven stage boundary. Stage means accumulate in at
least float32 precision and round once to the callback rate dtype. The caller
is responsible for making each step invariant for the supplied target.
Omitting both callbacks selects the existing cloud-adaptive random-walk
Metropolis mutation. Pass ordinary callbacks rather than pre-jitting a
function that accepts the target callable.
The built-in tempering and SMC² proposals retain trace-relative jitter for
ill-conditioned clouds. If no positive factor survives in the parameter
dtype, they use a machine-epsilon variance floor with squared parameter units,
so an identical parameter population remains a valid input.
Proposal noise is now drawn explicitly in the particle dtype. On x64-enabled
runs this corrects the former promotion of float32 clouds and therefore
changes fixed-key proposals and acceptance paths, including for
well-conditioned covariances. Following
[NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html), smcx
treats that wrong-dtype result as a direct bug fix.

`target_ess` is a ratio in
`(0, 1 - numpy.finfo(numpy.float32).eps]` (an upper bound of approximately
`0.99999988`). This backend-independent cap leaves the ESS search one float32
machine epsilon below the uniform-cloud maximum. The schedule scales the
ratio by the ESS computed for the represented uniform log weights, which is
mathematically the particle count. This prevents backend reduction rounding
from placing an accepted target above the computed maximum.

Exact one is rejected before model callbacks run, including for constant
likelihoods. The cap removes exact one's categorical no-positive-increment
case for heterogeneous likelihoods; it does not guarantee a positive
represented increment or completion within a fixed stage budget for every
finite likelihood scale. Targets near the cap can require a larger
`max_stages`, so choose the ratio and stage budget together.

## Optional Equinox representation

If an application already uses Equinox, a callable module slots into
the model record directly — its parameters can live in ``params`` or
inside the module, whichever the application prefers. This example
targets `equinox==0.13.8`; Equinox is not an smcx dependency. See the
Equinox [Module documentation][equinox-module] for its PyTree
behavior.

```python
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


transition_module = LinearGaussianTransition(
    rho=jnp.asarray(0.95),
    process_scale=jnp.asarray(0.3),
)
model = smcx.StateSpaceModel(
    sample_initial=lambda key, params, input_0: jr.normal(key, (1,)),
    sample_transition=lambda key, state, params, input_t: params(key, state),
    log_observation=lambda emission, state, params, input_t: (
        -0.5 * ((emission[0] - state[0]) / 0.7) ** 2
    ),
)

emissions = jnp.asarray([[0.2], [-0.1], [0.4]])
posterior = smcx.run_smc(
    jr.key(0),
    smcx.bootstrap_fk(model, transition_module, emissions),
    num_particles=4_096,
)
```

Here the Equinox module itself is the ``params`` PyTree, so smcx
threads it explicitly and `jax.grad` with respect to the module's
arrays works through the filter. smcx ships no Equinox-specific
adapter. This example does not need one.

[equinox-module]: https://docs.kidger.site/equinox/api/module/module/
