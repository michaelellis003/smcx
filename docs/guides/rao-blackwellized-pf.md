# Rao-Blackwellized particle filtering

A conditionally linear-Gaussian model has a nonlinear latent
component and, given its path, a linear-Gaussian substructure. The
Rao-Blackwellized particle filter runs particles only over the
nonlinear component and an exact Kalman recursion per particle over
the linear part — marginalizing analytically whatever can be
marginalized, which is where the variance reduction comes from
(Doucet, de Freitas, Murphy, and Russell 2000; Chen and Liu 2000).

smcx expresses this without any dedicated machinery, because
particle states are arbitrary PyTrees and every callback is
per-particle: a particle carries its own Kalman mean and covariance,
the transition advances them, and the observation potential returns
the per-particle Gaussian predictive density. One wiring trick makes
the Kalman update fit the bootstrap contract: the transition
callback needs the previous observation to complete the previous
step's update, and the inputs channel supplies it — feed the
emissions again, lagged by one row.

The example is a two-regime switching model: a stable AR(1) state
whose innovation scale depends on a hidden Markov regime, observed
in Gaussian noise. Particles track the regime; each particle's
Kalman pair tracks the state exactly given that regime path.

```python
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import smcx

A_COEFF, R_SD = 0.98, 0.5
Q_SDS = jnp.asarray([0.1, 1.0])
STAY = 0.95

rng = np.random.default_rng(7)
T = 60
regimes = [0]
for _ in range(T - 1):
    regimes.append(regimes[-1] if rng.uniform() < STAY else 1 - regimes[-1])
x = [rng.normal()]
for t in range(1, T):
    x.append(A_COEFF * x[-1] + float(Q_SDS[regimes[t]]) * rng.normal())
y = np.asarray(x) + R_SD * rng.normal(size=T)
observations = jnp.asarray(y)[:, None]
lagged = jnp.concatenate([jnp.zeros((1,)), jnp.asarray(y)[:-1]])[:, None]
```

`lagged[t]` is $y_{t-1}$ (row zero is a placeholder that
initialization ignores), so the input-aware transition at time $t$
can finish step $t-1$'s Kalman update before predicting.

```python
def initial_sampler(key, n, input_0):
    return {
        "regime": jr.bernoulli(key, 0.5, (n,)).astype(jnp.float32),
        "mean": jnp.zeros((n,)),
        "cov": jnp.ones((n,)),
    }


def transition_sampler(key, state, input_t):
    gain = state["cov"] / (state["cov"] + R_SD**2)
    mean_updated = state["mean"] + gain * (input_t[0] - state["mean"])
    cov_updated = (1.0 - gain) * state["cov"]
    switch = jr.bernoulli(key, 1.0 - STAY)
    regime = jnp.where(switch, 1.0 - state["regime"], state["regime"])
    q_sd = Q_SDS[regime.astype(jnp.int32)]
    return {
        "regime": regime,
        "mean": A_COEFF * mean_updated,
        "cov": A_COEFF**2 * cov_updated + q_sd**2,
    }


def log_observation_fn(emission, state, input_t):
    variance = state["cov"] + R_SD**2
    return -0.5 * (
        (emission[0] - state["mean"]) ** 2 / variance
        + jnp.log(2.0 * jnp.pi * variance)
    )


posterior = smcx.bootstrap_filter(
    jr.key(5),
    initial_sampler,
    transition_sampler,
    log_observation_fn,
    observations,
    512,
    inputs=lagged,
)
print("RBPF marginal loglik:", round(float(posterior.marginal_loglik), 3))
```

Each particle's `mean` and `cov` are its predicted Kalman moments,
so the observation potential is the exact one-step predictive
density given the particle's regime path — the Rao-Blackwellized
weight.

## Validation one: collapse to the exact filter

Pin the regime chain to one state and the marginalization must
reproduce `kalman_filter` exactly — every particle carries the same
deterministic Kalman recursion, so the evidence estimate has zero
Monte Carlo error:

```python
def initial_pinned(key, n, input_0):
    out = initial_sampler(key, n, input_0)
    out["regime"] = jnp.zeros_like(out["regime"])
    return out


def transition_pinned(key, state, input_t):
    out = transition_sampler(key, state, input_t)
    gain = state["cov"] / (state["cov"] + R_SD**2)
    cov_updated = (1.0 - gain) * state["cov"]
    out["regime"] = jnp.zeros_like(out["regime"])
    out["cov"] = A_COEFF**2 * cov_updated + Q_SDS[0] ** 2
    return out


pinned = smcx.bootstrap_filter(
    jr.key(5),
    initial_pinned,
    transition_pinned,
    log_observation_fn,
    observations,
    8,
    inputs=lagged,
)
exact = smcx.kalman_filter(
    jnp.zeros(1),
    jnp.eye(1),
    jnp.asarray([[A_COEFF]]),
    jnp.asarray([[float(Q_SDS[0]) ** 2]]),
    jnp.eye(1),
    jnp.asarray([[R_SD**2]]),
    observations,
)
gap = abs(float(pinned.marginal_loglik) - float(exact.marginal_loglik))
assert gap < 1e-4
print("pinned-vs-exact gap:", round(gap, 8))
```

## Validation two: the variance reduction

Against a plain bootstrap filter over both components — the regime
and a sampled state — at the same particle count, the
Rao-Blackwellized evidence estimate is markedly steadier. The
comparison replicates both filters over independent keys:

```python
def initial_plain(key, n):
    k1, k2 = jr.split(key)
    return {
        "regime": jr.bernoulli(k1, 0.5, (n,)).astype(jnp.float32),
        "x": jr.normal(k2, (n,)),
    }


def transition_plain(key, state):
    k1, k2 = jr.split(key)
    switch = jr.bernoulli(k1, 1.0 - STAY)
    regime = jnp.where(switch, 1.0 - state["regime"], state["regime"])
    q_sd = Q_SDS[regime.astype(jnp.int32)]
    return {
        "regime": regime,
        "x": A_COEFF * state["x"] + q_sd * jr.normal(k2, state["x"].shape),
    }


def log_observation_plain(emission, state):
    return -0.5 * (
        (emission[0] - state["x"]) ** 2 / R_SD**2
        + jnp.log(2.0 * jnp.pi * R_SD**2)
    )


history = []
plain_history = []
for i in range(24):
    history.append(
        float(
            smcx.bootstrap_filter(
                jr.key(100 + i),
                initial_sampler,
                transition_sampler,
                log_observation_fn,
                observations,
                512,
                inputs=lagged,
            ).marginal_loglik
        )
    )
    plain_history.append(
        float(
            smcx.bootstrap_filter(
                jr.key(100 + i),
                initial_plain,
                transition_plain,
                log_observation_plain,
                observations,
                512,
            ).marginal_loglik
        )
    )
rb_sd = float(np.std(history, ddof=1))
plain_sd = float(np.std(plain_history, ddof=1))
assert rb_sd < plain_sd
print(
    "loglik sd over 24 keys - RBPF:",
    round(rb_sd, 3),
    "plain:",
    round(plain_sd, 3),
)
```

At 512 particles over 24 replicate keys, both estimators agree on
the evidence while the Rao-Blackwellized standard deviation (0.083)
is under a third of the plain filter's (0.295) — more than a factor
of ten in variance, obtained by marginalizing one scalar dimension.
The gap widens with the dimension of the linear block: everything
the Kalman recursion absorbs is variance the particles no longer
pay for.

The same construction extends to vector linear blocks by carrying
per-particle mean vectors and covariance matrices in the PyTree, and
composes with `forecast_sample` and the diagnostics unchanged —
the posterior is an ordinary `ParticleFilterPosterior` whose state
happens to contain sufficient statistics rather than samples.
