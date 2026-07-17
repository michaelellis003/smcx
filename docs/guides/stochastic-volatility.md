# Stochastic volatility with online parameter learning

The filters in the [quickstart](quickstart.md) assume the model is
known. This guide relaxes that: we fit a stochastic-volatility model
whose mean log-variance is unknown and learn it *online*, updating a
posterior over the parameter at every step alongside the latent
volatility. The tool is the Liu-West filter, which augments the
particle state with the static parameter and controls the resulting
degeneracy with a shrinkage kernel.

## The model

Daily asset returns are near-uncorrelated but their variance moves
slowly and clusters. The canonical latent-volatility model writes
the log-variance as a stationary AR(1) and the return as Gaussian
with that variance,

$$
x_t = \mu + \phi\,(x_{t-1} - \mu) + \sigma\,\varepsilon_t, \qquad
y_t = \exp(x_t / 2)\,\eta_t,
$$

where $x_t$ is the log-variance at time $t$, and $\varepsilon_t,
\eta_t$ are independent standard normals. The persistence $\phi$ and
the volatility-of-volatility $\sigma$ we treat as known
($\phi = 0.95$, $\sigma = 0.25$); the long-run mean log-variance
$\mu$ is the unknown we learn. The observation is heteroskedastic but
its log-density is closed form,

$$
\log p(y_t \mid x_t)
  = -\tfrac{1}{2}\!\left(\log 2\pi + x_t + y_t^2\,e^{-x_t}\right).
$$

```python
import math
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import smcx

MU_TRUE, PHI, SIGMA = -0.5, 0.95, 0.25
LOG2PI = math.log(2 * math.pi)
T = 400


def simulate_sv(seed):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(MU_TRUE, SIGMA / math.sqrt(1 - PHI**2))
    for t in range(1, T):
        x[t] = MU_TRUE + PHI * (x[t - 1] - MU_TRUE) + SIGMA * rng.normal()
    y = np.exp(x / 2) * rng.normal(size=T)
    return x, y


x_true, y = simulate_sv(0)
emissions = jnp.asarray(y)[:, None]
```

We simulate with NumPy rather than `smcx.simulate` only to keep the
true latent path `x_true` for scoring at the end; the model closures
below are what the filter sees.

## Model closures with a parameter argument

The Liu-West closures take a third argument, `params`, holding one
parameter vector per particle. Here `params[0]` is that particle's
value of $\mu$. The transition and the observation read it; the
auxiliary function is the APF look-ahead, evaluating the observation
at the one-step-ahead mean of the transition to pre-weight particles
before they move.

```python
def initial_sampler(key, n):
    sd = SIGMA / math.sqrt(1 - PHI**2)
    return MU_TRUE + sd * jr.normal(key, (n, 1))


def transition_sampler(key, state, params):
    mu = params[0]
    mean = mu + PHI * (state[0] - mu)
    return mean + SIGMA * jr.normal(key, state.shape)


def log_observation_fn(y, state, params):
    x = state[0]
    return -0.5 * (LOG2PI + x + y[0] * y[0] * jnp.exp(-x))


def log_auxiliary_fn(y, state, params):
    mu = params[0]
    x_pred = mu + PHI * (state[0] - mu)
    return -0.5 * (LOG2PI + x_pred + y[0] * y[0] * jnp.exp(-x_pred))


def param_initial_sampler(key, n):
    # Diffuse prior over the mean log-variance: Uniform(-3, 1).
    return jr.uniform(key, (n, 1), minval=-3.0, maxval=1.0)
```

The prior on $\mu$ is deliberately vague — a uniform band four units
wide — so the concentration we see afterward is the data speaking,
not the prior.

## Run the filter

```python
post = smcx.liu_west_filter(
    jr.key(1),
    initial_sampler,
    transition_sampler,
    log_observation_fn,
    log_auxiliary_fn,
    param_initial_sampler,
    emissions,
    num_particles=20_000,
    shrinkage=0.97,
)
```

The one knob beyond the model is `shrinkage`, the Liu-West discount
$a \in (0, 1)$. Resampling the parameter alongside the state would
collapse the parameter cloud to a few duplicated values within a
handful of steps; Liu-West counters this by shrinking each resampled
parameter toward the ensemble mean and adding jitter of variance
$h^2 = 1 - a^2$ times the ensemble covariance. The construction is
variance-matched: it fights degeneracy without inflating the
marginal parameter posterior. Values near $0.97$–$0.99$ are typical;
smaller $a$ shrinks harder.

## Read the parameter posterior

`param_weighted_mean` and `param_weighted_quantile` summarize the
parameter cloud at every step, so we can watch the posterior form.

```python
mu_mean = np.array(smcx.param_weighted_mean(post))[:, 0]
mu_q = np.array(smcx.param_weighted_quantile(post, jnp.array([0.05, 0.95])))

print("true mu:", MU_TRUE)
print("posterior mean, first 3 steps:", mu_mean[:3].round(2))
print("posterior mean, last 3 steps:", mu_mean[-3:].round(2))
print("prior 90% width:", round(float(mu_q[0, 1, 0] - mu_q[0, 0, 0]), 2))
print("final 90% width:", round(float(mu_q[-1, 1, 0] - mu_q[-1, 0, 0]), 2))
print("final 90% interval:", mu_q[-1, :, 0].round(2))
```

The 90% interval starts near the full four-unit prior and narrows to
roughly one unit — about a fourfold concentration — with the true
$\mu = -0.5$ comfortably inside. The posterior mean does not sit
exactly on the truth, and it should not be expected to: with
$\phi = 0.95$ the log-variance mixes slowly, so four hundred
observations carry only modest information about its long-run mean,
and the Liu-West kernel adds a known, non-vanishing approximation
bias on top. The honest reading is the interval, not the point —
across independent simulations the 90% band covers $\mu$ reliably,
which is the behavior we want from an online parameter posterior.

## Score the volatility track

Learning the parameter is only useful if the state estimate holds up.
The filtered log-variance follows the latent path closely:

```python
vol_mean = np.array(smcx.weighted_mean(post))[:, 0]
corr = float(np.corrcoef(vol_mean, x_true)[0, 1])
print("filtered log-vol vs truth correlation:", round(corr, 2))
print("marginal loglik:", round(post.marginal_loglik.item(), 1))
```

A correlation around $0.7$ between the filtered and true
log-variance, from returns alone with the level parameter unknown,
is the filter earning its keep: it is tracking volatility and
locating $\mu$ at the same time, in one forward pass.

## Notes

- Liu-West is labeled approximate for a reason (the shrinkage bias
  above). When the parameter posterior matters more than online
  operation, an offline SMC sampler over the parameter — smcx's
  [`temper`](../api/) — trades the single pass for lower bias, and
  [`smc2`](../api/) nests a full particle filter inside it for exact
  pseudo-marginal parameter inference.
- The parameter here is unconstrained, which suits the Gaussian
  jitter. For a bounded parameter such as $\phi \in (-1, 1)$, learn
  it on an unconstrained scale (for instance $\operatorname{arctanh}
  \phi$) and transform inside the closures.
