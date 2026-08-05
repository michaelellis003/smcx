# Estimating linear-Gaussian parameters

smcx is the inference engine, not a model zoo: models enter as arrays
and callables the caller owns, so parameter estimation is a recipe
rather than a `fit` method. Every ingredient already exists —
`kalman_filter` returns a differentiable `marginal_loglik`,
`rts_smoother` returns smoothed moments, and
`smoothed_cross_covariances` supplies the adjacent-state expectations
that complete an EM E-step. This page fits one model both ways and
checks that the answers agree.

The example is the local level model with unknown variances: the
state is a random walk with innovation standard deviation
$\sigma_q = 0.3$, observed with noise standard deviation
$\sigma_r = 0.7$, for 300 steps.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import smcx

TRUE_Q_SD, TRUE_R_SD = 0.3, 0.7
states, observations = smcx.simulate(
    jr.key(11),
    lambda key: jr.normal(key, (1,)),
    lambda key, state: state + TRUE_Q_SD * jr.normal(key, (1,)),
    lambda key, state: state + TRUE_R_SD * jr.normal(key, (1,)),
    num_timesteps=300,
)


def build(q_var, r_var):
    return smcx.LinearGaussianModel(
        initial_mean=jnp.zeros(1),
        initial_covariance=jnp.eye(1),
        transition_matrix=jnp.eye(1),
        transition_covariance=jnp.reshape(q_var, (1, 1)),
        observation_matrix=jnp.eye(1),
        observation_covariance=jnp.reshape(r_var, (1, 1)),
    )
```

`build` closes the gap between free parameters and the model record;
because the record is a PyTree of arrays, gradients flow through its
leaves without ceremony.

## Direct maximum likelihood

The filter's `marginal_loglik` is the exact log-likelihood, so
maximum likelihood is `jax.grad` plus any optimizer. Two practical
points matter more than the choice of optimizer. Work with
log-variances, so that every gradient step stays inside the positive
cone — on the raw scale a single step can produce a negative
variance and the filter will rightly reject it. And scale the
learning rate by the series length, since the log-likelihood grows
linearly with $T$. The loop below is plain momentum gradient
descent; swap in an optax optimizer if you already depend on one —
these docs stay dependency-free.

```python
def negative_loglik(params):
    log_q, log_r = params
    model = build(jnp.exp(log_q), jnp.exp(log_r))
    return -smcx.kalman_filter(model, observations).marginal_loglik


grad_fn = jax.jit(jax.value_and_grad(negative_loglik))
params = jnp.log(jnp.asarray([0.25, 0.25]))
velocity = jnp.zeros(2)
for _ in range(400):
    loss, gradient = grad_fn(params)
    velocity = 0.9 * velocity - 0.02 * gradient / observations.shape[0]
    params = params + velocity

ml_q_sd, ml_r_sd = (float(jnp.exp(0.5 * value)) for value in params)
print("ML estimates:", round(ml_q_sd, 3), round(ml_r_sd, 3))
```

This prints `ML estimates: 0.293 0.656` against the simulating truth
of 0.3 and 0.7 — the gap is sampling variability at $T = 300$, not
optimizer error.

## Expectation-maximization

The E-step is one filter-smoother pass; with the smoothed means
$\hat m_t$, variances, and adjacent cross-covariances in hand, the
M-step for both variances is closed form. Writing
$s_t = \operatorname{E}[x_t^2 \mid y_{1:T}]$ and
$p_t = \operatorname{E}[x_t x_{t-1} \mid y_{1:T}]$, the updates are
the smoothed mean squares of the innovations and residuals:

$$
\hat\sigma_q^2 = \frac{1}{T-1}\sum_{t=2}^{T}
\bigl(s_t + s_{t-1} - 2 p_t\bigr),
\qquad
\hat\sigma_r^2 = \frac{1}{T}\sum_{t=1}^{T}
\bigl(y_t^2 - 2 y_t \hat m_t + s_t\bigr).
$$

The initial moments stay fixed here; estimating them too adds one
more closed-form update but little insight. The correctness gate is
monotonicity — every EM iteration must not decrease the
log-likelihood, and the loop asserts it.

```python
def em_step(q_var, r_var):
    model = build(q_var, r_var)
    filtered = smcx.kalman_filter(model, observations)
    smoothed = smcx.rts_smoother(filtered, model)
    cross = smcx.smoothed_cross_covariances(smoothed, model)
    means = smoothed.smoothed_means[:, 0]
    second = smoothed.smoothed_covariances[:, 0, 0] + means**2
    pair = cross[:, 0, 0] + means[1:] * means[:-1]
    q_new = jnp.mean(second[1:] + second[:-1] - 2.0 * pair)
    y = observations[:, 0]
    r_new = jnp.mean(y**2 - 2.0 * y * means + second)
    return float(q_new), float(r_new), float(filtered.marginal_loglik)


q_var, r_var = 0.25, 0.25
history = []
for _ in range(60):
    q_var, r_var, loglik = em_step(q_var, r_var)
    history.append(loglik)

assert all(
    later >= earlier - 1e-8 for earlier, later in zip(history, history[1:])
)
print("EM estimates:", round(q_var**0.5, 3), round(r_var**0.5, 3))
```

This prints `EM estimates: 0.291 0.656` — within 0.002 of the
gradient answer on both parameters, the same maximum approached from
different directions. Automating that cross-method agreement is the
cheapest correctness check available for your own fits.

## Which to use

EM needs no learning rate and every iteration provably improves, but
its steps shrink near the optimum: after the sixty sweeps above the
estimates are still moving in the third decimal, and high-precision
convergence takes hundreds more. Gradient methods reach the same optimum in
fewer passes once tuned, but the tuning is real work — the
log-variance reparameterization is not optional, and a learning rate
that ignores the $T$-scaling of the likelihood diverges. A pragmatic
default: a few EM sweeps for a stable neighborhood, then gradients
for polish — or either alone when the model is as small as this one.

Both recipes extend to full matrices: the M-step generalizations are
in Särkkä and Svensson (2023, chapter 16), and the gradient route
needs only a log-Cholesky reparameterization of each covariance in
`build`.
