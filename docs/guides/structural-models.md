# Composing structural models by superposition

West and Harrison's structural models are built by superposition: a
trend, a seasonal pattern, and a regression effect each contribute a
small state-space block, and the full model is their block-diagonal
sum — the composed state is the concatenation of component states,
and the observation row is the concatenation of component observation
rows (W&H chapter 6). smcx owns none of this: the engine sees exactly
one finished `LinearGaussianModel` of arrays you assemble, and this
page shows the assembly. The same stacked arrays can be transcribed
from any source — a model specified on paper, or another tool's
fitted parameters — so what follows is a recipe for bringing a
structural model *to* the engine, not a modeling layer inside it.

The example series has a slowly drifting level, a quarterly seasonal
pattern, and one regression covariate:

```python
import jax.numpy as jnp
import numpy as np
import smcx


def bdiag(*blocks):
    size = sum(block.shape[0] for block in blocks)
    out = jnp.zeros((size, size))
    start = 0
    for block in blocks:
        stop = start + block.shape[0]
        out = out.at[start:stop, start:stop].set(block)
        start = stop
    return out


rng = np.random.default_rng(21)
T = 160
level = np.cumsum(0.15 * rng.normal(size=T)) + 2.0
season_pattern = np.array([1.2, -0.4, -1.1, 0.3])
seasonal = np.tile(season_pattern, T // 4 + 1)[:T]
covariate = rng.normal(size=T)
beta_true = 0.8
y = level + seasonal + beta_true * covariate + 0.5 * rng.normal(size=T)
observations = jnp.asarray(y)[:, None]
```

Each component is a few concrete arrays. The local level is a random
walk. The seasonal block carries the last three seasonal effects and
forces them to sum to zero over a cycle; only its first state
receives evolution noise. The regression coefficient is a state with
(almost) no evolution noise, observed through the covariate — which
makes the observation row time-varying:

```python
G_level = jnp.asarray([[1.0]])
W_level = jnp.asarray([[0.15**2]])

G_seasonal = jnp.asarray([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
W_seasonal = jnp.zeros((3, 3)).at[0, 0].set(1e-6)

G_regression = jnp.asarray([[1.0]])
W_regression = jnp.asarray([[1e-8]])

transition = bdiag(G_level, G_seasonal, G_regression)
evolution = bdiag(W_level, W_seasonal, W_regression)
observation_rows = jnp.concatenate(
    [
        jnp.ones((T, 1)),
        jnp.tile(jnp.asarray([1.0, 0.0, 0.0]), (T, 1)),
        jnp.asarray(covariate)[:, None],
    ],
    axis=1,
)[:, None, :]
```

Superposition is nothing but this stacking — the composed transition
is literally the block-diagonal matrix you could write by hand, and
asserting that keeps the recipe honest:

```python
transition_by_hand = jnp.asarray([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])
np.testing.assert_array_equal(np.asarray(transition), transition_by_hand)
```

The composed model filters, smooths, and forecasts through the
ordinary entry points, and the smoothed state decomposes the series
back into its components — the level in state 0, the current
seasonal effect in state 1, and the regression coefficient in
state 4:

```python
model = smcx.LinearGaussianModel(
    initial_mean=jnp.zeros(5),
    initial_covariance=bdiag(
        4.0 * jnp.eye(1), 4.0 * jnp.eye(3), 4.0 * jnp.eye(1)
    ),
    transition_matrix=transition,
    transition_covariance=evolution,
    observation_matrix=observation_rows,
    observation_covariance=jnp.asarray([[0.5**2]]),
)
posterior = smcx.kalman_filter(model, observations)
smoothed = smcx.rts_smoother(posterior, model)

print("beta estimate:", round(float(smoothed.smoothed_means[-1, 4]), 3))
print(
    "seasonal pattern:",
    np.round(np.asarray(smoothed.smoothed_means[-4:, 1]), 2),
)
level_rmse = float(
    np.sqrt(np.mean((np.asarray(smoothed.smoothed_means[:, 0]) - level) ** 2))
)
print("level RMSE:", round(level_rmse, 3))
```

Against the simulating truth this prints a coefficient estimate of
0.822 for a true 0.8, recovers the quarterly pattern
(1.2, -0.4, -1.1, 0.3) as (1.23, -0.35, -1.14, 0.26), and tracks the
level with root mean squared error 0.226 — each component identified
from the one observed sum.

Forecasting needs the future observation rows, which for a
regression component means the future covariate values:

```python
future_covariate = rng.normal(size=4)
future_rows = jnp.concatenate(
    [
        jnp.ones((4, 1)),
        jnp.tile(jnp.asarray([1.0, 0.0, 0.0]), (4, 1)),
        jnp.asarray(future_covariate)[:, None],
    ],
    axis=1,
)[:, None, :]
forecast = smcx.kalman_forecast(
    posterior,
    transition,
    evolution,
    future_rows,
    jnp.asarray([[0.5**2]]),
    num_steps=4,
)
print(
    "4-step forecast means:",
    np.round(np.asarray(forecast.observation_means[:, 0]), 2),
)
```

Nothing here is special to these three components: an extra
covariate is one more column, a second seasonal cycle is one more
block, and a local linear trend replaces `G_level` with the familiar
2-by-2. The composition stays yours — smcx neither builds nor
combines models, and any proposal to move helpers like `bdiag` into
the library goes through the model-free boundary with its own
decision record.
