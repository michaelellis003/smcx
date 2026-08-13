# smcx

> [!NOTE]
> **AI-assisted development.** smcx is developed with substantial assistance
> from generative-AI tools, including planning, implementation, test
> development, documentation, and review. The maintainer makes design and
> release decisions and remains responsible for the project, but not every
> line has been manually inspected by a human. Automated tests and numerical
> cross-checks provide evidence, not a guarantee of correctness; independently
> validate results important to your work.


Sequential inference for state-space models in JAX: Kalman-family
and DLM/DGLM filters, smoothers, and SMC methods including particle
filtering and smoothing, tempered SMC, IBIS, and SMC². Algorithms consume
plain JAX callables and small typed records, keeping model definitions
separate from inference. smcx defines no probabilistic programming
language.
Models defined elsewhere can be used when the caller maps their
components to these callables or records.

An introduction to the Kalman and SMC methods is developed in
[the documentation](https://michaelellis003.github.io/smcx/guides/sequential-inference/).
Below is a quick start and a map of the methods.

```bash
pip install smcx
```

## Quick start

A simple model to start with is a linear-Gaussian model

$$
\begin{aligned}
y_t &\sim \mathcal{N}(\theta_t,\ 0.3), \\
\theta_t &\sim \mathcal{N}(0.8\,\theta_{t-1},\ 0.2), \qquad
\theta_0 \sim \mathcal{N}(0, 1).
\end{aligned}
$$

In this case, assuming the only unknown is the latent state, we can
calculate the exact filtering distribution in closed form using the
Kalman filter. The Kalman filter assumes the model is linear and
Gaussian, so all you need to provide are the model parameters.

```python
import jax.numpy as jnp
import jax.random as jr

import smcx

# fmt: off
y = jnp.array([
    -0.54, -1.09, -0.77, -0.03, 0.92, -0.45, 1.19, 0.24, 1.13,
    -0.42, 0.63, 1.18, 1.13, 0.64, 1.35, 2.25, 1.98, 1.65, 2.01,
    1.63, 0.80, 0.39, -0.68, -0.87, -0.96,
])[:, None]
# fmt: on

m0 = jnp.zeros(1)
C0 = jnp.eye(1)
G = 0.8 * jnp.eye(1)
W = 0.2 * jnp.eye(1)
F = jnp.eye(1)
V = 0.3 * jnp.eye(1)

kalman = smcx.kalman_filter(
    initial_mean=m0,
    initial_covariance=C0,
    transition_matrix=G,
    transition_covariance=W,
    observation_matrix=F,
    observation_covariance=V,
    emissions=y,
)
print(kalman.marginal_loglik)  # -29.26, exact
```

A particle filter instead needs a Markov state-space model given by
three functions: a sampler for the initial law, a sampler for the
transition, and an evaluable observation log density. Here is the
same model through the bootstrap particle filter:

```python
def sample_initial(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def sample_transition(key, state):
    return 0.8 * state + jnp.sqrt(0.2) * jr.normal(key, state.shape)


def log_observation(obs, state):
    residual = obs[0] - state[0]
    return -0.5 * (jnp.log(2 * jnp.pi * 0.3) + residual**2 / 0.3)


particle = smcx.bootstrap_filter(
    jr.key(0),
    sample_initial,
    sample_transition,
    log_observation,
    y,
    num_particles=10_000,
)
print(particle.marginal_loglik)  # -29.16, N = 10,000
```

The particle estimate approximates the exact Kalman value. At this
key and N = 10,000 the two log-likelihoods differ by 0.10. This
single key does not characterize Monte Carlo error. Repeated keys
are needed to estimate the bias and spread of the log-likelihood
error at N = 10,000. If our model leaves the linear-Gaussian
family, we can no longer use the Kalman filter. We only change the
three functions of the bootstrap call to the new densities. The
table below maps each model class to its methods, and the
[introduction in the documentation](https://michaelellis003.github.io/smcx/guides/sequential-inference/)
develops the theory with four worked examples, relaxing one
assumption at a time.

## Methods

Filtering conditions each state on the observations up to its own
time and runs online. Smoothing revisits every state once the
complete record is in hand: a backward pass consumes the stored
filter output, so `rts_smoother` takes the result of
`kalman_filter` rather than rerunning it. smcx implements the
standard sequential inference methods:

| Setting | Methods | Functions |
| --- | --- | --- |
| Linear-Gaussian, fully known | Kalman filter, RTS smoother, lag-one covariances, and joint posterior draws, exact | `kalman_filter`, `rts_smoother`, `smoothed_cross_covariances`, `posterior_sample` |
| Known nonlinear functions | Extended and unscented Kalman filters and RTS smoothers, approximate; the linearization strategy is an argument | `extended_kalman_filter`, `unscented_kalman_filter`, `gaussian_filter`, `gaussian_smoother` |
| Observation variance unknown, variance-scaled | Conjugate DLM filter and retrospective smoother, exact | `dlm_filter`, `dlm_smoother` |
| Count and binary observations | Conjugate/linear-Bayes DGLM filtering and retrospective state-moment smoothing, approximate | `dglm_filter` with `poisson()`, `bernoulli()`, or `binomial(trials=n)`; `dglm_smoother` |
| General densities | Bootstrap, auxiliary, and guided particle filters | `bootstrap_filter`, `auxiliary_filter`, `guided_filter` |
| General densities, retrospective | Genealogy paths and particle FFBS trajectory draws, approximate | `reconstruct_trajectories`, `backward_simulation` |
| Custom particle algorithms | Feynman–Kac derivations over one generic loop | `StateSpaceModel`, `FeynmanKac`, `run_smc`, `run_particle_filter` |
| Static parameters | Tempered SMC targets a fixed posterior through a temperature path. IBIS updates an exact-likelihood posterior in data order. SMC² nests a particle filter inside parameter-space SMC. Liu-West is approximate online parameter learning through kernel shrinkage | `temper`, `ibis`, `smc2`, `liu_west_filter` |
| Simulation and prediction | Model simulation and posterior predictive draws | `simulate`, `posterior_predictive_sample` |
| Resampling | Systematic, stratified, multinomial, residual | `systematic`, `stratified`, `multinomial`, `residual` |
| Diagnostics and reporting | ESS, scoring rules, ArviZ export | `diagnose`, `crps`, `to_arviz` |

smcx runs on CPU, CUDA, and TPU through JAX, and on Apple-silicon
GPUs through the optional
[jax-mps](https://github.com/tillahoffmann/jax-mps) backend.

## Installation

smcx requires Python 3.11 or later.

```bash
pip install smcx
```

Optional extras add Apple-silicon GPU execution or ArviZ reporting:

```bash
pip install "smcx[metal]"
pip install "smcx[arviz]"
```

## Documentation

Available at
[michaelellis003.github.io/smcx](https://michaelellis003.github.io/smcx/).

## Citation

If smcx contributes to academic work, please cite the release used.
The repository's **Cite this repository** menu uses
[`CITATION.cff`](https://github.com/michaelellis003/smcx/blob/main/CITATION.cff)
to provide BibTeX and APA entries; include the version and release
date in the final citation.

## See also

**State-space models and SMC**

- [dynamax](https://github.com/probml/dynamax): probabilistic state-space
models with learning via EM and SGD.
- [dynestyx](https://github.com/BasisResearch/dynestyx): NumPyro-based
inference for dynamical systems.
- [particles](https://github.com/nchopin/particles): the reference
Python companion to Chopin and Papaspiliopoulos (2020).
- [BlackJAX](https://github.com/blackjax-devs/blackjax): MCMC and SMC
samplers for JAX.

**The JAX ecosystem**

- [Equinox](https://github.com/patrick-kidger/equinox): neural networks
and PyTree modules.
- [Diffrax](https://github.com/patrick-kidger/diffrax): numerical
differential equation solvers.
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping): shape and
dtype annotations for arrays.
- [ArviZ](https://github.com/arviz-devs/arviz): exploratory analysis of
Bayesian models.

## Sources and attribution

The broader Feynman–Kac architecture follows Chopin and
Papaspiliopoulos's
[*An Introduction to Sequential Monte Carlo*](https://doi.org/10.1007/978-3-030-47845-2).
The caller-owned particle-filter runner and dependency-free tempering
mutation boundary were informed by BlackJAX's functional state/information
protocol and pinned
[SMC-from-MCMC split](https://github.com/blackjax-devs/blackjax/blob/a9ef478c69d730a2caa13ca4b2d735c580e0feec/blackjax/smc/from_mcmc.py),
and by the separation of orchestration from history in
[particles 0.4](https://github.com/nchopin/particles/releases/tag/v0.4).
These are design credits; no code was copied or translated.
The implemented methods draw on these primary sources:

- Exact linear-Gaussian state estimation:
  [Kalman (1960)](https://doi.org/10.1115/1.3662552) and
  [Rauch, Tung, and Striebel (1965)](https://doi.org/10.2514/3.3166), with
  joint draws from [Carter and Kohn (1994)](https://doi.org/10.1093/biomet/81.3.541)
  and [Frühwirth-Schnatter (1994)](https://doi.org/10.1111/j.1467-9892.1994.tb00184.x).
- Conjugate dynamic models:
  [West and Harrison (1997)](https://doi.org/10.1007/b98971) and
  [West, Harrison, and Migon (1985)](https://doi.org/10.1080/01621459.1985.10477131),
  with retrospective state moments from
  [Alves et al. (2025)](https://arxiv.org/html/2201.05387v4#S3.SS2).
- Nonlinear Gaussian filtering:
  [Schmidt (1966)](https://doi.org/10.1016/B978-1-4831-6716-9.50011-4) and
  [Julier (2002)](https://doi.org/10.1109/ACC.2002.1025369).
- Particle filters and smoothing:
  [Gordon, Salmond, and Smith (1993)](https://doi.org/10.1049/ip-f-2.1993.0015),
  [Pitt and Shephard (1999)](https://doi.org/10.1080/01621459.1999.10474153),
  [Doucet, Godsill, and Andrieu (2000)](https://doi.org/10.1023/A:1008935410038),
  [Godsill, Doucet, and West (2004)](https://doi.org/10.1198/016214504000000151),
  and [Liu and West (2001)](https://doi.org/10.1007/978-1-4757-3437-9_10).
- Static and parameter inference:
  [Chopin (2002)](https://doi.org/10.1093/biomet/89.3.539),
  [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x),
  and [Chopin, Jacob, and Papaspiliopoulos (2013)](https://doi.org/10.1111/j.1467-9868.2012.01046.x).
- Resampling and diagnostics:
  [Douc, Cappé, and Moulines (2005)](https://doi.org/10.1109/ISPA.2005.195385),
  [Lee and Whiteley (2018)](https://doi.org/10.1093/biomet/asy028),
  [Zhang and Stephens (2009)](https://doi.org/10.1198/TECH.2009.08017),
  and [Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html).
- Scoring rules:
  [Matheson and Winkler (1976)](https://doi.org/10.1287/mnsc.22.10.1087)
  and [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437).
- Reporting: [ArviZ](https://doi.org/10.21105/joss.01143).

## Contributing

Contributions are welcome. See
[`CONTRIBUTING.md`](https://github.com/michaelellis003/smcx/blob/main/CONTRIBUTING.md)
for the development setup and pull-request conventions.

## License

smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
