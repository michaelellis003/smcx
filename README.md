# smcx

Sequential inference in JAX: Kalman filters, particle filters, and
sequential Monte Carlo, with model code decoupled from inference code.

## A short tour of sequential inference

Start where everything is exact. A linear-Gaussian state-space model
moves a latent state $x_t$ by a matrix and measures it through noise:

$$
x_0 \sim \mathcal{N}(m_0, P_0), \qquad
x_t = A x_{t-1} + q_t, \qquad y_t = H x_t + r_t,
\qquad q_t \sim \mathcal{N}(0, Q), \quad r_t \sim \mathcal{N}(0, R),
$$

with independent noises. As observations arrive, the posterior
distribution of interest is the filtering distribution
$p(x_t \mid y_{1:t})$ — the current state given every observation so
far. Here it is Gaussian
([Durbin and Koopman, 2012](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)),
and it updates by a cycle in which each posterior becomes the prior
for the next observation: predict the state one step ahead through
$A$, then update on $y_t$. Both steps are a handful of matrix
operations: the Kalman filter
([Kalman, 1960](https://doi.org/10.1115/1.3662552)). Exact, and cheap
in any moderate state dimension.

Now relax the maps. Let the state move and be measured through
nonlinear functions,

$$
x_t = f(x_{t-1}) + q_t, \qquad y_t = h(x_t) + r_t,
$$

and the filtering distribution, in general, stops being Gaussian. A
Gaussian approximation of the same cycle survives: linearize $f$ and
$h$ locally (the extended Kalman filter), or propagate a handful of
sigma points through them (the unscented Kalman filter)
([Särkkä and Svensson, 2023](https://doi.org/10.1017/9781108917407)).

Now relax the distributions themselves. Discrete counts, bounded or
heavy-tailed measurements, multiplicative noise: in general the model
is a Markov state with conditionally independent observations,
specified by a state density and an observation density,

$$
x_t \mid x_{t-1} \sim p(\,\cdot \mid x_{t-1}), \qquad
y_t \mid x_t \sim p(\,\cdot \mid x_t),
$$

and sequential Bayesian inference rests on two relations — the
one-step-ahead forecast density, then Bayes' theorem turning it into
the next filtering distribution:

$$
p(x_t \mid y_{1:t-1}) = \int p(x_t \mid x_{t-1})\,
p(x_{t-1} \mid y_{1:t-1})\,dx_{t-1},
\qquad
p(x_t \mid y_{1:t}) \propto p(y_t \mid x_t)\,
p(x_t \mid y_{1:t-1}).
$$

Beyond the linear-Gaussian and finite-state cases, no closed form
computes that integral. A particle filter maintains a Monte Carlo
approximation instead: the filtering distribution is represented by
$N$ particles and their importance weights,

$$
p(dx_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)}\,
\delta_{x_t^{(i)}}(dx_t),
\qquad \sum_{i=1}^{N} w_t^{(i)} = 1,
$$

with each particle propagated through the state density, the update
paid in weight, and resampling curbing weight degeneracy
([Gordon, Salmond, and Smith, 1993](https://doi.org/10.1049/ip-f-2.1993.0015)),
while auxiliary resampling that looks one observation ahead
([Pitt and Shephard, 1999](https://doi.org/10.1080/01621459.1999.10474153))
and observation-informed proposals sharpen the approximation.

One relaxation remains. Suppose the parameters — the transition
coefficient, or the covariances $Q$ and $R$ themselves — are unknown,
so the model gains a prior:

$$
\theta \sim p(\theta), \qquad
x_t \mid x_{t-1} \sim p_\theta(\,\cdot \mid x_{t-1}), \qquad
y_t \mid x_t \sim p_\theta(\,\cdot \mid x_t).
$$

A static $\theta$ couples every time step to every other. In special
cases the exact posterior can still be updated by conjugate Bayesian
inference — a linear-Gaussian model whose one unknown variance scales
the evolution noise as well as the observation noise updates exactly
on two summary statistics
([West and Harrison, 1997](https://doi.org/10.1007/b98971), ch. 4) —
but for the general state-space model no fixed set of summary
statistics carries $p(x_t, \theta \mid y_{1:t})$ forward
([Kantas et al., 2015](https://doi.org/10.1214/14-STS511)). Nor does
the augmented state $(x_t, \theta)$ rescue the particle filter: the
recursion itself still holds, yet a static $\theta$ is never
refreshed by the state density, so the cloud's $\theta$-support only
ever shrinks. Sequential Monte Carlo answers with the same
weighted-particle machinery, aimed at any sequence of distributions
([Chopin and Papaspiliopoulos, 2020](https://doi.org/10.1007/978-3-030-47845-2)):
rejuvenate $\theta$ online beside the states
([Liu and West, 2001](https://doi.org/10.1007/978-1-4757-3437-9_10)),
run a filter for every parameter particle
(SMC²: [Chopin, Jacob, and Papaspiliopoulos, 2013](https://doi.org/10.1111/j.1467-9868.2012.01046.x)),
or leave time behind entirely and anneal from prior to posterior
through a ladder of temperatures
([Del Moral, Doucet, and Jasra, 2006](https://doi.org/10.1111/j.1467-9868.2006.00553.x)).

smcx implements this ladder end to end:

- exact linear-Gaussian Kalman filtering and RTS smoothing;
- extended and unscented Kalman filtering, unified by
  `gaussian_filter` over exchangeable linearization strategies;
- bootstrap, auxiliary, guided, and Liu–West particle filters;
- adaptive tempered SMC and SMC² for static-parameter inference;
- systematic, stratified, multinomial, and residual resampling;
- filtering diagnostics, scoring rules, trajectory reconstruction,
  and ArviZ export.

It runs on CPU, CUDA, and TPU through JAX, and on Apple-silicon GPUs
through the optional [jax-mps](https://github.com/tillahoffmann/jax-mps)
backend.

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

## Quick example

The tour ends at non-Gaussian observations and unknown parameters, so
here is a model with both: a latent log-intensity driving Poisson
counts. No Kalman variant can run it. A model is a record of pure
functions; its parameters are an ordinary PyTree that smcx threads
through every call.

```python
import jax
import jax.numpy as jnp
import jax.random as jr

import smcx

model = smcx.StateSpaceModel(
    sample_initial=lambda key, params, input_0: jr.normal(key, (1,)),
    sample_transition=lambda key, state, params, input_t: (
        params["rho"] * state + params["sigma"] * jr.normal(key, state.shape)
    ),
    log_observation=lambda emission, state, params, input_t: (
        emission[0] * state[0] - jnp.exp(state[0])
    ),
)
params = {"rho": jnp.asarray(0.9), "sigma": jnp.asarray(0.4)}
counts = jnp.asarray([[1], [0], [2], [1], [3]], dtype=jnp.int32)


def log_marginal(params):
    fk = smcx.bootstrap_fk(model, params, counts)
    return smcx.run_smc(jr.key(0), fk, num_particles=4_096).marginal_loglik


score = jax.grad(log_marginal)(params)
```

Here `bootstrap_fk` derives a Feynman–Kac model — the algorithm-facing
object — from the state-space model, and `run_smc` runs the one
generic resample–mutate–reweight loop underneath every particle filter
in the library. Because the parameters are an explicit argument rather
than something baked into the model, `jax.grad` differentiates the
marginal likelihood estimate with respect to $\rho$ and $\sigma$ in
one call. (The estimator's gradient is biased by the
non-differentiable resampling step — often tolerable for
optimization, and worth knowing about.)

## Design: model code is decoupled from inference code

A goal smcx shares with
[dynestyx](https://github.com/BasisResearch/dynestyx): modellers write
a model once and get every applicable inference method for free, while
methods researchers get a platform where a new algorithm slots in
beside established, independently validated ones. In smcx the boundary is concrete:

- a model is a `StateSpaceModel` — a record of pure JAX callables,
  with no base class, no distribution objects, and no framework;
- every particle algorithm is a `FeynmanKac` derivation over one
  `run_smc` loop, so a custom method reuses the resampling machinery,
  evidence accounting, diagnostics, and export;
- the components a researcher wants to swap are all arguments:
  resamplers, resampling criteria, proposals, potentials, look-ahead
  twists, mutation kernels, temperature schedules, and Gaussian
  linearization strategies.

## Citation

If smcx contributes to academic work, please cite the release used.
The repository's **Cite this repository** menu uses
[`CITATION.cff`](https://github.com/michaelellis003/smcx/blob/main/CITATION.cff)
to provide BibTeX and APA entries; include the version and release
date in the final citation.

## See also

**State-space models and SMC**

[dynamax](https://github.com/probml/dynamax): probabilistic state-space
models with learning via EM and SGD.
[dynestyx](https://github.com/BasisResearch/dynestyx): NumPyro-based
inference for dynamical systems.
[particles](https://github.com/nchopin/particles): the reference
Python companion to Chopin and Papaspiliopoulos (2020).
[BlackJAX](https://github.com/blackjax-devs/blackjax): MCMC and SMC
samplers for JAX.

**The JAX ecosystem**

[Equinox](https://github.com/patrick-kidger/equinox): neural networks
and PyTree modules.
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical
differential equation solvers.
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): shape and
dtype annotations for arrays.
[ArviZ](https://github.com/arviz-devs/arviz): exploratory analysis of
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
  [Rauch, Tung, and Striebel (1965)](https://doi.org/10.2514/3.3166).
- Nonlinear Gaussian filtering:
  [Schmidt (1966)](https://doi.org/10.1016/B978-1-4831-6716-9.50011-4) and
  [Julier (2002)](https://doi.org/10.1109/ACC.2002.1025369).
- Particle filters: [Gordon, Salmond, and Smith (1993)](https://doi.org/10.1049/ip-f-2.1993.0015),
  [Pitt and Shephard (1999)](https://doi.org/10.1080/01621459.1999.10474153),
  [Doucet, Godsill, and Andrieu (2000)](https://doi.org/10.1023/A:1008935410038),
  and [Liu and West (2001)](https://doi.org/10.1007/978-1-4757-3437-9_10).
- Static and parameter inference:
  [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x)
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

### Numerical validation references

The linear Kalman and RTS outputs are independently validated against
[Dynamax 1.0.2](https://github.com/probml/dynamax/releases/tag/1.0.2)
and
[statsmodels 0.14.6](https://github.com/statsmodels/statsmodels/releases/tag/v0.14.6);
the details are recorded with the
[frozen linear fixture](https://github.com/michaelellis003/smcx/blob/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/tests/_kalman_reference.py).

The extended and unscented Kalman outputs are independently validated against
[Stone Soup 1.9.1](https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1),
cross-checked with
[Dynamax 1.0.2](https://github.com/probml/dynamax/releases/tag/1.0.2),
and checked against
[SciPy 1.18.0](https://github.com/scipy/scipy/releases/tag/v1.18.0)
innovation log densities. Exact commits, environments, licenses, and
observed differences are recorded with the frozen
[extended](https://github.com/michaelellis003/smcx/blob/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/tests/_extended_kalman_reference.py)
and
[unscented](https://github.com/michaelellis003/smcx/blob/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/tests/_unscented_kalman_reference.py)
fixtures.

The auxiliary-guided runner recipe is distributionally cross-checked against
particles 0.4's pinned
[`AuxiliaryPF` composition](https://github.com/nchopin/particles/blob/c5fcb0b6d34b3c8efea6f6dc21d73e0e91287d9f/particles/state_space_models.py#L352-L428)
and [SMC bookkeeping](https://github.com/nchopin/particles/blob/c5fcb0b6d34b3c8efea6f6dc21d73e0e91287d9f/particles/core.py#L299-L359).

These projects are numerical comparison implementations, not code
lineage; no implementation code was copied or translated.

## Contributing

Contributions are welcome. See
[`CONTRIBUTING.md`](https://github.com/michaelellis003/smcx/blob/main/CONTRIBUTING.md)
for the development setup and pull-request conventions.

## License

smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
