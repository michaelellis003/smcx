# smcx

[![CI](https://github.com/michaelellis003/smcx/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smcx)](https://pypi.org/project/smcx/)
[![License](https://img.shields.io/github/license/michaelellis003/smcx)](LICENSE)

Sequential Monte Carlo for Apple silicon, built on [MLX](https://github.com/ml-explore/mlx).

The MLX-native sibling of [smcjax](https://github.com/michaelellis003/smcjax):
same algorithms, same API shape, different substrate. smcjax runs on JAX/XLA;
smcx targets the unified-memory architecture of M-series Macs.

## Thesis

Apple silicon's unified memory means zero-copy data movement between CPU and GPU.
General-purpose MCMC (NUTS) is a poor fit for this hardware — chains are
latency-bound and sequential, and Metal has no float64, which breaks Hamiltonian
integration on exactly the hierarchical models people reach for a PPL to fit.

SMC is the opposite case:

- **Throughput-shaped**: particle propagation, weighting, and resampling are
  embarrassingly parallel and vmap-friendly.
- **float32-tolerant**: no symplectic integration of ill-conditioned dynamics.
- **Unified-memory-friendly**: no PCIe staging, zero-copy host-side
  checks and diagnostics mid-filter, and full particle histories in
  one address space — the classic GPU-particle-filter pain points
  disappear on a laptop.
- **Unserved**: no MLX-native SMC exists. jax-metal is unmaintained; the only
  MLX sampler package (samplex) has no SMC and no PPL layer.

Open research question this project exists to probe: *which sampling algorithms
change character when CPU and GPU share memory with zero transfer cost?*

## Kill test (do this first)

Before building anything else: run the same particle-filter workloads in smcx
(MLX, M-series) and smcjax (JAX-CPU) at 10⁴–10⁶ particles. Same algorithms,
same API — a true apples-to-apples benchmark, under the pre-registered
success criterion in [benchmarks/PROTOCOL.md](benchmarks/PROTOCOL.md).
If the GPU doesn't show a clear win, the thesis dies quietly and cheaply.

## v0 scope

Mirror smcjax's core, MLX-native:

- Bootstrap (SIR) particle filter, with guided, auxiliary and Liu-West
  to follow
- Resamplers: multinomial, systematic, stratified, residual
  (differentiable resampling comes later — see ROADMAP)
- Adaptive tempering for static-model SMC (v0.2)
- A small set of distributions (~8, not 40) sufficient for state-space models
  and tempered targets
- Diagnostics/ArviZ export via numpy (port smcjax's diagnostics module)

Out of scope (deliberately): general PPL / effect handlers, NUTS,
NumPyro feature parity.

## Design

The v0 architecture — one internal Feynman-Kac core beneath a
smcjax-parity flat API — is documented in
[docs/design/v0-design.md](docs/design/v0-design.md). Individual
decisions (why native inverse-CDF resamplers, the float32 numerics
policy, explicit RNG keys) live as ADRs in [docs/adr/](docs/adr/).
Priorities and non-goals: [ROADMAP.md](ROADMAP.md).

## Known MLX hazards

Verified against mlx 0.32
([full audit](docs/research/mlx-audit.md)):

- `mx.random.categorical(num_samples=…)` is O(N·M) memory — unusable
  for resampling; smcx uses an inverse-CDF kernel with an in-library
  binary-search `searchsorted` (upstream: open invitation in
  mlx#1255, no implementation yet)
- No `lgamma`/`digamma` (PR declined upstream) — smcx ships a Lanczos
  `lgamma`; `mx.erf`/`mx.erfinv` do exist
- float64 raises on GPU; the CPU stream supports f64 for diagnostics,
  with device-pinning care
- Keyless RNG inside `mx.compile` is silently frozen — explicit keys
  everywhere
- No scan primitive — filter loops are Python over `mx.compile`d steps

## Development

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/michaelellis003/smcx.git
cd smcx
uv sync
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

A `Makefile` covers common tasks:

```bash
make test        # lint + pytest
make lint        # ruff check, format check, license headers, ty
make format      # add license headers, ruff format, ruff fix
make docs        # build docs
```

Releases are automated: `python-semantic-release` reads conventional
commits on merge to main, bumps the version, tags, and publishes.

## Acknowledgments

smcx's design draws on the SMC ecosystem: [smcjax](https://github.com/michaelellis003/smcjax)
(the sibling library and parity contract),
[particles](https://github.com/nchopin/particles) and Chopin &
Papaspiliopoulos's *An Introduction to Sequential Monte Carlo* (the
Feynman-Kac architecture), [BlackJAX](https://github.com/blackjax-devs/blackjax)
(the resampling contract), [Dynamax](https://github.com/probml/dynamax)
(container conventions), TensorFlow Probability (criterion/trace
hooks), and design lessons from PyMC, FilterPy, pfilter, pyfilter,
Stone Soup, pomp, nimbleSMC, ArviZ, and
[samplex](https://github.com/tedwards2412/samplex) (the MLX prior
art). See `CITATION.cff` for formal references and
`docs/research/licensing.md` for the full provenance record.

## Status

Alpha. The core works: weights, four native resamplers, the
Feynman-Kac loop, `bootstrap_filter` (with inputs channel and
missing-data support), and `simulate` — 85 tests against exact
Kalman oracles.

**Kill-test verdict (2026-07-14, pre-registered): the thesis holds
weakly.** At matched, oracle-gated accuracy, MLX-GPU beat JAX-CPU in
all 12 grid cells (1.1–4.8×), clearing the pre-registered ≥3× bar on
the resampling-bound LGSSM workload (4.0×/4.8× at 10⁵/10⁶ particles)
but not yet on the compute-heavier SV and tracking workloads —
where full-history materialization (up to 12 GB at 10⁶×T=500) is the
measured limiter, not compute. Full data:
[benchmarks/results/2026-07-14-kill-test.md](benchmarks/results/2026-07-14-kill-test.md).
Claim accordingly: a real but workload-dependent GPU advantage today,
with the O(1)-memory history option as the known next lever.
