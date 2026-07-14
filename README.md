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
- **Unified-memory-hungry**: particle counts (10⁵–10⁶+) that don't fit
  comfortably in discrete-GPU workflows run flat-out on a laptop.
- **Unserved**: no MLX-native SMC exists. jax-metal is unmaintained; the only
  MLX sampler package (samplex) has no SMC and no PPL layer.

Open research question this project exists to probe: *which sampling algorithms
change character when CPU and GPU share memory with zero transfer cost?*

## Kill test (do this first)

Before building anything else: run the same particle-filter workload in smcx
(MLX, M-series) and smcjax (JAX-CPU) at 10⁵–10⁶ particles. Same algorithms,
same API — a true apples-to-apples benchmark. If unified memory doesn't show
a clear win, the thesis dies quietly and cheaply.

## v0 scope

Mirror smcjax's core, MLX-native:

- Bootstrap (SIR) particle filter, with auxiliary and Liu-West to follow
- Resamplers: multinomial, systematic, stratified — plus differentiable resampling
- Adaptive tempering for static-model SMC
- A small set of distributions (~8, not 40) sufficient for state-space models
  and tempered targets
- Diagnostics/ArviZ export via numpy (port smcjax's diagnostics module)

Out of scope (deliberately): general PPL / effect handlers, NUTS,
NumPyro feature parity.

## Design notes vs smcjax

- smcjax uses `jax.lax.scan` for filter loops; MLX has no scan primitive —
  filter loops are plain Python over `mx.compile`-able step functions
- smcjax delegates resampling to BlackJAX; smcx implements resamplers natively
- No float64 on Metal: log-weight arithmetic must be f32-safe
  (log-sum-exp shifting, careful ESS computation)

## Known MLX hazards

- No `lgamma`/`digamma`/`erfinv` natively — Gamma/Beta/StudentT log-probs need
  Lanczos-style approximations or custom Metal kernels
- `vmap` coverage gaps, especially around `random.split`

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

## Status

Pre-alpha. Nothing works yet.
