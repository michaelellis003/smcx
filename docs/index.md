# smcjax

Sequential Monte Carlo and particle filtering in JAX.

**smcjax** extends [Dynamax](https://github.com/probml/dynamax) and
[BlackJAX](https://github.com/blackjax-devs/blackjax) with particle
filters and Bayesian workflow diagnostics that neither library provides.
All filters are JIT-compiled via `jax.lax.scan` and GPU-ready.

## Features

- Bootstrap (SIR) particle filter
- Auxiliary particle filter (Pitt & Shephard, 1999)
- Liu-West filter for joint state-parameter estimation
- Forward simulation from state-space models
- Diagnostics: weighted summaries, ESS traces, particle diversity,
  log evidence increments, replicated log-ML, log Bayes factors, CRPS
- All functions are `jit`- and `vmap`-compatible

## Installation

```bash
pip install smcjax
```

Or from source:

```bash
git clone https://github.com/michaelellis003/smcjax.git
cd smcjax
uv sync
```

Install the pre-commit hooks (one-time):

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

## Development

A `Makefile` collects the common development tasks:

| Target | What it does |
|---|---|
| `make test` | Lint, then run pytest |
| `make lint` | Ruff check, format check, license header check, ty |
| `make format` | Add license headers, ruff format, ruff fix |
| `make license` | Add missing SPDX license headers |
| `make docs` | Build documentation |
| `make serve-docs` | Serve documentation locally |
| `make install` | `uv sync` |
| `make clean` | `git clean` (preserves `.venv`) |

## License

Apache-2.0. See
[LICENSE](https://github.com/michaelellis003/smcjax/blob/main/LICENSE)
for the full text.
