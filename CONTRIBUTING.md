# Contributing

Contributions are welcome. For a user-visible defect or substantial
change, open an issue first to discuss the problem and scope.

## Setup

smcx requires Python 3.11 or later and
[`uv`](https://docs.astral.sh/uv/). Fork the repository, then install
the development environment:

```bash
git clone https://github.com/your-username/smcx.git
cd smcx
uv sync
uv run pre-commit install
```

The test suite uses CPU and float64 by default. On Apple silicon, install
the Metal backend with `uv sync --extra metal`; then
`SMCX_TEST_PLATFORM=mps uv run pytest` runs the suite on the physical
GPU in float32.

## Code changes

Write a failing test before changing behavior, then make the smallest
implementation that passes it. Before opening a pull request, run:

```bash
uv run pre-commit run --all-files
uv run ty check
uv run pytest --cov --cov-report=term-missing
make docs
```

Coverage must remain above the threshold in `pyproject.toml`. Every
Python file carries an Apache-2.0 header; `make license` adds it.

## Documentation changes

Build the site with `make docs`, or preview it at
`http://localhost:8000` with:

```bash
make serve-docs
```

## Pull requests

Branch from current `main` using `<type>/<issue-id>-<short-summary>`
when an issue exists, or `<type>/<short-summary>` for small maintenance
work. Rebase rather than merging `main`, and keep the change focused;
pull requests should normally stay below 400 changed lines.

The pull-request title becomes the squash commit and must follow
[Conventional Commits](https://www.conventionalcommits.org/), for
example `fix: handle empty observations` or `docs: clarify callbacks`.
This is also a release decision: `fix` publishes a patch, `feat`
publishes a minor release, and documentation, tests, and chores do not
publish a package. CI must pass before merge.
