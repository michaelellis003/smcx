# Contributing

Thanks for your interest in smcx.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pre-commit install
```

The suite runs on CPU by default (float64). On Apple silicon with
the `metal` extra installed, `SMCX_TEST_PLATFORM=mps uv run pytest`
runs it on the GPU in float32.

## Development

- Tests first: new behavior arrives together with the test that
  specifies it.
- `uv run pytest -v --cov` runs the suite; coverage must stay above
  the `fail_under` threshold in `pyproject.toml`.
- `uv run pre-commit run --all-files` must pass before you push. It
  runs Ruff lint and format (80 columns, single quotes, Google
  docstrings), the ty type checker, and the license-header check.
- Every `.py` file carries the Apache-2.0 header; `make license`
  adds it for you.

## Pull requests

- For anything larger than a small fix, open an issue first so the
  approach is agreed before you write code.
- Keep PRs small — under roughly 400 changed lines.
- PRs are squash-merged and the PR title becomes the commit
  message, so the title must follow
  [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, ...). A CI check enforces this.
  Versioning and releases are automated from these messages by
  python-semantic-release.
- CI (`ci-pass`) must be green before merge. Workflow runs on PRs
  from new contributors wait for maintainer approval before they
  start.

## Releases

Releases are automated and gated. After CI passes on `main`, the
release job waits for a manual approval that attests the full suite
passed on Metal GPU hardware; it then versions, tags, and publishes
to PyPI via trusted publishing.
