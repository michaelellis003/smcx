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

## Branching and pull requests

The repo runs trunk-based: `main` is the only long-lived branch and
every change lands through a short-lived branch and a squash-merged
PR.

- Branch from up-to-date `main`, named `<type>/<short-summary>`
  where `<type>` is the Conventional Commit type your change will
  carry (`fix/resampling-clamp`, `docs/quickstart-typo`).
- Keep branches short-lived; rebase on `main` rather than merging
  it in. Squash-merging keeps `main` linear either way.
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

Releases are fully automated: python-semantic-release reads the
commit types on `main`, so a merged `fix:` or `feat:` PR publishes
to PyPI (patch or minor) within minutes of CI going green, and
`docs:`/`test:`/`chore:` PRs release nothing. Your PR title is
therefore a release decision — maintainers may adjust the type
during review. Never edit version fields by hand; the release bot
owns them.
